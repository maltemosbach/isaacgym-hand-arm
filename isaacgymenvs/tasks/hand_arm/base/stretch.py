import hydra
from omegaconf import DictConfig
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.gymtorch import *
from isaacgymenvs.tasks.hand_arm.base.configurable_vec_task import ConfigurableVecTask
from isaacgymenvs.tasks.hand_arm.utils.actionables import Actionable
from isaacgymenvs.tasks.hand_arm.utils.observables import LowDimObservable, PosObservable, PoseObservable, SyntheticPointcloudObservable
from isaacgymenvs.tasks.hand_arm.utils.callbacks import  ActionableCallback, ObservableCallback

from isaacgym.torch_utils import scale, to_torch, quat_mul, quat_conjugate, quat_apply
import matplotlib
from typing import *
import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from urdfpy import URDF
import time

from functools import partial

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Int32MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
import actionlib
from control_msgs.msg import \
    FollowJointTrajectoryAction,\
    FollowJointTrajectoryGoal
import tf
import tabulate
import numpy as np
import trimesh


class Stretch(ConfigurableVecTask):
    _ur5_reset_pos = [0.6985, -1.4106, 1.2932, 0.1174, 0.6983, 1.5708]

    def __init__(
            self,
            cfg: Dict[str, Any],
            rl_device: str,
            sim_device: str,
            graphics_device_id: int,
            headless: bool,
            virtual_screen_capture: bool,
            force_render: bool = True,
            robot_filename: str = '../assets/hand_arm/stretch/stretch.urdf'
        ) -> None:
        
        self._robot_filename = robot_filename
        self.cfg_base = self._acquire_base_cfg()
        self._acquire_robot_urdf()
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, True)

    def _acquire_base_cfg(self) -> DictConfig:
        return hydra.compose(config_name='task/StretchBase.yaml')['task']
    
    def _acquire_robot_urdf(self) -> None:
        self._urdf_robot = URDF.load(self._robot_filename)

        self.stretch_actuated_dof_names = [t.joints[0].name for t in self._urdf_robot.transmissions]
        self.stretch_body_names = [l.name for l in self._urdf_robot.links]

        self.stretch_fingertip_body_names = [l.name for l in self._urdf_robot.links if l.name.startswith("link_gripper_fingertip")]
        self.stretch_fingertip_count = len(self.stretch_fingertip_body_names)

    def _acquire_robot_asset(self, rootpath: str, filename: str) -> None:
        robot_asset = self.gym.load_asset(self.sim, rootpath, filename, self.robot_asset_options)

        self.stretch_asset = robot_asset

        self.stretch_dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.stretch_dof_count = len(self.stretch_dof_names)

        self.stretch_actuated_dof_indices = [self.gym.find_asset_dof_index(robot_asset, name) for name in self.stretch_actuated_dof_names]
        self.stretch_rigid_body_count = self.gym.get_asset_rigid_body_count(robot_asset)
        self.stretch_rigid_shape_count = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.stretch_rigid_body_names = self.gym.get_asset_rigid_body_names(robot_asset)


        self.stretch_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.stretch_dof_lower_limits = []
        self.stretch_dof_upper_limits = []
        for dof_index in range(self.stretch_dof_count):
            self.stretch_dof_lower_limits.append(self.stretch_dof_props["lower"][dof_index])
            self.stretch_dof_upper_limits.append(self.stretch_dof_props["upper"][dof_index])
            self.stretch_dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_POS
            self.stretch_dof_props["stiffness"][dof_index] = self.cfg_base.asset.dof_properties.prop_gain[dof_index]
            self.stretch_dof_props["damping"][dof_index] = self.cfg_base.asset.dof_properties.deriv_gain[dof_index]
        self.stretch_dof_lower_limits = to_torch(self.stretch_dof_lower_limits, device=self.device)
        self.stretch_dof_upper_limits = to_torch(self.stretch_dof_upper_limits, device=self.device)
        self.stretch_actuated_dof_lower_limits = self.stretch_dof_lower_limits[self.stretch_actuated_dof_indices]
        self.stretch_actuated_dof_upper_limits = self.stretch_dof_upper_limits[self.stretch_actuated_dof_indices]

    def create_robot_actor(self, env_ptr, env_index, disable_self_collisions: bool, segmentation_id: int = 1):
        collision_filter = 0b1 if disable_self_collisions else 0b0
        actor_handle = self.gym.create_actor(env_ptr, self.stretch_asset, gymapi.Transform(p=gymapi.Vec3(0.2, 0.175, 0.), r=gymapi.Quat(0., 0., 1., 0.)), 'robot', env_index, collision_filter, segmentation_id)
        self.gym.set_actor_dof_properties(env_ptr, actor_handle, self.stretch_dof_props)
        return actor_handle

    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""
        # Set time-step size and substeps.
        self.sim_params.dt = self.cfg_base.sim.dt
        self.sim_params.substeps = self.cfg_base.sim.num_substeps

        # Use GPU-pipeline if the simulation device is a GPU.
        self.sim_params.use_gpu_pipeline = self.device.startswith("cuda")

        # Set orientation and gravity vector.
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -self.cfg_base.sim.gravity_mag

        # Set PhysX parameters.
        self.sim_params.physx.use_gpu = self.device.startswith("cuda")
        self.sim_params.physx.num_position_iterations = self.cfg_base.sim.num_pos_iters
        self.sim_params.physx.num_velocity_iterations = self.cfg_base.sim.num_vel_iters

        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 ** 2  # default = 1024^2
        self.sim_params.physx.default_buffer_size_multiplier = 8  # default = 1

        self.sim = super().create_sim(compute_device=self.device_id,
                                      graphics_device=self.graphics_device_id,
                                      physics_engine=self.physics_engine,
                                      sim_params=self.sim_params)
        self._create_ground_plane()
        self._create_envs()  # defined in subclass

    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0  # default = 0.0
        plane_params.static_friction = 1.0  # default = 1.0
        plane_params.dynamic_friction = 1.0  # default = 1.0
        plane_params.restitution = 0.0  # default = 0.0
        self.gym.add_ground(self.sim, plane_params)
    
    @property
    def robot_asset_options(self) -> gymapi.AssetOptions:
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = False
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        return asset_options

    def register_actionables(self) -> None:
        super().register_actionables()

        self.register_actionable(
            Actionable(
                name="stretch_relative_joint_pos",
                size=5,
                callback=ActionableCallback(
                    post_init=self._init_stretch_joint_pos_controller,
                    post_reset=self._reset_stretch_joint_pos_controller,
                    pre_step=partial(self._step_stretch_joint_pos_controller, mode="relative"),
                ),
            )
        )

    def register_observables(self) -> None:
        super().register_observables()

        # Register Stretch fingertip observables.
        self.register_observable(
            PosObservable(
                name="stretch_fingertip_pos",
                size=6,
                get_state=lambda: self.stretch_fingertip_pos,
                callback=ObservableCallback(
                    post_init=lambda: (self._init_stretch_fingertip_indices(), setattr(self, "stretch_fingertip_pos", torch.zeros((self.num_envs, 2, 3), device=self.device))),
                    post_step=lambda: self.stretch_fingertip_pos.copy_(self.body_pos[:, self.stretch_fingertip_body_env_indices, 0:3]),
                )
            )
        )

        self.register_observable(
            LowDimObservable(
                name="stretch_fingertip_linvel",
                size=6,
                get_state=lambda: self.stretch_fingertip_linvel,
                callback=ObservableCallback(
                    post_init=lambda: (self._init_stretch_fingertip_indices(), setattr(self, "stretch_fingertip_linvel", torch.zeros((self.num_envs, 2, 3), device=self.device))),
                    post_step=lambda: self.stretch_fingertip_linvel.copy_(self.body_linvel[:, self.stretch_fingertip_body_env_indices, 0:3]),
                )
            )
        )

        self.register_observable(
            LowDimObservable(
                name="dof_position_targets",
                size=9,
                get_state=lambda: self.dof_position_targets.to(self.device),
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "dof_position_targets", torch.zeros((self.num_envs, 9), device=self.device)),
                ),
                required=True
            )
        )

    
    def _init_stretch_joint_pos_controller(self) -> None:
        self.stretch_joint_pos_target = torch.zeros((self.num_envs, 9), device=self.device)

    def _reset_stretch_joint_pos_controller(self, env_ids) -> None:
        self.stretch_joint_pos_target[env_ids] = self.dof_pos[env_ids]

    def _step_stretch_joint_pos_controller(self, actions: torch.Tensor, mode: str, action_scale: float = 0.25) -> None:
        if mode == "absolute":
            assert False
            self.stretch_joint_pos_target[:] = scale(
                actions.to(self.device), self.stretch_joint_lower_limits, self.stretch_joint_upper_limits
                )
        elif mode == "relative":
            self.stretch_joint_pos_target[:, 0:2] += self.dt * action_scale * actions.to(self.device)[:, 0:2]
            self.stretch_joint_pos_target[:, 2:6] += self.dt * action_scale * actions.to(self.device)[:, 2:3]
            self.stretch_joint_pos_target[:, 6:7] += self.dt * 8 * action_scale * actions.to(self.device)[:, 3:4]
            self.stretch_joint_pos_target[:, 7:9] += self.dt * 6 * action_scale * actions.to(self.device)[:, 4:5]
            #self.stretch_joint_pos_target[:] += self.dt * action_scale * actions.to(self.device)

        self.dof_position_targets[:] = self.stretch_joint_pos_target  # This tensor sets the targets in the simulation.

    def _init_stretch_flange_pose(self) -> None:
        self.stretch_flange_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.stretch_flange_body_env_index = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.stretch_handles[0], "flange", gymapi.DOMAIN_ENV)

        if self.cfg_base.ros.activate and not hasattr(self, "transform_subscriber"):
            self.transform_subscriber = tf.TransformListener()
         
    def _init_stretch_fingertip_indices(self) -> None:
        stretch_fingertip_body_names = ["fingertip_left", "fingertip_right"]
        self.stretch_fingertip_body_env_indices = [self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.stretch_handles[0], n, gymapi.DOMAIN_ENV) for n in stretch_fingertip_body_names]
    
    def _reset_stretch(self, env_ids, reset_dof_pos: List) -> None:
        assert len(env_ids) == self.num_envs, "All environments should be reset simultaneously."

        self.dof_pos[env_ids, :self.stretch_dof_count] = torch.tensor(reset_dof_pos, device=self.device).unsqueeze(0).repeat((len(env_ids), 1))
        self.dof_vel[env_ids, :self.stretch_dof_count] = 0.0

        self.dof_position_targets[env_ids] = self.dof_pos[env_ids, :self.stretch_dof_count]

        reset_indices = self.stretch_actor_indices[env_ids]

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_position_targets), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )
