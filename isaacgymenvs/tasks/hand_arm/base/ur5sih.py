import hydra
from omegaconf import DictConfig
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.gymtorch import *
from isaacgymenvs.tasks.hand_arm.base.configurable_vec_task import ConfigurableVecTask
from isaacgymenvs.tasks.hand_arm.utils.actionables import Actionable
from isaacgymenvs.tasks.hand_arm.utils.observables import LowDimObservable, PosObservable, PoseObservable
from isaacgymenvs.tasks.hand_arm.utils.callbacks import  ActionableCallback, ObservableCallback

from isaacgym.torch_utils import scale, to_torch, quat_mul, quat_conjugate
import matplotlib
from typing import *
import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from urdfpy import URDF
import time

from functools import partial

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
import actionlib
from control_msgs.msg import \
    FollowJointTrajectoryAction,\
    FollowJointTrajectoryGoal
import tf
import tabulate


class Ur5Sih(ConfigurableVecTask):
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
            robot_filename: str = '../assets/hand_arm/robot/hand_arm_collision_is_visual.urdf'
        ) -> None:
        
        self._robot_filename = robot_filename
        self.cfg_base = self._acquire_base_cfg()
        self._acquire_robot_urdf()
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, True)

    def _acquire_base_cfg(self) -> DictConfig:
        return hydra.compose(config_name='task/Ur5SihBase.yaml')['task']
    
    def _acquire_robot_urdf(self) -> None:
        self._urdf_robot = URDF.load(self._robot_filename)

        #self.ur5sih_dof_names = [j.name for j in self._urdf_robot.joints if j.joint_type != "fixed"]
        #self.ur5sih_dof_count = len(self.ur5sih_dof_names)
        self.ur5sih_actuated_dof_names = [t.joints[0].name for t in self._urdf_robot.transmissions]
        #self.ur5sih_actuated_dof_count = len(self.ur5sih_actuated_dof_names)
        self.ur5sih_body_names = [l.name for l in self._urdf_robot.links]

        self.ur5sih_fingertip_body_names = [l.name for l in self._urdf_robot.links if l.name.endswith("fingertip")]
        self.ur5sih_fingertip_count = len(self.ur5sih_fingertip_body_names)

        self.ur5sih_body_meshes = {name: link.collision_mesh for name, link in zip(self.ur5sih_body_names, self._urdf_robot.links) if link.collision_mesh}
        self.ur5sih_body_areas = {name: link.collision_mesh.area for name, link in zip(self.ur5sih_body_names, self._urdf_robot.links) if link.collision_mesh}

    def _acquire_robot_asset(self, rootpath: str, filename: str) -> None:
        robot_asset = self.gym.load_asset(self.sim, rootpath, filename, self.robot_asset_options)

        self.ur5sih_asset = robot_asset

        self.ur5sih_dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.ur5sih_dof_count = len(self.ur5sih_dof_names)

        self.ur5sih_actuated_dof_indices = [self.gym.find_asset_dof_index(robot_asset, name) for name in self.ur5sih_actuated_dof_names]
        self.ur5sih_rigid_body_count = self.gym.get_asset_rigid_body_count(robot_asset)
        self.ur5sih_rigid_shape_count = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.ur5sih_rigid_body_names = self.gym.get_asset_rigid_body_names(robot_asset)

        


        self.ur5sih_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.ur5sih_dof_lower_limits = []
        self.ur5sih_dof_upper_limits = []
        for dof_index in range(self.ur5sih_dof_count):
            self.ur5sih_dof_lower_limits.append(self.ur5sih_dof_props["lower"][dof_index])
            self.ur5sih_dof_upper_limits.append(self.ur5sih_dof_props["upper"][dof_index])
            self.ur5sih_dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_POS
            self.ur5sih_dof_props["stiffness"][dof_index] = self.cfg_base.asset.dof_properties.prop_gain[dof_index]
            self.ur5sih_dof_props["damping"][dof_index] = self.cfg_base.asset.dof_properties.deriv_gain[dof_index]
        self.ur5sih_dof_lower_limits = to_torch(self.ur5sih_dof_lower_limits, device=self.device)
        self.ur5sih_dof_upper_limits = to_torch(self.ur5sih_dof_upper_limits, device=self.device)
        self.ur5sih_actuated_dof_lower_limits = self.ur5sih_dof_lower_limits[self.ur5sih_actuated_dof_indices]
        self.ur5sih_actuated_dof_upper_limits = self.ur5sih_dof_upper_limits[self.ur5sih_actuated_dof_indices]

    def create_robot_actor(self, env_ptr, env_index, disable_self_collisions: bool, segmentation_id: int = 1):
        collision_filter = 0b1 if disable_self_collisions else 0b0
        actor_handle = self.gym.create_actor(env_ptr, self.ur5sih_asset, gymapi.Transform(), 'robot', env_index, collision_filter, segmentation_id)
        self.gym.set_actor_dof_properties(env_ptr, actor_handle, self.ur5sih_dof_props)
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
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        return asset_options
    
    def actuated_dofs_to_all_dofs(self, actuated_angles: torch.Tensor) -> torch.Tensor:
        batch_size, num_actuated_dofs = actuated_angles.shape
        assert num_actuated_dofs == self.actuated_dof_count
        return torch.matmul(actuated_angles, self.mimic_multiplier.t()) + self.mimic_offset.expand(batch_size, -1)

    def register_actionables(self) -> None:
        super().register_actionables()

        self.register_actionable(
            Actionable(
                name="ur5_relative_joint_pos",
                size=6,
                callback=ActionableCallback(
                    post_init=self._init_ur5_joint_pos_controller,
                    post_reset=self._reset_ur5_joint_pos_controller,
                    pre_step=partial(self._step_ur5_joint_pos_controller, mode="relative"),
                ),
            )
        )

        self.register_actionable(
            Actionable(
                name="sih_absolute_servo_pos",
                size=5,
                callback=ActionableCallback(
                    post_init=self._init_sih_servo_pos_controller,
                    post_reset=self._reset_sih_servo_pos_controller,
                    pre_step=partial(self._step_sih_servo_pos_controller, mode="absolute"),
                )
            )
        )

        self.register_actionable(
            Actionable(
                name="sih_smoothed_relative_servo_pos",
                size=5,
                callback=ActionableCallback(
                    post_init=self._init_sih_servo_pos_controller,
                    post_reset=self._reset_sih_servo_pos_controller,
                    pre_step=partial(self._step_sih_servo_pos_controller, mode="smoothed_relative"),
                )
            )
        )

        # self.register_actionable(
        #     Actionable(
        #         name="sih_ema_relative_joint_pos",
        #         size=5,
        #         callback=ActionableCallback(
        #             post_init=self._init_sih_joint_pos_controller,
        #             post_reset=self._reset_sih_joint_pos_controller,
        #             pre_step=partial(self._step_sih_joint_pos_controller, mode="relative"),
        #         ),
        #     )
        # )

    def register_observables(self) -> None:
        super().register_observables()

        # Register UR5 observables.
        self.register_observable(
            PoseObservable(
                name="ur5_flange_pose",
                size=7,
                get_state=lambda: self.ur5_flange_pose,
                callback=ObservableCallback(
                    post_init=self._init_ur5_flange_pose,
                    post_step=self._refresh_ur5_flange_pose,
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="ur5_joint_state",
                size=12,
                get_state=lambda: self.ur5_joint_state,
                callback=ObservableCallback(
                    post_init=self._init_ur5_joint_state,
                    post_step=self._refresh_ur5_joint_state,
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="ur5_joint_pos",
                size=6,
                get_state=lambda: self.ur5_joint_state[:, 0:6],
                requires=["ur5_joint_state"],
            )
        )

        # Register SIH fingertip observables.
        self.register_observable(
            PosObservable(
                name="sih_fingertip_pos",
                size=15,
                get_state=lambda: self.sih_fingertip_pos,
                callback=ObservableCallback(
                    post_init=lambda: (self._init_sih_fingertip_indices(), setattr(self, "sih_fingertip_pos", torch.zeros((self.num_envs, 5, 3), device=self.device))),
                    post_step=lambda: self.sih_fingertip_pos.copy_(self.body_pos[:, self.sih_fingertip_body_env_indices, 0:3]),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="sih_fingertip_quat",
                size=20,
                get_state=lambda: self.sih_fingertip_quat,
                callback=ObservableCallback(
                    post_init=lambda: (self._init_sih_fingertip_indices(), setattr(self, "sih_fingertip_quat", torch.zeros((self.num_envs, 5, 4), device=self.device))),
                    post_step=lambda: self.sih_fingertip_quat.copy_(self.body_quat[:, self.sih_fingertip_body_env_indices, 0:4]),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="sih_fingertip_linvel",
                size=15,
                get_state=lambda: self.sih_fingertip_linvel,
                callback=ObservableCallback(
                    post_init=lambda: (self._init_sih_fingertip_indices(), setattr(self, "sih_fingertip_linvel", torch.zeros((self.num_envs, 5, 3), device=self.device))),
                    post_step=lambda: self.sih_fingertip_linvel.copy_(self.body_linvel[:, self.sih_fingertip_body_env_indices, 0:3]),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="sih_fingertip_angvel",
                size=15,
                get_state=lambda: self.sih_fingertip_angvel,
                callback=ObservableCallback(
                    post_init=lambda: (self._init_sih_fingertip_indices(), setattr(self, "sih_fingertip_angvel", torch.zeros((self.num_envs, 5, 3), device=self.device))),
                    post_step=lambda: self.sih_fingertip_angvel.copy_(self.body_angvel[:, self.sih_fingertip_body_env_indices, 0:3]),
                )
            )
        )

        self.register_observable(
            LowDimObservable(
                name="dof_position_targets",
                size=17,
                get_state=lambda: self.dof_position_targets.to(self.device),
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "dof_position_targets", torch.zeros((self.num_envs, 17), device=self.device)),
                ),
                required=True
            )
        )
    
    def _init_ur5_joint_pos_controller(self) -> None:
        self.ur5_joint_pos_target = torch.zeros((self.num_envs, 6), device=self.device)
        self.ur5_joint_lower_limits = self.ur5sih_actuated_dof_lower_limits[:6]
        self.ur5_joint_upper_limits = self.ur5sih_actuated_dof_upper_limits[:6]
        
        if self.cfg_base.ros.activate:
            self.ur5_joint_names = self.ur5sih_actuated_dof_names[:6]
            self.ur5_joint_trajectory_client = actionlib.SimpleActionClient(
            "/scaled_pos_joint_traj_controller/follow_joint_trajectory", FollowJointTrajectoryAction
            )
            self.ur5_goal_status_text = ['PENDING', 'ACTIVE', 'PREEMPTED', 'SUCCEEDED', 'ABORTED', 'REJECTED', 'PREEMPTING', 'RECALLING', 'RECALLED', 'LOST']

    def _reset_ur5_joint_pos_controller(self, env_ids) -> None:
        self.ur5_joint_pos_target[env_ids] = self.dof_pos[env_ids, :self.ur5sih_dof_count][:, self.ur5sih_actuated_dof_indices][:, 0:6]

        if self.cfg_base.ros.activate:
            input("Press Enter to reset the UR5 arm.")
            self._publish_ur5_joint_pos_controller(secs=10, nsecs=0)
            self.ur5_joint_trajectory_client.wait_for_result()
            input("UR5 arm reset. Press Enter to start episode.")

    def _step_ur5_joint_pos_controller(self, actions: torch.Tensor, mode: str, action_scale: float = 1.0) -> None:
        if mode == "absolute":
            self.ur5_joint_pos_target[:] = scale(
                actions.to(self.device), self.ur5_joint_lower_limits, self.ur5_joint_upper_limits
                )
        elif mode == "relative":
            self.ur5_joint_pos_target[:] += self.dt * action_scale * actions.to(self.device)

        self.dof_position_targets[:, 0:6] = self.ur5_joint_pos_target  # This tensor sets the targets in the simulation.

        if self.cfg_base.ros.activate:
            self._publish_ur5_joint_pos_controller(secs=0, nsecs=int(self.dt * 1e9))
             
    def _publish_ur5_joint_pos_controller(self, secs: int, nsecs: int, verbose: bool = False) -> None:
        self.ur5_joint_trajectory_client.cancel_all_goals()

        position_target_point = JointTrajectoryPoint()
        position_target_point.positions = self.ur5_joint_pos_target[0].cpu().numpy()
        position_target_point.time_from_start.secs = secs
        position_target_point.time_from_start.nsecs = nsecs

        joint_trajectory_msg = FollowJointTrajectoryGoal()
        joint_trajectory_msg.goal_time_tolerance = rospy.Time(0.1)
        joint_trajectory_msg.trajectory.header = Header()
        joint_trajectory_msg.trajectory.header.stamp = rospy.Time.now()
        joint_trajectory_msg.trajectory.joint_names = self.ur5_joint_names
        joint_trajectory_msg.trajectory.points.append(position_target_point)

        self.ur5_joint_trajectory_client.send_goal(joint_trajectory_msg)
        goal_status = self.joint_trajectory_client.get_state()

        if verbose:
            print(f"Trajectory goal {self.goal_status_text[goal_status]}.")
      
        if goal_status > 3:
            raise RuntimeError(f"Trajectory goal {self.goal_status_text[goal_status]}.")

    def _init_sih_servo_pos_controller(self) -> None:
        self.sih_servo_commands = torch.zeros((self.num_envs, 5), device=self.device)
        self.sih_servo_commands_lower_limits = torch.Tensor([0, -2000, -1250, -400, -1350]).to(self.device)
        self.sih_servo_commands_upper_limits = torch.Tensor([2650, 250, 1450, 2300, 1000]).to(self.device)

        self.thumb_proximal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-1850, -1175, -975, -600, -225]).to(self.device), torch.Tensor([-1.51, -1.31, -1.175, -0.6, 0.]).unsqueeze(-1).to(self.device)))
        self.thumb_distal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-1318.125, -906.25, -200]).to(self.device), torch.Tensor([-1.235, -0.855, 0.]).unsqueeze(-1).to(self.device)))
        self.thumb_proximal_coef = -625.0

        self.index_proximal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-1250, -250, 150, 350, 540, 730, 1085, 1400]).to(self.device), torch.Tensor([-1.53, -1.4425, -1.315, -1.25, -1.18, -1.15, -0.6, 0.]).unsqueeze(-1).to(self.device)))
        self.index_distal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-408.606, 793.515, 1400]).to(self.device), torch.Tensor([-1.665, -0.735, 0]).unsqueeze(-1).to(self.device)))
        self.index_proximal_coef = -582.61

        self.middle_proximal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-500, 500, 1350, 1625, 1700, 1980, 2240]).to(self.device), torch.Tensor([-1.571, -1.445, -1.055, -0.91, -0.9, -0.48, 0.]).unsqueeze(-1).to(self.device)))
        self.middle_distal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([442.6, 1147, 1750.6, 2240]).to(self.device), torch.Tensor([-1.65, -1.125, -0.62, 0.]).unsqueeze(-1).to(self.device)))
        self.middle_proximal_coef = -600.0

        self.ring_proximal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-1050, -500, -250, 0, 370, 500, 700, 940]).to(self.device), torch.Tensor([-1.571, -1.45, -1.35, -1.225, -0.95, -0.9, -0.533, 0.]).unsqueeze(-1).to(self.device)))
        self.ring_distal_spline = NaturalCubicSpline(natural_cubic_spline_coeffs(torch.Tensor([-719, 408.8, 686.8, 939.2]).to(self.device), torch.Tensor([-1.64, -0.69, -0.425, 0.]).unsqueeze(-1).to(self.device)))
        self.ring_proximal_coef = -488.0

        self.sih_actuated_dof_names = self.ur5sih_actuated_dof_names[6:]
        self.sih_dof_names = self.ur5sih_dof_names[6:]

        self.sih_smoothed_actions = torch.zeros((self.num_envs, 5), device=self.device)

    def _reset_sih_servo_pos_controller(self, env_ids) -> None:
        self.sih_servo_commands[env_ids] = self.sih_servo_commands_upper_limits  # Upper limits represent open hand with thumb facing downwards.

        self.dof_position_targets[env_ids, 6:] = 0.
        self.sih_smoothed_actions[env_ids] = 0.

        if self.cfg_base.ros.activate:
            input("Press Enter to reset the SIH hand.")
            self._publish_sih_servo_pos_controller()
            time.sleep(2)
            input("SIH hand reset. Press Enter to start episode.")

    def _publish_sih_servo_pos_controller(self, verbose: bool = False) -> None:
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = self.sih_joint_names
        joint_state_msg.position = self.sih_joint_commmands[0].cpu().numpy()
        self.sih_joint_target_publisher.publish(joint_state_msg)
    
    def _step_sih_servo_pos_controller(self, actions: torch.Tensor, mode: str, alpha: float = 0.8) -> None:
        if mode == "absolute":
            self.sih_servo_commands[:] = scale(
                actions.to(self.device), self.sih_servo_commands_lower_limits, self.sih_servo_commands_upper_limits
                )
        
        elif mode == "relative":
            self.sih_servo_commands[:] += 100 * actions.to(self.device)
            self.sih_servo_commands[:] = torch.clamp(self.sih_servo_commands, self.sih_servo_commands_lower_limits, self.sih_servo_commands_upper_limits)

        elif mode == "smoothed_relative":
            self.sih_smoothed_actions[:] = alpha * actions.to(self.device) + (1 - alpha) * self.sih_smoothed_actions
            self.sih_servo_commands[:] += 100 * self.sih_smoothed_actions
            self.sih_servo_commands[:] = torch.clamp(self.sih_servo_commands, self.sih_servo_commands_lower_limits, self.sih_servo_commands_upper_limits)

        else:
            assert False, f"Unknown mode {mode}."

        # Map servo commands to joint positions.
        sih_joint_pos = torch.zeros((self.num_envs, 11), device=self.device)

        sih_joint_pos[:, self.sih_dof_names.index("thumb_opposition")] = (-1.571 / 2675) * self.sih_servo_commands[:, 0]

        sih_joint_pos[:, self.sih_dof_names.index("thumb_flexion")] = -self.thumb_proximal_spline.evaluate(self.sih_servo_commands[:, 1]).squeeze(-1)
        thumb_distal_t_values = self.sih_servo_commands[:, 1] + self.thumb_proximal_coef * self.dof_pos[:, self.ur5sih_dof_names.index("thumb_flexion")]
        sih_joint_pos[:, self.sih_dof_names.index("th_inter_to_th_distal")] = -self.thumb_distal_spline.evaluate(thumb_distal_t_values).squeeze(-1)

        sih_joint_pos[:, self.sih_dof_names.index("index_finger")] = self.index_proximal_spline.evaluate(self.sih_servo_commands[:, 2]).squeeze(-1)
        index_distal_t_values = self.sih_servo_commands[:, 2] + self.index_proximal_coef * self.dof_pos[:, self.ur5sih_dof_names.index("index_finger")]
        sih_joint_pos[:, self.sih_dof_names.index("if_proximal_to_if_distal")] = self.index_distal_spline.evaluate(index_distal_t_values).squeeze(-1)

        sih_joint_pos[:, self.sih_dof_names.index("middle_finger")] = self.middle_proximal_spline.evaluate(self.sih_servo_commands[:, 3]).squeeze(-1)
        middle_distal_t_values = self.sih_servo_commands[:, 3] + self.middle_proximal_coef * self.dof_pos[:, self.ur5sih_dof_names.index("middle_finger")]
        sih_joint_pos[:, self.sih_dof_names.index("mf_proximal_to_mf_distal")] = self.middle_distal_spline.evaluate(middle_distal_t_values).squeeze(-1)

        sih_joint_pos[:, self.sih_dof_names.index("ring_finger")] = self.ring_proximal_spline.evaluate(self.sih_servo_commands[:, 4]).squeeze(-1)
        ring_distal_t_values = self.sih_servo_commands[:, 4] + self.ring_proximal_coef * self.dof_pos[:, self.ur5sih_dof_names.index("ring_finger")]
        sih_joint_pos[:, self.sih_dof_names.index("rf_proximal_to_rf_distal")] = self.ring_distal_spline.evaluate(ring_distal_t_values).squeeze(-1)

        sih_joint_pos[:, self.sih_dof_names.index("palm_to_lf_proximal")] = sih_joint_pos[:, self.sih_dof_names.index("ring_finger")]
        sih_joint_pos[:, self.sih_dof_names.index("lf_proximal_to_lf_distal")] = sih_joint_pos[:, self.sih_dof_names.index("rf_proximal_to_rf_distal")]

        self.dof_position_targets[:, 6:] = sih_joint_pos  # This tensor sets the targets in the simulation.


        if self.cfg_base.ros.activate:
            self._publish_sih_servo_pos_controller()

    def _init_ur5_flange_pose(self) -> None:
        self.ur5_flange_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.ur5_flange_body_env_index = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.ur5sih_handles[0], "flange", gymapi.DOMAIN_ENV)

        if self.cfg_base.ros.activate and not hasattr(self, "transform_subscriber"):
            self.transform_subscriber = tf.TransformListener()
    
    def _refresh_ur5_flange_pose(self, ros_compare: bool = True) -> None:
        if self.cfg_base.ros.activate:
            pos, quat = self.tf_sub.lookupTransform('base_link', 'flange', rospy.Time(0))
            self.ur5_flange_pose[:, 0:3] = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            self.ur5_flange_pose[:, 3:7] = torch.tensor(quat, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

            if ros_compare:
                pos_difference = torch.abs(self.ur5_flange_pose[:, 0:3] - self.body_pos[:, self.ur5_flange_body_env_index, 0:3])
                quat_difference = quat_mul(self.ur5_flange_pose[:, 3:7], quat_conjugate(self.body_quat[:, self.ur5_flange_body_env_index, 0:4]))
                angle_difference = 2.0 * torch.asin(torch.clamp(torch.norm(quat_difference[:, 0:3], p=2, dim=-1), max=1.0))

                table = [["Position", self.ur5_flange_pose[:, 0:3], self.body_pos[:, self.ur5_flange_body_env_index, 0:3], pos_difference, torch.norm(pos_difference, p=2, dim=-1)],
                         ["Quaternion", self.ur5_flange_pose[:, 3:7], self.body_quat[:, self.ur5_flange_body_env_index, 0:4], quat_difference, angle_difference],]
                print(tabulate(table, headers=["UR5-flange", "Real", "Sim", "Difference", "Difference Mag."]))

                if torch.any(pos_difference > 0.01):
                    raise ValueError("UR5 flange position from ROS and simulation do not match.")
                
                if torch.any(angle_difference > 0.01):
                    raise ValueError("UR5 flange orientation from ROS and simulation do not match.")
        
        else:
            self.ur5_flange_pose[:, 0:3] = self.body_pos[:, self.ur5_flange_body_env_index, 0:3]
            self.ur5_flange_pose[:, 3:7] = self.body_quat[:, self.ur5_flange_body_env_index, 0:4]
    
    def _init_ur5_joint_state(self) -> None:
        self.ur5_joint_state = torch.zeros((self.num_envs, 2 * 6), device=self.device)

        if self.cfg_base.ros.activate:
            self._ur5_joint_state_msg = None
            self._ur5_joint_state_sub = rospy.Subscriber("/joint_states", JointState, self._ur5_joint_state_callback)

    def _ur5_joint_state_callback(self, msg: JointState) -> None:
        # Check if received joint state is for th UR5.
        if set(msg.name) == set(("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")):
            # Store latest message
            self._ur5_joint_state_msg = msg

    def _refresh_ur5_joint_state(self, ros_compare: bool = True) -> None:
        if self.cfg_base.ros.activate:
            if self._ur5_joint_state_msg is None:
                raise ValueError("No joint state message received yet for UR5.")
            
            joint_state = torch.tensor(self._ur5_joint_state_msg.position + self._ur5_joint_state_msg.velocity, dtype=torch.float32, device=self.device)
            self.ur5_joint_state[:] = joint_state.unsqueeze(0).repeat(self.num_envs, 1)

            if ros_compare:
                # Check age of last received message.
                last_message_delay = (rospy.Time.now() - self._ur5_joint_state_msg.header.stamp).to_sec()

                # Compare joint state from ROS with simulation.
                pos_difference = torch.abs(self.ur5_joint_state[:, 0:self.arm_dof_count] - self.dof_pos[:, 0:6])
                vel_difference = torch.abs(self.ur5_joint_state[:, self.arm_dof_count:] - self.dof_vel[:, 0:6])

                table = [["Position", self.ur5_joint_state[:, 0:self.arm_dof_count], self.dof_pos[:, 0:6], pos_difference],
                 ["Velocity", self.ur5_joint_state[:, self.arm_dof_count:], self.dof_vel[:, 0:6], vel_difference],]

                print(tabulate(table, headers=["UR5",f"Real (delay={last_message_delay} [s])", "Sim", "Difference"]))

                if torch.any(pos_difference > 0.01):
                    raise ValueError("UR5 joint position from ROS and simulation do not match.")
                
                if last_message_delay > 0.1:
                    raise ValueError("Last UR5 joint state message is older than 0.1s.")
        
        else:
            self.ur5_joint_state[:, 0:6] = self.dof_pos[:, 0:6]
            self.ur5_joint_state[:, 6:] = self.dof_vel[:, 0:6]
         
    def _init_sih_fingertip_indices(self) -> None:
        if self.cfg_base.ros.activate:
            raise ValueError("SIH fingertip observations not supported with ROS.")
        
        sih_fingertip_body_names = ["thumb_fingertip", "index_fingertip", "middle_fingertip", "ring_fingertip", "little_fingertip"]
        self.sih_fingertip_body_env_indices = [self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.ur5sih_handles[0], n, gymapi.DOMAIN_ENV) for n in sih_fingertip_body_names]
    
    def _reset_ur5sih(self, env_ids, reset_dof_pos: List) -> None:
        assert len(env_ids) == self.num_envs, "All environments should be reset simultaneously."

        self.dof_pos[env_ids, :self.ur5sih_dof_count] = torch.tensor(reset_dof_pos, device=self.device).unsqueeze(0).repeat((len(env_ids), 1))
        self.dof_vel[env_ids, :self.ur5sih_dof_count] = 0.0

        self.dof_position_targets[env_ids] = self.dof_pos[env_ids, :self.ur5sih_dof_count]

        reset_indices = self.ur5sih_actor_indices[env_ids]

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_position_targets), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

    def visualize_contact_force(self, colormap: str = 'viridis', vmax: float = 2.5) -> None:
        cmap = matplotlib.colormaps[colormap]
        contact_force_mag = torch.clip(torch.norm(self.contact_force, dim=2) / vmax, max=1.0)
        rgb = cmap(contact_force_mag.cpu().numpy())[..., 0:3]
        num_rigid_bodies = self.ur5sih_rigid_body_count
        for rb_index in range(num_rigid_bodies):
            for env_index in range(self.num_envs):
                self.gym.set_rigid_body_color(
                    self.env_ptrs[env_index], self.ur5sih_handles[env_index], rb_index, gymapi.MESH_VISUAL, gymapi.Vec3(*rgb[env_index, rb_index]),
                )
