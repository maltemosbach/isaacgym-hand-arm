from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from omegaconf import DictConfig
import os
import torch
from typing import *
from urdfpy import URDF



class ActorMixin:
    def _compute_num_actions(self) -> int:
        return 11
    
    def acquire_action_tensors(self) -> None:
        self.current_dof_pos_targets = torch.zeros((self.num_envs, self.controller.dof_count), device=self.device)
        self.current_actuated_dof_pos_targets = torch.zeros((self.num_envs, self.controller.actuator_count), device=self.device)
        self.previous_actuated_dof_pos_targets = torch.zeros((self.num_envs, self.controller.actuator_count), device=self.device)
        self.actuated_dof_pos = self.dof_pos[:, self.controller.actuated_dof_indices]
        self.actuated_dof_vel = self.dof_vel[:, self.controller.actuated_dof_indices]

        # End-effector (EEF) body is the one controled via inverse kinematics if that is the control mode.
        self.eef_body_env_index = self.gym.find_actor_rigid_body_index(
            self.env_ptrs[0], self.controller_handles[0], self.cfg_base.control.end_effector.body_name, gymapi.DOMAIN_ENV
        )
        self.eef_body_pos = self.body_pos[:, self.eef_body_env_index, 0:3]
        self.eef_body_quat = self.body_quat[:, self.eef_body_env_index, 0:4]
        self.eef_body_linvel = self.body_linvel[:, self.eef_body_env_index, 0:3]
        self.eef_body_angvel = self.body_angvel[:, self.eef_body_env_index, 0:3]

    def refresh_action_tensors(self) -> None:
        pass

    def apply_controls(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().to(self.device)

        # actions 0-6 arm arm DoFs. 6 is thumb opposition 7 is thumb flextion 8 is index finger 9 is middle finger 10 is ring finger

        if self.cfg_base.control.type == "joint":

            if self.cfg_base.control.mode == "absolute":

                actuated_joint_targets = scale(
                self.actions, self.controller.actuated_dof_lower_limits, self.controller.actuated_dof_upper_limits
                )

                self.current_dof_pos_targets[:, :] = scale(
                    joint_actions, self.controller.dof_lower_limits, self.controller.dof_upper_limits
                )

                self.current_dof_pos_targets[:, :] = (
                    self.cfg_base.control.moving_average * self.current_dof_pos_targets
                    + (1.0 - self.cfg_base.control.moving_average) * self.previous_dof_pos_targets
                )
            
            elif self.cfg_base.control.mode == "relative":
                self.current_actuated_dof_pos_targets[:, 0:self.arm_dof_count] += self.dt * self.cfg_base.control.joint.arm_action_scale * self.actions[:, 0:self.arm_dof_count]
                self.current_actuated_dof_pos_targets[:, self.arm_dof_count:] += self.dt * self.cfg_base.control.joint.hand_action_scale * self.actions[:, self.arm_dof_count:]

                self.current_actuated_dof_pos_targets = tensor_clamp(
                    self.current_actuated_dof_pos_targets, self.controller.actuated_dof_lower_limits, self.controller.actuated_dof_upper_limits
                )
                self.current_dof_pos_targets[:, :] = self.controller.actuated_to_all(self.current_actuated_dof_pos_targets)

            else:
                assert False

            self.current_dof_pos_targets[:, :] = tensor_clamp(
                self.current_dof_pos_targets, self.controller.dof_lower_limits, self.controller.dof_upper_limits
            )

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.current_dof_pos_targets))
            self.previous_actuated_dof_pos_targets[:, :] = self.current_actuated_dof_pos_targets[:, :]
        
        else:
            assert False


class Robot:
    def __init__(self, rootpath: str, asset_cfg: DictConfig) -> None:
        self.rootpath = rootpath
        self.filename = asset_cfg.robot
        self.dof_cfg = asset_cfg.dof_properties
        self.urdf = URDF.load(os.path.join(self.rootpath, self.filename))

    @property
    def urdf(self) -> URDF:
        return self._urdf
    
    @urdf.setter
    def urdf(self, loadpath: Union[str, bytes, os.PathLike]) -> None:
        # Load robot description as urdfpy.URDF object.
        self._urdf = URDF.load(loadpath)
        
        # Acquire basic information about the robot.
        self.dof_names = [j.name for j in self.urdf.joints if j.joint_type != "fixed"]
        self.dof_count = len(self.dof_names)
        self.actuated_dof_names = [t.joints[0].name for t in self.urdf.transmissions]
        self.actuated_dof_count = len(self.actuated_dof_names)
        self.body_names = [l.name for l in self.urdf.links]

    def attach_simulation(self, gym, sim, device: torch.device) -> None:
        self.gym = gym
        self.sim = sim
        self.device = device
        self.asset = self.gym.load_asset(self.sim, self.rootpath, self.filename, self.asset_options)

    @property
    def asset(self) -> gymapi.Asset:
        return self._asset
    
    @asset.setter
    def asset(self, asset: gymapi.Asset) -> None:
        self._asset = asset

        self.actuated_dof_indices = [self.gym.find_asset_dof_index(asset, name) for name in self.actuated_dof_names]
        self.rigid_body_count = self.gym.get_asset_rigid_body_count(asset)
        self.rigid_shape_count = self.gym.get_asset_rigid_shape_count(asset)

        self.dof_props = self.gym.get_asset_dof_properties(asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for dof_index in range(self.dof_count):
            self.dof_lower_limits.append(self.dof_props["lower"][dof_index])
            self.dof_upper_limits.append(self.dof_props["upper"][dof_index])
            self.dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_POS
            self.dof_props["stiffness"][dof_index] = self.dof_cfg.prop_gain[dof_index]
            self.dof_props["damping"][dof_index] = self.dof_cfg.deriv_gain[dof_index]
        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.actuated_dof_lower_limits = self.dof_lower_limits[self.actuated_dof_indices]
        self.actuated_dof_upper_limits = self.dof_upper_limits[self.actuated_dof_indices]

        # Retrieve actuator joint mapping.
        self.mimic_multiplier = torch.zeros((self.dof_count, self.actuated_dof_count), device=self.device)
        self.mimic_offset = torch.zeros(self.dof_count, device=self.device)
        for i, actuated_joint_name in enumerate(self.actuated_dof_names):
            index = self.gym.find_asset_dof_index(asset, actuated_joint_name)
            self.mimic_multiplier[index, i] = 1.0  # Direct mapping for actuated joints.
        for joint in self.urdf.joints:
            if joint.mimic:
                index = self.gym.find_asset_dof_index(asset, joint.name)
                mimic_index = self.actuated_dof_names.index(joint.mimic.joint)
                if self.mimic_multiplier[:, mimic_index].sum() == 0:
                    raise ValueError(f"Mimicked joint {joint.mimic.joint.name} is not actuated.")
                self.mimic_multiplier[index, mimic_index] = joint.mimic.multiplier
                self.mimic_offset[index] = joint.mimic.offset

    @property
    def asset_options(self) -> gymapi.AssetOptions:
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
        assert num_actuated_dofs == self.actuator_count
        return torch.matmul(actuated_angles, self.mimic_multiplier.t()) + self.mimic_offset.expand(batch_size, -1)
    
    def map_actions_to_dof_pos_targets(actions: torch.Tensor) -> torch.Tensor:
        if self.cfg_base.control.type == "joint":

            if self.cfg_base.control.mode == "absolute":

                actuated_joint_targets = scale(
                self.actions, self.controller.actuated_dof_lower_limits, self.controller.actuated_dof_upper_limits
                )

                print("actuated_joint_targets:", actuated_joint_targets)
                
                self.current_dof_pos_targets[:, :] = scale(
                    joint_actions, self.controller.dof_lower_limits, self.controller.dof_upper_limits
                )

                self.current_dof_pos_targets[:, :] = (
                    self.cfg_base.control.moving_average * self.current_dof_pos_targets
                    + (1.0 - self.cfg_base.control.moving_average) * self.previous_dof_pos_targets
                )
            
            elif self.cfg_base.control.mode == "relative":
                self.current_actuated_dof_pos_targets[:, 0:self.arm_dof_count] += self.dt * self.cfg_base.control.joint.arm_action_scale * self.actions[:, 0:self.arm_dof_count]
                self.current_actuated_dof_pos_targets[:, self.arm_dof_count:] += self.dt * self.cfg_base.control.joint.hand_action_scale * self.actions[:, self.arm_dof_count:]

                self.current_actuated_dof_pos_targets = tensor_clamp(
                    self.current_actuated_dof_pos_targets, self.controller.actuated_dof_lower_limits, self.controller.actuated_dof_upper_limits
                )
                self.current_dof_pos_targets[:, :] = self.controller.actuated_to_all(self.current_actuated_dof_pos_targets)

            else:
                assert False

            self.current_dof_pos_targets[:, :] = tensor_clamp(
                self.current_dof_pos_targets, self.controller.dof_lower_limits, self.controller.dof_upper_limits
            )
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.current_dof_pos_targets))
            self.previous_actuated_dof_pos_targets[:, :] = self.current_actuated_dof_pos_targets[:, :]
        
        else:
            assert False

