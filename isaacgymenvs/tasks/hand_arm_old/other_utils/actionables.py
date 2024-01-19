from abc import ABC
from dataclasses import dataclass, field, InitVar
import networkx as nx
import torch
from typing import Callable, Dict, List, Tuple, Sequence, Optional
from isaacgymenvs.tasks.hand_arm.base.camera_sensor import ImageType, CameraSensorProperties
from isaacgymenvs.tasks.hand_arm.utils.entities import ReinforcementLearningEntity


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
    def urdf(self, urdf: URDF) -> None:
        # Load robot description as urdfpy.URDF object.
        self._urdf = urdf
        
        # Acquire basic information about the robot.
        self.dof_names = [j.name for j in self.urdf.joints if j.joint_type != "fixed"]
        self.dof_count = len(self.dof_names)
        self.actuated_dof_names = [t.joints[0].name for t in self.urdf.transmissions]
        self.actuated_dof_count = len(self.actuated_dof_names)
        self.body_names = [l.name for l in self.urdf.links]

        self.fingertip_body_names = [l.name for l in urdf.links if l.name.endswith("fingertip")]
        self.fingertip_count = len(self.fingertip_body_names)

        self.body_meshes = {name: link.collision_mesh for name, link in zip(self.body_names, urdf.links) if link.collision_mesh}
        self.body_areas = {name: link.collision_mesh.area for name, link in zip(self.body_names, urdf.links) if link.collision_mesh}

        print("self.body_areas:", self.body_areas)
        input()

        use_reduced_robot = True
        if use_reduced_robot:
            self.body_areas.pop("shoulder_link")
            self.body_areas.pop("upper_arm_link")
            self.body_areas.pop("forearm_link")
            self.body_areas.pop("wwrist_1_link")
            self.body_areas.pop("wrist_2_link")
            self.body_areas.pop("wrist_3_link")
            self.body_areas.pop("palm")
            self.body_meshes.pop("shoulder_link")
            self.body_meshes.pop("upper_arm_link")
            self.body_meshes.pop("forearm_link")
            self.body_meshes.pop("wwrist_1_link")
            self.body_meshes.pop("wrist_2_link")
            self.body_meshes.pop("wrist_3_link")
            self.body_meshes.pop("palm")

        density = 2000.0  # Number of samples per square meter.
        self.num_body_surface_samples = [int(density * area) for area in self.body_areas.values()]
        print("self.num_body_surface_samples:", self.num_body_surface_samples)
        
    @property
    def body_surface_samples(self) -> List[np.array]:
        body_surface_samples = []
        for body_index, body_mesh in enumerate(self.body_meshes.values()):
            body_surface_samples.append(np.array(trimesh.sample.sample_surface(body_mesh, count=self.num_body_surface_samples[body_index])[0]).astype(float))
        return body_surface_samples

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
        self.rigid_body_names = self.gym.get_asset_rigid_body_names(asset)

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
        assert num_actuated_dofs == self.actuated_dof_count
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

    def create_actor(self, env_ptr, env_index, filter: int = 1, segmentation_id: int = 1):
        actor_handle = self.gym.create_actor(env_ptr, self.asset, gymapi.Transform(), 'robot', env_index, filter, segmentation_id)
        self.gym.set_actor_dof_properties(env_ptr, actor_handle, self.dof_props)
        return actor_handle




# Actionables get access to the robot model they are supposed to actuate?
# Sounds reasonable.


class Actionable(ReinforcementLearningEntity, ABC):
    """Base class for actionable (things that can be actuated).

    Takes actions from the agent and applies them to the environment.
    
    Args:
        name (str): The name of the actionable.
        size (int): The size of the action tensor.
        set_state (callable): A callable that applies the action to the environment.
    """
    def __init__(
        self,
        name: str,
        size: int,
        robot: Robot,
        set_state: Callable[[torch.Tensor], None]
    ) -> None:
        super().__init__(name, size, lambda: None, set_state)


class JointPosActionable(Actionable):
    def __init__(
        self,
        name: str,
        size: int,
        set_state: Callable[[torch.Tensor], None],
        actuated_to_all: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ) -> None:
        super().__init__(name, size, set_state)

    def set_state(self, tensor_data: torch.Tensor) -> None:
        actuated_joint_pos_targets = tensor_data
        self.check_tensor_data(actuated_joint_pos_targets)
        joint_pos_targets = self.actuated_to_all(actuated_joint_pos_targets)
        self._set_state(tensor_data)
    

