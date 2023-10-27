from isaacgym import gymapi
from isaacgym.torch_utils import *
import os
from urdfpy import URDF
from omegaconf import DictConfig
from typing import *


class URDFRobot:
    def __init__(self, rootpath: str, filename: str) -> None:
        self.rootpath = rootpath
        self.filename = filename
        self.urdf_asset = URDF.load(os.path.join(self.rootpath, self.filename))
        self.actuated_dof_count = self.actuator_count

    @property
    def dof_count(self) -> int:
        return len(self.dof_names)
    
    @property
    def dof_names(self) -> List[str]:
        return [j.name for j in self.urdf_asset.joints if j.joint_type != "fixed"]
    
    @property
    def actuator_count(self) -> int:
        return len(self.urdf_asset.actuated_joints)
    
    @property
    def actuated_dof_names(self) -> List[str]:
        return [t.joints[0].name for t in self.urdf_asset.transmissions]
    
    @property
    def fingertip_body_names(self) -> List[str]:
        return [l.name for l in self.urdf_asset.links if l.name.endswith("fingertip")]
    
    @property
    def fingertip_count(self) -> int:
        return len(self.fingertip_body_names)
    
    def attach_simulation(self, gym, sim, dof_cfg: DictConfig, device: torch.device) -> gymapi.Asset:
        self.gym = gym
        self.sim = sim
        self.device = device
        self.isaacgym_asset = self.gym.load_asset(self.sim, self.rootpath, self.filename, self.asset_options)

        self.actuated_dof_indices = [self.gym.find_asset_dof_index(self.isaacgym_asset, name) for name in self.actuated_dof_names]
        
        self.dof_props = self.gym.get_asset_dof_properties(self.isaacgym_asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []
        self.actuated_dof_lower_limits = []
        self.actuated_dof_upper_limits = []

        for dof_index, dof_name in enumerate(self.dof_names):
            self.dof_lower_limits.append(self.dof_props["lower"][dof_index])
            self.dof_upper_limits.append(self.dof_props["upper"][dof_index])

            self.dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_POS
            self.dof_props["stiffness"][dof_index] = dof_cfg.prop_gain[dof_index]
            self.dof_props["damping"][dof_index] = dof_cfg.deriv_gain[dof_index]

        for actuated_dof_name in self.actuated_dof_names:
            actuated_dof_index = self.gym.find_asset_dof_index(self.isaacgym_asset, actuated_dof_name)
            self.actuated_dof_lower_limits.append(self.dof_props["lower"][actuated_dof_index])
            self.actuated_dof_upper_limits.append(self.dof_props["upper"][actuated_dof_index])
            
        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.actuated_dof_lower_limits = to_torch(self.actuated_dof_lower_limits, device=self.device)
        self.actuated_dof_upper_limits = to_torch(self.actuated_dof_upper_limits, device=self.device)

        self.populate_actuator_joint_mapping()

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
        asset_options.convex_decomposition_from_submeshes = True
        #asset_options.linear_damping = 0.01
        return asset_options

    @property
    def rigid_body_count(self) -> int:
        return self.gym.get_asset_rigid_body_count(self.isaacgym_asset)
    
    @property
    def rigid_shape_count(self) -> int:
        return self.gym.get_asset_rigid_shape_count(self.isaacgym_asset)
    
    def populate_actuator_joint_mapping(self) -> None:
        # Retrieve actuator joint mapping.
        self.mimic_multiplier = torch.zeros((self.dof_count, self.actuator_count), device=self.device)
        self.mimic_offset = torch.zeros(self.dof_count, device=self.device)

        for i, actuated_joint_name in enumerate(self.actuated_dof_names):
            index = self.gym.find_asset_dof_index(self.isaacgym_asset, actuated_joint_name)
            self.mimic_multiplier[index, i] = 1.0  # Direct mapping for actuated joints.

        for joint in self.urdf_asset.joints:
            if joint.mimic:
                index = self.gym.find_asset_dof_index(self.isaacgym_asset, joint.name)
                mimic_index = self.actuated_dof_names.index(joint.mimic.joint)

                print("joint:", joint.name, "mimics:", joint.mimic.joint, "multiplier:", joint.mimic.multiplier, "offset:", joint.mimic.offset)

                if self.mimic_multiplier[:, mimic_index].sum() == 0:
                    raise ValueError(f"Mimicked joint {joint.mimic.joint.name} is not actuated.")
                
                self.mimic_multiplier[index, mimic_index] = joint.mimic.multiplier
                self.mimic_offset[index] = joint.mimic.offset
    
    def actuated_to_all(self, actuated_angles: torch.Tensor) -> torch.Tensor:
        batch_size, num_actuated_dofs = actuated_angles.shape
        assert num_actuated_dofs == self.actuator_count
        return torch.matmul(actuated_angles, self.mimic_multiplier.t()) + self.mimic_offset.expand(batch_size, -1)
    
    def create_actor(self, env_ptr, env_index, filter: int = 1, segmentation_id: int = 1):
        actor_handle = self.gym.create_actor(env_ptr, self.isaacgym_asset, gymapi.Transform(), 'robot', env_index, filter, segmentation_id)
        self.gym.set_actor_dof_properties(env_ptr, actor_handle, self.dof_props)
        return actor_handle
   
