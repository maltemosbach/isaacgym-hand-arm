from isaacgym import gymapi
from isaacgym.torch_utils import *
import os
from urdfpy import URDF
from omegaconf import DictConfig


class HandArmRobot:
    def __init__(self, gym, sim, rootpath: str, cfg: DictConfig, device: torch.device) -> None:
        self.gym = gym
        self.sim = sim
        self.rootpath = rootpath
        self.cfg = cfg
        self.filename = cfg.asset.robot
        self.device = device

        self.isaacgym_asset = self.acquire_isaacgym_asset()
        self.urdf_asset = self.acquire_urdf_asset()

        self.populate_actuator_joint_mapping()
        self.populate_dof_props()
        
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

    def acquire_isaacgym_asset(self) -> gymapi.Asset:
        isaacgym_asset = self.gym.load_asset(self.sim, self.rootpath, self.filename, self.asset_options)

        # Retrieve basic asset infos.
        self.rigid_body_count = self.gym.get_asset_rigid_body_count(isaacgym_asset)
        self.rigid_shape_count = self.gym.get_asset_rigid_shape_count(isaacgym_asset)
        self.dof_count = self.gym.get_asset_dof_count(isaacgym_asset)
        self.dof_names = self.gym.get_asset_dof_names(isaacgym_asset)

        return isaacgym_asset
    
    def acquire_urdf_asset(self) -> URDF:
        # NOTE: Isaac Gym does not parse the URDF actuators correctly and ignores mimic joints. Hence, manual parsing with urdfpy is required.
        urdf_asset = URDF.load(os.path.join(self.rootpath, self.filename))

        # Retrieve actuator information.
        print("dir(urdf_asset):", dir(urdf_asset))
        print("urdf_asset.transmissions:", urdf_asset.transmissions)

        actuators = [t.actuators[0] for t in urdf_asset.transmissions if t.actuators is not None]

        print("actuators:", actuators)


        # This is the ordering I want to got by when interpreting the actions
        actuator_names = [a.name for a in actuators]

        actuator_joints = [t.joints[0] for t in urdf_asset.transmissions if t.joints is not None]

        self.actuator_joint_names = [j.name for j in actuator_joints]

        print("actuator_names:", actuator_names)
        print("actuator_joint_names:", self.actuator_joint_names)


        actuated_joints = urdf_asset.actuated_joints
        self.actuator_count = len(actuated_joints)
        self.actuated_dof_names = [j.name for j in actuated_joints]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(self.isaacgym_asset, name) for name in self.actuated_dof_names]

        return urdf_asset
    
    def populate_actuator_joint_mapping(self) -> None:
        # Retrieve actuator joint mapping.
        self.mimic_multiplier = torch.zeros((self.dof_count, self.actuator_count), device=self.device)
        self.mimic_offset = torch.zeros(self.dof_count, device=self.device)

        for i, actuated_joint_name in enumerate(self.actuator_joint_names):
            index = self.gym.find_asset_dof_index(self.isaacgym_asset, actuated_joint_name)
            self.mimic_multiplier[index, i] = 1.0  # Direct mapping for actuated joints.

        for joint in self.urdf_asset.joints:
            if joint.mimic:
                index = self.gym.find_asset_dof_index(self.isaacgym_asset, joint.name)
                mimic_index = self.actuator_joint_names.index(joint.mimic.joint)

                if self.mimic_multiplier[:, mimic_index].sum() == 0:
                    raise ValueError(f"Mimicked joint {joint.mimic.joint.name} is not actuated.")
                
                self.mimic_multiplier[index, :] += joint.mimic.multiplier * self.mimic_multiplier[mimic_index, :]
                self.mimic_offset[index] = joint.mimic.offset

        print("self.mimic_multiplier:", self.mimic_multiplier)
        print("self.mimic_offset:", self.mimic_offset)

        import time
        time.sleep(5)
    
    
    def populate_dof_props(self) -> None:
        self.dof_props = self.gym.get_asset_dof_properties(self.isaacgym_asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []

        for dof_index, dof_name in enumerate(self.dof_names):
            self.dof_lower_limits.append(self.dof_props["lower"][dof_index])
            self.dof_upper_limits.append(self.dof_props["upper"][dof_index])

            self.dof_props["driveMode"][dof_index] = gymapi.DOF_MODE_POS
            self.dof_props["stiffness"][dof_index] = self.cfg.asset.dof_properties.prop_gain[dof_index]
            self.dof_props["damping"][dof_index] = self.cfg.asset.dof_properties.deriv_gain[dof_index]

            
        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)


        # Update DoF properties.
        #self.dof_props["driveMode"][:] = gymapi.DOF_MODE_POS

        #print("self.dof_props:", self.dof_props)
        #print("self.dof_props.dtype:", self.dof_props.dtype)
        #print("dir(self.dof_props):", dir(self.dof_props))



        #import time
        #time.sleep(1000)
    
    def actuated_to_all(self, actuated_angles: torch.Tensor) -> torch.Tensor:
        batch_size, num_actuated_dofs = actuated_angles.shape
        assert num_actuated_dofs == self.actuator_count
        return torch.matmul(actuated_angles, self.mimic_multiplier.t()) + self.mimic_offset.expand(batch_size, -1)
    

    def create_actor(self, env_ptr, env_index, filter: int = 0, segmentation_id: int = 1):
        actor_handle = self.gym.create_actor(env_ptr, self.isaacgym_asset, gymapi.Transform(), 'robot', env_index, filter, segmentation_id)
        self.gym.set_actor_dof_properties(env_ptr, actor_handle, self.dof_props)
        return actor_handle

        


   
