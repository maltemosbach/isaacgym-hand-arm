
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.hand_arm.base.base import HandArmBase
import hydra
from omegaconf import DictConfig
import numpy as np
import glob
import os
from typing import *
import trimesh
from urdfpy import URDF
import random
import torch
from isaacgym import gymapi, gymtorch, torch_utils, gymutil


class ObjectAsset:
    def __init__(self, gym, sim, asset_root: str, asset_file: str) -> None:
        self._gym = gym
        self._sim = sim
        self._asset_root = asset_root
        self._asset_file = asset_file
        self.name = asset_file.split('/')[-1].split('.')[0]

        self._acquire_asset_options()
        self.asset = gym.load_asset(sim, asset_root, asset_file, self.asset_options)
        
    def _acquire_asset_options(self, vhacd_resolution: int = 100000) -> None:
        self._asset_options = gymapi.AssetOptions()
        self._asset_options.override_com = True
        self._asset_options.override_inertia = True
        self._asset_options.vhacd_enabled = True  # Enable convex decomposition
        self._asset_options.vhacd_params = gymapi.VhacdParams()
        self._asset_options.vhacd_params.resolution = vhacd_resolution

    def acquire_mesh(self) -> None:
        urdf = URDF.load(os.path.join(self._asset_root, self._asset_file))
        self.mesh = urdf.base_link.collision_mesh

    @property
    def asset_options(self) -> gymapi.AssetOptions:
        return self._asset_options

    @asset_options.setter
    def asset_options(self, asset_options: gymapi.AssetOptions) -> None:
        self._asset_options = asset_options

    @property
    def rigid_body_count(self) -> int:
        return self._gym.get_asset_rigid_body_count(self.asset)

    @property
    def rigid_shape_count(self) -> int:
        return self._gym.get_asset_rigid_shape_count(self.asset)

    @property
    def start_pose(self) -> gymapi.Transform:
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.5)
        return start_pose
    
    @property
    def surface_area(self) -> float:
        if not hasattr(self, "mesh"):
            self.acquire_mesh()
        return self.mesh.area

    def sample_points_from_mesh(self, num_samples: int) -> np.array:
        if not hasattr(self, "mesh"):
            self.acquire_mesh()
        points = np.array(trimesh.sample.sample_surface(
            self.mesh, count=num_samples)[0]).astype(float)
        return points
    
    def find_bounding_box_from_mesh(self) -> Tuple[np.array, np.array]:
        if not hasattr(self, "mesh"):
            self.acquire_mesh()
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        return to_origin, extents


class HandArmEnvMultiObject(HandArmBase):
    _env_cfg_path = 'task/HandArmEnvMultiObject.yaml'

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_env = self._acquire_env_cfg()
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_env_tensors()
        self.refresh_env_tensors()

    def _acquire_env_cfg(self) -> DictConfig:
        cfg_env = hydra.compose(config_name=self._env_cfg_path)['task']
        
        if cfg_env.bin.asset == 'no_bin':
            self.bin_info = {"extent": [[-0.15, -0.15, 0.0], [0.15, 0.15, 0.15]]}
        else:
            assert False

        return cfg_env

        
    

    def acquire_env_tensors(self):
        self.object_pos = self.root_pos[:, self.object_actor_env_indices, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_env_indices, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_env_indices, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_env_indices, 0:3]

        self.target_object_pos = self.object_pos[:, 0]
        self.target_object_quat = self.object_quat[:, 0]
        self.target_object_linvel = self.object_linvel[:, 0]
        self.target_object_angvel = self.object_angvel[:, 0]

    def refresh_env_tensors(self):
        # NOTE: Since object_pos, object_quat, etc. are obtained from the root state tensor through advanced slicing, they are separate tensors and have to be updated separately
        self.object_pos[:] = self.root_pos[:, self.object_actor_env_indices, 0:3]
        self.object_quat[:] = self.root_quat[:, self.object_actor_env_indices, 0:4]
        self.object_linvel[:] = self.root_linvel[:, self.object_actor_env_indices, 0:3]
        self.object_angvel[:] = self.root_angvel[:, self.object_actor_env_indices, 0:3]


        # Subsample target object from updated observations.
        self.target_object_pos = self.object_pos.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_quat = self.object_quat.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)
        self.target_object_linvel = self.object_linvel.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_angvel = self.object_angvel.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)


    def _acquire_objects(self) -> None:
        def solve_object_regex(regex: str, object_set: str) -> List[str]:
            root = os.path.join(self._asset_root, 'object_sets', 'urdf', object_set)
            ret_list = []
            regex_path_len = len(regex.split("/"))
            regex = os.path.normpath(os.path.join(root, regex))
            for path in glob.glob(regex):
                file_name = "/".join(path.split("/")[-regex_path_len:])
                if "." in file_name:
                    obj, extension = file_name.split(".")
                else:
                    obj = file_name
                    extension = ""
                if extension == "urdf":
                    ret_list.append(obj)
            return ret_list
        
        self.object_count = 0
        self.object_dict = {}
        for dataset in self.cfg_env.objects.dataset.keys():
            if isinstance(self.cfg_env.objects.dataset[dataset], str):
                object_names = [self.cfg_env.objects.dataset[dataset]]
            else:
                object_names = self.cfg_env.objects.dataset[dataset]
            dataset_object_list = []
            for object_name in object_names:
                if "*" in object_name:
                    dataset_object_list += solve_object_regex(object_name, dataset)
                else:
                    dataset_object_list.append(object_name)
            self.object_dict[dataset] = dataset_object_list
            self.object_count += len(dataset_object_list)
        
        self.objects = []
        for object_set, object_list in self.object_dict.items():
            for object_name in object_list:
                object = ObjectAsset(self.gym, self.sim, self._asset_root + 'object_sets', f'urdf/{object_set}/' + object_name + '.urdf')
                self.objects.append(object)

    def _create_envs(self) -> None:
        lower = gymapi.Vec3(-self.cfg_base.sim.env_spacing, -self.cfg_base.sim.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.sim.env_spacing, self.cfg_base.sim.env_spacing, self.cfg_base.sim.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset = self._acquire_robot()
        self._acquire_objects()

        self.robot_handles = []
        self.robot_actor_indices = []

        self.env_ptrs = []
        self.object_handles = [[] for _ in range(self.num_envs)]
        self.bin_handles = []
        self.object_actor_indices = [[] for _ in range(self.num_envs)]
        self.object_indices = []
        self.object_names = [[] for _ in range(self.num_envs)]

        self.goal_handles = []
        self.goal_actor_indices = []

        actor_count = 0
        for env_index in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)  # Create new env.

            # Select subset of size num_objects from all available objects.
            assert self.object_count >= self.cfg_env.objects.num_objects, \
                "Number of objects per environment cannot be larger that the total number of objects used."
            object_indices = random.sample(list(range(self.object_count)), self.cfg_env.objects.num_objects)
            self.object_indices.append(object_indices)
            self.object_names.append([self.objects[i].name for i in object_indices])
            objects_rigid_body_count = sum([o.rigid_body_count for o in [self.objects[i] for i in object_indices]])
            objects_rigid_shape_count = sum([o.rigid_shape_count for o in [self.objects[i] for i in object_indices]])

            # Aggregate all actors.
            max_rigid_bodies = self.gym.get_asset_rigid_body_count(robot_asset) + objects_rigid_body_count
            max_rigid_shapes = self.gym.get_asset_rigid_shape_count(robot_asset) + objects_rigid_shape_count
            self.gym.begin_aggregate(env_ptr, max_rigid_bodies, max_rigid_shapes, True)

            # Create robot actor.
            robot_handle = self.gym.create_actor(env_ptr, robot_asset, gymapi.Transform(), 'robot', env_index, 0, 1)
            self.robot_handles.append(robot_handle)
            self.robot_actor_indices.append(actor_count)
            actor_count += 1

            # Create object actors
            for i, object_index in enumerate(object_indices):
                used_object = self.objects[object_index]
                object_handle = self.gym.create_actor(env_ptr, used_object.asset, used_object.start_pose, used_object.name, env_index, 0, 3 + i)
                self.object_handles[env_index].append(object_handle)
                self.object_actor_indices[env_index].append(actor_count)
                actor_count += 1

            self.gym.end_aggregate(env_ptr)
            self.env_ptrs.append(env_ptr)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        self.robot_actor_indices = torch.tensor(self.robot_actor_indices, dtype=torch.int32, device=self.device)
        self.object_actor_indices = torch.tensor(self.object_actor_indices, dtype=torch.int32, device=self.device)

        self.object_actor_env_indices = [self.gym.find_actor_index(env_ptr, o.name, gymapi.DOMAIN_ENV) for o in [self.objects[i] for i in object_indices]]

        self.target_object_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.target_object_actor_env_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    # self.robot_actor_id_env = self.gym.find_actor_index(
    #     env_ptr, 'robot', gymapi.DOMAIN_ENV)
    # self.object_actor_id_env = [self.gym.find_actor_index(
    #     env_ptr, o.name, gymapi.DOMAIN_ENV)
    #     for o in [self.objects[idx] for idx in objects_idx]]

    def _disable_object_collisions(self, object_ids: List[int]):
        self._set_object_collisions(object_ids, collision_filter=-1)

    def _enable_object_collisions(self, object_ids: List[int]):
        self._set_object_collisions(object_ids, collision_filter=0)

    def _set_object_collisions(self, object_ids: List[int], collision_filter: int) -> None:
        print("object_ids: ", object_ids)

        print("self.object_handles:", self.object_handles)
        def set_collision_filter(env_id: int, actor_handle, collision_filter: int) -> None:
            print("env_id: ", env_id)
            actor_shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptrs[env_id], actor_handle)
            for shape_id in range(len(actor_shape_props)):
                actor_shape_props[shape_id].filter = collision_filter
            self.gym.set_actor_rigid_shape_properties(self.env_ptrs[env_id], actor_handle, actor_shape_props)

        # No tensor API to set actor rigid shape props, so a loop is required.
        for env_id in range(self.num_envs):
            for object_id in object_ids:
                set_collision_filter(env_id, self.object_handles[env_id][object_id], collision_filter)


    def objects_in_bin(self) -> torch.BoolTensor:
        x_lower = self.bin_info['extent'][0][0] + self.cfg_env.bin.pos[0]
        x_upper = self.bin_info['extent'][1][0] + self.cfg_env.bin.pos[0]
        y_lower = self.bin_info['extent'][0][1] + self.cfg_env.bin.pos[1]
        y_upper = self.bin_info['extent'][1][1] + self.cfg_env.bin.pos[1]
        z_lower = self.bin_info['extent'][0][2] + self.cfg_env.bin.pos[2]
        z_upper = self.bin_info['extent'][1][2] + self.cfg_env.bin.pos[2]
        in_bin = x_lower <= self.object_pos[..., 0]
        in_bin = torch.logical_and(in_bin, self.object_pos[..., 0] <= x_upper)
        in_bin = torch.logical_and(in_bin, y_lower <= self.object_pos[..., 1])
        in_bin = torch.logical_and(in_bin, self.object_pos[..., 1] <= y_upper)
        in_bin = torch.logical_and(in_bin, z_lower <= self.object_pos[..., 2])
        in_bin = torch.logical_and(in_bin, self.object_pos[..., 2] <= z_upper)
        return in_bin
    
    def visualize_bin_extent(self, env_id) -> None:
        bin_pose = gymapi.Transform(p=gymapi.Vec3(*self.cfg_env.bin.pos))
        bbox = gymutil.WireframeBBoxGeometry(torch.tensor(self.bin_info['extent']), pose=bin_pose, color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id], pose=gymapi.Transform())
