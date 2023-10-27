
from isaacgym import gymapi, gymutil
from isaacgymenvs.tasks.hand_arm.base.base import HandArmBase
from isaacgymenvs.tasks.hand_arm.base.observations import Observation
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
from scipy.spatial.transform import Rotation as R

from isaacgym.torch_utils import quat_apply, quat_mul


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

    def register_observations(self) -> None:
        super().register_observations()
        self.register_observation(
            'object_pos',
            Observation(
                size=(self.cfg_env.objects.num_objects * 3,),
                data=lambda: self.object_pos.flatten(1, 2),
                is_mandatory=True,  # Required to save initial object pose.
                acquire=lambda: setattr(self, "object_pos", self.root_pos[:, self.object_actor_env_indices, 0:3]),
                refresh=lambda: self.object_pos.copy_(self.root_pos[:, self.object_actor_env_indices, 0:3]),
                visualize=lambda: self.visualize_poses(self.object_pos, self.object_quat),
            )
        )
        self.register_observation(
            'object_quat',
            Observation(
                size=(self.cfg_env.objects.num_objects * 4,),
                data=lambda: self.object_quat.flatten(1, 2),
                is_mandatory=True,  # Required to save initial object pose.
                acquire=lambda: setattr(self, "object_quat", self.root_quat[:, self.object_actor_env_indices, 0:4]),
                refresh=lambda: self.object_quat.copy_(self.root_quat[:, self.object_actor_env_indices, 0:4]),
                visualize=lambda: self.visualize_poses(self.object_pos, self.object_quat), 
            )
        )
        self.register_observation(
            'object_linvel',
            Observation(
                size=(self.cfg_env.objects.num_objects * 3,),
                data=lambda: self.object_linvel.flatten(1, 2),
                is_mandatory=True,  # Required to compute rewards.
                acquire=lambda: setattr(self, "object_linvel", self.root_linvel[:, self.object_actor_env_indices, 0:3]),
                refresh=lambda: self.object_linvel.copy_(self.root_linvel[:, self.object_actor_env_indices, 0:3]),
            )
        )
        self.register_observation(
            'object_angvel',
            Observation(
                size=(self.cfg_env.objects.num_objects * 3,),
                data=lambda: self.object_angvel.flatten(1, 2),
                acquire=lambda: setattr(self, "object_angvel", self.root_angvel[:, self.object_actor_env_indices, 0:3]),
                refresh=lambda: self.object_angvel.copy_(self.root_angvel[:, self.object_actor_env_indices, 0:3]),
            )
        )

        self.register_observation(
            'target_object_pos',
            Observation(
                size=(3,),
                data=lambda: self.target_object_pos,
                is_mandatory=True,  # Required to compute reward and success criterion.
                acquire=lambda: setattr(self, "target_object_pos", self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                refresh=lambda: self.target_object_pos.copy_(self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                visualize=lambda: self.visualize_poses(self.target_object_pos, self.target_object_quat),
            )
        )
        self.register_observation(
            'target_object_quat',
            Observation(
                size=(4,),
                data=lambda: self.target_object_quat,
                acquire=lambda: setattr(self, "target_object_quat", self.root_quat.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)),
                refresh=lambda: self.target_object_quat.copy_(self.root_quat.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)),
                visualize=lambda: self.visualize_poses(self.target_object_pos, self.target_object_quat)
            )
        )

        self.register_observation(
            "fingertip_to_target_object_pos",
            Observation(
                size=(3 * self.controller.fingertip_count,),
                data=lambda: self.fingertip_to_target_object_pos.flatten(1, 2),
                acquire=lambda: setattr(self, "fingertip_to_target_object_pos", self.target_object_pos.unsqueeze(1).repeat(1, self.controller.fingertip_count, 1) - self.fingertip_pos),
                refresh=lambda: self.fingertip_to_target_object_pos.copy_(self.target_object_pos.unsqueeze(1).repeat(1, self.controller.fingertip_count, 1) - self.fingertip_pos),
                visualize=lambda: self.visualize_distance(self.fingertip_pos, self.fingertip_to_target_object_pos),
            )
        )

        self.register_observation(
            "object_bounding_box", 
            Observation(
                size=(self.cfg_env.objects.num_objects * 10,),  # pos, quat, extent
                data=lambda: self.object_bounding_box.flatten(1, 2),
                acquire=self._acquire_object_bounding_box,
                refresh=self._refresh_object_bounding_box,
                visualize=lambda: self.visualize_bounding_boxes(self.object_bounding_box[..., 0:3], self.object_bounding_box[..., 3:7], self.object_bounding_box[..., 7:10]),
            )
        )
        self.register_observation(
            "target_object_bounding_box", 
            Observation(
                size=(10,),
                data=lambda: self.target_object_bounding_box,
                requires=["object_bounding_box"],
                acquire=lambda: setattr(self, "target_object_bounding_box", torch.zeros((self.num_envs, 10)).to(self.device)),
                refresh=lambda: self.target_object_bounding_box.copy_(self.object_bounding_box.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 10)).squeeze(1)),
                visualize=lambda: self.visualize_bounding_boxes(self.target_object_bounding_box[..., 0:3], self.target_object_bounding_box[..., 3:7], self.target_object_bounding_box[..., 7:10]),
            )
        )

        self.register_observation(
            "object_synthetic_pointcloud",
            Observation(
                size=(self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 3,),
                key='pointcloud',
                data=lambda: self.object_synthetic_pointcloud,
                acquire=self._acquire_object_synthetic_pointcloud,
                refresh=self._refresh_object_synthetic_pointcloud,
                visualize=lambda: self.visualize_points(self.object_synthetic_pointcloud, size=0.0025)
            )
        )
        self.register_observation(
            "target_object_synthetic_pointcloud",
            Observation(
                size=(self.cfg_env.pointclouds.max_num_points, 3,),
                key='pointcloud',
                data=lambda: self.target_object_synthetic_pointcloud,
                requires=["object_synthetic_pointcloud"],
                acquire=lambda: setattr(self, "target_object_synthetic_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                refresh=lambda: self.target_object_synthetic_pointcloud.copy_(self.object_synthetic_pointcloud.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 4)).squeeze(1)),
                visualize=lambda: self.visualize_points(self.target_object_synthetic_pointcloud, size=0.0025)
            )
        )

        # self.register_observation(
        #     "object_synthetic_pointcloud_initial",
        #     Observation(
        #         size=(self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 3,),
        #         key='pointcloud',
        #         data=lambda: self.object_synthetic_pointcloud_initial,
        #         requires=["object_synthetic_pointcloud"],
        #         acquire=setattr(self, "object_synthetic_pointcloud_initial", torch.zeros((self.num_envs, self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
        #         refresh=self._refresh_object_synthetic_pointcloud_initial,
        #         visualize=lambda: self.visualize_points(self.object_synthetic_pointcloud, size=0.0025)
        #     )
        # )

        if self.cfg_task.rl.goal == "reposition":
            self.register_observation(
                "goal_pos",
                Observation(
                    size=(3,),
                    data=lambda: self.goal_pos,
                    is_mandatory=True,  # Required for reward computation.
                    acquire=lambda: setattr(self, "goal_pos", torch.zeros(self.num_envs, 3, device=self.device)),
                )
            )
            self.register_observation(
                "target_object_to_goal_pos",
                Observation(
                    size=(3,),
                    data=lambda: self.target_object_to_goal_pos,
                    acquire=lambda: setattr(self, "target_object_to_goal_pos", self.goal_pos - self.target_object_pos),
                    refresh=lambda: self.target_object_to_goal_pos.copy_(self.goal_pos - self.target_object_pos),
                    visualize=lambda: self.visualize_distance(self.target_object_pos, self.target_object_to_goal_pos),
                )
            )

    def _acquire_env_cfg(self) -> DictConfig:
        cfg_env = hydra.compose(config_name=self._env_cfg_path)['task']
        
        if cfg_env.bin.asset == 'no_bin':
            self.bin_info = {"extent": [[-0.25, -0.25, 0.0], [0.25, 0.25, 0.2]]}
        else:
            bin_info_path = f'../../assets/hand_arm/{cfg_env.bin.asset}/bin_info.yaml'
            self.bin_info = hydra.compose(config_name=bin_info_path)
            self.bin_info = self.bin_info['']['']['']['']['']['']['assets']['hand_arm'][cfg_env.bin.asset]

        return cfg_env

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

        self._acquire_objects()

        self.controller_handles = []
        self.controller_actor_indices = []

        self.env_ptrs = []
        self.object_handles = [[] for _ in range(self.num_envs)]
        self.object_actor_indices = [[] for _ in range(self.num_envs)]
        self.object_indices = []
        self.object_names = [[] for _ in range(self.num_envs)]

        self.goal_handles = []
        self.goal_actor_indices = []

        if self.cfg_env.bin.asset != 'no_bin':
            bin_options = gymapi.AssetOptions()
            bin_options.fix_base_link = True
            bin_options.use_mesh_materials = True
            bin_options.vhacd_enabled = True  # Enable convex decomposition
            bin_options.vhacd_params = gymapi.VhacdParams()
            bin_options.vhacd_params.resolution = 1000000
            bin_asset = self.gym.load_asset(self.sim, self._asset_root, self.cfg_env.bin.asset + '/bin.urdf', bin_options)
            bin_pose = gymapi.Transform(p=gymapi.Vec3(*self.cfg_env.bin.pos), r=gymapi.Quat(*self.cfg_env.bin.quat))
            bin_rigid_body_count = self.gym.get_asset_rigid_body_count(bin_asset)
            bin_rigid_shape_count = self.gym.get_asset_rigid_shape_count(bin_asset)

        if self.cfg_task.rl.goal == "reposition":
            goal_options = gymapi.AssetOptions()
            goal_options.fix_base_link = True
            goal_asset = self.gym.create_sphere(self.sim, 0.025, goal_options)
            goal_rigid_body_count = self.gym.get_asset_rigid_body_count(goal_asset)
            goal_rigid_shape_count = self.gym.get_asset_rigid_shape_count(goal_asset)
        else:
            assert False

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

            # Create goal actor.
            if not self.headless:
                goal_handle = self.gym.create_actor(env_ptr, goal_asset, gymapi.Transform(), "goal", self.num_envs, 0, 3)
                for rigid_body_index in range(self.gym.get_actor_rigid_body_count(env_ptr, goal_handle)):
                    self.gym.set_rigid_body_color(env_ptr, goal_handle, rigid_body_index, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))
                self.goal_handles.append(goal_handle)
                self.goal_actor_indices.append(actor_count)
                actor_count += 1

            # Aggregate all actors.
            max_rigid_bodies = self.controller.rigid_body_count + objects_rigid_body_count
            max_rigid_shapes = self.controller.rigid_shape_count + objects_rigid_shape_count
            if not self.headless:
                max_rigid_bodies += goal_rigid_body_count
                max_rigid_shapes += goal_rigid_shape_count

            if self.cfg_env.bin.asset != 'no_bin':
                max_rigid_bodies += bin_rigid_body_count
                max_rigid_shapes += bin_rigid_shape_count

            self.gym.begin_aggregate(env_ptr, max_rigid_bodies, max_rigid_shapes, True)

            # Create robot actor.
            robot_handle = self.controller.create_actor(env_ptr, env_index)
            self.controller_handles.append(robot_handle)
            self.controller_actor_indices.append(actor_count)
            actor_count += 1

            # Create cameras.
            for camera in self.camera_dict.values():
                camera.create_sensor(self.gym, self.sim, env_ptr)

            # Create bin actor.
            if self.cfg_env.bin.asset != 'no_bin':
                bin_handle = self.gym.create_actor(env_ptr, bin_asset, bin_pose, 'bin', env_index, 0, 2)
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

        self.controller_actor_indices = torch.tensor(self.controller_actor_indices, dtype=torch.int32, device=self.device)
        self.object_actor_indices = torch.tensor(self.object_actor_indices, dtype=torch.int32, device=self.device)

        self.object_actor_env_indices = [self.gym.find_actor_index(env_ptr, o.name, gymapi.DOMAIN_ENV) for o in [self.objects[i] for i in object_indices]]
        self.object_actor_env_indices_tensor = torch.tensor(self.object_actor_env_indices, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.object_indices = torch.tensor(self.object_indices, dtype=torch.long, device=self.device)

        self.target_object_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.target_object_actor_env_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        self.goal_actor_indices = torch.tensor(self.goal_actor_indices, dtype=torch.int32, device=self.device)
        self.goal_actor_env_index = self.gym.find_actor_index(env_ptr, "goal", gymapi.DOMAIN_ENV)

        # Create cameras.
        #for camera in self.camera_dict.values():
        #    for k, v in camera._camera_tensors.items():
        #        camera._camera_tensors[k] = torch.stack(v)

        # self.controller_actor_id_env = self.gym.find_actor_index(
        #     env_ptr, 'robot', gymapi.DOMAIN_ENV)
        # self.object_actor_id_env = [self.gym.find_actor_index(
        #     env_ptr, o.name, gymapi.DOMAIN_ENV)
        #     for o in [self.objects[idx] for idx in objects_idx]]
        

    def _disable_object_collisions(self, object_ids: List[int]):
        self._set_object_collisions(object_ids, collision_filter=-1)

    def _enable_object_collisions(self, object_ids: List[int]):
        self._set_object_collisions(object_ids, collision_filter=0)

    def _set_object_collisions(self, object_ids: List[int], collision_filter: int) -> None:
        def set_collision_filter(env_id: int, actor_handle, collision_filter: int) -> None:
            actor_shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptrs[env_id], actor_handle)
            for shape_id in range(len(actor_shape_props)):
                actor_shape_props[shape_id].filter = collision_filter
            self.gym.set_actor_rigid_shape_properties(self.env_ptrs[env_id], actor_handle, actor_shape_props)

        # No tensor API to set actor rigid shape props, so a loop is required.
        for env_id in range(self.num_envs):
            for object_id in object_ids:
                set_collision_filter(env_id, self.object_handles[env_id][object_id], collision_filter)

    def objects_in_bin(self, object_pos: torch.Tensor) -> torch.BoolTensor:
        x_lower = self.bin_info['extent'][0][0] + self.cfg_env.bin.pos[0]
        x_upper = self.bin_info['extent'][1][0] + self.cfg_env.bin.pos[0]
        y_lower = self.bin_info['extent'][0][1] + self.cfg_env.bin.pos[1]
        y_upper = self.bin_info['extent'][1][1] + self.cfg_env.bin.pos[1]
        z_lower = self.bin_info['extent'][0][2] + self.cfg_env.bin.pos[2]
        z_upper = self.bin_info['extent'][1][2] + self.cfg_env.bin.pos[2]
        in_bin = x_lower <= object_pos[..., 0]
        in_bin = torch.logical_and(in_bin, object_pos[..., 0] <= x_upper)
        in_bin = torch.logical_and(in_bin, y_lower <= object_pos[..., 1])
        in_bin = torch.logical_and(in_bin, object_pos[..., 1] <= y_upper)
        in_bin = torch.logical_and(in_bin, z_lower <= object_pos[..., 2])
        in_bin = torch.logical_and(in_bin, object_pos[..., 2] <= z_upper)
        return in_bin
    
    def visualize_bin_extent(self) -> None:
        for env_id in range(self.num_envs):
            bin_pose = gymapi.Transform(p=gymapi.Vec3(*self.cfg_env.bin.pos))
            bbox = gymutil.WireframeBBoxGeometry(torch.tensor(self.bin_info['extent']), pose=bin_pose, color=(0, 1, 1))
            gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id], pose=gymapi.Transform())

    def _acquire_object_bounding_box(self) -> None:
        self.object_bounding_box = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects, 10)).to(self.device)  # [pos, quat, extents] with shape (num_envs, num_objects_per_bin, 10)
        
        # Retrieve bounding box pose and extents for each object.
        object_bounding_box_extents = torch.zeros((len(self.objects), 3)).to(self.device)
        object_bounding_box_from_origin_pos = torch.zeros((len(self.objects), 3)).to(self.device)
        object_bounding_box_from_origin_quat = torch.zeros((len(self.objects), 4)).to(self.device)
        for i, obj in enumerate(self.objects):
            to_origin, extents = obj.find_bounding_box_from_mesh()
            from_origin = np.linalg.inv(to_origin)
            from_origin_pos = from_origin[0:3, 3]
            from_origin_quat = R.from_matrix(from_origin[0:3, 0:3]).as_quat()
            object_bounding_box_from_origin_pos[i] = torch.from_numpy(from_origin_pos)
            object_bounding_box_from_origin_quat[i] = torch.from_numpy(from_origin_quat)
            object_bounding_box_extents[i] = torch.from_numpy(extents)

        # Gather with linear indices avoids for-loop over the environments.
        self.object_bounding_box_from_origin_pos = object_bounding_box_from_origin_pos.view(-1).gather(
            0, (self.object_indices.unsqueeze(-1).expand(-1, -1, 3) * 3 + torch.arange(3, device=self.device)).reshape(-1)).view(
            self.num_envs, self.cfg_env.objects.num_objects, 3)
        self.object_bounding_box_from_origin_quat = object_bounding_box_from_origin_quat.view(-1).gather(
            0, (self.object_indices.unsqueeze(-1).expand(-1, -1, 4) * 4 + torch.arange(4, device=self.device)).reshape(-1)).view(
            self.num_envs, self.cfg_env.objects.num_objects, 4)
        self.object_bounding_box[..., 7:10] = object_bounding_box_extents.view(-1).gather(
            0, (self.object_indices.unsqueeze(-1).expand(-1, -1, 3) * 3 + torch.arange(3, device=self.device)).reshape(-1)).view(
            self.num_envs, self.cfg_env.objects.num_objects, 3)
        
    def _refresh_object_bounding_box(self) -> None:
        self.object_bounding_box[:, :, 0:3] = self.object_pos + quat_apply(self.object_quat, self.object_bounding_box_from_origin_pos)
        self.object_bounding_box[:, :, 3:7] = quat_mul(self.object_quat, self.object_bounding_box_from_origin_quat)
    
    def _acquire_object_synthetic_pointcloud(self) -> None:
        if self.cfg_env.pointclouds.sample_mode == 'uniform':
            num_samples = [self.cfg_env.pointclouds.average_num_points, ] * len(self.objects)
        elif self.cfg_env.pointclouds.sample_mode == 'area':
            areas = [obj.surface_area for obj in self.objects]
            mean_area = sum(areas) / len(areas)
            num_samples = [int(self.cfg_env.pointclouds.average_num_points * area / mean_area) for area in areas]

        self.object_surface_samples = torch.zeros((len(self.objects), self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)
        for i, obj in enumerate(self.objects):
            current_object_surface_samples = torch.from_numpy(obj.sample_points_from_mesh(num_samples=num_samples[i])).to(self.device, dtype=torch.float32)
            self.object_surface_samples[i, 0:min(num_samples[i], self.cfg_env.pointclouds.max_num_points), :3] = current_object_surface_samples[:self.cfg_env.pointclouds.max_num_points, :]
            self.object_surface_samples[i, 0:min(num_samples[i], self.cfg_env.pointclouds.max_num_points), 3] = 1
        self.object_surface_samples = self.object_surface_samples[self.object_indices]  # shape: (num_envs, num_objects_per_bin, max_num_points, 4)

        self.object_synthetic_pointcloud_ordered = self.object_surface_samples.clone()  # shape: (num_envs, num_objects_per_bin, max_num_points, 4)
        self.object_synthetic_pointcloud = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)  # shape: (num_envs, num_objects_per_bin, max_num_points, 4)

    def _refresh_object_synthetic_pointcloud(self) -> None:
        object_pos_expanded = self.object_pos.unsqueeze(2).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 1)
        object_quat_expanded = self.object_quat.unsqueeze(2).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 1)

        self.object_synthetic_pointcloud_ordered[:, :, :, 0:3] = object_pos_expanded + quat_apply(object_quat_expanded, self.object_surface_samples[:, :, :, 0:3])

        self.object_synthetic_pointcloud_ordered[..., 0:3] *= self.object_synthetic_pointcloud_ordered[..., 3:].repeat(1, 1, 1, 3)  # Set points that are only padding to zero.

        self.object_synthetic_pointcloud[:] = self.object_synthetic_pointcloud_ordered[:, :, torch.randperm(self.cfg_env.pointclouds.max_num_points), :]  # Randomly permute points.

        