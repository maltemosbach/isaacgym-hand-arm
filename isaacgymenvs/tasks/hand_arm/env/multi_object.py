
from isaacgym import gymapi, gymutil
from isaacgymenvs.tasks.hand_arm.base.ur5sih import Ur5Sih
from isaacgymenvs.tasks.hand_arm.utils.camera import PointType, ImageType
from isaacgymenvs.tasks.hand_arm.utils.observables import Observable, LowDimObservable, PosObservable, PoseObservable, BoundingBoxObservable, SyntheticPointcloudObservable
from isaacgymenvs.tasks.hand_arm.utils.callbacks import ObservableCallback
from isaacgymenvs. tasks.hand_arm.utils.urdf import generate_table_with_hole
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
import matplotlib.pyplot as plt

from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate, quat_conjugate
from functools import partial


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
    
    def create_actor(self, env_ptr, group: int, filter: int, segmentationId: int) -> None:
        actor_handle = self._gym.create_actor(env_ptr, self.asset, self.start_pose, self.name, group, filter, segmentationId)

        if not hasattr(self, "rigid_body_properties"):
            rigid_body_properties = self._gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
            assert len(rigid_body_properties) == 1, "Multiple rigid body properties in object asset."
            self.rigid_body_properties = rigid_body_properties[0]
        return actor_handle
    
    @property
    def mass(self) -> float:
        return self.rigid_body_properties.mass

    @property
    def com(self) -> gymapi.Vec3:
        return self.rigid_body_properties.com

    @property
    def inertia(self) -> gymapi.Mat33:
        return self.rigid_body_properties.inertia




class Ur5SihMultiObject(Ur5Sih):
    _env_cfg_path: str = 'task/Ur5SihMultiObject.yaml'

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_env = self._acquire_env_cfg()
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def register_observables(self) -> None:
        super().register_observables()
        
        # Register object pose and velocity observables.
        self.register_observable(
            PosObservable(
                name="object_pos",
                size=3 * self.cfg_env.objects.num_objects,
                get_state=lambda: self.object_pos,
                required=True,  # NOTE: Required to save initial object poses and compute rewards.
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "object_pos", self.root_pos[:, self.object_actor_env_indices, 0:3]),
                    post_step=lambda: self.object_pos.copy_(self.root_pos[:, self.object_actor_env_indices, 0:3]),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="object_quat",
                size=(4 * self.cfg_env.objects.num_objects,),
                get_state=lambda: self.object_quat,
                required=True,  # NOTE: Required to save initial object poses.
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "object_quat", self.root_quat[:, self.object_actor_env_indices, 0:4]),
                    post_step=lambda: self.object_quat.copy_(self.root_quat[:, self.object_actor_env_indices, 0:4]),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="object_linvel",
                size=(3 * self.cfg_env.objects.num_objects,),
                get_state=lambda: self.object_linvel,
                required=True,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "object_linvel", self.root_linvel[:, self.object_actor_env_indices, 0:3]),
                    post_step=lambda: self.object_linvel.copy_(self.root_linvel[:, self.object_actor_env_indices, 0:3]),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="object_angvel",
                size=(3 * self.cfg_env.objects.num_objects,),
                get_state=lambda: self.object_angvel,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "object_angvel", self.root_angvel[:, self.object_actor_env_indices, 0:3]),
                    post_step=lambda: self.object_angvel.copy_(self.root_angvel[:, self.object_actor_env_indices, 0:3]),
                )
            )
        )

        # Register observables of physical object properties.
        self.register_observable(
            LowDimObservable(
                name="object_mass",
                size=(self.cfg_env.objects.num_objects,),
                get_state=lambda: self.object_mass,
                required=True,  # NOTE: Required to compute randomizations.
                callback=ObservableCallback(
                    post_init=self._acquire_object_mass,
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="object_com",
                size=(3 * self.cfg_env.objects.num_objects,),
                get_state=lambda: self.object_com,
                callback=ObservableCallback(
                    post_init=self._acquire_object_com,
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="object_inertia",
                size=(9 * self.cfg_env.objects.num_objects,),
                get_state=lambda: self.object_inertia,
                callback=ObservableCallback(
                    post_init=self._acquire_object_inertia,
                )
            )
        )

        # Register target object observables.
        self.register_observable(
            PosObservable(
                name="target_object_pos",
                get_state=lambda: self.target_object_pos,
                required=True,  # NOTE: Required to compute rewards.
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_pos", self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                    post_step=lambda: self.target_object_pos.copy_(self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="target_object_quat",
                size=4,
                get_state=lambda: self.target_object_quat,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_quat", self.root_quat.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)),
                    post_step=lambda: self.target_object_quat.copy_(self.root_quat.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)),
                )
            )
        )
        self.register_observable(
            PosObservable(
                name="target_object_pos_initial",
                get_state=lambda: self.target_object_pos_initial,
                requires=["target_object_pos"],
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_pos_initial", self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                    post_step=lambda: self.target_object_pos_initial.copy_(self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                )
            )
        )

        # Register geometric object observables such as bounding boxes and synthetic point-clouds.
        self.register_observable(
            BoundingBoxObservable(
                name="object_bounding_box",
                size=10 * self.cfg_env.objects.num_objects,
                get_state=lambda: self.object_bounding_box,
                callback=ObservableCallback(
                    post_init=self._acquire_object_bounding_box,
                    post_step=self._refresh_object_bounding_box,
                )
            )
        )
        self.register_observable(
            BoundingBoxObservable(
                name="target_object_bounding_box",
                get_state=lambda: self.target_object_bounding_box,
                requires=["object_bounding_box"],
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_bounding_box", torch.zeros((self.num_envs, 10)).to(self.device)),
                    post_step=lambda: self.target_object_bounding_box.copy_(self.object_bounding_box.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 10)).squeeze(1)),
                )
            )
        )
        self.register_observable(
            SyntheticPointcloudObservable(
                name="object_synthetic_pointcloud",
                size=(self.cfg_env.objects.num_objects * self.cfg_env.pointclouds.max_num_points, 4),
                get_state=lambda: self.object_synthetic_pointcloud,
                callback=ObservableCallback(
                    post_init=self._acquire_object_synthetic_pointcloud,
                    post_step=self._refresh_object_synthetic_pointcloud,
                )
            )
        )
        self.register_observable(
            SyntheticPointcloudObservable(
                name="target_object_synthetic_pointcloud",
                size=(self.cfg_env.pointclouds.max_num_points, 4),
                get_state=lambda: self.target_object_synthetic_pointcloud,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_synthetic_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                    post_step=self._refresh_target_object_synthetic_pointcloud,
                ),
                requires=["object_synthetic_pointcloud"],
            )
        )
        self.register_observable(
            SyntheticPointcloudObservable(
                name="object_synthetic_initial_pointcloud",
                size=(self.cfg_env.objects.num_objects * self.cfg_env.pointclouds.max_num_points, 4),
                get_state=lambda: self.object_synthetic_initial_pointcloud,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "object_synthetic_initial_pointcloud", torch.zeros((self.num_envs, self.cfg_env.objects.drop.num_initial_poses, self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),  # NOTE: Initial object observations are overwritten automatically once after the objects have been dropped.
                ),
                requires=["object_synthetic_pointcloud"],
            )
        )
        self.register_observable(
            SyntheticPointcloudObservable(
                name="target_object_synthetic_initial_pointcloud",
                size=(self.cfg_env.pointclouds.max_num_points, 4),
                get_state=lambda: self.target_object_synthetic_initial_pointcloud,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_synthetic_initial_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                    post_reset=self._refresh_target_object_synthetic_initial_pointcloud,
                ),
                requires=["object_synthetic_initial_pointcloud"],
            )
        )

        # Register goal observables.
        self.register_observable(
            LowDimObservable(
                name="goal_pos",
                size=3,
                get_state=lambda: self.goal_pos,
                required=True,  # NOTE: Required to compute rewards.
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "goal_pos", torch.zeros(self.num_envs, 3, device=self.device)),
                )
            )
        )

        # Register task-specific observables (observations that make desired behaviors easier to learn).
        self.register_observable(
            LowDimObservable(
                name="sih_fingertip_to_target_object_pos",
                size=3 * 5,
                get_state=lambda: self.sih_fingertip_to_target_object_pos,
                required=True,  # NOTE: Required to compute rewards.
                requires=["sih_fingertip_pos", "target_object_pos"],
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "sih_fingertip_to_target_object_pos", self.target_object_pos.unsqueeze(1).repeat(1, 5, 1) - self.sih_fingertip_pos),
                    post_step=lambda: self.sih_fingertip_to_target_object_pos.copy_(self.target_object_pos.unsqueeze(1).repeat(1, 5, 1) - self.sih_fingertip_pos),
                )
            )
        )
        self.register_observable(
            LowDimObservable(
                name="target_object_to_goal_pos",
                size=3,
                get_state=lambda: self.target_object_to_goal_pos,
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "target_object_to_goal_pos", self.goal_pos - self.target_object_pos),
                    post_step=lambda: self.target_object_to_goal_pos.copy_(self.goal_pos - self.target_object_pos),
                )
            )
        )

        workspace_pointcloud_size=64
        self.register_observable(
            SyntheticPointcloudObservable(
                name="workspace_synthetic_pointcloud",
                size=(workspace_pointcloud_size, 4),
                get_state=lambda: self.workspace_synthetic_pointcloud,
                callback=ObservableCallback(
                    post_init=lambda: self._acquire_workspace_synthetic_pointcloud(num_samples=workspace_pointcloud_size),
                ),
            )
        )
        self.register_observable(
            SyntheticPointcloudObservable(
                name="goal_synthetic_pointcloud",
                size=(1, 4),
                get_state=lambda: torch.cat([self.goal_pos.unsqueeze(1), PointType.GOAL.value * torch.ones((self.num_envs, 1, 1), device=self.device)], dim=-1),
                requires=["goal_pos"],
            )
        )
        self.register_observable(
            SyntheticPointcloudObservable(
                name="relative_goal_synthetic_pointcloud",
                size=(1, 4),
                get_state=lambda: torch.cat([(self.relative_goal_pos).unsqueeze(1), PointType.GOAL.value * torch.ones((self.num_envs, 1, 1), device=self.device)], dim=-1),
                requires=["goal_pos"],
                callback=ObservableCallback(
                    post_init=lambda: setattr(self, "relative_goal_pos", torch.zeros((self.num_envs, 3), device=self.device)),
                    post_step=self._refresh_relative_goal_synthetic_pointcloud,
                )
            )
        )

        # Register camera observations that relate to the objects.
        for camera_name in self.cfg_env.cameras:
            self.register_observable(
                SyntheticPointcloudObservable(
                    name=f"{camera_name}_target_object_pointcloud",
                    size=(self.cfg_env.pointclouds.max_num_points, 4),
                    get_state=partial(getattr, self, f"{camera_name}_target_object_pointcloud"),
                    callback=ObservableCallback(
                        post_init=lambda camera_name=camera_name: setattr(self, f"{camera_name}_target_object_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                        post_step=lambda camera_name=camera_name: self._refresh_segmented_pointcloud(camera_name=camera_name, tensor_name="_target_object_pointcloud", target_segmentation_id=self.target_object_index + 3, pointcloud_id=PointType.TARGET.value),
                    ),
                    requires=[f"{camera_name}_pointcloud", f"{camera_name}_segmentation"]
                )
            )

    def _acquire_env_cfg(self) -> DictConfig:
        cfg_env = hydra.compose(config_name=self._env_cfg_path)['task']
        
        if cfg_env.bin.asset == 'no_bin':
            self.bin_info = {"extent": [[-0.25, -0.25, cfg_env.table_height], [0.25, 0.25, cfg_env.table_height + 0.2]]}
        else:
            bin_info_path = f'../../assets/hand_arm/{cfg_env.bin.asset}/bin_info.yaml'
            self.bin_info = hydra.compose(config_name=bin_info_path)
            self.bin_info = self.bin_info['']['']['']['']['']['']['assets']['hand_arm'][cfg_env.bin.asset]
            self.bin_info["extent"][0][2] += cfg_env.table_height
            self.bin_info["extent"][1][2] += cfg_env.table_height

        self.workspace = torch.tensor(cfg_env.workspace)
        self.workspace[:, 2] += cfg_env.table_height

        if self.cfg_task.rl.goal == "throw":
            self.goal_bin_half_extent = torch.Tensor([0.1, 0.1, 0.1])
        return cfg_env

    def _acquire_objects(self) -> None:
        def solve_object_regex(regex: str, object_set: str) -> List[str]:
            root = os.path.join('../assets/hand_arm/', 'object_sets', 'urdf', object_set)
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
                object = ObjectAsset(self.gym, self.sim, '../assets/hand_arm/' + 'object_sets', f'urdf/{object_set}/' + object_name + '.urdf')
                self.objects.append(object)

    def _create_envs(self) -> None:
        lower = gymapi.Vec3(-self.cfg_base.sim.env_spacing, -self.cfg_base.sim.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.sim.env_spacing, self.cfg_base.sim.env_spacing, self.cfg_base.sim.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self._acquire_objects()
        self._acquire_robot_asset("../assets/hand_arm", "robot/hand_arm_collision_is_visual.urdf")

        self.ur5sih_handles = []
        self.ur5sih_actor_indices = []

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
            bin_asset = self.gym.load_asset(self.sim, '../assets/hand_arm/', self.cfg_env.bin.asset + '/bin.urdf', bin_options)
            bin_pos = gymapi.Vec3(*self.cfg_env.bin.pos)
            bin_pos.z += self.cfg_env.table_height
            bin_pose = gymapi.Transform(p=bin_pos, r=gymapi.Quat(*self.cfg_env.bin.quat))
            bin_rigid_body_count = self.gym.get_asset_rigid_body_count(bin_asset)
            bin_rigid_shape_count = self.gym.get_asset_rigid_shape_count(bin_asset)

        if self.cfg_task.rl.goal == "reposition":
            goal_options = gymapi.AssetOptions()
            goal_options.fix_base_link = True
            goal_asset = self.gym.create_sphere(self.sim, 0.02, goal_options)
            goal_rigid_body_count = self.gym.get_asset_rigid_body_count(goal_asset)
            goal_rigid_shape_count = self.gym.get_asset_rigid_shape_count(goal_asset)

        elif self.cfg_task.rl.goal == "throw":
            goal_options = gymapi.AssetOptions()
            goal_options.fix_base_link = True
            goal_options.use_mesh_materials = True
            goal_bin_tmp_path = os.path.join(os.path.dirname(__file__), 'goal_bin.urdf')
            generate_cuboid_bin_urdf(0.2, 0.2, goal_bin_tmp_path)
            goal_asset = self.gym.load_asset(self.sim, str(os.path.dirname(__file__)), 'goal_bin.urdf', bin_options)
            os.remove(goal_bin_tmp_path)
            goal_rigid_body_count = self.gym.get_asset_rigid_body_count(goal_asset)
            goal_rigid_shape_count = self.gym.get_asset_rigid_shape_count(goal_asset)
        else:
            assert False

        if self.cfg_env.table_height > 0.0:
            table_options = gymapi.AssetOptions()
            table_options.fix_base_link = True

            if self.cfg_env.bin.asset == 'no_bin':
                table_asset = self.gym.create_box(self.sim, 0.75, 1.1, self.cfg_env.table_height, table_options)

            else:
                generate_table_with_hole(x_range=(-0.095, 0.655), y_range=(-0.17, 0.93), hole_x_range=(self.bin_info['extent'][0][0] + self.cfg_env.bin.pos[0], self.bin_info['extent'][1][0] + self.cfg_env.bin.pos[0]), hole_y_range=(self.bin_info['extent'][0][1] + self.cfg_env.bin.pos[1], self.bin_info['extent'][1][1] + self.cfg_env.bin.pos[1]), height=self.cfg_env.table_height, file_path='table_with_hole.urdf')
                table_asset = self.gym.load_asset(self.sim, ".", 'table_with_hole.urdf', table_options)

            table_rigid_body_count = self.gym.get_asset_rigid_body_count(table_asset)
            table_rigid_shape_count = self.gym.get_asset_rigid_shape_count(table_asset)




        if self.cfg_env.collision_boundaries.add_safety_walls:
            safety_walls_options = gymapi.AssetOptions()
            safety_walls_options.fix_base_link = True
            safety_walls_asset = self.gym.load_asset(self.sim, '../assets/hand_arm/', 'collision_boundaries/safety_walls_visible.urdf', safety_walls_options)
            safety_walls_rigid_body_count = self.gym.get_asset_rigid_body_count(safety_walls_asset)
            safety_walls_rigid_shape_count = self.gym.get_asset_rigid_shape_count(safety_walls_asset)

        if self.cfg_env.collision_boundaries.add_table_offset:
            table_offset_options = gymapi.AssetOptions()
            table_offset_options.fix_base_link = True
            table_offset_asset = self.gym.load_asset(self.sim, '../assets/hand_arm/', 'collision_boundaries/table_offset_visible.urdf', table_offset_options)
            table_offset_rigid_body_count = self.gym.get_asset_rigid_body_count(table_offset_asset)
            table_offset_rigid_shape_count = self.gym.get_asset_rigid_shape_count(table_offset_asset)

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

            # Create camera sensors.
            for camera in self._camera_dict.values():
                camera.connect_simulation(self.gym, self.sim, env_index, env_ptr, self.device, self.num_envs)

            # Create goal actor.
            if self.cfg_task.rl.goal == "reposition":
                goal_handle = self.gym.create_actor(env_ptr, goal_asset, gymapi.Transform(), "goal", self.num_envs, 0, 3 + self.cfg_env.objects.num_objects)
                for rigid_body_index in range(self.gym.get_actor_rigid_body_count(env_ptr, goal_handle)):
                    self.gym.set_rigid_body_color(env_ptr, goal_handle, rigid_body_index, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 1.0, 0.0))
                self.goal_handles.append(goal_handle)
                self.goal_actor_indices.append(actor_count)
                actor_count += 1
            
            elif self.cfg_task.rl.goal == "throw":
                goal_handle = self.gym.create_actor(env_ptr, goal_asset, gymapi.Transform(), "goal", self.num_envs, 0, 3 + self.cfg_env.objects.num_objects)
                self.goal_handles.append(goal_handle)
                self.goal_actor_indices.append(actor_count)
                actor_count += 1

            # Aggregate all actors.
            max_rigid_bodies = self.ur5sih_rigid_body_count + objects_rigid_body_count
            max_rigid_shapes = self.ur5sih_rigid_shape_count + objects_rigid_shape_count

            max_rigid_bodies += goal_rigid_body_count
            max_rigid_shapes += goal_rigid_shape_count

            if self.cfg_env.table_height > 0.0:
                max_rigid_bodies += table_rigid_body_count
                max_rigid_shapes += table_rigid_shape_count

            if self.cfg_env.bin.asset != 'no_bin':
                max_rigid_bodies += bin_rigid_body_count
                max_rigid_shapes += bin_rigid_shape_count

            if self.cfg_env.collision_boundaries.add_safety_walls:
                max_rigid_bodies += safety_walls_rigid_body_count
                max_rigid_shapes += safety_walls_rigid_shape_count
            
            if self.cfg_env.collision_boundaries.add_table_offset:
                max_rigid_bodies += table_offset_rigid_body_count
                max_rigid_shapes += table_offset_rigid_shape_count

            self.gym.begin_aggregate(env_ptr, max_rigid_bodies, max_rigid_shapes, True)

            # Create robot actor.
            robot_handle = self.create_robot_actor(env_ptr, env_index, disable_self_collisions=True)
            self.ur5sih_handles.append(robot_handle)
            self.ur5sih_actor_indices.append(actor_count)
            actor_count += 1

            # Create table actor.
            if self.cfg_env.table_height > 0.0:
                if self.cfg_env.bin.asset == 'no_bin':
                    table_pose = gymapi.Transform(p=gymapi.Vec3(0.2925, 0.38, self.cfg_env.table_height/2))
                else:
                    table_pose = gymapi.Transform(p=gymapi.Vec3(0., 0., self.cfg_env.table_height / 2))
                table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', env_index, 0)
                actor_count += 1

            # Create bin actor.
            if self.cfg_env.bin.asset != 'no_bin':
                bin_handle = self.gym.create_actor(env_ptr, bin_asset, bin_pose, 'bin', env_index, 0, 2)
                actor_count += 1

            # Create object actors
            for i, object_index in enumerate(object_indices):
                used_object = self.objects[object_index]
                object_handle = used_object.create_actor(env_ptr, env_index, 0, 3 + i)
                self.object_handles[env_index].append(object_handle)
                self.object_actor_indices[env_index].append(actor_count)
                actor_count += 1

            # Create collision boundaries actor.
            if self.cfg_env.collision_boundaries.add_safety_walls:
                self.gym.create_actor(env_ptr, safety_walls_asset, gymapi.Transform(p=gymapi.Vec3(0.28, 0.33, 0.)), 'safety_walls', env_index, 0, 0)
                actor_count += 1

            if self.cfg_env.collision_boundaries.add_table_offset:
                self.gym.create_actor(env_ptr, table_offset_asset, gymapi.Transform(p=gymapi.Vec3(0.28, 0.33 + 0.25, 0.)), 'table_offset', env_index, 0b1111111111111111111111111111110, 0)
                actor_count += 1

            self.gym.end_aggregate(env_ptr)
            self.env_ptrs.append(env_ptr)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        self.ur5sih_actor_indices = torch.tensor(self.ur5sih_actor_indices, dtype=torch.int32, device=self.device)
        self.ur5sih_rigid_body_env_indices = [self.gym.get_actor_rigid_body_index(self.env_ptrs[0], self.ur5sih_handles[0], i, gymapi.DOMAIN_ENV) for i in range(self.ur5sih_rigid_body_count)]

        self.object_actor_indices = torch.tensor(self.object_actor_indices, dtype=torch.int32, device=self.device)
        self.object_actor_env_indices = [self.gym.find_actor_index(env_ptr, o.name, gymapi.DOMAIN_ENV) for o in [self.objects[i] for i in object_indices]]
        self.object_actor_env_indices_tensor = torch.tensor(self.object_actor_env_indices, dtype=torch.int32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.object_indices = torch.tensor(self.object_indices, dtype=torch.long, device=self.device)
        self.object_rigid_body_env_indices = [self.gym.get_actor_rigid_body_index(self.env_ptrs[0], object_handle, 0, gymapi.DOMAIN_ENV) for object_handle in self.object_handles[0]]
        self.object_configuration_indices = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)

        self.target_object_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.target_object_actor_env_index = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        self.goal_actor_indices = torch.tensor(self.goal_actor_indices, dtype=torch.int32, device=self.device)
        self.goal_actor_env_index = self.gym.find_actor_index(env_ptr, "goal", gymapi.DOMAIN_ENV)

    def _disable_object_collisions(self, object_ids: List[int]):
        self._set_object_collisions(object_ids, collision_filters=[0b001 for _ in range(len(object_ids))])

    def _enable_object_collisions(self, object_ids: List[int]):
        if len(object_ids) > 30:
            raise ValueError("Cannot have more than 30 objects in the environment.")

        if self.cfg_base.ros.activate:
            OBJECT_COLLISION_FILTER = [0b001 for _ in object_ids]
        else:
            OBJECT_COLLISION_FILTER = [2**(i + 1) for i in object_ids]
        # Objects get non-overlapping biswise collision filters so they collide with each other. Robot collides with everything anyways, and the other things like an artificial collision boundary can now be tuned to collide with whatever is needed.
        self._set_object_collisions(object_ids, collision_filters=OBJECT_COLLISION_FILTER)

    def _set_object_collisions(self, object_ids: List[int], collision_filters: List[int]) -> None:
        def set_collision_filter(env_id: int, actor_handle, collision_filter: int) -> None:
            actor_shape_props = self.gym.get_actor_rigid_shape_properties(self.env_ptrs[env_id], actor_handle)
            for shape_id in range(len(actor_shape_props)):
                actor_shape_props[shape_id].filter = collision_filter
            self.gym.set_actor_rigid_shape_properties(self.env_ptrs[env_id], actor_handle, actor_shape_props)

        # No tensor API to set actor rigid shape props, so a loop is required.
        for env_id in range(self.num_envs):
            for object_id, collision_filter in zip(object_ids, collision_filters):
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
    
    def target_object_in_goal_bin(self) -> torch.BoolTensor:
        # Project object pos into goal bin frame.
        target_object_pos_in_goal_frame = quat_rotate(self.goal_quat, self.target_object_pos) - self.goal_pos

        half_extent = self.goal_bin_half_extent.unsqueeze(0).repeat(self.num_envs, 1).to(self.device)

        in_goal_bin = (target_object_pos_in_goal_frame >= - half_extent) & (target_object_pos_in_goal_frame <= half_extent)
        return in_goal_bin.all(dim=-1)
    
    def visualize_bin_extent(self) -> None:
        for env_id in range(self.num_envs):
            bin_pose = gymapi.Transform(p=gymapi.Vec3(*self.cfg_env.bin.pos))
            bbox = gymutil.WireframeBBoxGeometry(torch.tensor(self.bin_info['extent']), pose=bin_pose, color=(0, 1, 1))
            gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id], pose=gymapi.Transform())

    def visualize_goal_bin_extent(self) -> None:
        self.visualize_bounding_boxes(self.goal_pos, self.goal_quat, 2 * self.goal_bin_half_extent.unsqueeze(0).repeat(self.num_envs, 1))

    def visualize_workspace_extent(self) -> None:
        for env_id in range(self.num_envs):
            bbox = gymutil.WireframeBBoxGeometry(self.workspace, pose=gymapi.Transform(), color=(0, 1, 1))
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

    def _refresh_object_synthetic_pointcloud(self, relative: bool = False) -> None:
        object_pos = self.object_pos
        object_quat = self.object_quat

        object_pos_expanded = object_pos.unsqueeze(2).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 1)
        object_quat_expanded = object_quat.unsqueeze(2).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 1)

        self.object_synthetic_pointcloud_ordered[:, :, :, 0:3] = object_pos_expanded + quat_apply(object_quat_expanded, self.object_surface_samples[:, :, :, 0:3])

        if relative:
            self.object_synthetic_pointcloud_ordered[:, :, :, 0:3] -= self.ur5_flange_pose[:, 0:3].unsqueeze(1).unsqueeze(2).repeat(1, self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 1)
            self.object_synthetic_pointcloud_ordered[:, :, :, 0:3] = quat_apply(quat_conjugate(self.ur5_flange_pose[:, 3:7]).unsqueeze(1).unsqueeze(2).repeat(1, self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 1), self.object_synthetic_pointcloud_ordered[:, :, :, 0:3])

        self.object_synthetic_pointcloud_ordered[..., 0:3] *= self.object_synthetic_pointcloud_ordered[..., 3:].repeat(1, 1, 1, 3)  # Set points that are only padding to zero.
        self.object_synthetic_pointcloud[:] = self.object_synthetic_pointcloud_ordered[:, :, torch.randperm(self.cfg_env.pointclouds.max_num_points), :]  # Randomly permute points.
    
    def _refresh_target_object_synthetic_pointcloud(self) -> None:
        self.target_object_synthetic_pointcloud[:] = self.object_synthetic_pointcloud.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 4)).squeeze(1)
        self.target_object_synthetic_pointcloud[..., 3] *= PointType.TARGET.value

    def _refresh_target_object_synthetic_initial_pointcloud(self, env_ids) -> None:
        self.target_object_synthetic_initial_pointcloud[:] = self.object_synthetic_initial_pointcloud[torch.arange(self.num_envs), self.object_configuration_indices].gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 4)).squeeze(1)
        self.target_object_synthetic_initial_pointcloud[env_ids, ..., 3] *= PointType.TARGET.value

    def _refresh_relative_goal_synthetic_pointcloud(self) -> None:
        self.relative_goal_pos[:] = self.goal_pos
        self.relative_goal_pos[:] -= self.ur5_flange_pose[:, 0:3]
        self.relative_goal_pos[:] = quat_apply(quat_conjugate(self.ur5_flange_pose[:, 3:7]), self.relative_goal_pos)

    def _acquire_table_synthetic_pointcloud(self, num_samples: int) -> None:
        self.table_synthetic_pointcloud = torch.zeros((self.num_envs, num_samples, 4)).to(self.device)
        self.table_synthetic_pointcloud[:, :, 0].uniform_(-0.07, 0.63)
        self.table_synthetic_pointcloud[:, :, 1].uniform_(-0.17, 0.83)
        self.table_synthetic_pointcloud[:, :, 3] = 1

    def _acquire_workspace_synthetic_pointcloud(self, num_samples: int) -> None:
        self.workspace_synthetic_pointcloud = torch.zeros((self.num_envs, num_samples, 4)).to(self.device)
        self.workspace_synthetic_pointcloud[:, :, 0].uniform_(self.cfg_env.workspace[0][0], self.cfg_env.workspace[1][0])
        self.workspace_synthetic_pointcloud[:, :, 1].uniform_(self.cfg_env.workspace[0][1], self.cfg_env.workspace[1][1])
        self.workspace_synthetic_pointcloud[:, :, 3] = 1

    def _acquire_bin_synthetic_pointcloud(self, num_samples: int) -> None:
        self.bin_body_env_index = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.bin_handles[0], body_name, gymapi.DOMAIN_ENV)

    
    def _refresh_segmented_pointcloud(self, camera_name: str, tensor_name: str, target_segmentation_id: torch.Tensor, pointcloud_id: int = 1) -> None:
        segmentation = self._camera_dict[camera_name].current_sensor_observation[ImageType.SEGMENTATION].flatten(1, 2)
        pointcloud = self._camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD].clone().detach().flatten(1, 2)
  
        segmented_pointclouds = []
        for env_index in range(self.num_envs):
            target_object_mask = segmentation[env_index] == target_segmentation_id[env_index]
            segmented_pointcloud = pointcloud[env_index][target_object_mask]

            # More points are on the object than can fit into the padded point-cloud tensor. Subsample points.
            if len(segmented_pointcloud) > self.cfg_env.pointclouds.max_num_points:
                segmented_pointcloud = segmented_pointcloud[torch.randperm(len(segmented_pointcloud))[:self.cfg_env.pointclouds.max_num_points]]
            
            # Fewer points than the padded point-cloud tensor. Pad with zeros.
            else:
                segmented_pointcloud = torch.cat((segmented_pointcloud, torch.zeros((self.cfg_env.pointclouds.max_num_points - len(segmented_pointcloud), 4)).to(self.device)))
            segmented_pointclouds.append(segmented_pointcloud)
        getattr(self, camera_name + tensor_name)[:] = torch.stack(segmented_pointclouds)
        getattr(self, camera_name + tensor_name)[:, :, 3] *= pointcloud_id  # Used to set the pointcloud id of initial observations to a different one for example.

    def _refresh_segmented_pointcloud_mean(self, camera_name:str, tensor_name: str, target_segmentation_id: torch.Tensor) -> None:
        segmentation = self.camera_dict[camera_name].current_sensor_observation[ImageType.SEGMENTATION].flatten(1, 2)
        pointcloud = self.camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD].clone().detach().flatten(1, 2)

        points = pointcloud[:, :, 0:3]
        mask = segmentation == target_segmentation_id.unsqueeze(1).repeat(1, segmentation.shape[1])

        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        sum_observed = torch.where(expanded_mask, points, torch.zeros_like(points)).sum(dim=1)
        num_observed = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_observed = sum_observed / num_observed
        getattr(self, camera_name + tensor_name)[:] = mean_observed

    def _init_sam_pointcloud(self, camera_name: str) -> None:
        setattr(self, f"{camera_name}_sam_initial_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device))
        from isaacgymenvs.tasks.hand_arm.other_utils.lang_sam_interface import LangSamInterface
        self.lang_sam_interface = LangSamInterface()

    def _refresh_sam_pointcloud(self, camera_name: str, pointcloud_id: int = 2) -> None:
        rgb = self.camera_dict[camera_name].current_sensor_observation[ImageType.RGB]
        pointcloud = self.camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD].flatten(1, 2)

        segmented_pointclouds = []
        for env_index in range(self.num_envs):
            color_image_numpy = rgb[env_index].cpu().numpy()
            mask = self.lang_sam_interface.predict(color_image_numpy, title=f"Select target object for camera '{camera_name}' on env {env_index}.")

            target_object_mask = torch.from_numpy(mask).to(self.device).flatten(0, 1)
            segmented_pointcloud = pointcloud[env_index][target_object_mask]

            # More points are on the object than can fit into the padded point-cloud tensor. Subsample points.
            if len(segmented_pointcloud) > self.cfg_env.pointclouds.max_num_points:
                segmented_pointcloud = segmented_pointcloud[torch.randperm(len(segmented_pointcloud))[:self.cfg_env.pointclouds.max_num_points]]
            
            # Fewer points than the padded point-cloud tensor. Pad with zeros.
            else:
                segmented_pointcloud = torch.cat((segmented_pointcloud, torch.zeros((self.cfg_env.pointclouds.max_num_points - len(segmented_pointcloud), 4)).to(self.device)))
            segmented_pointclouds.append(segmented_pointcloud)
        getattr(self, camera_name + "_sam_initial_pointcloud")[:] = torch.stack(segmented_pointclouds)
        getattr(self, camera_name + "_sam_initial_pointcloud")[:, :, 3] *= pointcloud_id  # Used to set the pointcloud id of initial observations to a different one for example.

    def _refresh_sam_pointcloud_mean(self, camera_name: str) -> None:
        points = getattr(self, camera_name + "_sam_initial_pointcloud")[:, :, 0:3]
        mask = getattr(self, camera_name + "_sam_initial_pointcloud")[:, :, 3] > 0.5
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, 3)
        sum_observed = torch.where(expanded_mask, points, torch.zeros_like(points)).sum(dim=1)
        num_observed = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_observed = sum_observed / num_observed
        getattr(self, camera_name + "_sam_pos_initial")[:] = mean_observed

    def _acquire_object_mass(self) -> None:
        self.object_mass = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects)).to(self.device)
        for env_index in range(self.num_envs):
            for object_index in range(self.cfg_env.objects.num_objects):
                self.object_mass[env_index, object_index] = self.objects[self.object_indices[env_index, object_index]].mass

    def _acquire_object_com(self) -> None:
        self.object_com = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects, 3)).to(self.device)
        for env_index in range(self.num_envs):
            for object_index in range(self.cfg_env.objects.num_objects):
                com = self.objects[self.object_indices[env_index, object_index]].com
                self.object_com[env_index, object_index] = torch.Tensor([com.x, com.y, com.z])

    def _acquire_object_inertia(self) -> None:
        self.object_inertia = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects, 3, 3)).to(self.device)
        for env_index in range(self.num_envs):
            for object_index in range(self.cfg_env.objects.num_objects):
                inertia = self.objects[self.object_indices[env_index, object_index]].inertia
                self.object_inertia[env_index, object_index] = torch.Tensor([[inertia.x.x, inertia.x.y, inertia.x.z], [inertia.y.x, inertia.y.y, inertia.y.z], [inertia.z.x, inertia.z.y, inertia.z.z]])


        