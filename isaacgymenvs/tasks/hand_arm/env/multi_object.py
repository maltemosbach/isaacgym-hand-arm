
from isaacgym import gymapi, gymutil
from isaacgymenvs.tasks.hand_arm.base.base import HandArmBase
from isaacgymenvs.tasks.hand_arm.base.observations import Callback, PointcloudObservation, LowDimObservation
from functools import partial
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
from isaacgymenvs.tasks.hand_arm.base.camera_sensor import ImageType, CameraSensorProperties
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import TextBox, Button

from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate


def generate_cuboid_bin_urdf(height, depth_width, file_path):
    urdf_template = f'''
    <?xml version="1.0"?>
    <robot name="cuboid_bin">

      <!-- Materials -->
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>

      <!-- Links -->
      <link name="base_link"/>

      <link name="floor">
            <visual>
            <geometry>
                <box size="{depth_width} {depth_width} 0.01"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="{depth_width} {depth_width} 0.01"/>
            </geometry>
            </collision>
        </link>

      <link name="left_wall">
            <visual>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            </collision>
        </link>

        <link name="right_wall">
            <visual>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="0.01 {depth_width} {height}"/>
            </geometry>
            </collision>
        </link>

        <link name="front_wall">
            <visual>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            </collision>
        </link>

        <link name="back_wall">
            <visual>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            <material name="blue"/>
            </visual>
            <collision>
            <geometry>
                <box size="{depth_width} 0.01 {height}"/>
            </geometry>
            </collision>
        </link>


      <!-- Joints -->
      <joint name="floor_joint" type="fixed">
        <parent link="base_link"/>
        <child link="floor"/>
        <origin xyz="0 0 {-height/2}" rpy="0 0 0"/>
      </joint>

      <joint name="left_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="left_wall"/>
            <origin xyz="-{depth_width/2 + 0.005} 0 {height/2}" rpy="0 0 0"/>
        </joint>

        <joint name="right_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="right_wall"/>
            <origin xyz="{depth_width/2 + 0.005} 0 {height/2}" rpy="0 0 0"/>
        </joint>

        <joint name="front_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="front_wall"/>
            <origin xyz="0 {depth_width/2 + 0.005} {height/2}" rpy="0 0 0"/>
        </joint>

        <joint name="back_wall_joint" type="fixed">
            <parent link="floor"/>
            <child link="back_wall"/>
            <origin xyz="0 -{depth_width/2 + 0.005} {height/2}" rpy="0 0 0"/>
        </joint>

    </robot>
    '''

    urdf_string = urdf_template.format(height=height, depth_width=depth_width)

    with open(file_path, 'w') as f:
        f.write(urdf_string)




def show_mask(mask, ax, random_color=False):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


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




class HandArmEnvMultiObject(HandArmBase):
    _env_cfg_path: str = 'task/HandArmEnvMultiObject.yaml'

    _padding_semantic_id: int = 0
    _regular_semantic_id: int = 1
    _target_semantic_id: int = 2

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_env = self._acquire_env_cfg()
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def register_observations(self) -> None:
        super().register_observations()

        # Register object pose and velocity observations.
        self.register_observation(
            "object_pos", 
            LowDimObservation(
                size=(3 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_pos.flatten(1, 2),
                is_mandatory=True,  # NOTE: Required to save initial object poses and compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "object_pos", self.root_pos[:, self.object_actor_env_indices, 0:3]),
                    on_step=lambda: self.object_pos.copy_(self.root_pos[:, self.object_actor_env_indices, 0:3]),
                ),
                visualize=lambda: self.visualize_poses(self.object_pos, self.object_quat),
            )
        )
        self.register_observation(
            "object_quat", 
            LowDimObservation(
                size=(4 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_quat.flatten(1, 2),
                is_mandatory=True,  # NOTE: Required to save initial object poses and compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "object_quat", self.root_quat[:, self.object_actor_env_indices, 0:4]),
                    on_step=lambda: self.object_quat.copy_(self.root_quat[:, self.object_actor_env_indices, 0:4]),
                ),
                visualize=lambda: self.visualize_poses(self.object_pos, self.object_quat),
            )
        )
        self.register_observation(
            "object_linvel", 
            LowDimObservation(
                size=(3 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_linvel.flatten(1, 2),
                is_mandatory=True,  # NOTE: Required to compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "object_linvel", self.root_linvel[:, self.object_actor_env_indices, 0:3]),
                    on_step=lambda: self.object_linvel.copy_(self.root_linvel[:, self.object_actor_env_indices, 0:3]),
                )
            )
        )
        self.register_observation(
            "object_angvel", 
            LowDimObservation(
                size=(3 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_angvel.flatten(1, 2),
                is_mandatory=True,  # NOTE: Required to compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "object_angvel", self.root_angvel[:, self.object_actor_env_indices, 0:3]),
                    on_step=lambda: self.object_angvel.copy_(self.root_angvel[:, self.object_actor_env_indices, 0:3]),
                )
            )
        )

        # Register object physical properties observations.
        self.register_observation(
            "object_mass", 
            LowDimObservation(
                size=(self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_mass,
                callback=Callback(
                    on_init=self._acquire_object_mass,
                )
            )
        )
        self.register_observation(
            "object_com", 
            LowDimObservation(
                size=(3 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_com.flatten(1, 2),
                callback=Callback(
                    on_init=self._acquire_object_com,
                )
            )
        )
        self.register_observation(
            "object_inertia", 
            LowDimObservation(
                size=(9 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_inertia.flatten(1, 2),
                callback=Callback(
                    on_init=self._acquire_object_inertia,
                )
            )
        )

        # Register target object observations.
        self.register_observation(
            "target_object_pos", 
            LowDimObservation(
                size=(3,),
                as_tensor=lambda: self.target_object_pos,
                is_mandatory=True,  # NOTE: Required to compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_pos", self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                    on_step=lambda: self.target_object_pos.copy_(self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                ),
                visualize=lambda: self.visualize_poses(self.target_object_pos, self.target_object_quat),
            )
        )
        self.register_observation(
            "target_object_quat", 
            LowDimObservation(
                size=(4,),
                as_tensor=lambda: self.target_object_quat,
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_quat", self.root_quat.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)),
                    on_step=lambda: self.target_object_quat.copy_(self.root_quat.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)),
                ),
                visualize=lambda: self.visualize_poses(self.target_object_pos, self.target_object_quat),
            )
        )
        self.register_observation(
            "target_object_pos_initial", 
            LowDimObservation(
                size=(3,),
                as_tensor=lambda: self.target_object_pos_initial,
                requires=["target_object_pos"],
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_pos_initial", self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                    on_reset=lambda: self.target_object_pos_initial.copy_(self.root_pos.gather(1, self.target_object_actor_env_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)),
                ),
                visualize=lambda: self.visualize_poses(self.target_object_pos_initial, torch.Tensor([[0, 0, 0, 1]]).repeat(self.num_envs, 1).to(self.device)),
            )
        )

        # Register geometric object observations such as bounding boxes and synthetic point-clouds.
        self.register_observation(
            "object_bounding_box", 
            LowDimObservation(
                size=(10 * self.cfg_env.objects.num_objects,),
                as_tensor=lambda: self.object_bounding_box.flatten(1, 2),
                callback=Callback(
                    on_init=self._acquire_object_bounding_box,
                    on_step=self._refresh_object_bounding_box,
                ),
                visualize=lambda: self.visualize_bounding_boxes(self.object_bounding_box[..., 0:3], self.object_bounding_box[..., 3:7], self.object_bounding_box[..., 7:10]),
            )
        )
        self.register_observation(
            "target_object_bounding_box", 
            LowDimObservation(
                size=(10,),
                as_tensor=lambda: self.target_object_bounding_box,
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_bounding_box", torch.zeros((self.num_envs, 10)).to(self.device)),
                    on_step=lambda: self.target_object_bounding_box.copy_(self.object_bounding_box.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 10)).squeeze(1)),
                ),
                visualize=lambda: self.visualize_bounding_boxes(self.target_object_bounding_box[:, 0:3], self.target_object_bounding_box[:, 3:7], self.target_object_bounding_box[:, 7:10]),
            )
        )
        self.register_observation(
            "object_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="object_synthetic",
                size=(self.cfg_env.objects.num_objects * self.cfg_env.pointclouds.max_num_points, 4,),
                as_tensor=lambda: self.object_synthetic_pointcloud.flatten(1, 2),
                callback=Callback(
                    on_init=self._acquire_object_synthetic_pointcloud,
                    on_step=self._refresh_object_synthetic_pointcloud,
                ),
                visualize=lambda: self.visualize_points(self.object_synthetic_pointcloud, size=0.0025)
            )
        )
        self.register_observation(
            "object_synthetic_initial-pointcloud",
            PointcloudObservation(
                camera_name="object_synthetic_initial",
                size=(self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 4,),
                as_tensor=lambda: self.object_synthetic_pointcloud_initial[torch.arange(self.num_envs), self.object_configuration_indices],
                requires=["object_synthetic-pointcloud"],
                callback=Callback(
                    on_init=lambda: setattr(self, "object_synthetic_pointcloud_initial", torch.zeros((self.num_envs, self.cfg_env.objects.drop.num_initial_poses, self.cfg_env.objects.num_objects, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),  # NOTE: Initial object observations are overwritten automatically once after the objects have been dropped.
                ),
                visualize=lambda: self.visualize_points(self.object_synthetic_pointcloud_initial[torch.arange(self.num_envs), self.object_configuration_indices], size=0.0025)
            )
        )
        self.register_observation(
            "target_object_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="target_object_synthetic",
                size=(self.cfg_env.pointclouds.max_num_points, 4,),
                as_tensor=lambda: self.target_object_synthetic_pointcloud,
                requires=["object_synthetic-pointcloud"],
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_synthetic_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                    on_step=self._refresh_target_object_synthetic_pointcloud,
                ),
                visualize=lambda: self.visualize_points(self.target_object_synthetic_pointcloud, size=0.0025)
            )
        )
        self.register_observation(
            "target_object_synthetic_initial-pointcloud",
            PointcloudObservation(
                camera_name="target_object_synthetic_initial",
                size=(self.cfg_env.pointclouds.max_num_points, 4,),
                as_tensor=lambda: self.target_object_synthetic_pointcloud_initial,
                requires=["object_synthetic_initial-pointcloud"],
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_synthetic_pointcloud_initial", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                    on_reset=lambda: self.target_object_synthetic_pointcloud_initial.copy_(self.object_synthetic_pointcloud_initial[torch.arange(self.num_envs), self.object_configuration_indices].gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 4)).squeeze(1)),  # NOTE: Initial observations of the target object must only be refreshed on resets and do not change during the episode.
                ),
                visualize=lambda: self.visualize_points(self.target_object_synthetic_pointcloud_initial, size=0.0025)
            )
        )

        # Register goal observations.
        self.register_observation(
            "goal_pos",
            LowDimObservation(
                size=(3,),
                as_tensor=lambda: self.goal_pos,
                is_mandatory=True,  # NOTE: Required to compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "goal_pos", torch.zeros(self.num_envs, 3, device=self.device)),
                ),
            )
        )
        if self.cfg_task.rl.goal in ["throw", "oriented_reposition"]: # goal_quat either for the hand orientation or the goal bin orientation.
            self.register_observation(
                "goal_quat",
                LowDimObservation(
                    size=(3,),
                    as_tensor=lambda: self.goal_quat,
                    is_mandatory=True,  # NOTE: Required to compute rewards.
                    callback=Callback(
                        on_init=lambda: setattr(self, "goal_quat", torch.zeros(self.num_envs, 3, device=self.device)),
                    ),
                )
            )

        # Register task observation (observations that make desired bahaviors easier to learn).
        self.register_observation(
            "fingertip_to_target_object_pos", 
            LowDimObservation(
                size=(3 * self.controller.fingertip_count,),
                as_tensor=lambda: self.fingertip_to_target_object_pos.flatten(1, 2),
                is_mandatory=True,  # NOTE: Required to compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "fingertip_to_target_object_pos", self.target_object_pos.unsqueeze(1).repeat(1, self.controller.fingertip_count, 1) - self.fingertip_pos),
                    on_step=lambda: self.fingertip_to_target_object_pos.copy_(self.target_object_pos.unsqueeze(1).repeat(1, self.controller.fingertip_count, 1) - self.fingertip_pos),
                ),
                visualize=lambda: self.visualize_distance(self.fingertip_pos, self.fingertip_to_target_object_pos),
            )
        )
        self.register_observation(
            "target_object_to_goal_pos",
            LowDimObservation(
                size=(3,),
                as_tensor=lambda: self.target_object_to_goal_pos,
                callback=Callback(
                    on_init=lambda: setattr(self, "target_object_to_goal_pos", self.goal_pos - self.target_object_pos),
                    on_step=lambda: self.target_object_to_goal_pos.copy_(self.goal_pos - self.target_object_pos),
                ),
                visualize=lambda: self.visualize_distance(self.target_object_pos, self.target_object_to_goal_pos),
            )
        )

        # Register camera observations that relate to the objects.
        for camera_name in self.cfg_env.cameras:
            self.register_observation(
                f"{camera_name}_target_object-pointcloud", 
                PointcloudObservation(
                    camera_name=camera_name,
                    key=f"{camera_name}_target_object-pointcloud", # NOTE: This is required to avoid overwriting the pointcloud of the entire scene.
                    size=(self.cfg_env.pointclouds.max_num_points, 4),
                    as_tensor=lambda: getattr(self, f"{camera_name}_target_object_pointcloud"),
                    callback=Callback(
                        on_init=lambda: setattr(self, f"{camera_name}_target_object_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                        on_step=lambda: self._refresh_segmented_pointcloud(camera_name=camera_name, tensor_name="_target_object_pointcloud", target_segmentation_id=self.target_object_index + 3),
                    ),
                    requires=[f"{camera_name}-pointcloud", f"{camera_name}-segmentation"],  # NOTE: The segmentation image is required to compute the points on the target object.
                    visualize=lambda: self.visualize_points(getattr(self, f"{camera_name}_target_object_pointcloud"))
                )
            )
            self.register_observation(
                f"{camera_name}_target_object_initial-pointcloud", 
                PointcloudObservation(
                    camera_name=camera_name,
                    key=f"{camera_name}_target_object_initial-pointcloud", # NOTE: This is required to avoid overwriting the pointcloud of the entire scene.
                    size=(self.cfg_env.pointclouds.max_num_points, 4),
                    as_tensor=lambda: getattr(self, f"{camera_name}_target_object_initial_pointcloud"),
                    callback=Callback(
                        on_init=lambda: setattr(self, f"{camera_name}_target_object_initial_pointcloud", torch.zeros((self.num_envs, self.cfg_env.pointclouds.max_num_points, 4)).to(self.device)),
                        on_reset=lambda: self._refresh_segmented_pointcloud(camera_name=camera_name, tensor_name="_target_object_initial_pointcloud", target_segmentation_id=self.target_object_index + 3, pointcloud_id=2),
                    ),
                    requires=[f"{camera_name}-pointcloud", f"{camera_name}-segmentation"],  # NOTE: The segmentation image is required to compute the points on the target object.
                    visualize=lambda: self.visualize_points(getattr(self, f"{camera_name}_target_object_initial_pointcloud"))
                )
            )
            self.register_observation(
                f"{camera_name}_sam_initial-pointcloud", 
                PointcloudObservation(
                    camera_name=camera_name,
                    key=f"{camera_name}_sam_initial-pointcloud", # NOTE: This is required to avoid overwriting the pointcloud of the entire scene.
                    size=(self.cfg_env.pointclouds.max_num_points, 4),
                    as_tensor=lambda: getattr(self, f"{camera_name}_sam_initial_pointcloud"),
                    callback=Callback(
                        on_init=lambda: self._init_sam_pointcloud(camera_name=camera_name),
                        on_reset=lambda: self._refresh_sam_pointcloud(camera_name=camera_name),
                    ),
                    requires=[f"{camera_name}-pointcloud", f"{camera_name}-rgb"],  # NOTE: The segmentation image is required to compute the points on the target object.
                    visualize=lambda: self.visualize_points(getattr(self, f"{camera_name}_sam_initial_pointcloud"))
                )
            )
            self.register_observation(
                f"{camera_name}_target_object_pos_initial", 
                LowDimObservation(
                    size=(3,),
                    as_tensor=lambda: getattr(self, f"{camera_name}_target_object_pos_initial"),
                    requires=[f"{camera_name}-pointcloud"],
                    callback=Callback(
                        on_init=lambda: setattr(self, f"{camera_name}_target_object_pos_initial", torch.zeros((self.num_envs, 3)).to(self.device)),
                        on_reset=lambda: self._refresh_segmented_pointcloud_mean(camera_name=camera_name, tensor_name="_target_object_pos_initial", target_segmentation_id=self.target_object_index + 3),
                    ),
                    visualize=lambda: self.visualize_poses(getattr(self, f"{camera_name}_target_object_pos_initial"), torch.Tensor([[0, 0, 0, 1]]).repeat(self.num_envs, 1).to(self.device)),
                )
            )
            self.register_observation(
                f"{camera_name}_sam_pos_initial", 
                LowDimObservation(
                    size=(3,),
                    as_tensor=lambda: getattr(self, f"{camera_name}_sam_pos_initial"),
                    requires=[f"{camera_name}_sam_initial-pointcloud"],
                    callback=Callback(
                        on_init=lambda: setattr(self, f"{camera_name}_sam_pos_initial", torch.zeros((self.num_envs, 3)).to(self.device)),
                        on_reset=lambda: self._refresh_sam_pointcloud_mean(camera_name=camera_name),
                    ),
                    visualize=lambda: self.visualize_poses(getattr(self, f"{camera_name}_sam_pos_initial"), torch.Tensor([[0, 0, 0, 1]]).repeat(self.num_envs, 1).to(self.device)),
                )
            )

        # Register synthetic pointcloud observations that extend the object-pointclouds with scene elements.
        self.register_observation(
            "robot_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="robot_synthetic",
                size=(sum(self.controller.num_body_surface_samples), 4,),
                as_tensor=lambda: self.robot_synthetic_pointcloud,
                callback=Callback(
                    on_init=self._acquire_robot_synthetic_pointcloud,
                    on_step=self._refresh_robot_synthetic_pointcloud,
                ),
                visualize=lambda: self.visualize_points(self.robot_synthetic_pointcloud, size=0.0025)
            )
        )
        table_pointcloud_size = 128
        self.register_observation(
            "table_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="table_synthetic",
                size=(table_pointcloud_size, 4),
                as_tensor=lambda: self.table_synthetic_pointcloud,
                callback=Callback(
                    on_init=lambda: self._acquire_table_synthetic_pointcloud(num_samples=table_pointcloud_size),
                ),
                visualize=lambda: self.visualize_points(self.table_synthetic_pointcloud, size=0.0025)
            )
        )
        workspace_pointcloud_size = 128
        self.register_observation(
            "workspace_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="workspace_synthetic",
                size=(workspace_pointcloud_size, 4),
                as_tensor=lambda: self.workspace_synthetic_pointcloud,
                callback=Callback(
                    on_init=lambda: self._acquire_workspace_synthetic_pointcloud(num_samples=table_pointcloud_size),
                ),
                visualize=lambda: self.visualize_points(self.workspace_synthetic_pointcloud, size=0.0025)
            )
        )

        self.register_observation(
            "bin_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="bin_synthetic",
                size=(128, 4),
                as_tensor=lambda: self.bin_synthetic_pointcloud,
                callback=Callback(
                    on_init=lambda: self._acquire_bin_synthetic_pointcloud(num_samples=128),
                ),
                visualize=lambda: self.visualize_points(self.bin_synthetic_pointcloud, size=0.0025)
            )
        )
        self.register_observation(
            "scene_synthetic-pointcloud",
            PointcloudObservation(
                camera_name="scene_synthetic",
                size=((self.cfg_env.pointclouds.max_num_points * self.cfg_env.objects.num_objects) + workspace_pointcloud_size + sum(self.controller.num_body_surface_samples), 4),
                as_tensor=lambda: self.scene_synthetic_pointcloud,
                requires=["object_synthetic-pointcloud", "workspace_synthetic-pointcloud", "robot_synthetic-pointcloud"],
                callback=Callback(
                    on_init=lambda: setattr(self, "scene_synthetic_pointcloud", torch.zeros((self.num_envs, (self.cfg_env.pointclouds.max_num_points * self.cfg_env.objects.num_objects) + table_pointcloud_size + sum(self.controller.num_body_surface_samples), 4)).to(self.device)),
                    on_step=lambda: self.scene_synthetic_pointcloud.copy_(torch.cat([self.object_synthetic_pointcloud.flatten(1, 2), self.workspace_synthetic_pointcloud, self.robot_synthetic_pointcloud], dim=1)),
                ),
                visualize=lambda: self.visualize_points(self.scene_synthetic_pointcloud, size=0.0025)
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

        if self.cfg_task.rl.goal == "throw":
            self.goal_bin_half_extent = torch.Tensor([0.1, 0.1, 0.1])
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
            goal_asset = self.gym.create_sphere(self.sim, 0.00, goal_options)
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
            max_rigid_bodies = self.controller.rigid_body_count + objects_rigid_body_count
            max_rigid_shapes = self.controller.rigid_shape_count + objects_rigid_shape_count

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
                camera.connect_simulation(self.gym, self.sim, env_index, env_ptr, self.device, self.num_envs)

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

        self.object_configuration_indices = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)

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
            bbox = gymutil.WireframeBBoxGeometry(torch.tensor(self.cfg_env.workspace), pose=gymapi.Transform(), color=(0, 1, 1))
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
    
    def _refresh_target_object_synthetic_pointcloud(self) -> None:
        self.target_object_synthetic_pointcloud[:] = self.object_synthetic_pointcloud.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.cfg_env.pointclouds.max_num_points, 4)).squeeze(1)
        self.target_object_synthetic_pointcloud[..., 3] *= self._target_semantic_id

    def _acquire_robot_synthetic_pointcloud(self) -> None:
        self.robot_body_env_indices = [self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.controller_handles[0], body_name, gymapi.DOMAIN_ENV) for body_name in self.controller.body_areas.keys()]
        self.robot_body_pos = self.body_pos[:, self.robot_body_env_indices, 0:3]
        self.robot_body_quat = self.body_quat[:, self.robot_body_env_indices, 0:4]

        self.robot_surface_samples = []
        for body_index, body_mesh in enumerate(self.controller.body_meshes):
            current_body_surface_samples = self.controller.body_surface_samples[body_index]
            current_body_surface_samples = torch.from_numpy(current_body_surface_samples).to(self.device, dtype=torch.float32)
            self.robot_surface_samples.append(current_body_surface_samples.unsqueeze(0).repeat(self.num_envs, 1, 1))

        self.robot_synthetic_pointcloud = torch.zeros((self.num_envs, sum(self.controller.num_body_surface_samples), 4)).to(self.device)  # shape: (num_envs, num_robot_points, 4)
        self.robot_synthetic_pointcloud[:, :, 3] = 1

    def _refresh_robot_synthetic_pointcloud(self) -> None:
        # Refresh robot body poses.
        self.robot_body_pos[:] = self.body_pos[:, self.robot_body_env_indices, 0:3]
        self.robot_body_quat[:] = self.body_quat[:, self.robot_body_env_indices, 0:4]

        prev_index = 0
        for body_index, num_samples in enumerate(self.controller.num_body_surface_samples):
            self.robot_synthetic_pointcloud[:, prev_index:prev_index + num_samples, 0:3] = self.robot_body_pos[:, body_index].unsqueeze(1).repeat(1, num_samples, 1) + quat_apply(self.robot_body_quat[:, body_index].unsqueeze(1).repeat(1, num_samples, 1), self.robot_surface_samples[body_index])
            prev_index += num_samples

        # Check for in-workspace and set validity.
        is_valid = self.robot_synthetic_pointcloud[:, :, 0] >= self.cfg_env.workspace[0][0]
        is_valid = torch.logical_and(is_valid, self.robot_synthetic_pointcloud[:, :, 0] <= self.cfg_env.workspace[1][0])
        is_valid = torch.logical_and(is_valid, self.robot_synthetic_pointcloud[:, :, 1] >= self.cfg_env.workspace[0][1])
        is_valid = torch.logical_and(is_valid, self.robot_synthetic_pointcloud[:, :, 1] <= self.cfg_env.workspace[1][1])
        self.robot_synthetic_pointcloud[:, :, 3] = is_valid.float()

    def _acquire_table_synthetic_pointcloud(self, num_samples: int) -> None:
        self.table_synthetic_pointcloud = torch.zeros((self.num_envs, num_samples, 4)).to(self.device)
        self.table_synthetic_pointcloud[:, :, 0].uniform_(-0.07, 0.63)
        self.table_synthetic_pointcloud[:, :, 1].uniform_(-0.17, 0.83)
        self.table_synthetic_pointcloud[:, :, 3] = 1

    def _acquire_workspace_synthetic_pointcloud(self, num_samples: int) -> None:
        self.workspace_synthetic_pointcloud = torch.zeros((self.num_envs, num_samples, 4)).to(self.device)
        self.workspace_synthetic_pointcloud[:, :, 0].uniform_(-0.07, 0.63)
        self.workspace_synthetic_pointcloud[:, :, 1].uniform_(0.23, 0.83)
        self.workspace_synthetic_pointcloud[:, :, 3] = 1

    def _acquire_bin_synthetic_pointcloud(self, num_samples: int) -> None:
        self.bin_body_env_index = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.bin_handles[0], body_name, gymapi.DOMAIN_ENV)

    
    def _refresh_segmented_pointcloud(self, camera_name: str, tensor_name: str, target_segmentation_id: torch.Tensor, pointcloud_id: int = 1) -> None:
        segmentation = self.camera_dict[camera_name].current_sensor_observation[ImageType.SEGMENTATION].flatten(1, 2)
        pointcloud = self.camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD].clone().detach().flatten(1, 2)

        # Segmentation gets a semantic value of 2 compared to the base value of 1.
        #pointcloud[:, :, 3] = torch.where(pointcloud[:, :, 3] > 0.5, pointcloud_id, 0)
        #pointcloud[:, :, 3] *= self._target_semantic_id
  
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
        from lang_sam import LangSAM
        self.lang_sam = LangSAM("vit_b", "/home/user/mosbach/tools/sam_tracking/sam_tracking/ckpt/sam_vit_b_01ec64.pth")  # TODO: Check if there is a better model available. Speed is really not an issue here.

    def _refresh_sam_pointcloud(self, camera_name: str, pointcloud_id: int = 2) -> None:
        rgb = self.camera_dict[camera_name].current_sensor_observation[ImageType.RGB]
        pointcloud = self.camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD].flatten(1, 2)

        from PIL import Image

        segmented_pointclouds = []
        for env_index in range(self.num_envs):
            self.sam_selection_submitted = False
            color_image_numpy = rgb[env_index].cpu().numpy()
            self.input_points = []
            self.input_labels = []
            image_pil = Image.fromarray(color_image_numpy)

            def update_mask_on_click(event):
                if event.inaxes == ax_image:    
                    self.input_points.append([event.xdata, event.ydata])
                    self.input_labels.append(int(event.button == MouseButton.LEFT))
                    self.lang_sam.sam.set_image(color_image_numpy)
                    masks, scores, logits = self.lang_sam.sam.predict(point_coords=np.array(self.input_points).astype(np.int), point_labels=np.array(self.input_labels).astype(np.int), multimask_output=False)
                    self.mask = masks[0]
                    ax_image.imshow(color_image_numpy)
                    show_mask(self.mask, ax_image)

                    for point, label in zip(self.input_points, self.input_labels):
                        col = 'green' if label == 1 else 'red'
                        ax_image.scatter(point[0], point[1], color=col, edgecolors='white', s=50)
                        
                    sam_figure.canvas.draw()

            def update_mask_on_text(text):
                if text not in ['', 'Left-click to add positive marker, right-click to add negative marker.']:
                    if len(self.input_points) > 0:
                        print("Cannot use text input when markers have already been added.")
                    else:
                        masks, boxes, phrases, logits = self.lang_sam.predict(image_pil, text)
                        self.mask = torch.any(masks, dim=0).cpu().numpy()

                        #self.pred_mask, masked_frame = self.segtracker.detect_and_seg(color_image_numpy.copy(), text, 0.25, 0.25)
                        #selected_segmentation = draw_mask(color_image_numpy.copy(), masks, id_countour=True)
                        ax_image.imshow(color_image_numpy)
                        show_mask(self.mask, ax_image)
                        ax_image.axis('off')
                        sam_figure.canvas.draw()

            def on_hover_over_image(event):
                if event.inaxes == ax_image and event.xdata is not None and event.ydata is not None:
                    if text_box.text == "":
                        text_box.set_val('Left-click to add positive marker, right-click to add negative marker.')
                else:
                    if text_box.text == 'Left-click to add positive marker, right-click to add negative marker.':
                        text_box.set_val('')
                sam_figure.canvas.draw()

            def clear_button_clicked(event):
                self.input_points = []
                self.input_labels = []
                self.mask = np.zeros_like(color_image_numpy)
                ax_image.imshow(color_image_numpy)
                sam_figure.canvas.draw()

            def submit_button_clicked(event):
                self.sam_selection_submitted = True

            def on_resize(event):
                bbox = ax_image.get_position()
                ax_width = bbox.x1 - bbox.x0
                ax_x = bbox.x0

                ax_prompt.set_position([ax_x, 0.2, ax_width, 0.05])
                ax_clear.set_position([ax_x, 0.1, ax_width / 2, 0.075])
                ax_submit.set_position([ax_x + ax_width / 2, 0.1, ax_width / 2, 0.075])


            sam_figure = plt.figure(num=f"Select target object for camera '{camera_name}' on env {env_index}.")
            ax_image = sam_figure.add_subplot(111)
            ax_image.axis('off')
            sam_figure.subplots_adjust(bottom=0.2)
            ax_prompt = sam_figure.add_axes([0.1, 0.2, 0.8, 0.05])
            text_box = TextBox(ax_prompt, "", initial="")
            text_box.on_submit(update_mask_on_text)

            ax_clear = sam_figure.add_axes([0.1, 0.1, 0.4, 0.075])
            ax_submit = sam_figure.add_axes([0.5, 0.1, 0.4, 0.075])
            clear_button = Button(ax_clear, 'Clear')
            submit_button = Button(ax_submit, 'Submit')
            clear_button.on_clicked(clear_button_clicked)
            submit_button.on_clicked(submit_button_clicked)

            ax_image.imshow(color_image_numpy)
            ax_image.set_position([0.1, 0.3, 0.8, 0.6])
            sam_figure.canvas.mpl_connect('button_press_event', update_mask_on_click)
            sam_figure.canvas.mpl_connect('motion_notify_event', on_hover_over_image)

            sam_figure.canvas.mpl_connect('resize_event', on_resize)

            plt.show(block=False)
            
            while not self.sam_selection_submitted:
                plt.pause(0.01)

            target_object_mask = torch.from_numpy(self.mask).to(self.device).flatten(0, 1)
            segmented_pointcloud = pointcloud[env_index][target_object_mask]

            print("target_object_mask:", target_object_mask)

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


        