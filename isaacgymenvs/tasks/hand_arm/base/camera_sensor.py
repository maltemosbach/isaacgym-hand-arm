from abc import ABC, abstractmethod
from collections import defaultdict
import cv2
from dataclasses import dataclass, field
from enum import Enum
import itertools
from isaacgym import gymapi, gymtorch
import numpy as np
import hydra
import os
import torch
from typing import Dict, List, Optional, Sequence, Tuple
import warnings
from typing import Any




def subsample_pointcloud(pointcloud: torch.Tensor, num_samples: int) -> torch.Tensor:
    batch_size, num_points_padded, _ = pointcloud.shape

    valid_mask = pointcloud[:, :, 3] > 0.5

    valid_points = pointcloud[valid_mask][:, :3]
    valid_points = valid_points[torch.randperm(valid_points.shape[0])]

    valid_counts = valid_mask.sum(dim=1)


    cumsum_valid_counts = torch.cumsum(valid_counts, dim=0)
    indices = torch.arange(num_samples, device=pointcloud.device).unsqueeze(0).expand(batch_size, num_samples)

    indices = indices + cumsum_valid_counts.unsqueeze(1) - valid_counts.unsqueeze(1)
    indices = indices.clamp(min=0, max=len(valid_points) - 1)

    gathered_points = valid_points[indices]

    valid_points_mask = torch.ones_like(gathered_points[:, :, 0])

    padding_mask = torch.arange(num_samples, device=pointcloud.device).expand_as(indices) >= valid_counts.unsqueeze(1)
    valid_points_mask[padding_mask] = 0.

    gathered_points[padding_mask] = 0.

    gathered_points = torch.cat([gathered_points, valid_points_mask.unsqueeze(-1)], dim=-1)

    return gathered_points



# Extends IsaacGym's ImageType enum and specifies base type required for computation of the outpus.
class ImageType(Enum):
    RGB = ("RGB", (3,), torch.uint8, [gymapi.ImageType.IMAGE_COLOR])
    DEPTH = ("DEPTH", (), torch.float32, [gymapi.ImageType.IMAGE_DEPTH])
    SEGMENTATION = ("SEGMENTATION", (), torch.int32, [gymapi.ImageType.IMAGE_SEGMENTATION])
    POINTCLOUD = ("POINTCLOUD", (4,), torch.float32, [gymapi.ImageType.IMAGE_DEPTH])

    def __init__(self, name: str, size: Sequence[int], dtype: torch.dtype, required_isaacgym_image_types: List[gymapi.ImageType]) -> None:
        self._name = name
        self._size = size
        self._dtype = dtype
        self._required_isaacgym_image_types = required_isaacgym_image_types

    @property
    def size(self) -> Sequence[int]:
        return self._size
    
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
    
    @property
    def required_isaacgym_image_types(self) -> List[gymapi.ImageType]:
        return self._required_isaacgym_image_types


def depth_image_to_global_points(depth_image, proj_mat, view_mat, device: torch.device):
    batch_size, width, height = depth_image.shape
    sparse_depth = depth_image.to(device).to_sparse()
    indices = sparse_depth.indices()
    values = sparse_depth.values()
    xy_depth = torch.cat([indices.T[:, 1:].flip(1), values[..., None]], dim=-1)

    center_u = height / 2
    center_v = width / 2

    xy_depth[:, 0] = -(xy_depth[:, 0] - center_u) / height
    xy_depth[:, 1] = (xy_depth[:, 1] - center_v) / width
    xy_depth[:, 0] *= xy_depth[:, 2]
    xy_depth[:, 1] *= xy_depth[:, 2]

    x2 = xy_depth @ proj_mat

    x2_hom = torch.cat([x2, torch.ones_like(x2[:, 0:1])], dim=1).view(batch_size, -1, 4)
    xyz = torch.bmm(x2_hom, view_mat.inverse())[..., 0:3].view(batch_size, width, height, 3)
    return xyz


def global_to_environment_points(points: torch.Tensor, env_spacing: float = 1.):
    num_envs = len(points)
    # TODO: Make this more efficient by avoiding the for-loop.
    num_per_row = max(int(np.sqrt(num_envs)), 2)
    for env_index in range(num_envs):
        row = int(np.floor(env_index / num_per_row))
        column = env_index % num_per_row
        points[env_index, ..., 0] -= column * 2 * env_spacing
        points[env_index, ..., 1] -= row * 2 * env_spacing
    return points



class CameraSensorProperties:
    """Properties of a camera sensor."""
    _camera_asset_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'assets', 'hand_arm', 'cameras')

    def __init__(
            self,
            image_types: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        self.pos = pos
        self.quat = quat
        self.image_types = image_types

        if model is not None:
            self._acquire_properties_from_model(model)
        else:
            self._acquire_properties_from_args(fovx, resolution)
    
    def _acquire_properties_from_model(self, model: str) -> None:
        self._model = model
        camera_info_file = f'{self.model}/camera_info.yaml'
        camera_info = hydra.compose(config_name=os.path.join(self._camera_asset_root, camera_info_file))
        camera_info = camera_info['']['']['']['']['']['']['assets']['hand_arm']['cameras'][self.model]
        self._acquire_properties_from_args(camera_info.fovx, camera_info.resolution, camera_info.sensor_outputs, camera_info.source)

    def _acquire_properties_from_args(self, fovx: int, resolution: Tuple[int, int]) -> None:
        self.fovx = fovx
        self.resolution = resolution

    @property
    def pos(self) -> gymapi.Vec3:
        return self._pos

    @pos.setter
    def pos(self, value: Tuple[float, float, float]) -> None:
        self._pos = gymapi.Vec3(*value)

    @property
    def quat(self) -> gymapi.Quat:
        return self._quat

    @quat.setter
    def quat(self, value: Tuple[float, float, float, float]) -> None:
        self._quat = gymapi.Quat(*value)

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        available_camera_models = [f.name for f in os.scandir(self._camera_asset_root) if f.is_dir()]
        if value not in available_camera_models:
            raise ValueError(f"Camera model should be one of {available_camera_models}, but unknown model '{value}' was found.")
        self._model = value

    @property
    def fovx(self) -> int:
        return self._fovx

    @fovx.setter
    def fovx(self, value: int) -> None:
        if not 0 < value < 180:
            raise ValueError(f"Horizontal field-of-view (fovx) should be in [0, 180], but found '{value}'.")
        self._fovx = value

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        if not len(value) == 2 and all(isinstance(v, int) for v in value):
            raise ValueError(f"Resolution should be a tuple of 2 integer values, but found '{value}'.")
        self._resolution = value

    @property
    def image_types(self) -> List[ImageType]:
        return self._image_types
    
    @image_types.setter
    def image_types(self, value: List[str]) -> None:
        self._image_types = []
        for image_type in value:
            if image_type.upper() in ImageType.__members__:
                self._image_types.append(ImageType[image_type.upper()])
            else:
                warnings.warn(f"Invalid image type detected: '{image_type.upper()}'.")
        assert len(self._image_types) > 0, "No valid image_types provided."

    @property
    def width(self) -> int:
        return self._resolution[0]

    @property
    def height(self) -> int:
        return self._resolution[1]

    @property
    def camera_props(self) -> gymapi.CameraProperties:
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.width
        camera_props.height = self.height
        camera_props.horizontal_fov = self.fovx
        camera_props.enable_tensors = True
        return camera_props

    @property
    def transform(self) -> gymapi.Transform:
        return gymapi.Transform(p=self.pos, r=self.quat)
    
    @property
    def required_isaacgym_image_types(self) -> List[gymapi.ImageType]:
        required_isaacgym_image_types = []
        for image_type in self.image_types:
            for required_isaacgym_image_type in image_type.required_isaacgym_image_types:
                if required_isaacgym_image_type not in required_isaacgym_image_types:
                    required_isaacgym_image_types.append(required_isaacgym_image_type)
        return required_isaacgym_image_types


class CameraSensor(CameraSensorProperties, ABC):
    def __init__(
            self,
            sensor_outputs: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(sensor_outputs, pos, quat, model, fovx, resolution)
        self.env_ptrs = []
        
    def connect_simulation(self, gym, sim, env_index, env_ptr, device: torch.device, num_envs: int) -> None:
        if env_index == 0:
            self.gym = gym
            self.sim = sim
            self.device = device
            self.num_envs = num_envs
        self.env_ptrs.append(env_ptr)

        # Initialize observation tensors.
        if env_index == num_envs - 1:
            self.current_sensor_observation = {}
            for image_type in self.image_types:
                self.current_sensor_observation[image_type] = torch.zeros((num_envs, self.height, self.width, *image_type.size), device=device, dtype=image_type.dtype)

            # Extend by required base images.
            for image_type in [ImageType.RGB, ImageType.DEPTH, ImageType.SEGMENTATION]:
                if image_type.required_isaacgym_image_types[0] in self.required_isaacgym_image_types:
                    if image_type not in self.current_sensor_observation:
                        self.current_sensor_observation[image_type] = torch.zeros((num_envs, self.height, self.width, *image_type.size), device=device, dtype=image_type.dtype)
    
    def refresh_inside_gpu_access(self) -> None:
        pass

    def refresh_outside_gpu_access(self) -> None:
        pass


class IsaacGymCameraSensor(CameraSensor):
    def __init__(
            self,
            image_types: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(image_types, pos, quat, model, fovx, resolution)

        self._camera_handles = []
        self._camera_tensors = defaultdict(list)

    def connect_simulation(self, gym, sim, env_index, env_ptr, device: torch.device, num_envs: int) -> None:
        super().connect_simulation(gym, sim, env_index, env_ptr, device, num_envs)
        # Create new camera handle and set its transform.
        camera_handle = gym.create_camera_sensor(env_ptr, self.camera_props)
        gym.set_camera_transform(camera_handle, env_ptr, self.transform)
        self._camera_handles.append(camera_handle)

        # Retrieve camera's GPU tensors.
        for image_type in self.required_isaacgym_image_types:
            camera_tensor = gym.get_camera_image_gpu_tensor(sim, env_ptr, camera_handle, image_type)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            self._camera_tensors[image_type].append(torch_camera_tensor)

    def refresh_inside_gpu_access(self) -> None:
        for image_type in [ImageType.RGB, ImageType.DEPTH, ImageType.SEGMENTATION]:
            if image_type.required_isaacgym_image_types[0] in self._camera_tensors:
                if image_type == ImageType.RGB:
                    self.current_sensor_observation[image_type][:] = torch.stack(self._camera_tensors[image_type.required_isaacgym_image_types[0]])[..., 0:3]
                else:
                    self.current_sensor_observation[image_type][:] = torch.stack(self._camera_tensors[image_type.required_isaacgym_image_types[0]])

    def refresh_outside_gpu_access(self) -> None:
        if ImageType.POINTCLOUD in self.image_types:
            self.current_sensor_observation[ImageType.POINTCLOUD][:] = self._compute_pointcloud()

    def _compute_pointcloud(self, max_depth: float = 10) -> torch.Tensor:
        x_range = [-0.07, 0.63]
        y_range = [0.33, 0.83]
        is_valid = (self.current_sensor_observation[ImageType.DEPTH] > -max_depth).to(self.device).unsqueeze(-1)
        depth_image = torch.clamp(self.current_sensor_observation[ImageType.DEPTH], min=-max_depth)  # Clamp to avoid NaNs.
        global_points = depth_image_to_global_points(depth_image, self.projection_matrix, self.view_matrix, self.device)
        points = global_to_environment_points(global_points)
        in_workspace = ((points[..., 0] > x_range[0]) & (points[..., 0] < x_range[1]) & (points[..., 1] > y_range[0]) & (points[..., 1] < y_range[1])).unsqueeze(-1)
        is_valid = is_valid & in_workspace
        return torch.cat([points, is_valid], dim=-1)

    @property
    def projection_matrix(self) -> torch.Tensor:
        if not hasattr(self, '_projection_matrix'):
            proj_mat = torch.from_numpy(self.gym.get_camera_proj_matrix(self.sim, self.env_ptrs[0], self._camera_handles[0])).to(self.device)
            fu = 2 / proj_mat[0, 0]
            fv = 2 / proj_mat[1, 1]
            self._projection_matrix = torch.Tensor([[fu, 0., 0.], [0., fv, 0.], [0., 0., 1.]]).to(self.device)
        return self._projection_matrix
    
    @property
    def view_matrix(self) -> torch.Tensor:
        """Returns the batch of view matrices of shape: [len(env_ptrs), 4, 4].
        The camera view matrix is returned in global instead of env coordinates
        in IsaacGym."""
        if not hasattr(self, '_view_matrix'):
            view_mat = []
            for env_ptr, env_index in zip(self.env_ptrs, list(range(len(self.env_ptrs)))):
                view_mat.append(torch.from_numpy(self.gym.get_camera_view_matrix(self.sim, env_ptr, self._camera_handles[env_index])).to(self.device))
            self._view_matrix = torch.stack(view_mat)
        return self._view_matrix
    

class ROSCameraSensor(CameraSensor):
    def __init__(
            self,
            sensor_outputs: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        super().__init__(sensor_outputs, pos, quat, model, fovx, resolution)

    def get_image(self) -> torch.Tensor:
        pass

    def get_depth(self) -> torch.Tensor:
        pass

    def get_pointcloud(self, max_depth: float = 10) -> torch.Tensor:
        pass


class CameraMixin:
    def _acquire_camera_dict(self) -> Dict[str, CameraSensor]:

        camera_dict = {}
        if "cameras" in self.cfg_env.keys():
            for camera_name, camera_cfg in self.cfg_env.cameras.items():
                image_types = []
                for observation_name in self.ordered_observations:
                   if observation_name.startswith(camera_name):
                        image_types.append(observation_name.split("-")[-1])

                if image_types:
                    camera_dict[camera_name] = self.create_camera_sensor(**camera_cfg, image_types=image_types)
        return camera_dict

    def create_camera_sensor(
            self,
            image_types: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        if self.cfg_base.ros.activate:
            return ROSCameraSensor(image_types, pos, quat, model, fovx, resolution)
        else:
            return IsaacGymCameraSensor(image_types, pos, quat, model, fovx, resolution)

    def refresh_images(self):
        if not self.cfg_base.ros.activate:
            if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
                self.gym.clear_lines(self.viewer)

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
                
            #if self.headless:
            self.gym.step_graphics(self.sim)

            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            for camera_sensor in self.camera_dict.values():
                camera_sensor.refresh_inside_gpu_access()
            self.gym.end_access_image_tensors(self.sim)

            for camera_sensor in self.camera_dict.values():
                camera_sensor.refresh_outside_gpu_access()
        else:
            assert False

        
        if self.cfg_base.debug.camera.save_recordings:
            self._write_recordings()

    def _write_recordings(self) -> None:
        # Initialize recordings dict.
        if not hasattr(self, '_recordings_dict'):
            self._recordings_dict = {}
            self._episode_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            experiment_dir = os.path.join('runs', self.cfg['full_experiment_name'])
            self._recordings_dir = os.path.join(experiment_dir, 'videos')

            for camera_name, camera_sensor in self.camera_dict.items():
                self._recordings_dict[camera_name] = {}
                for image_type in camera_sensor.image_types:
                    self._recordings_dict[camera_name][image_type] = [[] for _ in range(self.num_envs)]

        # Append current sensor observations to recordings dict.
        for camera_name, camera_sensor in self.camera_dict.items():
            for image_type in camera_sensor.image_types:
                for env_index in range(self.num_envs):
                    image_np = camera_sensor.current_sensor_observation[image_type][env_index].cpu().numpy()
                    if image_type == ImageType.RGB:
                            self._recordings_dict[camera_name][image_type][env_index].append(image_np[..., ::-1])
                    elif image_type == ImageType.DEPTH:
                            depth_range = (0, 2.5)
                            image_np = np.clip(-image_np, *depth_range)
                            image_np = (image_np - depth_range[0]) / (depth_range[1] - depth_range[0])
                            image_np = (np.stack([image_np] * 3, axis=-1) * 255).astype(np.uint8)
                            self._recordings_dict[camera_name][image_type][env_index].append(image_np)

                            # TODO: Implement generic depth and segmentation to RGB mappings as I have already used for the visualization functions.
                else:
                    raise NotImplementedError
        
        # Write recordings to file at the end of the episode.
        fps = 1 / (self.cfg_base.sim.dt * self.cfg_task.env.controlFrequencyInv)
        done_env_indices = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_indices) > 0:
            for done_env_index in done_env_indices:
                self._episode_count[done_env_index] += 1
                for camera_name, camera_sensor in self.camera_dict.items():
                    for image_type in self.camera_sensor.image_types:
                        video_writer = cv2.VideoWriter(
                            os.path.join(
                                self._recordings_dir, f"{camera_name}_{image_type}_env_{env_index}_episode_{self._episode_count[env_index]}.mp4"
                            ),
                            self.fourcc, fps, (camera_sensor.width, camera_sensor.height)
                        )
                        for image_np in self._recordings_dict[camera_name][image_type][env_index]:
                            video_writer.write(image_np)
                        video_writer.release()
                        
                        self._recordings_dict[camera_name][image_type][env_index] = []
