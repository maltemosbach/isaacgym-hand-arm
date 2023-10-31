from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from isaacgym import gymapi, gymtorch
import numpy as np
import hydra
import os
import torch
from typing import Dict, List, Optional, Tuple
import warnings
from typing import Any
from isaacgymenvs.tasks.hand_arm.base.observations import Observation


@dataclass
class IsaacGymSensorOutput:
    required_images: List[gymapi.ImageType] = field(default_factory=list)

@dataclass
class ROSSensorOutput:
    message_topic: str = None
    message_type: Any = None


@dataclass
class SensorOutput(IsaacGymSensorOutput, ROSSensorOutput):
    pass



class SensorOutputs(Enum):
    IMAGE = SensorOutput(required_images=[gymapi.ImageType.IMAGE_COLOR], message_topic='/camera/color/image_raw')
    DEPTH = SensorOutput(required_images=[gymapi.ImageType.IMAGE_DEPTH], message_topic='/camera/depth/image_raw')
    POINTCLOUD = SensorOutput(required_images=[gymapi.ImageType.IMAGE_COLOR, gymapi.ImageType.IMAGE_DEPTH], message_topic='/camera/points')


@torch.jit.script
def depth_image_to_xyz(depth_image, proj_mat, view_mat, device: torch.device):
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
    xyz = torch.bmm(x2_hom, view_mat.inverse())[..., 0:3]
    return xyz


class CameraSensorProperties:
    """Properties of a camera sensor."""
    _camera_asset_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'assets', 'hand_arm', 'cameras')

    def __init__(
            self,
            sensor_outputs: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        self.pos = pos
        self.quat = quat
        self.sensor_outputs = sensor_outputs

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
    def sensor_outputs(self) -> List[SensorOutput]:
        return self._sensor_outputs
    
    @sensor_outputs.setter
    def sensor_outputs(self, value: List[str]) -> None:
        self._sensor_outputs = []
        for sensor_output in value:
            if sensor_output.upper() in SensorOutputs.__members__:
                self._sensor_outputs.append(SensorOutputs[sensor_output.upper()].value)
            else:
                warnings.warn(f"Invalid sensor output detected: '{sensor_output.upper()}'.")
        assert len(self._sensor_outputs) > 0, "No valid sensor outputs provided."

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
    def required_images(self) -> List[gymapi.ImageType]:
        required_images = []
        for sensor_output in self.sensor_outputs:
            required_images.extend(sensor_output.required_images)
        return required_images


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
        
    def connect_simulation(self, gym, sim, env_ptr, device: torch.device) -> None:
        if not hasattr(self, 'gym'):
            self.gym = gym
            self.sim = sim
            self.device = device
        self.env_ptrs.append(env_ptr)

    @abstractmethod
    def get_image(self) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_depth(self) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_pointcloud(self) -> torch.Tensor:
        raise NotImplementedError


class IsaacGymCameraSensor(CameraSensor):
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

        self._camera_handles = []
        self._camera_tensors = defaultdict(list)
        self._retrieved_tensors = {}

    def connect_simulation(self, gym, sim, env_ptr, device: torch.device) -> None:
        super().connect_simulation(gym, sim, env_ptr, device)
        # Create new camera handle and set its transform.
        camera_handle = gym.create_camera_sensor(env_ptr, self.camera_props)
        gym.set_camera_transform(camera_handle, env_ptr, self.transform)
        self._camera_handles.append(camera_handle)

        # Retrieve camera's GPU tensors.
        for image_type in self.required_images:
            camera_tensor = gym.get_camera_image_gpu_tensor(sim, env_ptr, camera_handle, image_type)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            self._camera_tensors[image_type].append(torch_camera_tensor)

    def retrieve_images(self) -> None:
        for image_type in self.required_images:
            self._retrieved_tensors[image_type] = torch.stack(self._camera_tensors[image_type])
    
    def get_image(self) -> torch.Tensor:
        return self._retrieved_tensors[gymapi.ImageType.IMAGE_COLOR][..., 0:3]

    def get_depth(self) -> torch.Tensor:
        return self._retrieved_tensors[gymapi.ImageType.IMAGE_DEPTH]

    def get_pointcloud(self, max_depth: float = 10) -> torch.Tensor:
        is_valid = (self._retrieved_tensors[gymapi.ImageType.IMAGE_DEPTH] > -max_depth).to(self.device).view(len(self.env_ptrs), -1, 1)
        depth_image = torch.clamp(self._retrieved_tensors[gymapi.ImageType.IMAGE_DEPTH], min=-max_depth)  # Clamp to avoid NaNs.
        xyz = depth_image_to_xyz(depth_image, self.projection_matrix, self.view_matrix, self.device)
        xyz = self.global_to_environment_xyz(xyz)
        return torch.cat([xyz, is_valid], dim=-1)
    
    def global_to_environment_xyz(self, xyz, env_spacing: float = 1.):
        """View matrices are returned in global instead of environment
        coordinates in IsaacGym. This function projects the point-clouds into
        their environment-specific frame, which is usually desired."""
        # TODO: Make this more efficient by avoiding the for-loop.
        num_per_row = max(int(np.sqrt(len(self.env_ptrs))), 2)
        for env_id in range(len(self.env_ptrs)):
            row = int(np.floor(env_id / num_per_row))
            column = env_id % num_per_row
            xyz[env_id, :, 0] -= column * 2 * env_spacing
            xyz[env_id, :, 1] -= row * 2 * env_spacing
        return xyz

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
                sensor_outputs = []
                for observation_name in self.cfg_task.env.observations:
                   if observation_name.startswith(camera_name):
                        sensor_outputs.append(observation_name.split("-")[-1])

                if sensor_outputs:
                    camera_dict[camera_name] = self.create_camera_sensor(**camera_cfg, sensor_outputs=sensor_outputs)
        return camera_dict

    def create_camera_sensor(
            self,
            sensor_outputs: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        if self.cfg_base.ros.activate:
            return ROSCameraSensor(sensor_outputs, pos, quat, model, fovx, resolution)
        else:
            return IsaacGymCameraSensor(sensor_outputs, pos, quat, model, fovx, resolution)

    def refresh_images(self):
        if not self.cfg_base.ros.activate:
            if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
                self.gym.clear_lines(self.viewer)
            if self.headless:
                self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            for camera_sensor in self.camera_dict.values():
                camera_sensor.retrieve_images()
            self.gym.end_access_image_tensors(self.sim)