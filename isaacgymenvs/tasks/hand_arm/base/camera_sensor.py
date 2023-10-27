from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from isaacgym import gymapi, gymtorch
import hydra
import os
import torch
from typing import Dict, List, Optional, Tuple
import warnings


@dataclass
class SensorOutput:
    name: str
    required_image_types: List[gymapi.ImageType]
    ros_topic: str


class ValidSensorOutputs(Enum):
    COLOR = SensorOutput(name='color', required_image_types=[gymapi.ImageType.IMAGE_COLOR], ros_topic='/camera/color/image_raw')
    POINTS = SensorOutput(name='points', required_image_types=[gymapi.ImageType.IMAGE_COLOR, gymapi.ImageType.IMAGE_DEPTH], ros_topic='/camera/points')


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
            pos: Tuple[float, float, float],
            quat: Tuple[float, float, float, float],
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None,
            sensor_outputs: Optional[List[str]] = None,
            source: Optional[str] = 'isaacgym',  # Should be either 'isaacgym' or 'ros'.
    ) -> None:
        self.pos = pos
        self.quat = quat

        if model is not None:
            self._acquire_properties_from_model(model)
        else:
            self._acquire_properties_from_args(fovx, resolution, sensor_outputs, source)
    
    def _acquire_properties_from_model(self, model: str) -> None:
        self._model = model
        camera_info_file = f'{self.model}/camera_info.yaml'
        camera_info = hydra.compose(config_name=os.path.join(self._camera_asset_root, camera_info_file))
        camera_info = camera_info['']['']['']['']['']['']['assets']['hand_arm']['cameras'][self.model]
        self._acquire_properties_from_args(camera_info.fovx, camera_info.resolution, camera_info.sensor_outputs, camera_info.source)

    def _acquire_properties_from_args(self, fovx: int, resolution: Tuple[int, int], sensor_outputs: List[str], source: str) -> None:
        self.fovx = fovx
        self.resolution = resolution
        self.sensor_outputs = sensor_outputs
        self.source = source

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
            if sensor_output.upper() in ValidSensorOutputs.__members__:
                self._sensor_outputs.append(ValidSensorOutputs[sensor_output.upper()].value)
            else:
                warnings.warn(f"Invalid sensor output detected: '{sensor_output.upper()}'.")
        assert len(self._sensor_outputs) > 0, "No valid sensor outputs provided."

    @property
    def source(self) -> str:
        return self._source
    
    @source.setter
    def source(self, value: str) -> None:
        if value not in ['isaacgym', 'ros']:
            raise ValueError(f"Source should be either 'isaacgym' or 'ros', but found '{value}'.")
        self._source = value

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
    def required_image_types(self) -> List[gymapi.ImageType]:
        required_image_types = []
        for sensor_output in self.sensor_outputs:
            required_image_types.extend(sensor_output.required_image_types)
        return required_image_types


class CameraSensor(CameraSensorProperties):
    """Camera Sensor that wraps the same functionality as the IsaacGym camera sensors more convieniently."""
    def __init__(
            self,
            pos: Tuple[float, float, float],
            quat: Tuple[float, float, float, float],
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None,
            sensor_outputs: Optional[List[str]] = None,
    ) -> None:

        super().__init__(pos, quat, model, fovx, resolution, sensor_outputs)

        self._camera_handles = []
        self._camera_tensors = defaultdict(list)

        if ValidSensorOutputs.POINTS in self.sensor_outputs:
            self.projection_matrix = self._acquire_projection_matrix()
            self.view_matrix = self._acquire_view_matrix()
            #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #print("Using device:", self.device)

    def create_sensor(self, gym, sim, env_ptr) -> None:
        # Create new camera handle and set its transform.
        camera_handle = gym.create_camera_sensor(env_ptr, self.camera_props)
        gym.set_camera_transform(camera_handle, env_ptr, self.transform)
        self._camera_handles.append(camera_handle)

        # Retrieve camera's GPU tensors.
        for image_type in self.required_image_types:
            camera_tensor = gym.get_camera_image_gpu_tensor(sim, env_ptr, camera_handle, image_type)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            self._camera_tensors[image_type].append(torch_camera_tensor)

    def _get_sensor_output(self, sensor_output: SensorOutput) -> torch.Tensor:
        if sensor_output in [ValidSensorOutputs.COLOR.value, ]:
            return self._camera_tensors[sensor_output.required_image_types[0]]
        
        elif sensor_output == ValidSensorOutputs.POINTS.value:
            depth_image = self._camera_tensors[gymapi.ImageType.IMAGE_DEPTH]
            print("depth_image.shape:", depth_image.shape)

            depth_image = torch.clamp(depth_image, min=-10.)  # Clamp to avoid NaNs.

            xyz = depth_image_to_xyz(depth_image, self.projection_matrix, self.view_matrix, self.device)
            return xyz

        else:
            assert False

    def _acquire_projection_matrix(self) -> torch.Tensor:
        proj_mat = torch.from_numpy(self.gym.get_camera_proj_matrix(self.sim, self.env_ptrs[0], self._camera_handles[0])).to(self.device)
        fu = 2 / proj_mat[0, 0]
        fv = 2 / proj_mat[1, 1]
        return torch.Tensor([[fu, 0., 0.],
                             [0., fv, 0.],
                             [0., 0., 1.]]).to(self.device)
    
    def _acquire_view_matrix(self) -> torch.Tensor:
        """Returns the batch of view matrices of shape: [len(env_ptrs), 4, 4].
        The camera view matrix is returned in global instead of env coordinates
        in IsaacGym."""
        view_mat = []
        for env_ptr, env_idx in zip(self.env_ptrs, self.env_ids):
            view_mat.append(torch.from_numpy(self.gym.get_camera_view_matrix(self.sim, env_ptr, self._camera_handles[env_idx])).to(self.device))
        return torch.stack(view_mat)

    def get_outputs(self) -> Dict[SensorOutput, torch.Tensor]:
        image_dict = {}

        for sensor_output in self.sensor_outputs:
            image_dict[sensor_output.name] = self._get_sensor_output(sensor_output)

        return image_dict

