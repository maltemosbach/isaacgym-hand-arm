from abc import ABC
from dataclasses import dataclass, field, InitVar
from isaacgym import gymapi
import matplotlib.pyplot as plt
import networkx as nx
import torch
from typing import Callable, Dict, List, Tuple, Sequence
from isaacgymenvs.tasks.hand_arm.base.camera_sensor import ImageType, CameraSensorProperties


PASS = lambda: None


@dataclass
class Callback:
    on_init: Callable[[], None]
    on_step: Callable[[], None] = PASS
    on_reset: Callable[[], None] = PASS

@dataclass
class Observation(ABC):
    """Base class for observations.
    
    Args:
        size: The size/shape of the observation.
        as_tensor: A callable that returns the observation as a tensor.
        callback: A callback that is called at specific points in time to acquire and refresh the observation.
        is_mandatory: Whether the observation is mandatory for the task, i.e., computed irrespectively of whether it is actually passed to the agent.
        requires: A list of observation names that are required to compute this observation.
    """
    size: Sequence[int]
    as_tensor: InitVar[Callable[[], torch.Tensor]]
    callback: Callback = Callback(on_init=PASS, on_step=PASS, on_reset=PASS)
    is_mandatory: bool = False
    requires: Sequence[str] = field(default_factory=list)
    visualize: Callable[[], None] = PASS

    def __post_init__(self, as_tensor: Callable[[], torch.Tensor]) -> None:
        self._as_tensor = as_tensor

    def as_tensor(self) -> torch.Tensor:
        tensor_data = self._as_tensor()

        if not isinstance(tensor_data, torch.Tensor):
            raise TypeError(f"Expected data of type torch.Tensor, but got {type(tensor_data)}.")
        
        if tensor_data.shape[1:] != self.size:
            raise ValueError(f"Expected data of shape {self.size}, but got {tensor_data.shape[1:]}.")
        
        return tensor_data
    

@dataclass
class DictObservation(Observation, ABC):
    key: str = ""

  

@dataclass
class LowDimObservation(DictObservation):
    def __post_init__(self, as_tensor: Callable[[], torch.Tensor]) -> None:
        super().__post_init__(as_tensor)
        if not len(self.size) == 1:
            raise ValueError("Low-dimensional observations must be one-dimensional.")
        
        self.key = "obs"


@dataclass
class CameraObservation(DictObservation, ABC):
    camera_name: str = ""


@dataclass
class RGBObservation(CameraObservation):
    def __post_init__(self, as_tensor: Callable[[], torch.Tensor]) -> None:
        super().__post_init__(as_tensor)
        #if not (self.size == (self.camera_dict[self.camera_name].height, self.camera_dict[self.camera_name].width, 3)):
        #    raise ValueError("Image observations must be of shape [H, W, 3].")
        
        if not self.key:
            self.key = f"{self.camera_name}_image"
        

@dataclass
class DepthObservation(CameraObservation):
    def __post_init__(self, as_tensor: Callable[[], torch.Tensor]) -> None:
        super().__post_init__(as_tensor)
        #if not (self.size == (self.camera_dict[self.camera_name].height, self.camera_dict[self.camera_name].width)):
        #    raise ValueError("Image observations must be of shape [H, W].")
        
        if not self.key:
            self.key = f"{self.camera_name}_depth"


@dataclass
class SegmentationObservation(CameraObservation):
    def __post_init__(self, as_tensor: Callable[[], torch.Tensor]) -> None:
        super().__post_init__(as_tensor)
        #if not (self.size == (self.camera_dict[self.camera_name].height, self.camera_dict[self.camera_name].width)):
        #    raise ValueError("Image observations must be of shape [H, W].")
        
        if not self.key:
            self.key = f"{self.camera_name}_segmentation"
        

@dataclass
class PointcloudObservation(CameraObservation):
    def __post_init__(self, as_tensor: Callable[[], torch.Tensor]) -> None:
        super().__post_init__(as_tensor)
        if not self.size[-1] == 4:
            raise ValueError("Pointcloud observations must be of shape [..., 4].")

        if not self.key:
            self.key = f"{self.camera_name}_pointcloud"



class ObserverMixin:
    _observations: Dict[str, Observation] = {}
    _mandatory_observations: List[str] = []

    def register_observation(self, name: str, observation: Observation) -> None:
        if name in self.cfg_env.cameras.keys():
            return
        self._observations[name] = observation
        if observation.is_mandatory:
            self._mandatory_observations.append(name)

    def register_observations(self) -> None:
        self.register_proprioceptive_observations()
        self.register_camera_observations()

    def register_proprioceptive_observations(self) -> None:
        # Register DoF observations.
        self.register_observation(
            "actuated_dof_pos", 
            LowDimObservation(
                size=(self.controller.actuated_dof_count,),
                as_tensor=lambda: self.actuated_dof_pos,
                callback=Callback(
                    on_init=lambda: setattr(self, "actuated_dof_pos", self.dof_pos[:, self.controller.actuated_dof_indices]),
                    on_step=lambda: self.actuated_dof_pos.copy_(self.dof_pos[:, self.controller.actuated_dof_indices]),
                )
            )
        )
        self.register_observation(
            "actuated_dof_vel", 
            LowDimObservation(
                size=(self.controller.actuated_dof_count,),
                as_tensor=lambda: self.actuated_dof_vel,
                callback=Callback(
                    on_init=lambda: setattr(self, "actuated_dof_vel", self.dof_vel[:, self.controller.actuated_dof_indices]),
                    on_step=lambda: self.actuated_dof_vel.copy_(self.dof_vel[:, self.controller.actuated_dof_indices]),
                )
            )
        )

        # Register fingertip keypoints observations.
        self.register_observation(
            "fingertip_pos", 
            LowDimObservation(
                size=(3 * self.controller.fingertip_count,),
                as_tensor=lambda: self.fingertip_pos.flatten(1, 2),
                is_mandatory=True,  # NOTE: Required to compute rewards.
                callback=Callback(
                    on_init=lambda: setattr(self, "fingertip_pos", self.body_pos[:, self.fingertip_body_env_indices, 0:3]),
                    on_step=lambda: self.fingertip_pos.copy_(self.body_pos[:, self.fingertip_body_env_indices, 0:3]),
                ),
                visualize=lambda: self.visualize_pose(self.fingertip_pos, self.fingertip_quat)
            )
        )
        self.register_observation(
            "fingertip_quat", 
            LowDimObservation(
                size=(4 * self.controller.fingertip_count,),
                as_tensor=lambda: self.fingertip_quat.flatten(1, 2),
                callback=Callback(
                    on_init=lambda: setattr(self, "fingertip_quat", self.body_quat[:, self.fingertip_body_env_indices, 0:4]),
                    on_step=lambda: self.fingertip_quat.copy_(self.body_quat[:, self.fingertip_body_env_indices, 0:4]),
                ),
                visualize=lambda: self.visualize_pose(self.fingertip_pos, self.fingertip_quat)
            )
        )
        self.register_observation(
            "fingertip_linvel", 
            LowDimObservation(
                size=(3 * self.controller.fingertip_count,),
                as_tensor=lambda: self.fingertip_linvel.flatten(1, 2),
                callback=Callback(
                    on_init=lambda: setattr(self, "fingertip_linvel", self.body_linvel[:, self.fingertip_body_env_indices, 0:3]),
                    on_step=lambda: self.fingertip_linvel.copy_(self.body_linvel[:, self.fingertip_body_env_indices, 0:3]),
                )
            )
        )
        self.register_observation(
            "fingertip_angvel", 
            LowDimObservation(
                size=(3 * self.controller.fingertip_count,),
                as_tensor=lambda: self.fingertip_angvel.flatten(1, 2),
                callback=Callback(
                    on_init=lambda: setattr(self, "fingertip_angvel", self.body_angvel[:, self.fingertip_body_env_indices, 0:3]),
                    on_step=lambda: self.fingertip_angvel.copy_(self.body_angvel[:, self.fingertip_body_env_indices, 0:3]),
                )
            )
        )

    def register_camera_observations(self) -> None:
        for camera_name in self.cfg_env.cameras:
            camera_properties = CameraSensorProperties(**self.cfg_env.cameras[camera_name], image_types=["rgb"])
            for image_type in ["rgb", "depth", "segmentation", "pointcloud"]:
                observation_name = f"{camera_name}-{image_type}"
                if image_type == "rgb":
                    self.register_observation(
                        observation_name, 
                        RGBObservation(
                            camera_name=camera_name,
                            size=(camera_properties.height, camera_properties.width, 3),
                            as_tensor=lambda: self.camera_dict[camera_name].current_sensor_observation[ImageType.RGB],
                            visualize=lambda: self.visualize_image(self.camera_dict[camera_name].current_sensor_observation[ImageType.RGB], window_name=observation_name),
                        )
                    )
                elif image_type == "depth":
                    self.register_observation(
                        observation_name, 
                        DepthObservation(
                            camera_name=camera_name,
                            size=(camera_properties.height, camera_properties.width),
                            as_tensor=lambda: self.camera_dict[camera_name].current_sensor_observation[ImageType.DEPTH],
                            visualize=lambda: self.visualize_depth(self.camera_dict[camera_name].current_sensor_observation[ImageType.DEPTH], window_name=observation_name)
                        )
                    )
                elif image_type == "segmentation":
                    self.register_observation(
                        observation_name, 
                        SegmentationObservation(
                            camera_name=camera_name,
                            size=(camera_properties.height, camera_properties.width),
                            as_tensor=lambda: self.camera_dict[camera_name].current_sensor_observation[ImageType.SEGMENTATION],
                            visualize=lambda: self.visualize_segmentation(self.camera_dict[camera_name].current_sensor_observation[ImageType.SEGMENTATION], window_name=observation_name)
                        )
                    )
                elif image_type == "pointcloud":
                    self.register_observation(
                        observation_name, 
                        PointcloudObservation(
                            camera_name=camera_name,
                            size=(camera_properties.height, camera_properties.width, 4),
                            as_tensor=lambda: self.camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD],
                            visualize=lambda: self.visualize_points(self.camera_dict[camera_name].current_sensor_observation[ImageType.POINTCLOUD])
                        )
                    )

    @property
    def observation_dependency_graph(self) -> nx.DiGraph:
        dependency_dict = {}
        for observation_name in self.cfg_task.env.observations:
            if observation_name in self.cfg_env.cameras.keys():
                continue
            if not self._observations[observation_name].is_mandatory:  # Mandatory observations are per definition independent.
                dependency_dict[observation_name] = self._observations[observation_name].requires
        
        return nx.DiGraph(dependency_dict)
    

    @property
    def ordered_observations(self) -> List[str]:
        return self._mandatory_observations + list(reversed(list(nx.topological_sort(self.observation_dependency_graph))))
    
    def _compute_num_observations(self, observations: List[str]) -> Tuple[int, Dict[str, Tuple[int, int]]]:
        num_observations = 0
        observations_start_end = {}

        print("self._observations", self._observations.keys())
        #nx.draw_networkx(self.observation_dependency_graph)

        #plt.show()

        for observation_name in observations:
            if observation_name in self.cfg_env.cameras.keys():
                continue
            observation = self._observations[observation_name]
            if observation.key == "obs":  # Only observations with key obs are included in the observation vector.
                assert len(observation.size) == 1, "Observations with key obs must be 1D."
                observations_start_end[observation_name] = (num_observations, num_observations + observation.size[0])
                num_observations += observation.size[0]
            
        return num_observations, observations_start_end
    
    def compute_observations(self) -> None:
        obs_tensors = []
        for observation_name in self.cfg_task.env.observations:
            observation = self._observations[observation_name]
            if observation.key == "obs":
                obs_tensors.append(observation.as_tensor())
            else:
                if observation.key not in self.obs_dict.keys():
                    self.obs_dict[observation.key] = {}
                self.obs_dict[observation.key] = observation.as_tensor()
            
        self.obs_buf = torch.cat(obs_tensors, dim=-1)
    
    def acquire_observation_tensors(self) -> None:
        # Fingertips provide a more direct task-space for the policy and enable useful reward formulations.
        self.fingertip_body_env_indices = [self.gym.find_actor_rigid_body_index(
            self.env_ptrs[0], self.controller_handles[0], ft_body_name, gymapi.DOMAIN_ENV
        ) for ft_body_name in self.controller.fingertip_body_names]

        for observation_name in self.ordered_observations:
            self._observations[observation_name].callback.on_init()

    def refresh_observation_tensors(self) -> None:
        for observation_name in self.ordered_observations:
            self._observations[observation_name].callback.on_step()

    def reset_observation_tensors(self) -> None:
        for observation_name in self.ordered_observations:
            self._observations[observation_name].callback.on_reset()
    