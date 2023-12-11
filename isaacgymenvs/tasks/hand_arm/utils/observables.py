from abc import ABC
from collections import OrderedDict
from functools import partial
import matplotlib.pyplot as plt
import networkx as nx
import torch
from typing import Callable, Sequence, Optional, Union
from isaacgymenvs.tasks.hand_arm.utils.visualization import *
from isaacgymenvs.tasks.hand_arm.utils.callbacks import  CameraCallback, ObservableCallback
from isaacgymenvs.tasks.hand_arm.utils.camera import ImageType, CameraSensor
from isaacgymenvs.tasks.hand_arm.utils.transforms import Transform, ToVector


class Observable(ABC):
    """Base class for observables (things that can be observed).

    Exposes information about the state of the environment, e.g., joint-angles, images, etc., to the agent.
    
    Args:
        name (str): The name of the observable.
        observation_key (str): The key of the observable in the observation dictionary.
        size (int or sequence[int]): The size/shape of the observation tensor.
        transfrom (Transform, optional): A function that is applied to the observation before returning it as a tensor.
        refresh (RefreshCallback, optional): Callbacks that are called when the environment is initialized, reset, or stepped.
        required (bool, optional): Whether the observable is required for the environment to function.
        requires (sequence[str], optional): A list of names of observables that are required for this observable to function.
        visualize (callable, optional): Debug function that visualizes the observation when called.
    """
    def __init__(
        self,
        name: str,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]],
        observation_key: str,
        transform: Optional[Transform] = None,
        callback: ObservableCallback = ObservableCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
        visualize: Optional[Callable[[], None]] = lambda: None
    ) -> None:
        self.name = name
        self._get_state = get_state
        self.size = size if isinstance(size, Sequence) else (size,)
        self.observation_key = observation_key
        self.transform = transform
        self.callback = callback
        self.required = required
        self.requires = requires if requires else []
        self.visualize = visualize

    def check_tensor_data(self, tensor_data: torch.Tensor) -> None:
        if not isinstance(tensor_data, torch.Tensor):
            raise TypeError(f"Expected data of type torch.Tensor, but got {type(tensor_data)} for {self.name}.")
        
        if tensor_data.shape[1:] != self.size:
            raise ValueError(f"Expected data of shape {self.size}, but got {tensor_data.shape[1:]} for {self.name}.")
    
    def get_state(self) -> torch.Tensor:
        tensor_data = self._get_state()
        if self.transform:
            tensor_data = self.transform(tensor_data)

        self.check_tensor_data(tensor_data)
        return tensor_data
    

class LowDimObservable(Observable):
    def __init__(
        self,
        name: str,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]],
        callback: ObservableCallback = ObservableCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
        visualize: Optional[Callable[[], None]] = lambda: None
    ) -> None:
        super().__init__(name, get_state, size, "obs", ToVector(), callback, required, requires, visualize)

        if not len(self.size) == 1:
            raise ValueError("Low-dimensional observations must be one-dimensional, but got size {self.size} for {self.name}.")


class PosObservable(LowDimObservable):
    def __init__(
        self,
        name: str,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]] = 3,
        callback: ObservableCallback = ObservableCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name, get_state, size, callback, required, requires, partial(visualize_pos, get_pos=get_state))

        if not self.size[0] % 3 == 0:
            raise ValueError("Position observations should be divisible by three, but got size {self.size} for {self.name}.")

    
class PoseObservable(LowDimObservable):
    def __init__(
        self,
        name: str,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]] = 7,
        callback: ObservableCallback = ObservableCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name, get_state, size, callback, required, requires, partial(visualize_pose, get_pose=get_state))

        if not self.size[0] % 7 == 0:
            raise ValueError("Pose observations should be divisible by seven, but got size {self.size} for {self.name}.")


class BoundingBoxObservable(LowDimObservable):
    def __init__(
        self,
        name: str,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]] = 10,
        callback: ObservableCallback = ObservableCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name, get_state, size, callback, required, requires, partial(visualize_bounding_box, get_bounding_box=get_state))

        if not self.size[0] % 10 == 0:
            raise ValueError("Bounding box observations should be divisible by ten (pos + quat + extents), but got size {self.size} for {self.name}.")
        

class CameraObservable(Observable):
    def __init__(
        self,
        name: str,
        camera_sensor: CameraSensor,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]],
        transform: Optional[Transform] = None,
        callback: CameraCallback = CameraCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
        visualize: Optional[Callable[[], None]] = lambda: None
    ) -> None:
        super().__init__(name, get_state, size, name, transform, callback, required, requires, visualize)  # Pass name as observation_key to store image in observation dictionary.

        #if not any(name.endswith(image_type.value) for image_type in ImageType):
        #    raise ValueError(f"Camera observables must end with one of {ImageType}, but got {self.name}.")


class ColorObservable(CameraObservable):
    def __init__(
        self,
        name: str,
        camera_sensor: CameraSensor,
        transform: Optional[Transform] = None,
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name, camera_sensor, lambda: camera_sensor.current_sensor_observation[ImageType.COLOR], (camera_sensor.height, camera_sensor.width, 3), transform, CameraCallback(post_step_inside_gpu_access=camera_sensor.refresh_color), required, requires, partial(visualize_color_image, get_color_image=lambda: camera_sensor.current_sensor_observation[ImageType.COLOR], window_name=name))


class PointcloudObservable(CameraObservable):
    def __init__(
        self,
        name: str,
        camera_sensor: CameraSensor,
        transform: Optional[Transform] = None,
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name, camera_sensor, lambda: camera_sensor.current_sensor_observation[ImageType.POINTCLOUD], (camera_sensor.height, camera_sensor.width, 4), transform, CameraCallback(post_step_inside_gpu_access=camera_sensor.refresh_depth, post_step_outside_gpu_access=camera_sensor.refresh_pointcloud), required, requires, partial(visualize_pos, get_pos=lambda: camera_sensor.current_sensor_observation[ImageType.POINTCLOUD], marker="sphere", size=0.005))


class SyntheticPointcloudObservable(Observable):
    def __init__(
        self,
        name: str,
        get_state: Callable[[], torch.Tensor],
        size: Union[int, Sequence[int]],
        transform: Optional[Transform] = None,
        callback: CameraCallback = CameraCallback(),
        required: bool = False,
        requires: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(name, get_state, size, name, transform, callback, required, requires, partial(visualize_pos, get_pos=get_state, marker="sphere", size=0.005))  # Pass name as observation_key to store synthetic pointcloud in observation dictionary.

        if not name.endswith("_pointcloud"):
            raise ValueError(f"Synthetic pointcloud observables must end with '_pointcloud', but got {self.name}.")
        
        if not size[-1] == 4:
            raise ValueError(f"Synthetic pointcloud observables must have four channels, but got {self.size[-1]} channels for {self.name}.")


class ActiveObservables:
    def __init__(self) -> None:
        self._active_observables = {}

    def add(self, observables: Union[Observable, Sequence[Observable]]) -> None:
        if isinstance(observables, Observable):
            observables = [observables]
        
        for observable in observables:
            if observable.name not in self._active_observables:
                self._active_observables[observable.name] = observable

    def sort(self, registered_observables, debug: bool = False) -> OrderedDict:
        dependency_graph = self.build_observables_dependency_graph(registered_observables)
        sorted_observable_names = list(reversed(list(nx.topological_sort(dependency_graph))))

        if debug:
            print("Sorted observables: ", sorted_observable_names)
            nx.draw_networkx(dependency_graph)
            plt.show()

        sorted_observables = OrderedDict()
        for observable_name in sorted_observable_names:
            sorted_observables[observable_name] = registered_observables[observable_name]
        return sorted_observables

    def build_observables_dependency_graph(self, registered_observables) -> nx.DiGraph:
        dependency_graph = {}

        observations_to_explore = [observation_name for observation_name in self._active_observables.keys()]
        while observations_to_explore:
            current_observation = observations_to_explore.pop()
            dependency_graph[current_observation] = []
            for current_required_observation in registered_observables[current_observation].requires:
                if current_required_observation not in dependency_graph:
                    observations_to_explore.append(current_required_observation)
                dependency_graph[current_observation].append(current_required_observation)
        
        return nx.DiGraph(dependency_graph)
    