from dataclasses import dataclass
import torch
from typing import Callable, List, Optional


@dataclass
class ActionableCallback:
    post_init: Optional[Callable[[], None]] = lambda: None
    post_reset: Optional[Callable[[List[int]], None]] = lambda env_ids: None
    pre_step: Optional[Callable[[torch.Tensor], None]] = lambda actions: None


@dataclass
class ObservableCallback:
    post_init: Optional[Callable[[], None]] = lambda: None
    post_reset: Optional[Callable[[List[int]], None]] = lambda env_ids: None
    post_step: Optional[Callable[[], None]] = lambda: None

@dataclass
class CameraCallback(ObservableCallback):
    post_step_inside_gpu_access: Optional[Callable[[], None]] = lambda: None
    post_step_outside_gpu_access: Optional[Callable[[], None]] = lambda: None
