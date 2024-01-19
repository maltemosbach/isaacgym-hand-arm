from abc import ABC
from collections import defaultdict
import contextlib
from dataclasses import dataclass
import numpy as np
import time
import torch    
from typing import Any, Callable, Dict, Optional, Sequence, Union



class ReinforcementLearningEntity(ABC):
    """Base class for reinforcement learning entities.
    
    Actionables (things that can be actuated) and observable (things that can be observed) inherit from this class.

    Args:
        name (str): The name of the entity.
        size (int or sequence[int]): The size/shape of the entity tensor.
        state_cache (dict): A dictionary containing the state of the simulation and potentially other entities.
        get_state (callable): A callable that returns the state of the entity.
        set_state (callable): A callable that sets the state of the entity.
        visualize (callable, optional): Debug function that visualizes the entity when called.
    """
    def __init__(
        self,
        name: str,
        size: Union[int, Sequence[int]],
        get_state: Callable[[], torch.Tensor],
        set_state: Callable[[torch.Tensor], None],
        visualize: Optional[Callable[[], None]] = lambda: None
    ) -> None:
        self.name = name
        self.size = size if isinstance(size, Sequence) else (size,)
        self._get_state = get_state
        self._set_state = set_state
        self.visualize = visualize
    
    def check_tensor_data(self, tensor_data: torch.Tensor) -> None:
        if not isinstance(tensor_data, torch.Tensor):
            raise TypeError(f"Expected data of type torch.Tensor, but got {type(tensor_data)} for {self.name}.")
        
        if tensor_data.shape[1:] != self.size:
            raise ValueError(f"Expected data of shape {self.size}, but got {tensor_data.shape[1:]} for {self.name}.")
        
        return tensor_data
    
    def get_state(self) -> torch.Tensor:
        tensor_data = self._get_state()
        return self.check_tensor_data(tensor_data)
    
    def set_state(self, tensor_data: torch.Tensor) -> None:
        self.check_tensor_data(tensor_data)
        self._set_state(tensor_data)


@dataclass
class RefreshCallback:
    on_init: Optional[Callable[[Dict[str, Any]], None]] = lambda: None
    on_step: Optional[Callable[[Dict[str, Any]], None]] = lambda: None
    on_reset: Optional[Callable[[Dict[str, Any]], None]] = lambda: None

    def 



class Statistic:
    def __init__(self) -> None:
        self.min_duration = np.inf
        self.max_duration = 0.0
        self.total_duration = 0.0
        self.count = 0

    def add(self, duration: float) -> None:
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.total_duration += duration
        self.count += 1

    def summary(self) -> Dict[str, Any]:
        return {
            'minimum': self.min_duration,
            'maximum': self.max_duration,
            'average': self.total_duration / self.count,
            'count': self.count,
        }


class Profiler:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.stats = defaultdict(Statistic)
        self.start = time.time()

    @contextlib.contextmanager
    def section(self, name: str):
        start = time.time()

        try:
            yield
        finally:
            duration = time.time() - start

        self.stats[name].add(duration)

        if self.verbose:
            print(f"{name}: Last FPS: {1 / duration:.2f} | Average FPS: {self.stats[name].count / self.stats[name].total_duration:.2f}")

    def wrap(self, name: str, obj: Any, methods: Sequence[str]) -> None:
        for method in methods:
            decorator = self.section(f"{name}.{method}")
            setattr(obj, method, decorator(getattr(obj, method)))

    def summary(self) -> Dict[str, Any]:
        return {name: stat.summary() for name, stat in self.stats.items()}
