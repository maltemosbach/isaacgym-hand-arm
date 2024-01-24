from abc import ABC
import torch
from typing import Dict, Optional


class Transform(ABC):
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ToVector(Transform):
    """Flatten the input tensor to a vector after the batch dimension."""
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return sample.flatten(start_dim=1)


class FlattenPointcloud(Transform):
    """Flatten pointcloud to be of shape [batch_size, num_points, 4]."""
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        return sample.flatten(start_dim=1, end_dim=-2)


class InvervalSample(Transform):
    """Corrupter function that makes observations only visible at a certain interval. To test history-awareness."""
    def __init__(self, interval: int) -> None:
        self.interval = interval
        self.step = -1

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        self.step += 1
        if self.step % self.interval == 0:
            return sample
        else:
            return torch.zeros_like(sample)


# Could add randomization etc. here later.

# Add compose transforms here later.

