from abc import ABC
from typing import Optional
from isaacgymenvs.tasks.hand_arm.utils.callbacks import  ActionableCallback


class Actionable(ABC):
    """Base class for actionables (things that can be actuated).

    Represents an element of the action-space and interacts with the simulation or real-robot based on the action.

    Args:
    """
    def __init__(
        self,
        name: str,
        size: int,
        callback: Optional[ActionableCallback] = ActionableCallback(),
    ) -> None:
        self.name = name
        self.size = size
        self.callback = callback
    