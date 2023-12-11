from collections import OrderedDict
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.hand_arm.utils.actionables import Actionable
from typing import List



class ActionableVecTask:
    def __init__(self):
        self.register_actionables()
        self._sorted_actions = OrderedDict()

        for action in self.cfg["env"]["actions"]:
            self._sorted_actions[action] = self._registered_actionables[action]
        
        self.cfg["env"]["numActions"] = self._compute_num_actions(self.cfg["env"]["actions"])

    def register_actionables(self) -> None:
        self._registered_actionables = {}

    def register_actionable(self, actionable: Actionable) -> None:
        self._registered_actionables[actionable.name] = actionable

    def _compute_num_actions(self, actionables: List[str]) -> int:
        num_actions = 0

        for actionable_name in actionables:
            actionable = self._registered_actionables[actionable_name]
            num_actions += actionable.size
            
        return num_actions
    
    def apply_actions_to_sim(self) -> None:
        self.set_dof_position_targets()

        # Other action modalities such as applying forces to bodies can be added here.
    
    def set_dof_position_targets(self) -> None:
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))
