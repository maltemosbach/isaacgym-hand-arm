from dataclasses import dataclass, field
from isaacgym import gymapi
import networkx as nx
import matplotlib.pyplot as plt
import torch
from typing import Callable, Dict, List, Tuple


@dataclass
class Observation:
    size: Tuple[int, ...]
    data: Callable[[], torch.Tensor]
    key: str = field(default="obs")
    is_mandatory: bool = field(default=False)
    requires: List[str] = field(default_factory=list)
    acquire: Callable[[], None] = field(default=lambda: None)
    refresh: Callable[[], None] = field(default=lambda: None)
    visualize: Callable[[], None] = field(default=lambda: None)

class ObserverMixin:
    observations: Dict[str, Observation] = {}
    mandatory_observations: List[str] = []

    def register_observation(self, name: str, observation: Observation) -> None:
        self.observations[name] = observation
        if observation.is_mandatory:
            self.mandatory_observations.append(name)

    def register_observations(self) -> None:
        """Registers common proprioceptive observations."""

        self.register_observation(
            "actuated_dof_pos", 
            Observation(
                size=(self.controller.actuated_dof_count,),
                data=lambda: self.actuated_dof_pos,
                acquire=lambda: setattr(self, "actuated_dof_pos", self.dof_pos[:, self.controller.actuated_dof_indices]),
                refresh=lambda: self.actuated_dof_pos.copy_(self.dof_pos[:, self.controller.actuated_dof_indices])
            )
        )
        self.register_observation(
            "actuated_dof_vel", 
            Observation(
                size=(self.controller.actuated_dof_count,),
                data=lambda: self.actuated_dof_vel,
                acquire=lambda: setattr(self, "actuated_dof_vel", self.dof_vel[:, self.controller.actuated_dof_indices]),
                refresh=lambda: self.actuated_dof_vel.copy_(self.dof_vel[:, self.controller.actuated_dof_indices])
            )
        )

        self.register_observation(
            "fingertip_pos",
            Observation(
                size=(3 * self.controller.fingertip_count,),
                data=lambda: self.fingertip_pos.flatten(1, 2),
                is_mandatory=True,  # Required for reward computation.
                acquire=lambda: setattr(self, "fingertip_pos", self.body_pos[:, self.fingertip_body_env_indices, 0:3]),
                refresh=lambda: self.fingertip_pos.copy_(self.body_pos[:, self.fingertip_body_env_indices, 0:3]),
                visualize=lambda: self.visualize_pose(self.fingertip_pos, self.fingertip_quat),
                
            )
        )
        self.register_observation(
            "fingertip_quat",
            Observation(
                size=(4 * self.controller.fingertip_count,),
                data=lambda: self.fingertip_quat.flatten(1, 2),
                acquire=lambda: setattr(self, "fingertip_quat", self.body_quat[:, self.fingertip_body_env_indices, 0:4]),
                refresh=lambda: self.fingertip_quat.copy_(self.body_quat[:, self.fingertip_body_env_indices, 0:4]),
                visualize=lambda: self.visualize_pose(self.fingertip_pos, self.fingertip_quat),
            )
        )
        self.register_observation(
            "fingertip_linvel",
            Observation(
                size=(3 * self.controller.fingertip_count,),
                data=lambda: self.fingertip_linvel.flatten(1, 2),
                acquire=lambda: setattr(self, "fingertip_linvel", self.body_linvel[:, self.fingertip_body_env_indices, 0:3]),
                refresh=lambda: self.fingertip_linvel.copy_(self.body_linvel[:, self.fingertip_body_env_indices, 0:3]),
            )
        )
        self.register_observation(
            "fingertip_angvel",
            Observation(
                size=(3 * self.controller.fingertip_count,),
                data=lambda: self.fingertip_angvel.flatten(1, 2),
                acquire=lambda: setattr(self, "fingertip_angvel", self.body_angvel[:, self.fingertip_body_env_indices, 0:3]),
                refresh=lambda: self.fingertip_angvel.copy_(self.body_angvel[:, self.fingertip_body_env_indices, 0:3]),
            )
        )

    @property
    def observation_dependency_graph(self) -> nx.DiGraph:
        dependency_dict = {}
        for observation_name in self.cfg_task.env.observations:
            if not self.observations[observation_name].is_mandatory:  # Mandatory observations are per definition independent.
                dependency_dict[observation_name] = self.observations[observation_name].requires
        
        return nx.DiGraph(dependency_dict)
    
    def _compute_num_observations(self, observations: List[str]) -> Tuple[int, Dict[str, Tuple[int, int]]]:
        num_observations = 0
        observations_start_end = {}

        for observation_name in observations:
            if observation_name in self.cfg_env.cameras.keys():
                continue
            observation = self.observations[observation_name]
            if observation.key == "obs":  # Only observations with key obs are included in the observation vector.
                assert len(observation.size) == 1, "Observations with key obs must be 1D."
                observations_start_end[observation_name] = (num_observations, num_observations + observation.size[0])
                num_observations += observation.size[0]
            
        return num_observations, observations_start_end
    
    def compute_observations(self) -> None:
        obs_tensors = []
        for observation_name in self.cfg_task.env.observations:
            if observation_name in self.cfg_env.cameras.keys():
                continue
            observation = self.observations[observation_name]
            if observation.key == "obs":  
                obs_tensors.append(observation.data())
            else:
                if observation.key not in self.obs_dict.keys():
                    self.obs_dict[observation.key] = {}
                
                self.obs_dict[observation.key][observation_name] = observation.data()
            
        self.obs_buf = torch.cat(obs_tensors, dim=-1)
    
    def acquire_observation_tensors(self) -> None:
        # Fingertips provide a more direct task-space for the policy and enable useful reward formulations.
        self.fingertip_body_env_indices = [self.gym.find_actor_rigid_body_index(
            self.env_ptrs[0], self.controller_handles[0], ft_body_name, gymapi.DOMAIN_ENV
        ) for ft_body_name in self.controller.fingertip_body_names]

        self.ordered_observations = self.mandatory_observations + list(reversed(list(nx.topological_sort(self.observation_dependency_graph))))

        for observation_name in self.ordered_observations:
            self.observations[observation_name].acquire()

    def refresh_observation_tensors(self) -> None:
        for observation_name in self.ordered_observations:
            self.observations[observation_name].refresh()
    