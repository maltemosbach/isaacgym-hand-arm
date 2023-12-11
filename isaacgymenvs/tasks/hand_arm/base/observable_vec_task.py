from isaacgym import gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.hand_arm.utils.camera import CameraSensor, IsaacGymCameraSensor, ROSCameraSensor, CameraSensorProperties, ImageType
from isaacgymenvs.tasks.hand_arm.utils.observables import Observable, ActiveObservables, ColorObservable, PointcloudObservable
import torch
from typing import Dict, List, Tuple, Optional



class ObservableVecTask(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self._active_observations = ActiveObservables()
        self.register_observables()

        # Acquire active/used observations (and teacher observations).
        self._active_observations.add([self._registered_observables[name] for name in self.cfg["env"]["observations"]])
        if "teacher_observations" in self.cfg["env"].keys():
            self._active_observations.add([self._registered_observables[name] for name in self.cfg["env"]["teacher_observations"]])

        # Acquire number of observations (and teacher observations).
        self.cfg["env"]["numObservations"], self.observations_start_end = self._compute_num_observations(cfg["env"]["observations"])
        if "teacher_observations" in self.cfg["env"].keys():
            self.cfg["env"]["numTeacherObservations"], self.teacher_observations_start_end = self._compute_num_observations(cfg["env"]["teacher_observations"])

        # Find order in wich observations are computed.
        self._sorted_observations = self._active_observations.sort(self._registered_observables)
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, True)

    def register_observables(self) -> None:
        self._registered_observables = {}

        self._register_camera_observables()

    def _register_camera_observables(self) -> None:
        # Fill camera dict with camera sensors that are used for observations.
        self._camera_dict = {}
        if "cameras" in self.cfg_env.keys():
            for camera_name, camera_cfg in self.cfg_env.cameras.items():
                image_types = []
                for observation_name in (self.cfg["env"]["observations"] + self.cfg["env"]["teacher_observations"] if "teacher_observations" in self.cfg["env"].keys() else self.cfg["env"]["observations"]):
                   if observation_name.startswith(camera_name):
                        image_types.append(observation_name.split("_")[-1])

                if image_types:
                    self._camera_dict[camera_name] = self.create_camera_sensor(**camera_cfg, image_types=image_types)

        # Register corresponding camera observables.
        for camera_name, camera_sensor in self._camera_dict.items():
            #camera_properties = CameraSensorProperties(**self.cfg_env.cameras[camera_name], image_types=["color"])
            for image_type in ["color", "depth", "segmentation", "pointcloud"]:
                if image_type == "color":
                    self.register_observable(
                        ColorObservable(
                            name=f"{camera_name}_{image_type}",
                            camera_sensor=camera_sensor,
                        )
                    )
                elif image_type == "pointcloud":
                    self.register_observable(
                        PointcloudObservable(
                            name=f"{camera_name}_{image_type}",
                            camera_sensor=camera_sensor,
                        )
                    )
                else:
                    pass

    def _acquire_camera_dict(self) -> Dict[str, CameraSensor]:
        camera_dict = {}
        if "cameras" in self.cfg_env.keys():
            for camera_name, camera_cfg in self.cfg_env.cameras.items():
                image_types = []
                for observation_name in self._sorted_observations:
                   if observation_name.startswith(camera_name):
                        image_types.append(observation_name.split("_")[-1])

                if image_types:
                    camera_dict[camera_name] = self.create_camera_sensor(**camera_cfg, image_types=image_types)
        return camera_dict
    
    def create_camera_sensor(
            self,
            image_types: List[str],
            pos: Tuple[float, float, float] = (0, 0, 0),
            quat: Tuple[float, float, float, float] = (0, 0, 0, 1),
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None
    ) -> None:
        if self.cfg_base.ros.activate:
            return ROSCameraSensor(image_types, pos, quat, model, fovx, resolution)
        else:
            return IsaacGymCameraSensor(image_types, pos, quat, model, fovx, resolution)
    
    def register_observable(self, observable: Observable) -> None:
        self._registered_observables[observable.name] = observable
        if observable.required:
            self._active_observations.add(observable)

    def _compute_num_observations(self, observations: List[str]) -> Tuple[int, Dict[str, Tuple[int, int]]]:
        num_observations = 0
        observations_start_end = {}

        for observation_name in observations:
            observable = self._registered_observables[observation_name]
            if observable.observation_key == "obs":  # Only observables with key obs are included in the observation vector.
                assert len(observable.size) == 1, "Observations with key obs must be 1D."
                observations_start_end[observation_name] = (num_observations, num_observations + observable.size[0])
                num_observations += observable.size[0]
            
        return num_observations, observations_start_end

    def acquire_simulation_tensors(self) -> None:
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)
        _contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies, 3)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.contact_force = gymtorch.wrap_tensor(_contact_force)

        # if self.cfg_base.control.type == "end_effector":
        #     _jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'robot')  # shape = (num envs, num_bodies - 1, 6, num_dofs)  -1 because the base is fixed
        #     _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'robot')  # shape = (num_envs, num_dofs, num_dofs)
        #     self.gym.refresh_jacobian_tensors(self.sim)
        #     self.gym.refresh_mass_matrix_tensors(self.sim)
        #     self.jacobian = gymtorch.wrap_tensor(_jacobian)
        #     self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 0:3]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 10:13]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_force = self.contact_force.view(self.num_envs, self.num_bodies, 3)[..., 0:3]

        # # Store actuation tensors.
        # self.current_dof_pos_targets = torch.zeros((self.num_envs, self.controller.dof_count), device=self.device)
        # self.current_actuated_dof_pos_targets = torch.zeros((self.num_envs, self.controller.actuated_dof_count), device=self.device)
        # self.previous_actuated_dof_pos_targets = torch.zeros((self.num_envs, self.controller.actuated_dof_count), device=self.device)
        # self.actuated_dof_pos = self.dof_pos[:, self.controller.actuated_dof_indices]
        # self.actuated_dof_vel = self.dof_vel[:, self.controller.actuated_dof_indices]

        # # End-effector (EEF) body is the one controled via inverse kinematics if that is the control mode.
        # self.eef_body_env_index = self.gym.find_actor_rigid_body_index(
        #     self.env_ptrs[0], self.controller_handles[0], self.cfg_base.control.end_effector.body_name, gymapi.DOMAIN_ENV
        # )
        # self.eef_body_pos = self.body_pos[:, self.eef_body_env_index, 0:3]
        # self.eef_body_quat = self.body_quat[:, self.eef_body_env_index, 0:4]
        # self.eef_body_linvel = self.body_linvel[:, self.eef_body_env_index, 0:3]
        # self.eef_body_angvel = self.body_angvel[:, self.eef_body_env_index, 0:3]

    def refresh_simulation_tensors(self) -> None:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # if self.cfg_base.control.type == "end_effector":
        #     self.gym.refresh_jacobian_tensors(self.sim)
        #     self.gym.refresh_mass_matrix_tensors(self.sim)
    
    def compute_observations(self) -> None:
        obs_tensors = []
        for observation_name in self.cfg["env"]["observations"]:
            observable = self._registered_observables[observation_name]

            if observable.observation_key == "obs":
                obs_tensors.append(observable.get_state())
            else:
                self.obs_dict[observable.observation_key] = observable.get_state()
        self.obs_buf[:] = torch.cat(obs_tensors, dim=-1)

        if "teacher_observations" in self.cfg["env"].keys():
            teacher_obs_tensors = []
            self.obs_dict["teacher"] = {}
            for observation_name in self.cfg["env"]["teacher_observations"]:
                observable = self._registered_observables[observation_name]
                if observable.observation_key == "obs":
                    teacher_obs_tensors.append(observable.get_state())
                else:
                    self.obs_dict["teacher"][observable.observation_key] = observable.get_state()
            self.teacher_obs_buf[:] = torch.cat(teacher_obs_tensors, dim=-1)
    
    @property
    def observation_keys(self) -> List[str]:
        keys = []
        for observation_name in self.cfg["env"]["observations"]:
            if self._registered_observables[observation_name].observation_key not in keys:
                keys.append(self._registered_observables[observation_name].observation_key)
        return keys

    def draw_visualizations(self, visualizations: List[str]) -> None:
        for visualization in visualizations:
            # If an observation is visualized.
            if visualization in self._active_observations._active_observables.keys():
                self._registered_observables[visualization].visualize(self.gym, self.viewer, self.env_ptrs)

            # Call any other visualization functions (e.g. workspace extent, etc.).
            else:
                getattr(self, "visualize_" + visualization)()

    
