from isaacgym import gymapi
from isaacgymenvs.tasks.hand_arm.base.actionable_vec_task import ActionableVecTask
from isaacgymenvs.tasks.hand_arm.base.observable_vec_task import ObservableVecTask
from isaacgymenvs.tasks.hand_arm.utils.observables import CameraObservable
import torch
from typing import Dict, Any


class ConfigurableVecTask(ObservableVecTask, ActionableVecTask):
    """Implements a configurable vectorized MDP, where the action and observation space can be 
    defined in a config file.
    """
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg  # Required for VecTask.
        self.cfg["headless"] = headless  # Required for VecTask.
        
        ActionableVecTask.__init__(self)
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, True)

        self.acquire_simulation_tensors()  # Acquire base simulation tensors.

        if self.viewer is not None:
            self._set_viewer_params()

        for actionable in self._sorted_actions.values():
            actionable.callback.post_init()

        for observable in self._sorted_observations.values():
            observable.callback.post_init()

        self.log_data = {}  

        for observable in self._sorted_observations.values():
            observable.callback.post_step()

        if self._camera_dict:
            if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
                self.gym.clear_lines(self.viewer)

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
                
            if self.headless:
                self.gym.step_graphics(self.sim)

            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            for observable in self._sorted_observations.values():
                if isinstance(observable, CameraObservable):
                    observable.callback.post_step_inside_gpu_access()
            self.gym.end_access_image_tensors(self.sim)

            for observable in self._sorted_observations.values():
                if isinstance(observable, CameraObservable):
                    observable.callback.post_step_outside_gpu_access()

    def log(self, data: Dict[str, Any]) -> None:
            self.log_data.update(data)

    def pre_physics_step(self, actions: torch.Tensor):
        reset_env_indices = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        action_start_index = 0
        for actionable in self._sorted_actions.values():
            actionable.callback.pre_step(actions[:, action_start_index:action_start_index + actionable.size])
            action_start_index += actionable.size
        self.apply_actions_to_sim()
        
        if len(reset_env_indices) > 0:
            self.reset_idx(reset_env_indices)

    def post_physics_step(self):
        self.progress_buf[:] += 1

        self.refresh_simulation_tensors()  # Refresh base simulation tensors.

        for observable in self._sorted_observations.values():
            observable.callback.post_step()

        if self._camera_dict:
            if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
                self.gym.clear_lines(self.viewer)

            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
                
            if self.headless:
                self.gym.step_graphics(self.sim)

            self.gym.render_all_camera_sensors(self.sim)

            self.gym.start_access_image_tensors(self.sim)
            for observable in self._sorted_observations.values():
                if isinstance(observable, CameraObservable):
                    observable.callback.post_step_inside_gpu_access()
            self.gym.end_access_image_tensors(self.sim)

            for observable in self._sorted_observations.values():
                if isinstance(observable, CameraObservable):
                    observable.callback.post_step_outside_gpu_access()

        self.compute_reward()
        self.compute_observations()

        if len(self.cfg_base["debug"]["visualize"]) > 0 and not self.headless:
            self.gym.clear_lines(self.viewer)

            self.draw_visualizations(self.cfg_base["debug"]["visualize"])

    def compute_reward(self):
        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        raise NotImplementedError
    
    def _update_rew_buf(self):
        raise NotImplementedError
    
    def _reset_buffers(self, env_ids) -> None:
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def reset_idx(self, env_ids):
        for actionable in self._sorted_actions.values():
            actionable.callback.post_reset(env_ids)

        for observable in self._sorted_observations.values():
            observable.callback.post_reset(env_ids)

        self._reset_buffers(env_ids)

    def _set_viewer_params(self):
        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

