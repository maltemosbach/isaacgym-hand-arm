from isaacgymenvs.tasks.hand_arm.env.multi_object import HandArmEnvMultiObject
import hydra
from omegaconf import DictConfig
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
import numpy as np


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))



class HandArmTaskMultiObjectManipulation(HandArmEnvMultiObject):
    _task_cfg_path = 'task/HandArmTaskMultiObjectManipulation.yaml'

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_task = self._acquire_task_cfg()
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required for VecTask
        self.objects_dropped = False
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)


    def _acquire_task_cfg(self) -> DictConfig:
        return hydra.compose(config_name=self._task_cfg_path)['task']
    

    def reset_idx(self, env_ids):
        if self.objects_dropped:
            self._reset_objects(env_ids)
        else:
            self._reset_robot(env_ids, reset_dof_pos=self.cfg_base.asset.joint_configurations.bringup)
            self._disable_object_collisions(object_ids=range(self.cfg_env.objects.num_objects))
            self._place_objects_before_bin()
            self._drop_objects(env_ids)
            self.objects_dropped = True
            self._reset_objects(env_ids)

        #self._reset_goal(env_ids)
        self._reset_robot(env_ids, reset_dof_pos=self.cfg_base.asset.joint_configurations.reset)
        self._reset_buffers(env_ids)

    def _reset_objects(self, env_ids):
        for i, object_index in enumerate(self.object_actor_env_indices):
            self.root_pos[env_ids, object_index] = self.object_pos_initial[env_ids, i]
            self.root_quat[env_ids, object_index] = self.object_quat_initial[env_ids, i]
            self.root_linvel[env_ids, object_index] = 0.0
            self.root_angvel[env_ids, object_index] = 0.0

        # Reset goal to random pose.
        # if "reposition" in self.cfg_task.rl.goal:
        #     self.goal_pos[env_ids] = self._get_random_goal_pos(env_ids)

        #     if self.cfg_task.rl.goal == "oriented_reposition":
        #         self.goal_manipulator_quat[env_ids] = self._get_random_quat(env_ids)

        # if len(self.goal_handles) > 0:
        #     if "reposition" in self.cfg_task.rl.goal:
        #         self.root_pos[env_ids, self.goal_actor_id_env] = self.goal_pos[env_ids]

        #     #self.root_quat[env_ids, self.goal_actor_id_env] = self.goal_quat[env_ids]
        #     reset_indices = torch.cat((self.object_actor_ids_sim[env_ids].to(torch.int32).flatten(), self.goal_actor_ids_sim[env_ids].to(torch.int32)))
        # else:
            
        reset_indices = self.object_actor_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

    def _drop_objects(self, env_ids):
        all_env_ids = env_ids.clone()
        objects_in_bin = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects), dtype=torch.bool, device=self.device)
        
        self.object_pos_initial = self.root_pos[:, self.object_actor_env_indices].detach().clone()
        self.object_quat_initial = self.root_quat[:, self.object_actor_env_indices].detach().clone()

        while not torch.all(objects_in_bin):
            # loop through object actors per bin
            for i in range(self.cfg_env.objects.num_objects):
                self._enable_object_collisions(object_ids=[i])
                # Check for which env_ids this object must still be dropped
                env_ids = torch.masked_select(all_env_ids, ~objects_in_bin[:, i])

                if len(env_ids) > 0:
                    print(f"Objects {i} must still be dropped in envs:", env_ids)

                    # Randomize drop position and orientation of object
                    object_pos_drop = self._get_random_drop_pos(env_ids)
                    object_quat_drop = self._get_random_quat(env_ids)

                    # Set root state tensor of the simulation
                    self.root_pos[env_ids, self.object_actor_env_indices[i]] = object_pos_drop
                    self.root_quat[env_ids, self.object_actor_env_indices[i]] = object_quat_drop
                    self.root_linvel[env_ids, self.object_actor_env_indices[i]] = 0.0
                    self.root_angvel[env_ids, self.object_actor_env_indices[i]] = 0.0

                    object_actor_indices = self.object_actor_indices[env_ids, i]
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(object_actor_indices), len(object_actor_indices)
                    )

                    # Step simulation to drop objects
                    for _ in range(self.cfg_env.objects.drop.num_steps):
                        self.gym.simulate(self.sim)
                        self.render()
                        self.refresh_base_tensors()
                        self.refresh_env_tensors()
                        if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
                            self.gym.clear_lines(self.viewer)
                            self.draw_visualizations(self.cfg_base.debug.visualize)

            # Refresh tensor and save initial object poses.
            self.refresh_base_tensors()
            #self.refresh_env_tensors()

            objects_in_bin[:] = self.objects_in_bin()

            print("objects_in_bin", objects_in_bin)

        # Let the scene settle.
        for _ in range(600):
            self.gym.simulate(self.sim)
            self.render()
            self.refresh_base_tensors()
            self.refresh_env_tensors()

        self.object_pos_initial[:] = self.object_pos.detach().clone()
        self.object_quat_initial[:] = self.object_quat.detach().clone()


    def _place_objects_before_bin(self):
        # Place the objects in front of the bin
        self.root_pos[:, self.object_actor_env_indices, 0] = 1.1
        self.root_pos[:, self.object_actor_env_indices, 1] = 0.0
        self.root_pos[:, self.object_actor_env_indices, 2] = 0.5
        self.root_linvel[:, self.object_actor_env_indices] = 0.0
        self.root_angvel[:, self.object_actor_env_indices] = 0.0

        reset_indices = self.object_actor_indices.flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

        for _ in range(1):
            self.gym.simulate(self.sim)
            self.render()

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        object_drop_pos = torch.tensor(self.cfg_env.objects.drop.pos, device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        object_drop_noise = 2 * (torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        object_drop_noise = object_drop_noise @ torch.diag(torch.tensor(self.cfg_env.objects.drop.noise, device=self.device))
        object_drop_pos += object_drop_noise
        return object_drop_pos

    def _get_random_quat(self, env_ids) -> torch.Tensor:
        x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_quat = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)
        return object_quat
    

    def _update_reset_buf(self):
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def _update_rew_buf(self):
        pass
