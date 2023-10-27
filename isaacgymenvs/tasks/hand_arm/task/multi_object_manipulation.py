from isaacgymenvs.tasks.hand_arm.env.multi_object import HandArmEnvMultiObject
from isaacgymenvs.tasks.hand_arm.base.observations import Observation
import hydra
from omegaconf import DictConfig
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
import numpy as np
from typing import *


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))



class HandArmTaskMultiObjectManipulation(HandArmEnvMultiObject):
    _task_cfg_path = 'task/HandArmTaskMultiObjectManipulation.yaml'

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_task = self._acquire_task_cfg()
        self.max_episode_length = self.cfg_task.rl.reset.max_episode_length  # required for VecTask

        self.objects_dropped = False

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _acquire_task_cfg(self) -> DictConfig:
        return hydra.compose(config_name=self._task_cfg_path)['task']
    
    def register_observations(self) -> None:
        super().register_observations()
        self.register_observation(
            'fingertip_closest_distance',
            Observation(
                size=(1,),
                data=lambda: self.fingertip_closest_distance,
                is_mandatory=True,  # Required to save initial object pose.
                acquire=lambda: setattr(self, "object_pos", self.root_pos[:, self.object_actor_env_indices, 0:3]),
                refresh=lambda: self.object_pos.copy_(self.root_pos[:, self.object_actor_env_indices, 0:3]),
                visualize=lambda: self.visualize_poses(self.object_pos, self.object_quat),
            )
        )

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

        self._reset_target_object(env_ids)
        self._reset_goal(env_ids)
        self._reset_robot(env_ids, reset_dof_pos=self.cfg_base.asset.joint_configurations.reset)

        self.gym.simulate(self.sim)  # Simulate for one step and refresh base tensors to update the buffers (i.e. fingertip positions etc.)
        self.refresh_base_tensors()

        self._reset_buffers(env_ids)

    def _reset_objects(self, env_ids):
        for i, object_index in enumerate(self.object_actor_env_indices):
            self.root_pos[env_ids, object_index] = self.object_pos_initial[env_ids, i]
            self.root_quat[env_ids, object_index] = self.object_quat_initial[env_ids, i]
            self.root_linvel[env_ids, object_index] = 0.0
            self.root_angvel[env_ids, object_index] = 0.0
            
        reset_indices = self.object_actor_indices[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

    def _drop_objects(self, env_ids):
        all_env_ids = env_ids.clone()
        objects_in_bin = torch.zeros((self.num_envs, self.cfg_env.objects.num_objects), dtype=torch.bool, device=self.device)

        while not torch.all(objects_in_bin):
            # loop through object actors per bin
            for i in range(self.cfg_env.objects.num_objects):
                self._enable_object_collisions(object_ids=[i])
                # Check for which env_ids this object must still be dropped
                env_ids = torch.masked_select(all_env_ids, ~objects_in_bin[:, i])

                if len(env_ids) > 0:
                    print(f"Objects {i} must still be dropped in envs:", env_ids)

                    # Randomize drop position and orientation of object
                    object_pos_drop = self._get_random_object_pos(env_ids, 'drop')
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
                        if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
                            self.gym.clear_lines(self.viewer)
                            self.draw_visualizations(self.cfg_base.debug.visualize)

            # Refresh tensor and save initial object poses.
            self.refresh_base_tensors()
            objects_in_bin[:] = self.objects_in_bin(self.object_pos)

        # Let the scene settle.
        for _ in range(400):
            self.gym.simulate(self.sim)
            self.render()
        
        self.refresh_base_tensors()

        for attribute in dir(self):
            if attribute.startswith("object_") and isinstance(getattr(self, attribute), torch.Tensor):
                setattr(self, attribute + "_initial", getattr(self, attribute).detach().clone())

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

    def _get_random_object_pos(self, env_ids, key: str) -> torch.Tensor:
        pos = torch.tensor(getattr(self.cfg_env.objects, key).pos, device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        noise = 2 * (torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        noise = noise @ torch.diag(torch.tensor(getattr(self.cfg_env.objects, key).noise, device=self.device))
        pos += noise
        return pos

    def _get_random_quat(self, env_ids) -> torch.Tensor:
        x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((len(env_ids), 1))
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_quat = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)
        return object_quat
    
    def _reset_target_object(self, env_ids, verbose: bool = True):
        if self.cfg_env.debug.highlight_target_object and not self.headless:
            for env_id in env_ids:
                old_target_object_handle = self.object_handles[env_id][self.target_object_index[env_id]]
                self.gym.reset_actor_materials(self.env_ptrs[env_id], old_target_object_handle, gymapi.MESH_VISUAL)
                
        self.target_object_index[:] = torch.randint(0, self.cfg_env.objects.num_objects, (len(env_ids),), dtype=torch.int64, device=self.device)
        self.target_object_actor_env_index[:] = torch.Tensor(self.object_actor_env_indices).to(dtype=torch.int64, device=self.device)[self.target_object_index]

        if self.cfg_env.debug.highlight_target_object and not self.headless:
            for env_id in env_ids:
                target_object_handle = self.object_handles[env_id][self.target_object_index[env_id]]
                self.gym.set_rigid_body_color(self.env_ptrs[env_id], target_object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*self.cfg_env.debug.target_object_color))

        if verbose and not self.headless:
            for env_id in env_ids:
                print(f"Target object in env {env_id} is '{self.objects[self.object_indices[env_id][self.target_object_index[env_id]]].name}'.")

    def _reset_goal(self, env_ids) -> None:
        self.goal_pos[env_ids] = self._get_random_object_pos(env_ids, 'goal')
        self.root_pos[env_ids, self.goal_actor_env_index] = self.goal_pos[env_ids]

        if self.cfg_task.rl.goal == "oriented_reposition":
            self.goal_manipulator_quat[env_ids] = self._get_random_quat(env_ids)

        reset_indices = self.object_actor_indices[env_ids].flatten()
            
        if not self.headless:
            reset_indices = torch.cat([reset_indices, self.goal_actor_indices[env_ids]])
        
        # Set actor root state is required to reset the objects and the goals but can only be called once. Hence no resets are applied in the reset_objects() function.
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

    def _get_random_goal_pos(self, env_ids) -> torch.Tensor:
        object_pos_goal = torch.tensor(self.object_pos_goal, device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        object_pos_goal_noise = 2 * (torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        object_pos_goal_noise = object_pos_goal_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.object_pos_goal_noise, device=self.device))
        object_pos_goal += object_pos_goal_noise
        return object_pos_goal


    def _update_reset_buf(self):
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length, torch.ones_like(self.reset_buf), self.reset_buf
        )

    def _update_rew_buf(self):
        self.rew_buf[:] = 0.

        object_goal_distance = self._compute_object_goal_distance()
        goal_reached = object_goal_distance < self.cfg_task.rl.goal_threshold
        self._update_success_rate(goal_reached)

        height_increase, object_lifted  = self._compute_object_lifting()

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            if reward_term == 'lifting':
                delta_h = torch.clip(self.cfg_task.rl.lifting_threshold - height_increase, min=0.) / self.cfg_task.rl.lifting_threshold  # Goes from 1: not lifted at all, to 0: lifted to threshold or above.

                #print("delta_h", delta_h)
                lifting_reward_temp = 3.0
                objects_in_bin = self.objects_in_bin(self.target_object_pos)
                reward = scale * objects_in_bin * (torch.exp(-lifting_reward_temp * delta_h) - torch.exp(-lifting_reward_temp * torch.ones_like(delta_h)))
            
            elif reward_term == 'reaching':
                target_object_pos_expanded = self.target_object_pos.unsqueeze(1).repeat(1, self.fingertip_pos.shape[1], 1)
                fingertip_distance = torch.min(torch.norm(self.fingertip_pos - target_object_pos_expanded, dim=2), dim=1).values

                print("fingertip_distance", fingertip_distance)

                reaching_reward_temp = 3.0
                reward = scale * torch.exp(-reaching_reward_temp * fingertip_distance)

            elif reward_term == 'goal':
                goal_reward_temp = 10.0
                reward = scale * object_lifted * torch.exp(-goal_reward_temp * object_goal_distance)

            elif reward_term == 'object_velocity_penalty':
                object_velocity = torch.sum(torch.norm(self.object_linvel, dim=2), dim=1)
                object_velocity_threshold = 0.25
                object_velocity_penalty_temp = 1.0
                reward = -scale * torch.where(
                    object_velocity > object_velocity_threshold, 
                    torch.exp(object_velocity_penalty_temp * (object_velocity - object_velocity_threshold)) - 1.0, 
                    torch.zeros_like(object_velocity)
                )
                reward = torch.clip(reward, -10, 0)

            elif reward_term == 'dof_velocity_penalty':
                vel_max, _ = torch.max(self.dof_vel[:, 0:self.arm_dof_count], dim=1)
                dof_vel = torch.abs(vel_max)
                dof_vel_threshold = 0.5
                dof_vel_penalty_temp = 1.0
                reward = -scale * torch.where(
                    dof_vel > dof_vel_threshold, 
                    torch.exp(dof_vel_penalty_temp * (dof_vel - dof_vel_threshold)) - 1.0, 
                    torch.zeros_like(dof_vel)
                )
                reward = torch.clip(reward, -10, 0)
                
            elif reward_term == 'fingertip_progression':
                target_object_pos_expanded = self.target_object_pos.unsqueeze(1).repeat(1, self.fingertip_pos.shape[1], 1)
                fingertip_dist = torch.norm(self.fingertip_pos - target_object_pos_expanded, dim=-1).sum(dim=-1)
                fingertip_delta_closest = self.fingertip_closest_distance - fingertip_dist  # This is positive if the the keypoints got closer to the object and negative if they got further away than the closest distance so far.
                self.fingertip_closest_distance[:] = torch.minimum(self.fingertip_closest_distance, fingertip_dist)  # Update closest distance the keypoints have gotten.

                fingertip_delta_furthest = self.fingertip_furthest_distance - fingertip_dist
                self.fingertip_furthest_distance[:] = torch.maximum(self.fingertip_furthest_distance, fingertip_dist)  # Update furthest distance the keypoints have gotten.
                
                reward = scale * ~self.object_lifted_before * torch.clip(fingertip_delta_closest, 0, 10)  # Reward getting closer than the closest we have gotten.
                #reward += 0.1 * scale * ~self.object_lifted_before * torch.clip(fingertip_delta_furthest, -10, 0)  # Penalize getting further away than the furthest we have gotten.

            elif reward_term == 'goal_progression':
                object_goal_distance_delta = self.object_goal_closest_distance - object_goal_distance
                self.object_goal_closest_distance[:] = torch.minimum(self.object_goal_closest_distance, object_goal_distance)
                reward = scale * self.object_lifted_before * torch.clip(object_goal_distance_delta, 0, 100)  # Reward getting closer than the closest we have gotten.

            elif reward_term == 'success_bonus':
                reward = scale * goal_reached

            else:
                assert False
            
            self.rew_buf[:] += reward
            reward_terms['reward_terms/' + reward_term] = reward.mean().item()

        print(reward_terms)
        self.log(reward_terms)

    

    def _update_success_rate(self, goal_reached: torch.Tensor) -> None:
        self.goal_reached_before[:] = torch.logical_or(goal_reached, self.goal_reached_before)                      

        # Log exponentially weighted moving average (EWMA) of the success rate
        if not hasattr(self, "success_rate_ewma"):
            self.success_rate_ewma = 0.
            self.total_num_successes = 0
            self.total_num_resets = 0
        num_resets = torch.sum(self.reset_buf)  # Reset buffer indicates environments that will be reset on the next step.

        # Update success rate if resets have actually occurred
        if num_resets > 0:
            num_successes = torch.sum(self.goal_reached_before)
            curr_success_rate = num_successes / num_resets
            alpha = 0.02 * (num_resets / self.num_envs)
            self.success_rate_ewma = alpha * curr_success_rate + (1 - alpha) * self.success_rate_ewma
            self.log({"success_rate_ewma/overall": self.success_rate_ewma})

            self.total_num_resets += num_resets
            self.total_num_successes += num_successes

        # Log exponentially weighted moving average (EWMA) of the object-wise success rate
        for i, obj in enumerate(self.objects):
            if not hasattr(self, obj.name + "_success_rate_ewma"):
                setattr(self, obj.name + "_success_rate_ewma", 0.)

            target_object_index_in_all_objects = self.object_indices[torch.arange(self.num_envs), self.target_object_index]
            num_resets = torch.sum(self.reset_buf[target_object_index_in_all_objects == i])

            if num_resets > 0:
                num_successes = torch.sum(self.goal_reached_before[target_object_index_in_all_objects == i])
                current_success_rate = num_successes / num_resets
                alpha = 0.02 * (num_resets / self.num_envs)
                setattr(self, obj.name + "_success_rate_ewma", alpha * current_success_rate + (1 - alpha) * getattr(self, obj.name + "_success_rate_ewma"))
                self.log({"success_rate_ewma/" + obj.name: getattr(self, obj.name + "_success_rate_ewma")})

    def _compute_object_goal_distance(self) -> torch.Tensor:
        if self.cfg_task.rl.goal == "lift":
            object_goal_distance = torch.clamp(self.target_object_pos[:, 2] - self.goal_height, min=0.0)
        elif "reposition" in self.cfg_task.rl.goal:
            object_goal_distance = torch.norm(self.target_object_pos - self.goal_pos, dim=-1)
            if self.cfg_task.rl.goal == "oriented_reposition":
                eef_body_quat_diff = quat_mul(self.eef_body_goal_quat, quat_conjugate(self.eef_body_quat))
                eef_body_rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(eef_body_quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
                object_goal_distance += 0.1 * eef_body_rot_dist
        else:
            assert False, f"Unknown goal configuration {self.cfg_task.rl.goal} given."
            
        return object_goal_distance
    
    def _compute_object_lifting(self) -> Tuple[torch.BoolTensor, torch.Tensor]:
        # Compute achieved height increase of the object.
        current_height = self.target_object_pos[:, 2]
        initial_height = self.object_pos_initial.gather(1, self.target_object_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)[:, 2]
        height_increase = torch.clamp(current_height - initial_height, min=0.0)

        object_lifted = height_increase > self.cfg_task.rl.lifting_threshold

        #just_lifted = torch.logical_and(height_increase > self.cfg_task.rl.lifting_threshold, ~self.object_lifted_before)

        #self.object_lifted_before[:] = torch.logical_or(lifted, self.object_lifted_before)
        #self.object_lifted_before = lifted
        return height_increase, object_lifted
    
    def _reset_buffers(self, env_ids) -> None:
        super()._reset_buffers(env_ids)

        if not hasattr(self, 'object_lifted_before'):
            self.goal_reached_before = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.object_lifted_before = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            self.fingertip_closest_distance = torch.zeros((self.num_envs,), device=self.device)

        self.object_lifted_before[env_ids] = False
        target_object_pos_expanded = self.target_object_pos.unsqueeze(1).repeat(1, self.fingertip_pos.shape[1], 1)
        fingertip_dist = torch.norm(self.fingertip_pos - target_object_pos_expanded, dim=-1).sum(dim=-1)
        self.fingertip_closest_distance[env_ids] = fingertip_dist.clone().detach()[env_ids]
