import hydra
from omegaconf import DictConfig
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym.torch_utils import *
from typing import *


class HandArmBase(VecTask):
    _asset_root = '../assets/hand_arm/'
    _base_cfg_path = 'task/HandArmBase.yaml'

    arm_dof_count = 6

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_base = self._acquire_base_cfg()

        

        if self.cfg_base.ros.activate:
            self._acquire_ros_interface()
        
        self.cfg = cfg
        self.cfg["env"]["numActions"] = 11
        self.headless = headless
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()
        self.refresh_base_tensors()

        if self.viewer is not None:
            self._set_viewer_params()


    def _acquire_base_cfg(self) -> DictConfig:
        return hydra.compose(config_name=self._base_cfg_path)['task']
    
    def _acquire_robot(self):
        from isaacgymenvs.tasks.hand_arm.utils import HandArmRobot
        self.robot = HandArmRobot(self.gym, self.sim, self._asset_root, self.cfg_base, self.device)
        
    def acquire_base_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # Acquire general simulation tensors
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)
        #_dof_force = self.gym.acquire_dof_force_tensor(self.sim)  # shape = (num_envs * num_dofs, 1)
        _contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies, 3)
        #_jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'robot')  # shape = (num envs, num_bodies - 1, 6, num_dofs)  -1 because the base is fixed
        #_mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'robot')  # shape = (num_envs, num_dofs, num_dofs)

        # Refresh simulation tensors.
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        #self.dof_force = gymtorch.wrap_tensor(_dof_force)
        self.contact_force = gymtorch.wrap_tensor(_contact_force)
        #self.jacobian = gymtorch.wrap_tensor(_jacobian)
        #self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

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
        #self.dof_force_view = self.dof_force.view(self.num_envs, self.num_dofs, 1)[..., 0]
        self.contact_force = self.contact_force.view(self.num_envs, self.num_bodies, 3)[..., 0:3]

        # Initialize torque or position targets for all DoFs
        self.current_dof_pos_targets = torch.zeros((self.num_envs, self.robot.dof_count), device=self.device)
        self.previous_dof_pos_targets = torch.zeros((self.num_envs, self.robot.dof_count), device=self.device)
        self.actuated_dof_pos_targets = torch.zeros((self.num_envs, self.robot.actuator_count), device=self.device)

        self.actuated_dof_pos = self.dof_pos[:, self.robot.actuated_dof_indices]
        self.actuated_dof_vel = self.dof_vel[:, self.robot.actuated_dof_indices]



    def refresh_base_tensors(self):
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        #self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #self.gym.refresh_jacobian_tensors(self.sim)
        #self.gym.refresh_mass_matrix_tensors(self.sim)

        #NEEDED?
        #self.actuated_dof_pos[:] = self.dof_pos[:, self.actuated_dof_indices]
        #self.actuated_dof_vel[:] = self.dof_vel[:, self.actuated_dof_indices]

    def _acquire_ros_interface(self):
        raise NotImplementedError
    
    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""
        # Set time-step size and substeps.
        self.sim_params.dt = self.cfg_base.sim.dt
        self.sim_params.substeps = self.cfg_base.sim.num_substeps

        # Use GPU-pipeline if the simulation device is a GPU.
        self.sim_params.use_gpu_pipeline = self.device.startswith("cuda")

        # Set orientation and gravity vector.
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -self.cfg_base.sim.gravity_mag

        # Set PhysX parameters.
        self.sim_params.physx.use_gpu = self.device.startswith("cuda")
        self.sim_params.physx.num_position_iterations = self.cfg_base.sim.num_pos_iters
        self.sim_params.physx.num_velocity_iterations = self.cfg_base.sim.num_vel_iters

        self.sim = super().create_sim(compute_device=self.device_id,
                                      graphics_device=self.graphics_device_id,
                                      physics_engine=self.physics_engine,
                                      sim_params=self.sim_params)
        self._create_ground_plane()
        self._create_envs()  # defined in subclass

    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0  # default = 0.0
        plane_params.static_friction = 1.0  # default = 1.0
        plane_params.dynamic_friction = 1.0  # default = 1.0
        plane_params.restitution = 0.0  # default = 0.0

        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        #reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        #self.reset_target_pose(reset_goal_env_ids)

        self.actions = actions.clone().to(self.device) * 0.


        # actions 0-6 arm arm DoFs. 6 is thumb opposition 7 is thumb flextion 8 is index finger 9 is middle finger 10 is ring finger

        self.actions[0, 6] = 0.1

        print("actions.shape:", actions.shape)

        print("actions.device:", actions.device)
        print("self.current_dof_pos_targets:", self.current_dof_pos_targets.device)


        if self.cfg_base.control.type == "joint":
            joint_actions = self.robot.actuated_to_all(self.actions)
            print("joint_actions:", joint_actions)

            if self.cfg_base.control.mode == "absolute":
                self.current_dof_pos_targets[:, :] = scale(
                    joint_actions, self.robot.dof_lower_limits, self.robot.dof_upper_limits
                )

                self.current_dof_pos_targets[:, :] = (
                    self.cfg_base.control.moving_average * self.current_dof_pos_targets
                    + (1.0 - self.cfg_base.control.moving_average) * self.previous_dof_pos_targets
                )
            
            elif self.cfg_base.control.mode == "relative":
                self.current_dof_pos_targets[:, 0:self.arm_dof_count] = self.previous_dof_pos_targets[:, 0:self.arm_dof_count] + self.cfg_base.control.joint.arm_action_scale * self.dt * joint_actions[:, 0:self.arm_dof_count]
                self.current_dof_pos_targets[:, self.arm_dof_count:] = self.previous_dof_pos_targets[:, self.arm_dof_count:] + self.cfg_base.control.joint.hand_action_scale * self.dt * joint_actions[:, self.arm_dof_count:]

            else:
                assert False

            self.current_dof_pos_targets[:, :] = tensor_clamp(
                self.current_dof_pos_targets, self.robot.dof_lower_limits, self.robot.dof_upper_limits
            )
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.current_dof_pos_targets))
            self.previous_dof_pos_targets[:, :] = self.current_dof_pos_targets[:, :]
        
        else:
            assert False



        print("self.progress_buf:", self.progress_buf)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)


    def compute_reward(self):
        self._update_reset_buf()
        self._update_rew_buf()
        

    def post_physics_step(self):
        self.progress_buf[:] += 1


        self.compute_reward()

        if len(self.cfg_base.debug.visualize) > 0 and not self.headless:
            self.gym.clear_lines(self.viewer)
            self.draw_visualizations(self.cfg_base.debug.visualize)

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    def _reset_robot(self, env_ids, reset_dof_pos: List) -> None:
        """Reset DOF states and DOF targets of robot.
        """
        assert len(env_ids) == self.num_envs, \
            "All environments should be reset simultaneously."

        self.dof_pos[env_ids, :self.robot.dof_count] = torch.tensor(reset_dof_pos, device=self.device).unsqueeze(0).repeat((len(env_ids), 1))
        self.dof_vel[env_ids, :self.robot.dof_count] = 0.0
        self.current_dof_pos_targets[env_ids] = self.dof_pos[env_ids, :self.robot.dof_count]
        self.previous_dof_pos_targets[env_ids] = self.dof_pos[env_ids, :self.robot.dof_count]
        
        #self.actuated_dof_pos_targets[env_ids] = unscale(self.dof_pos[env_ids, :self.robot_dof_count], self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.actuated_dof_indices]

        reset_indices = self.robot_actor_indices[env_ids]
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.current_dof_pos_targets), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(reset_indices), len(reset_indices)
        )

    def _reset_buffers(self, env_ids) -> None:
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def draw_visualizations(self, visualizations: List[str]) -> None:
        for env_id in range(self.num_envs):
            for visualization in visualizations:
                split_visualization = visualization.split('_')
                # Visualize positions with sphere geom.
                if split_visualization[-1] == 'pos':
                    self.visualize_pos(visualization, env_id)
                # Visualize poses with axes geom.
                elif split_visualization[-1] == 'pose':
                    self.visualize_pose(visualization, env_id)
                # Call any other visualization functions (e.g. workspace extent, etc.).
                else:
                    getattr(self, "visualize_" + visualization)(env_id)