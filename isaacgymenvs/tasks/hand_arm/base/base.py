import hydra
from omegaconf import DictConfig
from isaacgym import gymapi, gymtorch, gymutil
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.hand_arm.base.actions import ActorMixin, Robot
from isaacgymenvs.tasks.hand_arm.base.camera_sensor import CameraMixin
from isaacgymenvs.tasks.hand_arm.base.logger import LoggerMixin
from isaacgymenvs.tasks.hand_arm.base.observations import ObserverMixin
from isaacgymenvs.tasks.hand_arm.base.simulation import SimulationMixin
from isaacgym.torch_utils import *
import matplotlib
from typing import *
import cv2




class HandArmBase(VecTask, ActorMixin, ObserverMixin, SimulationMixin, LoggerMixin, CameraMixin):
    _asset_root: str = '../assets/hand_arm/'
    _base_cfg_path: str = 'task/HandArmBase.yaml'

    arm_dof_count = 6

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.cfg_base = self._acquire_base_cfg()
        
        #self.controller = URDFRobot(self._asset_root, self.cfg_base.asset.robot)
        self.controller = Robot(self._asset_root, self.cfg_base.asset)

        self.all_observations = cfg["env"]["observations"]
        if "teacher_observations" in cfg["env"].keys():
            self.all_observations = list(set(self.all_observations + cfg["env"]["teacher_observations"]))

        self.register_observations()

        self.cfg["env"]["numActions"] = self._compute_num_actions()
        self.cfg["env"]["numObservations"], self._observation_start_end = self._compute_num_observations(cfg["env"]["observations"])

        if "teacher_observations" in cfg["env"].keys():
            self.cfg["env"]["numTeacherObservations"], self._teacher_observation_start_end = self._compute_num_observations(cfg["env"]["teacher_observations"])
        
        
        self.headless = headless

        self.camera_dict = self._acquire_camera_dict()
        

        if self.cfg_base.ros.activate:
            self._acquire_ros_interface()
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, True)

        self.acquire_base_tensors()
        self.refresh_base_tensors()

        if self.viewer is not None:
            self._set_viewer_params()

    def _acquire_base_cfg(self) -> DictConfig:
        return hydra.compose(config_name=self._base_cfg_path)['task']

    def acquire_base_tensors(self):
        self.acquire_simulation_tensors()
        self.acquire_action_tensors()
        self.acquire_observation_tensors()

    def refresh_base_tensors(self):
        self.refresh_simulation_tensors()
        self.refresh_images()
        self.refresh_action_tensors()
        self.refresh_observation_tensors()

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

        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 ** 2  # default = 1024^2
        self.sim_params.physx.default_buffer_size_multiplier = 8  # default = 1

        self.sim = super().create_sim(compute_device=self.device_id,
                                      graphics_device=self.graphics_device_id,
                                      physics_engine=self.physics_engine,
                                      sim_params=self.sim_params)
        self._create_ground_plane()
        self.controller.attach_simulation(self.gym, self.sim, self.device)
        self._create_envs()  # defined in subclass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        #reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        #self.reset_target_pose(reset_goal_env_ids)

        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        self.apply_controls(actions)


        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)


    def compute_reward(self):
        self._update_reset_buf()
        self._update_rew_buf()

    def post_physics_step(self):
        self.progress_buf[:] += 1

        self.refresh_base_tensors()

        self.compute_reward()
        self.compute_observations()
        #if self.camera_dict:
        #    self.obs_dict["image"] = self.compute_sensor_outputs()

        # print("self.obs_dict:", self.obs_dict)


        # import matplotlib.pyplot as plt

        # plt.imshow(self.obs_dict["image"]["rgb_frontview"]["color"][0].cpu().numpy())
        # plt.show()

        if len(self.cfg_base.debug.visualize) > 0: # and not self.headless:
            self.gym.clear_lines(self.viewer)
            self.draw_visualizations(self.cfg_base.debug.visualize)

    def _reset_robot(self, env_ids, reset_dof_pos: List) -> None:
        """Reset DOF states and DOF targets of robot.
        """
        assert len(env_ids) == self.num_envs, \
            "All environments should be reset simultaneously."

        self.dof_pos[env_ids, :self.controller.dof_count] = torch.tensor(reset_dof_pos, device=self.device).unsqueeze(0).repeat((len(env_ids), 1))
        self.dof_vel[env_ids, :self.controller.dof_count] = 0.0
        self.current_dof_pos_targets[env_ids] = self.dof_pos[env_ids, :self.controller.dof_count]
        #self.previous_dof_pos_targets[env_ids] = self.dof_pos[env_ids, :self.controller.dof_count]
        self.current_actuated_dof_pos_targets[env_ids] = self.dof_pos[env_ids, :self.controller.dof_count][:, self.controller.actuated_dof_indices]
        self.previous_actuated_dof_pos_targets[env_ids] = self.dof_pos[env_ids, :self.controller.dof_count][:, self.controller.actuated_dof_indices]
        
        #self.actuated_dof_pos_targets[env_ids] = unscale(self.dof_pos[env_ids, :self.controller_dof_count], self.controller_dof_lower_limits, self.controller_dof_upper_limits)[:, self.actuated_dof_indices]

        reset_indices = self.controller_actor_indices[env_ids]
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
        for visualization in visualizations:
            # If an observation is visualized.
            if visualization in self.all_observations:
                self._observations[visualization].visualize()

            # Call any other visualization functions (e.g. workspace extent, etc.).
            else:
                getattr(self, "visualize_" + visualization)()

    def visualize_points(self, pos: torch.Tensor, size: float = 0.01, color: Tuple[float, float, float] = (1, 0, 0)) -> None:
        while len(pos.shape) < 4:
            pos = pos.unsqueeze(1)

        for env_index in range(self.num_envs):
            for actor_index in range(pos.shape[1]):
                for keypoint_index in range(pos.shape[2]):
                    if pos.shape[3] == 4 and pos[env_index, actor_index, keypoint_index, 3] < 0.5: # Points have a mask-dimension and this is a padding point.
                        continue
                    pose = gymapi.Transform(gymapi.Vec3(*pos[env_index, actor_index, keypoint_index, 0:3]))
                    color = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][pos[env_index, actor_index, keypoint_index, 3].long() - 1]  # Color is determined by the point's class/ID.
                    sphere_geom = gymutil.WireframeSphereGeometry(size, 4, 4, color=color)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.env_ptrs[env_index], pose)

    def visualize_voxelgrid(self, observation_name: str, voxel_size: float = 0.01, color: Tuple[float, float, float] = (1, 0, 0)) -> None:
        sparse_voxelgrid = self._observations[observation_name].as_tensor()
        #print("sparse_voxelgrid.shape:", sparse_voxelgrid.shape)
        #input()
        for env_index in range(self.num_envs):
            mask = sparse_voxelgrid[:, 0] == env_index
            voxels = sparse_voxelgrid[mask, 1:4]
            points = voxels * voxel_size

            for point in points:
                pose = gymapi.Transform(gymapi.Vec3(*point))
                #color = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][pos[env_index, actor_index, keypoint_index, 3].long() - 1]  # Color is determined by the point's class/ID.
                #sphere_geom = gymutil.WireframeSphereGeometry(size, 4, 4, color=color)
                box_geom = gymutil.WireframeBoxGeometry(xdim=voxel_size, ydim=voxel_size, zdim=voxel_size, color=color)
                gymutil.draw_lines(box_geom, self.gym, self.viewer, self.env_ptrs[env_index], pose)



    
    def visualize_poses(self, pos: torch.Tensor, quat: torch.Tensor, axis_length: float = 0.1) -> None:
        while len(pos.shape) < 4:
            pos = pos.unsqueeze(1)
            quat = quat.unsqueeze(1)

        for env_index in range(self.num_envs):
            for actor_index in range(pos.shape[1]):
                for keypoint_index in range(pos.shape[2]):
                    pose = gymapi.Transform(gymapi.Vec3(*pos[env_index, actor_index, keypoint_index]), gymapi.Quat(*quat[env_index, actor_index, keypoint_index]))
                    axes_geom = gymutil.AxesGeometry(axis_length)
                    gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env_ptrs[env_index], pose)

    def visualize_bounding_boxes(self, pos: torch.Tensor, quat: torch.Tensor, extents: torch.Tensor, color: Tuple[float, float, float] = (1, 0, 0)) -> None:
        while len(pos.shape) < 4:
            pos = pos.unsqueeze(1)
            quat = quat.unsqueeze(1)
            extents = extents.unsqueeze(1)

        for env_index in range(self.num_envs):
            for actor_index in range(pos.shape[1]):
                for keypoint_index in range(pos.shape[2]):
                    bounding_box_range = torch.stack([-0.5 * extents[env_index, actor_index, keypoint_index], 0.5 * extents[env_index, actor_index, keypoint_index]], dim=0)
                    bounding_box_geom = gymutil.WireframeBBoxGeometry(bounding_box_range, pose=gymapi.Transform(gymapi.Vec3(*pos[env_index, actor_index, keypoint_index]), gymapi.Quat(*quat[env_index, actor_index, keypoint_index])), color=color)
                    gymutil.draw_lines(bounding_box_geom, self.gym, self.viewer, self.env_ptrs[env_index], gymapi.Transform())

    def visualize_distance(self, anchor_pos: torch.Tensor, delta_pos: torch.Tensor) -> None:
        if len(anchor_pos.shape) == 2:
            anchor_pos = anchor_pos.unsqueeze(1)
            delta_pos = delta_pos.unsqueeze(1)

        goal_pos = anchor_pos + delta_pos

        for env_id in range(self.num_envs):
            for keypoint_index in range(anchor_pos.shape[1]):
                gymutil.draw_line(
                    gymapi.Vec3(*anchor_pos[env_id, keypoint_index]), gymapi.Vec3(*goal_pos[env_id, keypoint_index]), gymapi.Vec3(1, 0, 0), self.gym, self.viewer, self.env_ptrs[env_id]
                )

    def visualize_contact_force(self, colormap: str = 'viridis', vmax: float = 10.) -> None:
        cmap = matplotlib.colormaps[colormap]
        contact_force_mag = torch.clip(torch.norm(self.contact_force, dim=2) / vmax, max=1.0)
        rgb = cmap(contact_force_mag.cpu().numpy())[..., 0:3]
        num_rigid_bodies = self.controller.rigid_body_count
        for rb_index in range(num_rigid_bodies):
            for env_index in range(self.num_envs):
                self.gym.set_rigid_body_color(
                    self.env_ptrs[env_index], self.controller_handles[env_index], rb_index, gymapi.MESH_VISUAL, gymapi.Vec3(*rgb[env_index, rb_index]),
                )

    def visualize_image(self, image_tensor: torch.Tensor, window_name: str = "image", env_indices: Sequence[int] = (0,)) -> None:
        for env_index in env_indices:
            image_cv2 = cv2.cvtColor(image_tensor[env_index].cpu().numpy(), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name + f" (Env {env_index})", image_cv2)
            cv2.waitKey(1)

    def visualize_depth(self, depth_tensor: torch.Tensor, window_name: str = "depth", env_indices: Sequence[int] = (0,), max_depth: float = 2.0) -> None:
        for env_index in env_indices:
            depth_np = ((-depth_tensor[env_index].cpu().numpy() / max_depth).clip(0.0, 1.0) * 255).astype(np.uint8)
            cv2.imshow(window_name + f" (Env {env_index})", depth_np)
            cv2.waitKey(1)

    def visualize_segmentation(self, segmentation_tensor: torch.Tensor, window_name: str = "segmentation", env_indices: Sequence[int] = (0,)) -> None:
        def get_pascal_voc_colormap(max_num_classes: int):
            def bitget(byteval, idx):
                return (byteval & (1 << idx)) != 0

            colormap = np.zeros((max_num_classes, 3), dtype=int)
            for i in range(0, max_num_classes):
                r = g = b = 0
                c = i
                for j in range(8):
                    r = r | (bitget(c, 0) << 7 - j)
                    g = g | (bitget(c, 1) << 7 - j)
                    b = b | (bitget(c, 2) << 7 - j)
                    c = c >> 3

                colormap[i] = np.array([r, g, b])
            return colormap
        
        colormap = get_pascal_voc_colormap(torch.max(segmentation_tensor) + 1)

        rgb_tensor = torch.zeros_like(segmentation_tensor, dtype=torch.uint8).unsqueeze(-1).repeat(1, 1, 1, 3)

        for label in range(torch.max(segmentation_tensor) + 1):
            print("label:", label)
            print("colormap[label]:", colormap[label])
            rgb_tensor[segmentation_tensor == label] = torch.tensor(colormap[label], dtype=torch.uint8).to(segmentation_tensor.device)
        
        for env_index in env_indices:
            image_cv2 = cv2.cvtColor(rgb_tensor[env_index].cpu().numpy(), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name + f" (Env {env_index})", image_cv2)
            cv2.waitKey(1)

