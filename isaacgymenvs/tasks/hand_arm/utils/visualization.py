import cv2
from isaacgym import gymapi, gymutil
import numpy as np
import torch
from typing import Callable, Sequence, Tuple


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


def visualize_pos(gym, viewer, env_ptrs, get_pos: Callable[[],torch.Tensor], marker: str = "axis", size: float = 0.1) -> None:
    pos = get_pos()

    while len(pos.shape) < 4:
        pos = pos.unsqueeze(1)

    for env_index, env_ptr in enumerate(env_ptrs):
        for actor_index in range(pos.shape[1]):
            for keypoint_index in range(pos.shape[2]):
                if pos.shape[-1] == 4 and pos[env_index, actor_index, keypoint_index, 3] < 0.5:  # Skip points that have a is_valid flag and are invalid.
                    continue
                pose = gymapi.Transform(gymapi.Vec3(*pos[env_index, actor_index, keypoint_index][0:3]))

                if marker == "axis":
                    geom = gymutil.AxesGeometry(size)
                elif marker == "sphere":
                    geom = gymutil.WireframeSphereGeometry(size, 4, 4)
                elif marker == "cube":
                    geom = gymutil.WireframeBoxGeometry(size, size, size)
                else:
                    raise ValueError(f"Unknown marker {marker}.")

                gymutil.draw_lines(geom, gym, viewer, env_ptr, pose)


def visualize_pose(gym, viewer, env_ptrs, get_pose: Callable[[],torch.Tensor], axis_length: float = 0.1) -> None:
    pose = get_pose()
    
    while len(pose.shape) < 4:
        pose = pose.unsqueeze(1)
    
    pos = pose[..., 0:3]
    quat = pose[..., 3:7]

    for env_index, env_ptr in enumerate(env_ptrs):
        for actor_index in range(pos.shape[1]):
            for keypoint_index in range(pos.shape[2]):
                pose = gymapi.Transform(gymapi.Vec3(*pos[env_index, actor_index, keypoint_index]), gymapi.Quat(*quat[env_index, actor_index, keypoint_index]))
                axes_geom = gymutil.AxesGeometry(axis_length)
                gymutil.draw_lines(axes_geom, gym, viewer, env_ptr, pose)


def visualize_bounding_box(gym, viewer, env_ptrs, get_bounding_box: Callable[[], torch.Tensor], color: Tuple[float, float, float] = (1, 0, 0)) -> None:
    bounding_box = get_bounding_box()

    while len(bounding_box.shape) < 4:
        bounding_box = bounding_box.unsqueeze(1)
    
    pos = bounding_box[..., 0:3]
    quat = bounding_box[..., 3:7]
    extents = bounding_box[..., 7:10]

    for env_index, env_ptr in enumerate(env_ptrs):
        for actor_index in range(pos.shape[1]):
            for keypoint_index in range(pos.shape[2]):
                bounding_box_range = torch.stack([-0.5 * extents[env_index, actor_index, keypoint_index], 0.5 * extents[env_index, actor_index, keypoint_index]], dim=0)
                bounding_box_geom = gymutil.WireframeBBoxGeometry(bounding_box_range, pose=gymapi.Transform(gymapi.Vec3(*pos[env_index, actor_index, keypoint_index]), gymapi.Quat(*quat[env_index, actor_index, keypoint_index])), color=color)
                gymutil.draw_lines(bounding_box_geom, gym, viewer, env_ptr, gymapi.Transform())


def visualize_color_image(
    gym,
    viewer,
    env_ptrs,
    get_color_image: Callable[[], torch.Tensor],
    window_name: str = "color image",
    env_indices: Sequence[int] = (0,)
    ) -> None:
    color_image = get_color_image()
    for env_index in env_indices:
            image_cv2 = cv2.cvtColor(color_image[env_index].cpu().numpy(), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name + f" (Env {env_index})", image_cv2)
            cv2.waitKey(1)


def visualize_depth_image(
    gym,
    viewer,
    env_ptrs,
    get_depth_image: Callable[[], torch.Tensor],
    window_name: str = "depth image",
    env_indices: Sequence[int] = (0,),
    max_depth: float = 2.0
    ) -> None:
    color_image = ((-depth_image / max_depth).clip(0.0, 1.0) * 255).astype(np.uint8)
    visualize_color_image(color_image, window_name, env_indices)


def visualize_segmentation_image(
    gym,
    viewer,
    env_ptrs,
    get_segmentation_image: Callable[[], torch.Tensor],
    window_name: str = "semantic image",
    env_indices: Sequence[int] = (0,)
    ) -> None:
    segmentation_image = get_segmentation_image()
    colormap = get_pascal_voc_colormap(torch.max(segmentation_image) + 1)
    color_image = torch.zeros_like(segmentation_image, dtype=torch.uint8).unsqueeze(-1).repeat(1, 1, 1, 3)

    for label in range(torch.max(segmentation_image) + 1):
        color_image[segmentation_image == label] = torch.tensor(colormap[label], dtype=torch.uint8).to(segmentation_image.device)
    
    visualize_color_image(gym, viewer, env_ptrs, lambda: color_image, window_name, env_indices)
