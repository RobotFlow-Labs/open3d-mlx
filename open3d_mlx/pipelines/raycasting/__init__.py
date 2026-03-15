"""Volume raycasting pipeline.

Provides ray-TSDF marching for rendering synthetic depth maps and normal
maps from integrated TSDF volumes.
"""

from open3d_mlx.pipelines.raycasting.ray_utils import generate_rays, generate_rays_flat
from open3d_mlx.pipelines.raycasting.raycasting_scene import RaycastingScene

__all__ = [
    "RaycastingScene",
    "generate_rays",
    "generate_rays_flat",
]
