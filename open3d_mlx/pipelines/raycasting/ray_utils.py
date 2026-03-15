"""Ray generation utilities for volume raycasting.

Generates rays from pinhole camera parameters for use in TSDF ray marching.
Uses MLX for GPU-accelerated ray direction computation and world-frame rotation.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from open3d_mlx.camera.intrinsics import PinholeCameraIntrinsic


def generate_rays(
    intrinsic: PinholeCameraIntrinsic,
    extrinsic,
    width: int | None = None,
    height: int | None = None,
) -> mx.array:
    """Generate rays for all pixels in a camera view.

    Uses MLX for GPU-accelerated direction computation and rotation
    to world frame. The pixel grid is constructed in numpy (meshgrid)
    and converted to MLX once.

    Parameters
    ----------
    intrinsic : PinholeCameraIntrinsic
        Camera intrinsics (focal length, principal point, image size).
    extrinsic : array-like
        (4, 4) camera-to-world transformation matrix.
    width : int or None
        Image width override (default: intrinsic.width).
    height : int or None
        Image height override (default: intrinsic.height).

    Returns
    -------
    rays : mx.array
        (H, W, 6) array where ``rays[v, u] = [ox, oy, oz, dx, dy, dz]``.
        Origins are in world space, directions are unit vectors.
    """
    W = width or intrinsic.width
    H = height or intrinsic.height

    # Pixel grid — numpy meshgrid, normalize in float64 for precision
    uu, vv = np.meshgrid(
        np.arange(W, dtype=np.float64),
        np.arange(H, dtype=np.float64),
    )

    # Unproject to camera-frame directions (float64 for normalization precision)
    dx = (uu - intrinsic.cx) / intrinsic.fx
    dy = (vv - intrinsic.cy) / intrinsic.fy
    dz = np.ones_like(dx)
    dirs_cam = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3) float64
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam = dirs_cam / norms  # unit vectors in float64

    # Rotate to world frame — MLX GPU matmul (the expensive part)
    ext = np.asarray(extrinsic, dtype=np.float64)
    R_mat = ext[:3, :3]
    t_vec = ext[:3, 3]

    # Convert to MLX float32 for GPU-accelerated rotation
    dirs_cam_mx = mx.array(dirs_cam.reshape(-1, 3).astype(np.float32))
    R_mx = mx.array(R_mat.astype(np.float32))
    t_mx = mx.array(t_vec.astype(np.float32))

    # (H*W, 3) @ (3, 3)^T → world-frame directions on GPU
    dirs_world = dirs_cam_mx @ R_mx.T  # MLX GPU matmul

    # Re-normalize after float32 rotation to maintain unit length
    world_norms = mx.sqrt(mx.sum(dirs_world * dirs_world, axis=-1, keepdims=True))
    dirs_world = dirs_world / mx.maximum(world_norms, 1e-10)

    dirs_world = mx.reshape(dirs_world, (H, W, 3))

    # Broadcast origin to all pixels
    origins = mx.broadcast_to(t_mx, dirs_world.shape)

    rays = mx.concatenate([origins, dirs_world], axis=-1)  # (H, W, 6)
    mx.eval(rays)
    return rays


def generate_rays_flat(
    intrinsic: PinholeCameraIntrinsic,
    extrinsic,
) -> mx.array:
    """Generate rays as flat (H*W, 6) array for batch processing.

    Parameters
    ----------
    intrinsic : PinholeCameraIntrinsic
        Camera intrinsics.
    extrinsic : array-like
        (4, 4) camera-to-world transformation matrix.

    Returns
    -------
    rays : mx.array
        (H*W, 6) array of ray origins and directions.
    """
    rays = generate_rays(intrinsic, extrinsic)
    H, W = rays.shape[:2]
    return rays.reshape(H * W, 6)
