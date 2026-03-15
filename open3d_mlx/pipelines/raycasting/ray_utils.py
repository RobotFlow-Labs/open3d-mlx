"""Ray generation utilities for volume raycasting.

Generates rays from pinhole camera parameters for use in TSDF ray marching.
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

    # Pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H, W) each

    # Unproject to camera-frame directions
    dx = (uu - intrinsic.cx) / intrinsic.fx
    dy = (vv - intrinsic.cy) / intrinsic.fy
    dz = np.ones_like(dx)
    dirs_cam = np.stack([dx, dy, dz], axis=-1)  # (H, W, 3)
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam = dirs_cam / norms

    # Transform to world frame
    ext = np.asarray(extrinsic, dtype=np.float64)
    R = ext[:3, :3]
    t = ext[:3, 3]

    # Rotate directions to world frame: dirs_world = dirs_cam @ R^T
    dirs_world = (dirs_cam.reshape(-1, 3) @ R.T).reshape(H, W, 3).astype(np.float32)
    origins = np.broadcast_to(t.astype(np.float32), dirs_world.shape).copy()

    rays = np.concatenate([origins, dirs_world], axis=-1)  # (H, W, 6)
    return mx.array(rays)


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
