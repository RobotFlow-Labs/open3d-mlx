"""Conversion between Open3D-MLX and vanilla Open3D data types.

Enables hybrid workflows where Open3D-MLX handles GPU-accelerated computation
on Apple Silicon and vanilla Open3D provides visualization or legacy features.

All conversion functions lazily import ``open3d`` so that the interop module
can be imported even when Open3D is not installed -- a clear
``ImportError`` is raised at call time instead.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from open3d_mlx.geometry import PointCloud


def to_open3d(pcd: PointCloud) -> "o3d.geometry.PointCloud":  # noqa: F821
    """Convert an Open3D-MLX PointCloud to an Open3D legacy PointCloud.

    Parameters
    ----------
    pcd : PointCloud
        Source point cloud (MLX-backed).

    Returns
    -------
    o3d.geometry.PointCloud
        Open3D legacy point cloud with numpy data.

    Raises
    ------
    ImportError
        If the ``open3d`` package is not installed.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for interop conversions. "
            "Install it with: pip install open3d"
        )

    o3d_pcd = o3d.geometry.PointCloud()
    points_np = np.array(pcd.points, copy=False)
    if points_np.shape[0] > 0:
        o3d_pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
    if pcd.has_normals():
        o3d_pcd.normals = o3d.utility.Vector3dVector(
            np.array(pcd.normals, copy=False).astype(np.float64)
        )
    if pcd.has_colors():
        o3d_pcd.colors = o3d.utility.Vector3dVector(
            np.array(pcd.colors, copy=False).astype(np.float64)
        )
    return o3d_pcd


def from_open3d(o3d_pcd: "o3d.geometry.PointCloud") -> PointCloud:  # noqa: F821
    """Convert an Open3D legacy PointCloud to an Open3D-MLX PointCloud.

    Parameters
    ----------
    o3d_pcd : o3d.geometry.PointCloud
        Source Open3D point cloud.

    Returns
    -------
    PointCloud
        Open3D-MLX point cloud with MLX arrays.
    """
    points_np = np.asarray(o3d_pcd.points).astype(np.float32)
    pcd = PointCloud(mx.array(points_np) if len(points_np) > 0 else None)
    if o3d_pcd.has_normals():
        pcd.normals = mx.array(np.asarray(o3d_pcd.normals).astype(np.float32))
    if o3d_pcd.has_colors():
        pcd.colors = mx.array(np.asarray(o3d_pcd.colors).astype(np.float32))
    return pcd


def to_open3d_tensor(pcd: PointCloud) -> "o3d.t.geometry.PointCloud":  # noqa: F821
    """Convert an Open3D-MLX PointCloud to an Open3D tensor-based PointCloud.

    Parameters
    ----------
    pcd : PointCloud
        Source point cloud (MLX-backed).

    Returns
    -------
    o3d.t.geometry.PointCloud
        Open3D tensor-based point cloud.

    Raises
    ------
    ImportError
        If the ``open3d`` package is not installed.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for interop conversions. "
            "Install it with: pip install open3d"
        )

    o3d_pcd = o3d.t.geometry.PointCloud()
    points_np = np.array(pcd.points, copy=False)
    if points_np.shape[0] > 0:
        o3d_pcd.point.positions = o3d.core.Tensor(points_np.astype(np.float64))
    if pcd.has_normals():
        o3d_pcd.point.normals = o3d.core.Tensor(
            np.array(pcd.normals, copy=False).astype(np.float64)
        )
    if pcd.has_colors():
        o3d_pcd.point.colors = o3d.core.Tensor(
            np.array(pcd.colors, copy=False).astype(np.float64)
        )
    return o3d_pcd


__all__ = ["to_open3d", "from_open3d", "to_open3d_tensor"]
