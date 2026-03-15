"""Axis-aligned bounding box for Open3D-MLX.

Provides an AxisAlignedBoundingBox class that mirrors upstream Open3D's
o3d.geometry.AxisAlignedBoundingBox API.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


class AxisAlignedBoundingBox:
    """Axis-aligned bounding box defined by min and max bounds.

    Parameters
    ----------
    min_bound : mx.array or array-like or None
        ``(3,)`` minimum corner of the box.
    max_bound : mx.array or array-like or None
        ``(3,)`` maximum corner of the box.
    """

    def __init__(self, min_bound=None, max_bound=None):
        if min_bound is not None:
            if not isinstance(min_bound, mx.array):
                min_bound = mx.array(np.asarray(min_bound, dtype=np.float32))
            self.min_bound = min_bound.reshape((3,))
        else:
            self.min_bound = mx.zeros((3,))

        if max_bound is not None:
            if not isinstance(max_bound, mx.array):
                max_bound = mx.array(np.asarray(max_bound, dtype=np.float32))
            self.max_bound = max_bound.reshape((3,))
        else:
            self.max_bound = mx.zeros((3,))

    @classmethod
    def create_from_points(cls, points: mx.array) -> "AxisAlignedBoundingBox":
        """Create a bounding box that encloses all given points.

        Parameters
        ----------
        points : mx.array
            ``(N, 3)`` point positions.

        Returns
        -------
        AxisAlignedBoundingBox
        """
        return cls(mx.min(points, axis=0), mx.max(points, axis=0))

    def get_center(self) -> mx.array:
        """Return ``(3,)`` center of the bounding box."""
        return (self.min_bound + self.max_bound) / 2.0

    def get_extent(self) -> mx.array:
        """Return ``(3,)`` extent (max - min) of the bounding box."""
        return self.max_bound - self.min_bound

    def get_half_extent(self) -> mx.array:
        """Return ``(3,)`` half-extent of the bounding box."""
        return self.get_extent() / 2.0

    def volume(self) -> float:
        """Return the volume of the bounding box."""
        ext = np.asarray(self.get_extent(), dtype=np.float64)
        return float(np.prod(ext))

    def contains(self, points: mx.array) -> mx.array:
        """Test which points are inside the bounding box.

        Parameters
        ----------
        points : mx.array
            ``(N, 3)`` point positions.

        Returns
        -------
        mx.array
            ``(N,)`` boolean array. True if inside (inclusive of boundary).
        """
        pts_np = np.asarray(points, dtype=np.float32)
        min_b = np.asarray(self.min_bound, dtype=np.float32)
        max_b = np.asarray(self.max_bound, dtype=np.float32)
        mask = np.all((pts_np >= min_b) & (pts_np <= max_b), axis=1)
        return mx.array(mask)

    def __repr__(self) -> str:
        return (
            f"AxisAlignedBoundingBox(min={np.asarray(self.min_bound).tolist()}, "
            f"max={np.asarray(self.max_bound).tolist()})"
        )
