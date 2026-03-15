"""Pinhole camera intrinsic parameters.

Matches Open3D: o3d.camera.PinholeCameraIntrinsic
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class PinholeCameraIntrinsic:
    """Pinhole camera intrinsic parameters.

    Stores image dimensions and focal length / principal point,
    and provides the 3x3 intrinsic matrix K.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def intrinsic_matrix(self) -> mx.array:
        """3x3 intrinsic matrix K."""
        return mx.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=mx.float32,
        )

    @classmethod
    def prime_sense_default(cls) -> "PinholeCameraIntrinsic":
        """PrimeSense / Kinect default intrinsics (640x480)."""
        return cls(640, 480, 525.0, 525.0, 319.5, 239.5)
