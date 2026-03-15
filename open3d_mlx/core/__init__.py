"""Core types and utilities for Open3D-MLX.

This package provides the foundational type system that maps Open3D's
``Tensor``, ``Device``, and ``Dtype`` concepts to MLX equivalents.  Instead
of wrapping ``mlx.core.array`` in a custom tensor class, we expose utility
functions that operate on bare MLX arrays -- zero wrapper overhead.
"""

from open3d_mlx.core.device import Device
from open3d_mlx.core.dtype import (
    DTYPE_MAP,
    MLX_TO_NAME,
    ensure_float32,
    ensure_int32,
    to_mlx_dtype,
)
from open3d_mlx.core.tensor_utils import (
    check_points_shape,
    check_shape,
    eye4,
    from_numpy,
    is_finite,
    to_numpy,
)

__all__ = [
    "Device",
    "DTYPE_MAP",
    "MLX_TO_NAME",
    "ensure_float32",
    "ensure_int32",
    "to_mlx_dtype",
    "check_points_shape",
    "check_shape",
    "eye4",
    "from_numpy",
    "is_finite",
    "to_numpy",
]
