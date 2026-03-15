"""Utility functions for MLX array operations.

We use ``mlx.core.array`` directly as the tensor type -- no wrapper class.
These utilities provide convenience functions that match Open3D's common
tensor operations: NumPy interop, identity matrices, shape validation, and
finite-value checks.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np


def to_numpy(arr: mx.array) -> np.ndarray:
    """Convert an MLX array to a NumPy ``ndarray``.

    Uses ``np.array()`` which is zero-copy when the memory layout allows it.

    Args:
        arr: Input MLX array.

    Returns:
        A NumPy array with the same data.
    """
    return np.array(arr)


def from_numpy(arr: np.ndarray, dtype: Optional[mx.Dtype] = None) -> mx.array:
    """Convert a NumPy ``ndarray`` to an MLX array.

    Args:
        arr: Input NumPy array.
        dtype: Optional MLX dtype to cast to.  If ``None`` the dtype is
            inferred from the NumPy array (with float64 -> float32 and
            int64 -> int32 handled by MLX automatically).

    Returns:
        An MLX array.
    """
    result = mx.array(arr)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def eye4(dtype: mx.Dtype = mx.float32) -> mx.array:
    """Return a 4x4 identity matrix.

    This is a convenience for the ubiquitous transformation matrices used
    in registration and TSDF pipelines.

    Args:
        dtype: Desired dtype.  Defaults to ``mx.float32``.

    Returns:
        A ``(4, 4)`` identity matrix.
    """
    return mx.eye(4, dtype=dtype)


def is_finite(arr: mx.array) -> mx.array:
    """Element-wise check for finite values (not NaN, not Inf).

    Args:
        arr: Input MLX array.

    Returns:
        A boolean MLX array of the same shape, ``True`` where the value
        is finite.
    """
    return mx.logical_not(mx.logical_or(mx.isnan(arr), mx.isinf(arr)))


def check_shape(
    arr: mx.array,
    expected_shape: Union[Tuple[int, ...], list],
    name: str = "array",
) -> None:
    """Validate that an array has the expected shape.

    Args:
        arr: Input MLX array.
        expected_shape: The required shape as a tuple of ints.
        name: Descriptive name used in the error message.

    Raises:
        ValueError: If ``arr.shape`` does not match ``expected_shape``.
    """
    expected = tuple(expected_shape)
    if arr.shape != expected:
        raise ValueError(
            f"Expected {name} shape {expected}, got {arr.shape}"
        )


def check_points_shape(arr: mx.array, name: str = "points") -> None:
    """Validate that an array has shape ``(N, 3)`` for 3-D points.

    Args:
        arr: Input MLX array.
        name: Descriptive name used in the error message.

    Raises:
        ValueError: If the array is not 2-D or the second dimension is not 3.
    """
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected {name} shape (N, 3), got {arr.shape}"
        )
