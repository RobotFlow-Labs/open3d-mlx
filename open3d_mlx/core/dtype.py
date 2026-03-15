"""Dtype mapping between Open3D conventions and MLX.

MLX has limited support for float64 and int64 on GPU (Metal). This module
provides safe mappings that downcast unsupported types to their 32-bit
equivalents, matching Open3D's dtype naming conventions.
"""

from __future__ import annotations

from typing import Union

import mlx.core as mx
import numpy as np

# Open3D dtype name -> MLX dtype.
# float64 -> float32 and int64 -> int32 because MLX Metal backend
# does not support 64-bit types on GPU.
DTYPE_MAP: dict[str, mx.Dtype] = {
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float32,  # downcast: no float64 on Metal GPU
    "bfloat16": mx.bfloat16,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int32,  # downcast: no int64 on Metal GPU
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "bool": mx.bool_,
}

# Reverse map: MLX dtype -> canonical name string.
# Built from DTYPE_MAP but skipping the downcast entries so we get
# the "true" name for each MLX dtype.
MLX_TO_NAME: dict[mx.Dtype, str] = {
    mx.float16: "float16",
    mx.float32: "float32",
    mx.bfloat16: "bfloat16",
    mx.int8: "int8",
    mx.int16: "int16",
    mx.int32: "int32",
    mx.uint8: "uint8",
    mx.uint16: "uint16",
    mx.uint32: "uint32",
    mx.bool_: "bool",
}

# NumPy dtype -> MLX dtype mapping (for to_mlx_dtype).
_NP_TO_MLX: dict[np.dtype, mx.Dtype] = {
    np.dtype("float16"): mx.float16,
    np.dtype("float32"): mx.float32,
    np.dtype("float64"): mx.float32,  # downcast
    np.dtype("int8"): mx.int8,
    np.dtype("int16"): mx.int16,
    np.dtype("int32"): mx.int32,
    np.dtype("int64"): mx.int32,  # downcast
    np.dtype("uint8"): mx.uint8,
    np.dtype("uint16"): mx.uint16,
    np.dtype("uint32"): mx.uint32,
    np.dtype("bool"): mx.bool_,
}


def to_mlx_dtype(dtype_str_or_np: Union[str, np.dtype, mx.Dtype]) -> mx.Dtype:
    """Convert an Open3D dtype string, numpy dtype, or MLX dtype to MLX dtype.

    Handles the downcasting of float64 -> float32 and int64 -> int32 that
    is required for MLX's Metal GPU backend.

    Args:
        dtype_str_or_np: One of:
            - A string like ``"float32"``, ``"int64"``, ``"bool"``
            - A ``numpy.dtype`` instance (e.g. ``np.float32``)
            - An ``mlx.core.Dtype`` instance (returned as-is)

    Returns:
        The corresponding ``mlx.core.Dtype``.

    Raises:
        ValueError: If the dtype is not recognized.

    Examples:
        >>> to_mlx_dtype("float32")
        mlx.core.float32
        >>> to_mlx_dtype(np.dtype("float64"))
        mlx.core.float32
        >>> to_mlx_dtype(mx.int32)
        mlx.core.int32
    """
    # Already an MLX dtype — pass through.
    if isinstance(dtype_str_or_np, mx.Dtype):
        return dtype_str_or_np

    # String lookup.
    if isinstance(dtype_str_or_np, str):
        key = dtype_str_or_np.lower().strip()
        if key in DTYPE_MAP:
            return DTYPE_MAP[key]
        raise ValueError(
            f"Unrecognized dtype string '{dtype_str_or_np}'. "
            f"Valid options: {list(DTYPE_MAP.keys())}"
        )

    # NumPy dtype lookup.
    if isinstance(dtype_str_or_np, np.dtype):
        if dtype_str_or_np in _NP_TO_MLX:
            return _NP_TO_MLX[dtype_str_or_np]
        raise ValueError(
            f"Unrecognized numpy dtype '{dtype_str_or_np}'. "
            f"Valid options: {[str(d) for d in _NP_TO_MLX.keys()]}"
        )

    # Also handle numpy type objects like np.float32 (which are types, not dtype instances).
    if isinstance(dtype_str_or_np, type) and issubclass(dtype_str_or_np, np.generic):
        return to_mlx_dtype(np.dtype(dtype_str_or_np))

    raise TypeError(
        f"Expected str, numpy.dtype, or mlx.core.Dtype, got {type(dtype_str_or_np)}"
    )


def ensure_float32(arr: mx.array) -> mx.array:
    """Ensure an MLX array has dtype float32.

    This is the workhorse dtype for MLX GPU operations. Arrays with other
    float types (float16, bfloat16) or integer types are cast to float32.
    If the array is already float32, it is returned as-is (no copy).

    Args:
        arr: Input MLX array.

    Returns:
        The same array if already float32, otherwise a new float32 array.
    """
    if arr.dtype != mx.float32:
        return arr.astype(mx.float32)
    return arr


def ensure_int32(arr: mx.array) -> mx.array:
    """Ensure an MLX array has dtype int32.

    MLX has no int64 support on the Metal GPU backend. This function
    downcasts int64 arrays to int32. Other integer types are also
    cast to int32. If the array is already int32, it is returned as-is.

    Args:
        arr: Input MLX array.

    Returns:
        The same array if already int32, otherwise a new int32 array.
    """
    if arr.dtype != mx.int32:
        return arr.astype(mx.int32)
    return arr
