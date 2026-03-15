# PRD-01: Core Types & Device Abstraction

## Status: P0 — Foundation
## Priority: P0
## Phase: 1 — Foundation
## Estimated Effort: 1 day
## Depends On: PRD-00
## Blocks: PRD-02, PRD-03, PRD-04, PRD-05, PRD-06, PRD-07, PRD-08, PRD-09, PRD-10

---

## 1. Objective

Implement the core type system that maps Open3D's `Tensor`, `Device`, and `Dtype` concepts to MLX equivalents. Unlike Open3D which wraps tensors in a custom `Tensor` class, we use `mlx.core.array` directly — zero wrapper overhead.

---

## 2. Upstream Reference

| Our File | Upstream File | Notes |
|----------|--------------|-------|
| `core/dtype.py` | `cpp/open3d/core/Dtype.h` (146 lines) | Dtype enum mapping |
| `core/device.py` | `cpp/open3d/core/Device.h` (126 lines) | Device abstraction |
| `core/tensor_utils.py` | `cpp/open3d/core/Tensor.h` (1,442 lines) | Utility functions only |

---

## 3. Design Decision: No Tensor Wrapper

Open3D wraps DLPack tensors in `o3d.core.Tensor` with `.device`, `.dtype`, `.shape` etc. We skip this because:

1. `mlx.core.array` already has `.dtype`, `.shape`, `.ndim`
2. Zero-copy NumPy interop: `np.array(mx_arr)` and `mx.array(np_arr)` just work
3. No overhead, no confusion about which tensor type to use
4. MLX unified memory eliminates the need for `.to(device)` transfers

```python
# Open3D way:
tensor = o3d.core.Tensor(data, dtype=o3d.core.Float32, device=o3d.core.Device("CUDA:0"))
tensor.cpu()  # explicit transfer

# Our way:
tensor = mx.array(data, dtype=mx.float32)  # always on Apple GPU, unified memory
```

---

## 4. Module: `open3d_mlx/core/dtype.py`

### 4.1 Dtype Mapping

```python
"""Dtype mapping between Open3D conventions and MLX."""

import mlx.core as mx

# Open3D dtype name → MLX dtype
DTYPE_MAP = {
    "float32": mx.float32,
    "float64": mx.float32,   # MLX has limited float64 GPU support; downcast
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int32,       # MLX has no int64 on GPU; downcast
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "bool": mx.bool_,
}

# Reverse map
MLX_TO_NAME = {v: k for k, v in DTYPE_MAP.items()}


def to_mlx_dtype(dtype_str_or_np):
    """Convert Open3D dtype string or numpy dtype to MLX dtype.

    Args:
        dtype_str_or_np: String like "float32", numpy dtype, or MLX dtype.

    Returns:
        mlx.core.Dtype
    """
    ...


def ensure_float32(arr):
    """Ensure array is float32 (the workhorse dtype for MLX GPU ops)."""
    if arr.dtype != mx.float32:
        return arr.astype(mx.float32)
    return arr


def ensure_int32(arr):
    """Ensure array is int32 (MLX has no int64 on GPU)."""
    if arr.dtype in (mx.int64,) or (hasattr(arr, 'dtype') and arr.dtype == mx.int64):
        return arr.astype(mx.int32)
    return arr
```

### 4.2 Float64 Strategy

Open3D uses float64 for transformation matrices (4x4) and convergence criteria. MLX float64 support is limited on GPU. Our strategy:

- **Transformation matrices**: Keep as float64 on CPU (NumPy), convert to float32 for GPU ops
- **Point coordinates**: Always float32 on MLX
- **Accumulations** (RMSE, fitness): Compute in float32, return as Python float

---

## 5. Module: `open3d_mlx/core/device.py`

```python
"""Device abstraction for MLX.

MLX uses unified memory — there is no CPU/GPU distinction for data.
This module exists for API compatibility with Open3D patterns.
"""

import mlx.core as mx


class Device:
    """Device descriptor, compatible with Open3D's device strings.

    On MLX, all computation happens on the default device (Apple GPU).
    This class exists so code that passes device="GPU:0" etc. doesn't break.
    """

    def __init__(self, device_str: str = "MLX:0"):
        self._device_str = device_str.upper()

    @property
    def type(self) -> str:
        return self._device_str.split(":")[0]

    def __repr__(self) -> str:
        return f"Device('{self._device_str}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return self._device_str == other.upper()
        if isinstance(other, Device):
            return self._device_str == other._device_str
        return False

    @staticmethod
    def get_default() -> "Device":
        return Device("MLX:0")

    @staticmethod
    def is_available() -> bool:
        """Check if MLX Metal backend is available."""
        try:
            mx.eval(mx.zeros(1))
            return True
        except Exception:
            return False
```

---

## 6. Module: `open3d_mlx/core/tensor_utils.py`

```python
"""Utility functions for MLX array operations.

We use mx.array directly as the tensor type. These utilities provide
convenience wrappers that match Open3D's common tensor operations.
"""

import mlx.core as mx
import numpy as np


def to_numpy(arr: mx.array) -> np.ndarray:
    """Convert MLX array to NumPy. Zero-copy when possible."""
    return np.array(arr)


def from_numpy(arr: np.ndarray, dtype=None) -> mx.array:
    """Convert NumPy array to MLX array."""
    result = mx.array(arr)
    if dtype is not None:
        result = result.astype(dtype)
    return result


def eye4(dtype=mx.float32) -> mx.array:
    """4x4 identity matrix (common for transformations)."""
    return mx.eye(4, dtype=dtype)


def is_finite(arr: mx.array) -> mx.array:
    """Check if all elements are finite (no nan, no inf)."""
    return mx.logical_not(mx.logical_or(mx.isnan(arr), mx.isinf(arr)))


def check_shape(arr: mx.array, expected_shape: tuple, name: str = "array"):
    """Validate array shape, raise ValueError if mismatch."""
    if arr.shape != expected_shape:
        raise ValueError(
            f"Expected {name} shape {expected_shape}, got {arr.shape}"
        )


def check_points_shape(arr: mx.array, name: str = "points"):
    """Validate (N, 3) point array shape."""
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Expected {name} shape (N, 3), got {arr.shape}"
        )
```

---

## 7. Module: `open3d_mlx/core/__init__.py`

```python
"""Core types and utilities for Open3D-MLX."""

from open3d_mlx.core.device import Device
from open3d_mlx.core.dtype import DTYPE_MAP, ensure_float32, ensure_int32, to_mlx_dtype
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
```

---

## 8. Tests

### `tests/test_core/test_dtype.py`

```python
def test_dtype_map_completeness():
    """All Open3D dtypes have MLX equivalents."""

def test_ensure_float32_passthrough():
    """float32 arrays pass through without copy."""

def test_ensure_float32_conversion():
    """float64 arrays are converted to float32."""

def test_ensure_int32_from_int64():
    """int64 arrays are downcast to int32."""

def test_to_mlx_dtype_from_string():
    """String dtype names resolve correctly."""

def test_to_mlx_dtype_from_numpy():
    """NumPy dtypes resolve correctly."""
```

### `tests/test_core/test_device.py`

```python
def test_device_default():
    """Default device is MLX:0."""

def test_device_is_available():
    """MLX device is available on Apple Silicon."""

def test_device_equality_string():
    """Device compares equal to matching string."""
```

### `tests/test_core/test_tensor_utils.py`

```python
def test_numpy_roundtrip():
    """MLX → NumPy → MLX preserves values."""

def test_eye4_shape():
    """eye4 returns (4, 4) identity matrix."""

def test_check_shape_valid():
    """Valid shapes pass without error."""

def test_check_shape_invalid():
    """Invalid shapes raise ValueError."""

def test_check_points_shape_valid():
    """(N, 3) arrays pass validation."""

def test_check_points_shape_invalid():
    """Non-(N, 3) arrays raise ValueError."""
```

---

## 9. MLX Gotchas Applied

1. **No `mlx.__version__`** — use `importlib.metadata.version("mlx")` if version checking needed
2. **No int64 on GPU** — `ensure_int32()` applied to all index arrays
3. **No float64 on GPU** — `ensure_float32()` for computation, Python float for scalar results
4. **Functional API** — all operations return new arrays, no in-place mutation
5. **Lazy evaluation** — `mx.eval()` required to materialize results for timing

---

## 10. Acceptance Criteria

- [ ] `from open3d_mlx.core import Device, ensure_float32, eye4` works
- [ ] `to_mlx_dtype("float32")` returns `mx.float32`
- [ ] `ensure_float32(mx.array([1.0], dtype=mx.float16))` returns float32 array
- [ ] `ensure_int32(mx.array([1], dtype=mx.int64))` returns int32 array (if mx.int64 exists)
- [ ] `eye4()` returns 4x4 identity of dtype float32
- [ ] NumPy roundtrip preserves values within float32 precision
- [ ] All tests in `tests/test_core/` pass
- [ ] Zero external dependencies beyond `mlx` and `numpy`
