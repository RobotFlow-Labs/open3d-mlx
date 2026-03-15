"""Tests for open3d_mlx.core.dtype module."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.core.dtype import (
    DTYPE_MAP,
    MLX_TO_NAME,
    ensure_float32,
    ensure_int32,
    to_mlx_dtype,
)


# ---------------------------------------------------------------------------
# DTYPE_MAP
# ---------------------------------------------------------------------------

class TestDtypeMap:
    """Tests for the DTYPE_MAP constant."""

    def test_dtype_map_completeness(self):
        """All essential Open3D dtypes have MLX equivalents."""
        required_keys = [
            "float32", "float64", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "bool",
        ]
        for key in required_keys:
            assert key in DTYPE_MAP, f"Missing dtype mapping for '{key}'"

    def test_dtype_map_values_are_mlx_dtypes(self):
        """Every value in DTYPE_MAP is an mlx.core.Dtype instance."""
        for key, value in DTYPE_MAP.items():
            assert isinstance(value, mx.Dtype), (
                f"DTYPE_MAP['{key}'] = {value!r} is not an mlx.core.Dtype"
            )

    def test_float64_downcasts_to_float32(self):
        """float64 maps to float32 (no float64 on Metal GPU)."""
        assert DTYPE_MAP["float64"] == mx.float32

    def test_int64_downcasts_to_int32(self):
        """int64 maps to int32 (no int64 on Metal GPU)."""
        assert DTYPE_MAP["int64"] == mx.int32

    def test_bool_maps_to_bool_(self):
        """'bool' string maps to mx.bool_."""
        assert DTYPE_MAP["bool"] == mx.bool_


class TestMlxToName:
    """Tests for the MLX_TO_NAME reverse mapping."""

    def test_reverse_map_has_all_non_downcast_types(self):
        """All non-downcast MLX types appear in the reverse map."""
        expected_types = {mx.float32, mx.int32, mx.int8, mx.int16,
                         mx.uint8, mx.uint16, mx.uint32, mx.bool_}
        for dt in expected_types:
            assert dt in MLX_TO_NAME, f"Missing reverse mapping for {dt}"

    def test_reverse_map_returns_strings(self):
        """All values are strings."""
        for dt, name in MLX_TO_NAME.items():
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# to_mlx_dtype
# ---------------------------------------------------------------------------

class TestToMlxDtype:
    """Tests for to_mlx_dtype()."""

    @pytest.mark.parametrize("name,expected", [
        ("float32", mx.float32),
        ("float64", mx.float32),
        ("int32", mx.int32),
        ("int64", mx.int32),
        ("uint8", mx.uint8),
        ("bool", mx.bool_),
        ("float16", mx.float16),
    ])
    def test_from_string(self, name, expected):
        """String dtype names resolve correctly."""
        assert to_mlx_dtype(name) == expected

    def test_from_string_case_insensitive(self):
        """Strings are case-insensitive."""
        assert to_mlx_dtype("FLOAT32") == mx.float32
        assert to_mlx_dtype("Float32") == mx.float32

    def test_from_string_with_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        assert to_mlx_dtype("  float32  ") == mx.float32

    def test_from_string_invalid(self):
        """Unknown string raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized dtype string"):
            to_mlx_dtype("complex128")

    @pytest.mark.parametrize("np_dtype,expected", [
        (np.dtype("float32"), mx.float32),
        (np.dtype("float64"), mx.float32),
        (np.dtype("int32"), mx.int32),
        (np.dtype("int64"), mx.int32),
        (np.dtype("uint8"), mx.uint8),
        (np.dtype("bool"), mx.bool_),
    ])
    def test_from_numpy_dtype(self, np_dtype, expected):
        """NumPy dtype instances resolve correctly."""
        assert to_mlx_dtype(np_dtype) == expected

    def test_from_numpy_type_object(self):
        """NumPy type objects (e.g. np.float32) resolve correctly."""
        assert to_mlx_dtype(np.float32) == mx.float32
        assert to_mlx_dtype(np.float64) == mx.float32
        assert to_mlx_dtype(np.int64) == mx.int32

    def test_from_mlx_dtype_passthrough(self):
        """MLX dtype instances pass through unchanged."""
        assert to_mlx_dtype(mx.float32) is mx.float32
        assert to_mlx_dtype(mx.int32) is mx.int32

    def test_from_numpy_dtype_invalid(self):
        """Unsupported numpy dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized numpy dtype"):
            to_mlx_dtype(np.dtype("complex64"))

    def test_invalid_type(self):
        """Non-string, non-dtype input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            to_mlx_dtype(42)


# ---------------------------------------------------------------------------
# ensure_float32
# ---------------------------------------------------------------------------

class TestEnsureFloat32:
    """Tests for ensure_float32()."""

    def test_passthrough_float32(self):
        """float32 arrays are returned as-is (no copy)."""
        a = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
        b = ensure_float32(a)
        # Should be the exact same object since dtype already matches.
        assert b.dtype == mx.float32

    def test_conversion_from_float16(self):
        """float16 arrays are converted to float32."""
        a = mx.array([1.0, 2.0], dtype=mx.float16)
        b = ensure_float32(a)
        assert b.dtype == mx.float32
        mx.eval(b)
        np.testing.assert_allclose(np.array(b), [1.0, 2.0], atol=1e-3)

    def test_conversion_from_int32(self):
        """Integer arrays are cast to float32."""
        a = mx.array([1, 2, 3], dtype=mx.int32)
        b = ensure_float32(a)
        assert b.dtype == mx.float32
        mx.eval(b)
        np.testing.assert_array_equal(np.array(b), [1.0, 2.0, 3.0])

    def test_preserves_values(self):
        """Values are preserved within float32 precision."""
        a = mx.array([1.5, -2.5, 0.0, 3.14], dtype=mx.float32)
        b = ensure_float32(a)
        mx.eval(b)
        np.testing.assert_allclose(np.array(b), [1.5, -2.5, 0.0, 3.14], atol=1e-6)


# ---------------------------------------------------------------------------
# ensure_int32
# ---------------------------------------------------------------------------

class TestEnsureInt32:
    """Tests for ensure_int32()."""

    def test_passthrough_int32(self):
        """int32 arrays are returned as-is."""
        a = mx.array([1, 2, 3], dtype=mx.int32)
        b = ensure_int32(a)
        assert b.dtype == mx.int32

    def test_conversion_from_int64(self):
        """int64 arrays are downcast to int32."""
        a = mx.array([10, 20, 30], dtype=mx.int64)
        b = ensure_int32(a)
        assert b.dtype == mx.int32
        mx.eval(b)
        np.testing.assert_array_equal(np.array(b), [10, 20, 30])

    def test_conversion_from_int16(self):
        """int16 arrays are cast to int32."""
        a = mx.array([1, 2], dtype=mx.int16)
        b = ensure_int32(a)
        assert b.dtype == mx.int32

    def test_conversion_from_uint8(self):
        """uint8 arrays are cast to int32."""
        a = mx.array([0, 255], dtype=mx.uint8)
        b = ensure_int32(a)
        assert b.dtype == mx.int32
        mx.eval(b)
        np.testing.assert_array_equal(np.array(b), [0, 255])
