"""Tests for open3d_mlx.core.tensor_utils module."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.core.tensor_utils import (
    check_points_shape,
    check_shape,
    eye4,
    from_numpy,
    is_finite,
    to_numpy,
)


# ---------------------------------------------------------------------------
# to_numpy / from_numpy roundtrip
# ---------------------------------------------------------------------------

class TestNumpyInterop:
    """Tests for to_numpy() and from_numpy()."""

    def test_numpy_roundtrip_float32(self):
        """MLX -> NumPy -> MLX preserves float32 values."""
        original = mx.array([1.0, 2.5, -3.14], dtype=mx.float32)
        np_arr = to_numpy(original)
        assert isinstance(np_arr, np.ndarray)
        assert np_arr.dtype == np.float32
        restored = from_numpy(np_arr)
        mx.eval(restored)
        np.testing.assert_allclose(np.array(restored), np.array(original), atol=1e-6)

    def test_numpy_roundtrip_int32(self):
        """MLX -> NumPy -> MLX preserves int32 values."""
        original = mx.array([0, 1, -42, 100], dtype=mx.int32)
        np_arr = to_numpy(original)
        assert np_arr.dtype == np.int32
        restored = from_numpy(np_arr)
        mx.eval(restored)
        np.testing.assert_array_equal(np.array(restored), np.array(original))

    def test_numpy_roundtrip_2d(self):
        """2D arrays survive the roundtrip."""
        original = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
        np_arr = to_numpy(original)
        assert np_arr.shape == (2, 2)
        restored = from_numpy(np_arr)
        mx.eval(restored)
        np.testing.assert_allclose(np.array(restored), np.array(original))

    def test_from_numpy_with_dtype_cast(self):
        """from_numpy with explicit dtype casts correctly."""
        np_arr = np.array([1, 2, 3], dtype=np.int64)
        result = from_numpy(np_arr, dtype=mx.float32)
        assert result.dtype == mx.float32
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), [1.0, 2.0, 3.0])

    def test_from_numpy_without_dtype(self):
        """from_numpy without dtype infers from the numpy array."""
        np_arr = np.array([1.0, 2.0], dtype=np.float32)
        result = from_numpy(np_arr)
        assert result.dtype == mx.float32

    def test_to_numpy_empty_array(self):
        """Empty MLX arrays convert to empty numpy arrays."""
        a = mx.array([], dtype=mx.float32)
        np_arr = to_numpy(a)
        assert np_arr.shape == (0,)


# ---------------------------------------------------------------------------
# eye4
# ---------------------------------------------------------------------------

class TestEye4:
    """Tests for eye4()."""

    def test_shape(self):
        """eye4 returns (4, 4) matrix."""
        result = eye4()
        assert result.shape == (4, 4)

    def test_dtype_default_float32(self):
        """Default dtype is float32."""
        result = eye4()
        assert result.dtype == mx.float32

    def test_dtype_custom(self):
        """Custom dtype is respected."""
        result = eye4(dtype=mx.float16)
        assert result.dtype == mx.float16

    def test_identity_values(self):
        """Diagonal is 1, off-diagonal is 0."""
        result = eye4()
        mx.eval(result)
        expected = np.eye(4, dtype=np.float32)
        np.testing.assert_array_equal(np.array(result), expected)


# ---------------------------------------------------------------------------
# is_finite
# ---------------------------------------------------------------------------

class TestIsFinite:
    """Tests for is_finite()."""

    def test_all_finite(self):
        """Normal values are all finite."""
        a = mx.array([1.0, -2.0, 0.0, 3.14])
        result = is_finite(a)
        mx.eval(result)
        assert np.all(np.array(result))

    def test_nan_detected(self):
        """NaN values are not finite."""
        a = mx.array([1.0, float("nan"), 3.0])
        result = is_finite(a)
        mx.eval(result)
        np_result = np.array(result)
        assert np_result[0] is np.True_
        assert np_result[1] is np.False_
        assert np_result[2] is np.True_

    def test_inf_detected(self):
        """Inf values are not finite."""
        a = mx.array([float("inf"), 1.0, float("-inf")])
        result = is_finite(a)
        mx.eval(result)
        np_result = np.array(result)
        assert np_result[0] is np.False_
        assert np_result[1] is np.True_
        assert np_result[2] is np.False_

    def test_output_shape_preserved(self):
        """Output shape matches input shape."""
        a = mx.array([[1.0, 2.0], [float("nan"), 4.0]])
        result = is_finite(a)
        assert result.shape == a.shape


# ---------------------------------------------------------------------------
# check_shape
# ---------------------------------------------------------------------------

class TestCheckShape:
    """Tests for check_shape()."""

    def test_valid_shape(self):
        """Valid shapes pass without error."""
        a = mx.zeros((3, 4))
        check_shape(a, (3, 4))  # Should not raise.

    def test_valid_shape_1d(self):
        """1D shape validation works."""
        a = mx.zeros((5,))
        check_shape(a, (5,))

    def test_invalid_shape(self):
        """Mismatched shape raises ValueError."""
        a = mx.zeros((3, 4))
        with pytest.raises(ValueError, match="Expected array shape"):
            check_shape(a, (4, 3))

    def test_custom_name_in_error(self):
        """Custom name appears in the error message."""
        a = mx.zeros((2,))
        with pytest.raises(ValueError, match="Expected my_tensor shape"):
            check_shape(a, (3,), name="my_tensor")

    def test_shape_from_list(self):
        """Expected shape can be passed as a list."""
        a = mx.zeros((2, 3))
        check_shape(a, [2, 3])


# ---------------------------------------------------------------------------
# check_points_shape
# ---------------------------------------------------------------------------

class TestCheckPointsShape:
    """Tests for check_points_shape()."""

    def test_valid_n3(self):
        """(N, 3) arrays pass validation."""
        a = mx.zeros((100, 3))
        check_points_shape(a)  # Should not raise.

    def test_valid_single_point(self):
        """(1, 3) single-point array passes."""
        a = mx.zeros((1, 3))
        check_points_shape(a)

    def test_invalid_1d(self):
        """1D arrays raise ValueError."""
        a = mx.zeros((9,))
        with pytest.raises(ValueError, match=r"Expected points shape \(N, 3\)"):
            check_points_shape(a)

    def test_invalid_wrong_columns(self):
        """(N, 4) arrays raise ValueError."""
        a = mx.zeros((10, 4))
        with pytest.raises(ValueError, match=r"Expected points shape \(N, 3\)"):
            check_points_shape(a)

    def test_invalid_3d(self):
        """3D arrays raise ValueError."""
        a = mx.zeros((2, 3, 4))
        with pytest.raises(ValueError, match=r"Expected points shape \(N, 3\)"):
            check_points_shape(a)

    def test_custom_name_in_error(self):
        """Custom name appears in the error message."""
        a = mx.zeros((5, 2))
        with pytest.raises(ValueError, match="Expected normals shape"):
            check_points_shape(a, name="normals")

    def test_empty_points(self):
        """(0, 3) empty point cloud passes."""
        a = mx.zeros((0, 3))
        check_points_shape(a)
