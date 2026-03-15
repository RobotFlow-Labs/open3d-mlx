"""Tests for PinholeCameraIntrinsic."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.camera import PinholeCameraIntrinsic


class TestPinholeCameraIntrinsic:
    """Camera intrinsic tests."""

    def test_intrinsic_matrix_shape(self):
        cam = PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
        K = cam.intrinsic_matrix
        assert K.shape == (3, 3)
        assert K.dtype == mx.float32

    def test_intrinsic_matrix_values(self):
        cam = PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
        K = np.array(cam.intrinsic_matrix)
        np.testing.assert_allclose(K[0, 0], 500.0)
        np.testing.assert_allclose(K[1, 1], 500.0)
        np.testing.assert_allclose(K[0, 2], 320.0)
        np.testing.assert_allclose(K[1, 2], 240.0)
        np.testing.assert_allclose(K[2, 2], 1.0)
        # Off-diagonals that should be zero
        np.testing.assert_allclose(K[0, 1], 0.0)
        np.testing.assert_allclose(K[1, 0], 0.0)
        np.testing.assert_allclose(K[2, 0], 0.0)
        np.testing.assert_allclose(K[2, 1], 0.0)

    def test_prime_sense_default(self):
        cam = PinholeCameraIntrinsic.prime_sense_default()
        assert cam.width == 640
        assert cam.height == 480
        assert cam.fx == 525.0
        assert cam.fy == 525.0
        assert cam.cx == 319.5
        assert cam.cy == 239.5

    def test_projection(self):
        """K @ [x,y,z]/z should give correct pixel coordinates."""
        cam = PinholeCameraIntrinsic(640, 480, 500.0, 500.0, 320.0, 240.0)
        # A 3D point in camera frame
        point_cam = np.array([0.1, -0.2, 1.5])
        # Expected pixel: u = fx * x/z + cx, v = fy * y/z + cy
        u_exp = 500.0 * 0.1 / 1.5 + 320.0
        v_exp = 500.0 * (-0.2) / 1.5 + 240.0

        K = np.array(cam.intrinsic_matrix)
        projected = K @ (point_cam / point_cam[2])
        np.testing.assert_allclose(projected[0], u_exp, atol=1e-4)
        np.testing.assert_allclose(projected[1], v_exp, atol=1e-4)
        np.testing.assert_allclose(projected[2], 1.0, atol=1e-6)
