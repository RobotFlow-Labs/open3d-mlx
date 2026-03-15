"""Tests for SVD-based and point-to-plane transformation estimation."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.pipelines.registration.transformation import (
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
    _rotation_from_euler_small,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rotation_z(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around Z axis."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _make_rotation_x(angle_deg: float) -> np.ndarray:
    """3x3 rotation matrix around X axis."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _random_points(n: int = 200, seed: int = 42) -> np.ndarray:
    """Generate random 3D points."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float64)


def _apply_transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply R @ p + t."""
    return (R @ pts.T).T + t


def _full_correspondences(n: int) -> mx.array:
    """All-valid correspondence vector [0, 1, 2, ..., n-1]."""
    return mx.array(np.arange(n, dtype=np.int32))


# ---------------------------------------------------------------------------
# Point-to-Point SVD tests
# ---------------------------------------------------------------------------


class TestTransformationEstimationPointToPoint:
    """Tests for SVD-based point-to-point estimation."""

    def test_identical_points_gives_identity(self):
        """Identical source and target should yield identity transform."""
        pts = _random_points(100)
        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(pts.astype(np.float32))
        corr = _full_correspondences(100)

        est = TransformationEstimationPointToPoint()
        T = est.compute_transformation(src, tgt, corr)

        T_np = np.array(T, dtype=np.float64)
        np.testing.assert_allclose(T_np, np.eye(4), atol=1e-4)

    def test_known_translation(self):
        """Recover a known translation [1, 0, 0]."""
        pts = _random_points(200)
        t_true = np.array([1.0, 0.0, 0.0])
        R_true = np.eye(3)

        src_np = pts.copy()
        tgt_np = _apply_transform(pts, R_true, t_true)

        src = mx.array(src_np.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        corr = _full_correspondences(200)

        est = TransformationEstimationPointToPoint()
        T = est.compute_transformation(src, tgt, corr)
        T_np = np.array(T, dtype=np.float64)

        # Check rotation is identity
        np.testing.assert_allclose(T_np[:3, :3], np.eye(3), atol=1e-4)
        # Check translation recovered
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=1e-4)

    def test_known_rotation_45_z(self):
        """Recover a known 45-degree rotation around Z."""
        pts = _random_points(200)
        R_true = _make_rotation_z(45.0)
        t_true = np.zeros(3)

        src_np = pts.copy()
        tgt_np = _apply_transform(pts, R_true, t_true)

        src = mx.array(src_np.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        corr = _full_correspondences(200)

        est = TransformationEstimationPointToPoint()
        T = est.compute_transformation(src, tgt, corr)
        T_np = np.array(T, dtype=np.float64)

        np.testing.assert_allclose(T_np[:3, :3], R_true, atol=1e-3)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=1e-3)

    def test_rotation_and_translation(self):
        """Recover rotation + translation combined."""
        pts = _random_points(300)
        R_true = _make_rotation_z(30.0)
        t_true = np.array([2.0, -1.0, 0.5])

        src_np = pts.copy()
        tgt_np = _apply_transform(pts, R_true, t_true)

        src = mx.array(src_np.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        corr = _full_correspondences(300)

        est = TransformationEstimationPointToPoint()
        T = est.compute_transformation(src, tgt, corr)
        T_np = np.array(T, dtype=np.float64)

        np.testing.assert_allclose(T_np[:3, :3], R_true, atol=1e-3)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=1e-3)

    def test_reflection_handling(self):
        """Ensure the determinant check prevents reflections."""
        # Create a pathological case where without the det-check, we could
        # get a reflection. Use coplanar points (all Z=0).
        rng = np.random.default_rng(99)
        pts = np.zeros((50, 3), dtype=np.float64)
        pts[:, :2] = rng.standard_normal((50, 2))

        R_true = _make_rotation_z(15.0)
        t_true = np.array([0.5, 0.3, 0.0])
        tgt_np = _apply_transform(pts, R_true, t_true)

        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        corr = _full_correspondences(50)

        est = TransformationEstimationPointToPoint()
        T = est.compute_transformation(src, tgt, corr)
        T_np = np.array(T, dtype=np.float64)

        # The resulting rotation must be proper (det = +1)
        det = np.linalg.det(T_np[:3, :3])
        assert det > 0, f"Expected det > 0, got {det}"
        np.testing.assert_allclose(abs(det), 1.0, atol=1e-4)

    def test_compute_rmse(self):
        """Test RMSE computation."""
        pts = _random_points(100)
        t_true = np.array([0.5, 0.0, 0.0])
        tgt_np = pts + t_true

        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        corr = _full_correspondences(100)

        est = TransformationEstimationPointToPoint()
        fitness, rmse = est.compute_rmse(src, tgt, corr)

        assert fitness == 1.0
        np.testing.assert_allclose(rmse, 0.5, atol=1e-4)

    def test_partial_correspondences(self):
        """Only some points have correspondences (others are -1)."""
        pts = _random_points(100)
        t_true = np.array([1.0, 0.0, 0.0])
        tgt_np = pts + t_true

        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))

        # Only first 50 points have correspondences
        corr_np = np.full(100, -1, dtype=np.int32)
        corr_np[:50] = np.arange(50, dtype=np.int32)
        corr = mx.array(corr_np)

        est = TransformationEstimationPointToPoint()
        T = est.compute_transformation(src, tgt, corr)
        T_np = np.array(T, dtype=np.float64)

        # Should still recover the translation from the 50 valid pairs
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=1e-3)


# ---------------------------------------------------------------------------
# Point-to-Plane tests
# ---------------------------------------------------------------------------


class TestTransformationEstimationPointToPlane:
    """Tests for linearized point-to-plane estimation."""

    def test_identity_aligned(self):
        """Aligned points with normals should yield near-identity."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((200, 3)).astype(np.float64)
        # Simple normals pointing up
        normals = np.tile([0.0, 0.0, 1.0], (200, 1))

        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(pts.astype(np.float32))
        nrm = mx.array(normals.astype(np.float32))
        corr = _full_correspondences(200)

        est = TransformationEstimationPointToPlane()
        T = est.compute_transformation(src, tgt, nrm, corr)
        T_np = np.array(T, dtype=np.float64)

        np.testing.assert_allclose(T_np, np.eye(4), atol=1e-4)

    def test_known_translation(self):
        """Recover a known translation using point-to-plane."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((300, 3)).astype(np.float64)
        t_true = np.array([0.5, 0.3, 0.2])
        tgt_np = pts + t_true

        # Use varied normals for a well-conditioned system
        normals = rng.standard_normal((300, 3)).astype(np.float64)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        nrm = mx.array(normals.astype(np.float32))
        corr = _full_correspondences(300)

        est = TransformationEstimationPointToPlane()
        T = est.compute_transformation(src, tgt, nrm, corr)
        T_np = np.array(T, dtype=np.float64)

        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=1e-3)

    def test_known_small_rotation(self):
        """Recover a small rotation (10 degrees around Z)."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((300, 3)).astype(np.float64)
        R_true = _make_rotation_z(10.0)
        tgt_np = (R_true @ pts.T).T

        normals = rng.standard_normal((300, 3)).astype(np.float64)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        # Transform normals to match target
        normals_tgt = (R_true @ normals.T).T

        src = mx.array(pts.astype(np.float32))
        tgt = mx.array(tgt_np.astype(np.float32))
        nrm = mx.array(normals_tgt.astype(np.float32))
        corr = _full_correspondences(300)

        est = TransformationEstimationPointToPlane()
        T = est.compute_transformation(src, tgt, nrm, corr)
        T_np = np.array(T, dtype=np.float64)

        # Check rotation is close
        np.testing.assert_allclose(T_np[:3, :3], R_true, atol=5e-2)


# ---------------------------------------------------------------------------
# Rodrigues helper tests
# ---------------------------------------------------------------------------


class TestRodriguesRotation:
    """Tests for the Rodrigues rotation helper."""

    def test_zero_angle_gives_identity(self):
        R = _rotation_from_euler_small(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_small_angle_close_to_skew(self):
        """For very small angles, R should be close to I + [w]x."""
        alpha, beta, gamma = 0.001, 0.002, -0.001
        R = _rotation_from_euler_small(alpha, beta, gamma)
        # Should be a valid rotation
        assert abs(np.linalg.det(R) - 1.0) < 1e-8
        # R^T R should be identity
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)

    def test_produces_valid_rotation(self):
        """Output must be orthonormal with det=+1."""
        R = _rotation_from_euler_small(0.3, -0.2, 0.1)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-8)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-8)
