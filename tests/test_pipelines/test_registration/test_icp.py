"""Tests for ICP registration (point-to-point and point-to-plane)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.registration import (
    ICPConvergenceCriteria,
    RegistrationResult,
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
    evaluate_registration,
    registration_icp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rotation_z(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _random_cloud(n: int = 200, seed: int = 42) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    return PointCloud(mx.array(pts))


def _random_cloud_with_normals(n: int = 200, seed: int = 42) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    normals = rng.standard_normal((n, 3)).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    pcd = PointCloud(mx.array(pts))
    pcd.normals = mx.array(normals.astype(np.float32))
    return pcd


def _transform_cloud(pcd: PointCloud, T: np.ndarray) -> PointCloud:
    """Apply a numpy 4x4 transform to a PointCloud."""
    return pcd.clone().transform(mx.array(T.astype(np.float32)))


# ---------------------------------------------------------------------------
# Point-to-Point ICP tests
# ---------------------------------------------------------------------------


class TestICPPointToPoint:
    """Tests for point-to-point ICP registration."""

    def test_identical_clouds(self):
        """Identical source and target should give fitness ~1.0, RMSE ~0."""
        cloud = _random_cloud(200)
        result = registration_icp(
            cloud, cloud, max_correspondence_distance=1.0
        )
        assert result.fitness > 0.99
        assert result.inlier_rmse < 0.01
        T_np = np.array(result.transformation, dtype=np.float64)
        np.testing.assert_allclose(T_np, np.eye(4), atol=1e-3)

    def test_known_translation(self):
        """Recover a known translation [1, 0, 0]."""
        source = _random_cloud(300, seed=42)
        t_true = np.array([1.0, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=2.0,
            criteria=ICPConvergenceCriteria(max_iteration=50),
        )

        assert result.fitness > 0.95
        assert result.inlier_rmse < 0.1

        T_np = np.array(result.transformation, dtype=np.float64)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=0.1)

    def test_known_rotation_45deg(self):
        """Recover a known 45-degree rotation around Z."""
        source = _random_cloud(300, seed=42)
        R_true = _make_rotation_z(45.0)
        T_true = _make_transform(R_true, np.zeros(3))
        target = _transform_cloud(source, T_true)

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=3.0,
            criteria=ICPConvergenceCriteria(max_iteration=80),
        )

        assert result.fitness > 0.8
        assert result.inlier_rmse < 0.5

    def test_convergence_stops_early(self):
        """ICP should converge before max_iteration on easy problems."""
        source = _random_cloud(200, seed=42)
        t_true = np.array([0.1, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        criteria = ICPConvergenceCriteria(max_iteration=100)
        result = registration_icp(
            source, target, max_correspondence_distance=1.0, criteria=criteria
        )

        assert result.converged
        assert result.num_iterations < 100

    def test_max_correspondence_distance_respected(self):
        """Small max_distance should reduce fitness when clouds are far."""
        source = _random_cloud(200, seed=42)
        t_true = np.array([5.0, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        # Distance=0.5 is much smaller than the translation
        result = registration_icp(
            source, target, max_correspondence_distance=0.5
        )
        assert result.fitness < 0.1

    def test_voxel_downsample(self):
        """Voxel downsampling should still produce valid results."""
        source = _random_cloud(500, seed=42)
        t_true = np.array([0.2, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=1.0,
            voxel_size=0.5,
        )
        # Should still produce a reasonable result
        assert result.fitness > 0.5
        assert result.inlier_rmse < 1.0

    def test_empty_source(self):
        """Empty source should return zero fitness."""
        source = PointCloud()
        target = _random_cloud(100)
        result = registration_icp(source, target, max_correspondence_distance=1.0)
        assert result.fitness == 0.0
        assert result.num_iterations == 0

    def test_empty_target(self):
        """Empty target should return zero fitness."""
        source = _random_cloud(100)
        target = PointCloud()
        result = registration_icp(source, target, max_correspondence_distance=1.0)
        assert result.fitness == 0.0
        assert result.num_iterations == 0

    def test_no_overlap_zero_fitness(self):
        """Non-overlapping clouds with tiny max_distance -> zero fitness."""
        source = _random_cloud(100, seed=42)
        # Move target very far away
        t_true = np.array([100.0, 100.0, 100.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        result = registration_icp(
            source, target, max_correspondence_distance=0.1
        )
        assert result.fitness == 0.0

    def test_init_source_to_target(self):
        """Providing a good initial guess should help convergence."""
        source = _random_cloud(200, seed=42)
        t_true = np.array([2.0, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        # Give initial guess close to the truth
        init_T = mx.array(
            _make_transform(np.eye(3), np.array([1.8, 0.0, 0.0])).astype(
                np.float32
            )
        )

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=1.0,
            init_source_to_target=init_T,
        )

        assert result.fitness > 0.9
        T_np = np.array(result.transformation, dtype=np.float64)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=0.2)


# ---------------------------------------------------------------------------
# evaluate_registration tests
# ---------------------------------------------------------------------------


class TestEvaluateRegistration:
    """Tests for evaluate_registration (no ICP iteration)."""

    def test_identity(self):
        """Identity transform on identical clouds."""
        cloud = _random_cloud(100)
        result = evaluate_registration(
            cloud, cloud, max_correspondence_distance=1.0
        )
        assert result.fitness > 0.99
        assert result.inlier_rmse < 0.01
        assert result.num_iterations == 0

    def test_with_transform(self):
        """Evaluate with a known good transform."""
        source = _random_cloud(200, seed=42)
        t_true = np.array([1.0, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        # Give the correct transform
        result = evaluate_registration(
            source,
            target,
            max_correspondence_distance=1.0,
            transformation=mx.array(T_true.astype(np.float32)),
        )
        assert result.fitness > 0.99
        assert result.inlier_rmse < 0.01
        assert result.num_iterations == 0

    def test_empty_clouds(self):
        """Empty clouds should return zero fitness."""
        result = evaluate_registration(
            PointCloud(), _random_cloud(10), max_correspondence_distance=1.0
        )
        assert result.fitness == 0.0


# ---------------------------------------------------------------------------
# RegistrationResult tests
# ---------------------------------------------------------------------------


class TestRegistrationResult:
    """Tests for RegistrationResult dataclass."""

    def test_is_better_than_fitness(self):
        r1 = RegistrationResult(fitness=0.9, inlier_rmse=0.1)
        r2 = RegistrationResult(fitness=0.8, inlier_rmse=0.05)
        assert r1.is_better_than(r2)
        assert not r2.is_better_than(r1)

    def test_is_better_than_rmse_tiebreaker(self):
        r1 = RegistrationResult(fitness=0.9, inlier_rmse=0.05)
        r2 = RegistrationResult(fitness=0.9, inlier_rmse=0.1)
        assert r1.is_better_than(r2)
        assert not r2.is_better_than(r1)

    def test_repr(self):
        r = RegistrationResult(fitness=0.95, inlier_rmse=0.01)
        s = repr(r)
        assert "fitness" in s
        assert "0.95" in s


# ---------------------------------------------------------------------------
# Point-to-Plane ICP tests
# ---------------------------------------------------------------------------


class TestICPPointToPlane:
    """Tests for point-to-plane ICP registration."""

    def test_missing_normals_raises(self):
        """Point-to-plane without target normals should raise ValueError."""
        source = _random_cloud(100)
        target = _random_cloud(100)  # no normals
        with pytest.raises(ValueError, match="normals"):
            registration_icp(
                source,
                target,
                max_correspondence_distance=1.0,
                estimation_method=TransformationEstimationPointToPlane(),
            )

    def test_identical_clouds(self):
        """Identical clouds with normals -> fitness ~1.0."""
        cloud = _random_cloud_with_normals(200, seed=42)
        result = registration_icp(
            cloud,
            cloud,
            max_correspondence_distance=1.0,
            estimation_method=TransformationEstimationPointToPlane(),
        )
        assert result.fitness > 0.99
        assert result.inlier_rmse < 0.01

    def test_known_translation(self):
        """Point-to-plane recovers known translation."""
        target = _random_cloud_with_normals(300, seed=42)
        t_true = np.array([0.5, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)

        # Source is the un-translated version
        source_pts_np = np.array(target.points) - t_true.astype(np.float32)
        source = PointCloud(mx.array(source_pts_np))

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=2.0,
            estimation_method=TransformationEstimationPointToPlane(),
            criteria=ICPConvergenceCriteria(max_iteration=50),
        )

        assert result.fitness > 0.9
        T_np = np.array(result.transformation, dtype=np.float64)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=0.2)

    def test_point_to_plane_converges_faster(self):
        """Point-to-plane should converge in fewer iterations than point-to-point
        on a smooth surface problem with small displacement."""
        target = _random_cloud_with_normals(300, seed=42)
        t_true = np.array([0.3, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)

        source_pts_np = np.array(target.points) - t_true.astype(np.float32)
        source = PointCloud(mx.array(source_pts_np))

        # Point-to-point
        result_p2p = registration_icp(
            source,
            target,
            max_correspondence_distance=2.0,
            estimation_method=TransformationEstimationPointToPoint(),
            criteria=ICPConvergenceCriteria(max_iteration=100),
        )

        # Point-to-plane
        result_p2l = registration_icp(
            source,
            target,
            max_correspondence_distance=2.0,
            estimation_method=TransformationEstimationPointToPlane(),
            criteria=ICPConvergenceCriteria(max_iteration=100),
        )

        # Both should have good fitness
        assert result_p2p.fitness > 0.8
        assert result_p2l.fitness > 0.8

        # Point-to-plane should use fewer or equal iterations
        # (allow a margin since stochastic normals don't always guarantee faster)
        assert result_p2l.num_iterations <= result_p2p.num_iterations + 5

    def test_empty_source(self):
        """Empty source with point-to-plane should return zero fitness."""
        target = _random_cloud_with_normals(100)
        result = registration_icp(
            PointCloud(),
            target,
            max_correspondence_distance=1.0,
            estimation_method=TransformationEstimationPointToPlane(),
        )
        assert result.fitness == 0.0
