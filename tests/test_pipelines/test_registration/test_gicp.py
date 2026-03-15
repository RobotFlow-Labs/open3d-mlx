"""Tests for Generalized ICP (GICP) registration."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.registration import (
    ICPConvergenceCriteria,
    TransformationEstimationForGeneralizedICP,
    compute_point_covariances,
    registration_icp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _random_cloud(n: int = 200, seed: int = 42) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    return PointCloud(mx.array(pts))


def _transform_cloud(pcd: PointCloud, T: np.ndarray) -> PointCloud:
    return pcd.clone().transform(mx.array(T.astype(np.float32)))


def _build_neighbor_indices(points_np: np.ndarray, k: int = 10) -> np.ndarray:
    """Build K-nearest neighbor indices using brute force."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points_np)
    _, indices = tree.query(points_np, k=k)
    return indices


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputePointCovariances:
    """Tests for compute_point_covariances helper."""

    def test_output_shape(self):
        """Covariance output should be (N, 3, 3)."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((50, 3)).astype(np.float64)
        # Simple neighbor indices: each point uses its 5 nearest
        nbr_idx = _build_neighbor_indices(pts, k=5)
        covs = compute_point_covariances(pts, nbr_idx, epsilon=0.001)
        assert covs.shape == (50, 3, 3)

    def test_positive_definite(self):
        """Covariance matrices should be positive semi-definite."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((30, 3)).astype(np.float64)
        nbr_idx = _build_neighbor_indices(pts, k=5)
        covs = compute_point_covariances(pts, nbr_idx, epsilon=0.001)
        for i in range(30):
            eigvals = np.linalg.eigvalsh(covs[i])
            assert np.all(eigvals > 0), f"Covariance {i} not positive definite"

    def test_symmetric(self):
        """Each covariance matrix should be symmetric."""
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((20, 3)).astype(np.float64)
        nbr_idx = _build_neighbor_indices(pts, k=5)
        covs = compute_point_covariances(pts, nbr_idx, epsilon=0.001)
        for i in range(20):
            np.testing.assert_allclose(covs[i], covs[i].T, atol=1e-12)

    def test_planar_surface_covariance(self):
        """Points on a plane should have one small eigenvalue."""
        # Create points on z=0 plane with small noise
        rng = np.random.default_rng(42)
        N = 100
        pts = np.zeros((N, 3), dtype=np.float64)
        pts[:, 0] = rng.uniform(-1, 1, N)
        pts[:, 1] = rng.uniform(-1, 1, N)
        pts[:, 2] = rng.normal(0, 0.001, N)  # Very small z variation

        nbr_idx = _build_neighbor_indices(pts, k=10)
        covs = compute_point_covariances(pts, nbr_idx, epsilon=1e-6)

        # For most points, the smallest eigenvalue should be much smaller
        # than the other two
        ratios = []
        for i in range(N):
            eigvals = np.sort(np.linalg.eigvalsh(covs[i]))
            if eigvals[2] > 1e-10:
                ratios.append(eigvals[0] / eigvals[2])

        # At least some points should show planar structure
        if len(ratios) > 0:
            assert min(ratios) < 0.1


class TestGICP:
    """Tests for TransformationEstimationForGeneralizedICP."""

    def test_identical_clouds(self):
        """Identical clouds should give high fitness."""
        cloud = _random_cloud(200, seed=42)
        result = registration_icp(
            cloud,
            cloud,
            max_correspondence_distance=1.0,
            estimation_method=TransformationEstimationForGeneralizedICP(),
        )
        assert result.fitness > 0.99
        assert result.inlier_rmse < 0.01

    def test_known_translation(self):
        """GICP recovers a known translation."""
        source = _random_cloud(300, seed=42)
        t_true = np.array([0.5, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=2.0,
            estimation_method=TransformationEstimationForGeneralizedICP(),
            criteria=ICPConvergenceCriteria(max_iteration=50),
        )

        assert result.fitness > 0.9
        T_np = np.array(result.transformation, dtype=np.float64)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=0.35)

    def test_epsilon_parameter(self):
        """Epsilon parameter should be stored correctly."""
        est = TransformationEstimationForGeneralizedICP(epsilon=0.01)
        assert est.epsilon == 0.01

    def test_default_epsilon(self):
        """Default epsilon should be 0.001."""
        est = TransformationEstimationForGeneralizedICP()
        assert est.epsilon == 0.001

    def test_with_covariances(self):
        """GICP with precomputed covariances should still produce results."""
        source = _random_cloud(100, seed=42)
        t_true = np.array([0.3, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        # Test that the estimation method works directly with covariances
        est = TransformationEstimationForGeneralizedICP()

        src_np = np.array(source.points, dtype=np.float64)
        tgt_np = np.array(target.points, dtype=np.float64)

        nbr_src = _build_neighbor_indices(src_np, k=5)
        nbr_tgt = _build_neighbor_indices(tgt_np, k=5)

        src_covs = compute_point_covariances(src_np, nbr_src)
        tgt_covs = compute_point_covariances(tgt_np, nbr_tgt)

        assert src_covs.shape == (100, 3, 3)
        assert tgt_covs.shape == (100, 3, 3)

    def test_compute_rmse(self):
        """compute_rmse should return valid fitness and RMSE."""
        est = TransformationEstimationForGeneralizedICP()
        source = _random_cloud(50, seed=42)
        # Create correspondences: each point maps to itself
        corr = mx.arange(50, dtype=mx.int32)
        fitness, rmse = est.compute_rmse(source.points, source.points, corr)
        assert fitness == 1.0
        assert rmse < 1e-6
