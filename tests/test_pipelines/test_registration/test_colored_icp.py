"""Tests for Colored ICP registration."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.registration import (
    ICPConvergenceCriteria,
    TransformationEstimationForColoredICP,
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


def _random_cloud_with_colors_and_normals(
    n: int = 200, seed: int = 42
) -> PointCloud:
    """Create a random point cloud with colors and normals."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    normals = rng.standard_normal((n, 3)).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    pcd = PointCloud(mx.array(pts))
    pcd.colors = mx.array(colors)
    pcd.normals = mx.array(normals.astype(np.float32))
    return pcd


def _random_cloud_no_colors(n: int = 200, seed: int = 42) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    normals = rng.standard_normal((n, 3)).astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    pcd = PointCloud(mx.array(pts))
    pcd.normals = mx.array(normals.astype(np.float32))
    return pcd


def _transform_cloud(pcd: PointCloud, T: np.ndarray) -> PointCloud:
    return pcd.clone().transform(mx.array(T.astype(np.float32)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestColoredICP:
    """Tests for TransformationEstimationForColoredICP."""

    def test_requires_colors_source(self):
        """Colored ICP should raise if source lacks colors."""
        source = _random_cloud_no_colors(100)
        target = _random_cloud_with_colors_and_normals(100)
        with pytest.raises(ValueError, match="colors"):
            registration_icp(
                source,
                target,
                max_correspondence_distance=1.0,
                estimation_method=TransformationEstimationForColoredICP(),
            )

    def test_requires_colors_target(self):
        """Colored ICP should raise if target lacks colors."""
        source = _random_cloud_with_colors_and_normals(100)
        target = _random_cloud_no_colors(100)
        with pytest.raises(ValueError, match="colors"):
            registration_icp(
                source,
                target,
                max_correspondence_distance=1.0,
                estimation_method=TransformationEstimationForColoredICP(),
            )

    def test_requires_target_normals(self):
        """Colored ICP should raise if target lacks normals."""
        source = _random_cloud_with_colors_and_normals(100)
        # Target with colors but no normals
        rng = np.random.default_rng(99)
        pts = rng.standard_normal((100, 3)).astype(np.float32)
        target = PointCloud(mx.array(pts))
        target.colors = mx.array(rng.uniform(0, 1, (100, 3)).astype(np.float32))
        with pytest.raises(ValueError, match="normals"):
            registration_icp(
                source,
                target,
                max_correspondence_distance=1.0,
                estimation_method=TransformationEstimationForColoredICP(),
            )

    def test_intensity_computation(self):
        """Check that intensity weights are correct (ITU-R BT.709)."""
        est = TransformationEstimationForColoredICP()
        # Pure red -> I = 0.2126
        # Pure green -> I = 0.7152
        # Pure blue -> I = 0.0722
        weights = np.array([0.2126, 0.7152, 0.0722])
        assert abs(weights.sum() - 1.0) < 1e-4
        # White -> I = 1.0
        white = np.array([1.0, 1.0, 1.0])
        assert abs(white @ weights - 1.0) < 1e-6

    def test_identical_textured_clouds(self):
        """Identical textured clouds should give high fitness."""
        cloud = _random_cloud_with_colors_and_normals(200, seed=42)
        result = registration_icp(
            cloud,
            cloud,
            max_correspondence_distance=1.0,
            estimation_method=TransformationEstimationForColoredICP(),
        )
        assert result.fitness > 0.99
        assert result.inlier_rmse < 0.01

    def test_small_translation_recovery(self):
        """Colored ICP recovers a small known translation."""
        target = _random_cloud_with_colors_and_normals(300, seed=42)
        t_true = np.array([0.3, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)

        source_pts_np = np.array(target.points) - t_true.astype(np.float32)
        source = PointCloud(mx.array(source_pts_np))
        source.colors = target.colors
        source.normals = target.normals

        result = registration_icp(
            source,
            target,
            max_correspondence_distance=2.0,
            estimation_method=TransformationEstimationForColoredICP(),
            criteria=ICPConvergenceCriteria(max_iteration=50),
        )

        assert result.fitness > 0.8
        T_np = np.array(result.transformation, dtype=np.float64)
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=0.5)

    def test_lambda_geometric_range(self):
        """Lambda geometric parameter should be in [0, 1]."""
        est = TransformationEstimationForColoredICP(lambda_geometric=0.5)
        assert est.lambda_geometric == 0.5
        est2 = TransformationEstimationForColoredICP(lambda_geometric=0.968)
        assert est2.lambda_geometric == 0.968
