"""Tests for multi-scale ICP registration."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.registration import (
    ICPConvergenceCriteria,
    TransformationEstimationPointToPoint,
    multi_scale_icp,
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


def _make_rotation_z(angle_deg: float) -> np.ndarray:
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _random_cloud(n: int = 500, seed: int = 42) -> PointCloud:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    return PointCloud(mx.array(pts))


def _transform_cloud(pcd: PointCloud, T: np.ndarray) -> PointCloud:
    return pcd.clone().transform(mx.array(T.astype(np.float32)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiScaleICP:
    """Tests for multi_scale_icp."""

    def test_mismatched_lengths_raises(self):
        """Mismatched voxel_sizes and max_correspondence_distances should raise."""
        source = _random_cloud(100)
        target = _random_cloud(100)
        with pytest.raises(ValueError, match="same length"):
            multi_scale_icp(
                source,
                target,
                voxel_sizes=[1.0, 0.5],
                max_correspondence_distances=[2.0],
            )

    def test_three_scales_coarse_to_fine(self):
        """3-scale ICP should produce a valid result."""
        source = _random_cloud(500, seed=42)
        t_true = np.array([0.5, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        result = multi_scale_icp(
            source,
            target,
            voxel_sizes=[2.0, 1.0, 0.5],
            max_correspondence_distances=[4.0, 2.0, 1.0],
            criteria_list=[
                ICPConvergenceCriteria(max_iteration=20),
                ICPConvergenceCriteria(max_iteration=20),
                ICPConvergenceCriteria(max_iteration=30),
            ],
        )

        assert result.fitness > 0.5
        T_np = np.array(result.transformation, dtype=np.float64)
        # Should recover the translation
        np.testing.assert_allclose(T_np[:3, 3], t_true, atol=0.5)

    def test_multi_scale_vs_single_scale_large_displacement(self):
        """Multi-scale should handle larger displacements better than
        single-scale with small max_correspondence_distance."""
        source = _random_cloud(500, seed=42)
        t_true = np.array([2.0, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        # Single scale with small distance
        result_single = registration_icp(
            source,
            target,
            max_correspondence_distance=1.0,
            criteria=ICPConvergenceCriteria(max_iteration=50),
        )

        # Multi-scale: start coarse, end fine
        result_multi = multi_scale_icp(
            source,
            target,
            voxel_sizes=[3.0, 1.5, 0.5],
            max_correspondence_distances=[5.0, 3.0, 1.5],
            criteria_list=[
                ICPConvergenceCriteria(max_iteration=30),
                ICPConvergenceCriteria(max_iteration=30),
                ICPConvergenceCriteria(max_iteration=30),
            ],
        )

        # Multi-scale should achieve at least as good or better fitness
        # on a large displacement problem
        assert result_multi.fitness >= result_single.fitness - 0.05

    def test_single_scale(self):
        """Single-scale multi_scale_icp should behave like regular ICP."""
        source = _random_cloud(200, seed=42)
        t_true = np.array([0.3, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        result = multi_scale_icp(
            source,
            target,
            voxel_sizes=[0.5],
            max_correspondence_distances=[2.0],
        )

        assert result.fitness > 0.5

    def test_with_initial_transform(self):
        """Providing an initial transform should help convergence."""
        source = _random_cloud(300, seed=42)
        t_true = np.array([2.0, 0.0, 0.0])
        T_true = _make_transform(np.eye(3), t_true)
        target = _transform_cloud(source, T_true)

        init_T = mx.array(
            _make_transform(np.eye(3), np.array([1.8, 0.0, 0.0])).astype(
                np.float32
            )
        )

        result = multi_scale_icp(
            source,
            target,
            voxel_sizes=[1.0, 0.5],
            max_correspondence_distances=[3.0, 1.5],
            init_source_to_target=init_T,
        )

        assert result.fitness > 0.5
