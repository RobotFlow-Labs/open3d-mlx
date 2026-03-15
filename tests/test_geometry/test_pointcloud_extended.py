"""Extended tests for PointCloud: crop, farthest_point_down_sample, search_param API."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import (
    AxisAlignedBoundingBox,
    KDTreeSearchParamHybrid,
    KDTreeSearchParamKNN,
    KDTreeSearchParamRadius,
    PointCloud,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_cloud(n=10) -> PointCloud:
    """Create a grid of points in [0, 1]^3."""
    coords = np.linspace(0, 1, n, dtype=np.float32)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)
    return PointCloud(mx.array(pts))


# ---------------------------------------------------------------------------
# crop()
# ---------------------------------------------------------------------------


class TestCrop:
    def test_crop_filters_points(self):
        pcd = _make_grid_cloud(10)
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.2, 0.2, 0.2]),
            max_bound=mx.array([0.8, 0.8, 0.8]),
        )
        cropped = pcd.crop(bbox)
        assert len(cropped) > 0
        assert len(cropped) < len(pcd)

        # All cropped points should be inside the box
        pts = np.asarray(cropped.points)
        assert np.all(pts >= 0.2 - 1e-6)
        assert np.all(pts <= 0.8 + 1e-6)

    def test_crop_preserves_normals(self):
        pcd = _make_grid_cloud(5)
        n = len(pcd)
        pcd.normals = mx.broadcast_to(mx.array([[0, 0, 1.0]]), (n, 3))
        # Force contiguous
        pcd.normals = mx.array(np.asarray(pcd.normals, dtype=np.float32))

        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([0.5, 0.5, 0.5]),
        )
        cropped = pcd.crop(bbox)
        assert cropped.has_normals()
        assert len(cropped.normals) == len(cropped)

    def test_crop_empty_cloud(self):
        pcd = PointCloud()
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([1.0, 1.0, 1.0]),
        )
        cropped = pcd.crop(bbox)
        assert cropped.is_empty()

    def test_crop_nothing_inside(self):
        pcd = _make_grid_cloud(5)
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([10.0, 10.0, 10.0]),
            max_bound=mx.array([20.0, 20.0, 20.0]),
        )
        cropped = pcd.crop(bbox)
        assert cropped.is_empty()


# ---------------------------------------------------------------------------
# farthest_point_down_sample()
# ---------------------------------------------------------------------------


class TestFarthestPointDownSample:
    def test_correct_count(self):
        pcd = _make_grid_cloud(5)  # 125 points
        sampled = pcd.farthest_point_down_sample(10)
        assert len(sampled) == 10

    def test_spread(self):
        """FPS should produce well-spread points."""
        pcd = _make_grid_cloud(10)  # 1000 points
        sampled = pcd.farthest_point_down_sample(8)
        pts = np.asarray(sampled.points)

        # Check spread: min pairwise distance should be reasonable
        from scipy.spatial.distance import cdist
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        # With 8 points in [0,1]^3, min spacing should be > 0.3
        assert min_dist > 0.3

    def test_empty_cloud(self):
        pcd = PointCloud()
        sampled = pcd.farthest_point_down_sample(10)
        assert sampled.is_empty()

    def test_num_samples_exceeds_points(self):
        pcd = PointCloud(mx.array([[0, 0, 0], [1, 1, 1]], dtype=mx.float32))
        sampled = pcd.farthest_point_down_sample(10)
        assert len(sampled) == 2


# ---------------------------------------------------------------------------
# estimate_normals with search_param=
# ---------------------------------------------------------------------------


class TestEstimateNormalsSearchParam:
    def test_with_hybrid_param(self):
        pcd = _make_grid_cloud(5)
        param = KDTreeSearchParamHybrid(radius=0.5, max_nn=15)
        pcd.estimate_normals(search_param=param)
        assert pcd.has_normals()
        assert pcd.normals.shape == pcd.points.shape

    def test_with_knn_param(self):
        pcd = _make_grid_cloud(5)
        param = KDTreeSearchParamKNN(knn=10)
        pcd.estimate_normals(search_param=param)
        assert pcd.has_normals()

    def test_with_radius_param(self):
        pcd = _make_grid_cloud(5)
        param = KDTreeSearchParamRadius(radius=0.5)
        pcd.estimate_normals(search_param=param)
        assert pcd.has_normals()

    def test_backward_compat_positional(self):
        """Original positional API should still work."""
        pcd = _make_grid_cloud(5)
        pcd.estimate_normals(max_nn=10)
        assert pcd.has_normals()
