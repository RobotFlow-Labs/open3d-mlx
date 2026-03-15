"""Benchmarks for ICP registration pipeline."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.registration import (
    registration_icp,
    find_correspondences,
    TransformationEstimationPointToPlane,
)
from open3d_mlx.ops.fixed_radius_nn import FixedRadiusIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(point_count, rng):
    """Create source/target pair with a known small translation."""
    pts = rng.standard_normal((point_count, 3)).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [0.1, 0.05, -0.02]
    target_pts = (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)

    source = PointCloud(mx.array(pts))
    target = PointCloud(mx.array(target_pts))
    return source, target


# ---------------------------------------------------------------------------
# ICP Point-to-Point
# ---------------------------------------------------------------------------


def test_bench_icp_point_to_point(benchmark, point_count):
    rng = np.random.default_rng(42)
    source, target = _make_pair(point_count, rng)

    def run():
        return registration_icp(
            source, target, max_correspondence_distance=0.5
        )

    result = benchmark(run)
    assert result.fitness > 0.5


# ---------------------------------------------------------------------------
# ICP Point-to-Plane (skip 100K — needs normals, slower)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1000, 10000])
def plane_point_count(request):
    return request.param


def test_bench_icp_point_to_plane(benchmark, plane_point_count):
    from open3d_mlx.geometry import KDTreeSearchParamKNN

    rng = np.random.default_rng(42)
    source, target = _make_pair(plane_point_count, rng)

    # Normals required for point-to-plane
    search_param = KDTreeSearchParamKNN(knn=30)
    source.estimate_normals(search_param=search_param)
    target.estimate_normals(search_param=search_param)

    estimation = TransformationEstimationPointToPlane()

    def run():
        return registration_icp(
            source,
            target,
            max_correspondence_distance=0.5,
            estimation_method=estimation,
        )

    result = benchmark(run)
    assert result.fitness > 0.5


# ---------------------------------------------------------------------------
# Correspondence search
# ---------------------------------------------------------------------------


def test_bench_correspondence_search(benchmark, point_count):
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((point_count, 3)).astype(np.float32)
    query_pts = (pts + 0.05).astype(np.float32)

    target_mx = mx.array(pts)
    query_mx = mx.array(query_pts)
    radius = 0.5

    # Build index once (not benchmarked)
    index = FixedRadiusIndex(target_mx, radius=radius)

    def run():
        return find_correspondences(query_mx, index, max_distance=radius)

    benchmark(run)
