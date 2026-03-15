"""Benchmarks for PointCloud operations: downsample, transform, normals."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.geometry import PointCloud


# ---------------------------------------------------------------------------
# Voxel downsampling
# ---------------------------------------------------------------------------


def test_bench_voxel_downsample(benchmark, point_count):
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((point_count, 3)).astype(np.float32)
    pcd = PointCloud(mx.array(pts))
    voxel_size = 0.1

    result = benchmark(pcd.voxel_down_sample, voxel_size)

    assert len(result.points) > 0
    assert len(result.points) <= point_count


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def test_bench_transform(benchmark, point_count):
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((point_count, 3)).astype(np.float32)
    pcd = PointCloud(mx.array(pts))

    T = mx.array(np.eye(4, dtype=np.float32))

    result = benchmark(pcd.transform, T)

    assert len(result.points) == point_count


# ---------------------------------------------------------------------------
# Normal estimation (skip 100K — too slow for CI)
# ---------------------------------------------------------------------------


@pytest.fixture(params=[1000, 10000])
def normal_point_count(request):
    return request.param


def test_bench_normal_estimation(benchmark, normal_point_count):
    from open3d_mlx.geometry import KDTreeSearchParamKNN

    rng = np.random.default_rng(42)
    pts = rng.standard_normal((normal_point_count, 3)).astype(np.float32)
    pcd = PointCloud(mx.array(pts))

    search_param = KDTreeSearchParamKNN(knn=30)

    def run():
        pcd.estimate_normals(search_param=search_param)

    benchmark(run)

    assert pcd.has_normals()
