"""Shared fixtures for Open3D-MLX benchmarks."""

import numpy as np
import mlx.core as mx
import pytest


@pytest.fixture(params=[1000, 10000, 100000])
def point_count(request):
    """Parametrized point count: 1K, 10K, 100K."""
    return request.param


@pytest.fixture
def random_points(point_count):
    """Generate random (N, 3) float32 MLX points with a fixed seed."""
    rng = np.random.default_rng(42)
    return mx.array(rng.standard_normal((point_count, 3)).astype(np.float32))


@pytest.fixture
def random_pointcloud(random_points):
    """PointCloud from random_points."""
    from open3d_mlx.geometry import PointCloud

    return PointCloud(random_points)


@pytest.fixture
def rng():
    """Seeded numpy random generator."""
    return np.random.default_rng(42)
