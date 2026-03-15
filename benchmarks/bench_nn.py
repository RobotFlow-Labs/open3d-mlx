"""Benchmarks for nearest neighbor search (KNN and fixed-radius)."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.ops import NearestNeighborSearch, FixedRadiusIndex


# ---------------------------------------------------------------------------
# Shared fixtures — only 10K and 100K for NN benchmarks
# ---------------------------------------------------------------------------


@pytest.fixture(params=[10000, 100000])
def nn_point_count(request):
    return request.param


# ---------------------------------------------------------------------------
# KNN Search (k=30)
# ---------------------------------------------------------------------------


def test_bench_knn_search(benchmark, nn_point_count):
    rng = np.random.default_rng(42)
    pts = mx.array(rng.standard_normal((nn_point_count, 3)).astype(np.float32))

    # Build index (not benchmarked)
    nns = NearestNeighborSearch(pts)

    # Query a subset of 1000 points
    query = pts[:1000]

    def run():
        return nns.knn_search(query, k=30)

    indices, distances = benchmark(run)
    assert indices.shape == (1000, 30)


# ---------------------------------------------------------------------------
# Fixed-Radius Search
# ---------------------------------------------------------------------------


def test_bench_fixed_radius_search(benchmark, nn_point_count):
    rng = np.random.default_rng(42)
    pts = mx.array(rng.standard_normal((nn_point_count, 3)).astype(np.float32))

    radius = 0.3

    # Build index (not benchmarked)
    fri = FixedRadiusIndex(pts, radius=radius)

    # Query a subset of 1000 points
    query = pts[:1000]

    def run():
        return fri.search(query, max_nn=1)

    benchmark(run)
