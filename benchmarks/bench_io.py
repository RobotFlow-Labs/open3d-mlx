"""Benchmarks for point cloud I/O (PLY read/write)."""

import tempfile
import os

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.io import write_point_cloud, read_point_cloud


# ---------------------------------------------------------------------------
# Write PLY
# ---------------------------------------------------------------------------


def test_bench_write_ply(benchmark, point_count, tmp_path):
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((point_count, 3)).astype(np.float32)
    pcd = PointCloud(mx.array(pts))
    filepath = str(tmp_path / "bench.ply")

    def run():
        write_point_cloud(filepath, pcd)

    benchmark(run)
    assert os.path.exists(filepath)


# ---------------------------------------------------------------------------
# Read PLY (write once, then benchmark reads)
# ---------------------------------------------------------------------------


def test_bench_read_ply(benchmark, point_count, tmp_path):
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((point_count, 3)).astype(np.float32)
    pcd = PointCloud(mx.array(pts))
    filepath = str(tmp_path / "bench_read.ply")
    write_point_cloud(filepath, pcd)

    def run():
        return read_point_cloud(filepath)

    result = benchmark(run)
    assert len(result.points) == point_count
