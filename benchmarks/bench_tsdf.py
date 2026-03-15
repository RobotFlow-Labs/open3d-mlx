"""Benchmarks for TSDF volume integration and extraction."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.pipelines.integration import UniformTSDFVolume
from open3d_mlx.camera import PinholeCameraIntrinsic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_depth_frame(rng, h=480, w=640):
    """Synthetic depth frame (uint16 millimeters, 0.5-3.0 m range)."""
    return mx.array(rng.integers(500, 3000, size=(h, w), dtype=np.uint16))


# ---------------------------------------------------------------------------
# TSDF Integrate — 64^3
# ---------------------------------------------------------------------------


def test_bench_tsdf_integrate_64(benchmark):
    rng = np.random.default_rng(42)
    volume = UniformTSDFVolume(length=4.0, resolution=64, sdf_trunc=0.04)
    intrinsic = PinholeCameraIntrinsic.prime_sense_default()
    depth = _make_depth_frame(rng)
    extrinsic = mx.eye(4)

    benchmark(volume.integrate, depth, intrinsic, extrinsic)


# ---------------------------------------------------------------------------
# TSDF Integrate — 128^3
# ---------------------------------------------------------------------------


def test_bench_tsdf_integrate_128(benchmark):
    rng = np.random.default_rng(42)
    volume = UniformTSDFVolume(length=4.0, resolution=128, sdf_trunc=0.04)
    intrinsic = PinholeCameraIntrinsic.prime_sense_default()
    depth = _make_depth_frame(rng)
    extrinsic = mx.eye(4)

    benchmark(volume.integrate, depth, intrinsic, extrinsic)


# ---------------------------------------------------------------------------
# TSDF Extract PointCloud — 64^3, 128^3
# ---------------------------------------------------------------------------


@pytest.fixture(params=[64, 128])
def integrated_volume(request):
    """Pre-integrated TSDF volume at given resolution."""
    res = request.param
    rng = np.random.default_rng(42)
    volume = UniformTSDFVolume(length=4.0, resolution=res, sdf_trunc=0.04)
    intrinsic = PinholeCameraIntrinsic.prime_sense_default()
    depth = _make_depth_frame(rng)
    extrinsic = mx.eye(4)
    volume.integrate(depth, intrinsic, extrinsic)
    return volume


def test_bench_tsdf_extract_pointcloud(benchmark, integrated_volume):
    result = benchmark(integrated_volume.extract_point_cloud)
    # May be empty for random depth but should not raise
    assert result is not None
