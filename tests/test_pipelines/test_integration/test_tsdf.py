"""Tests for UniformTSDFVolume."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.camera import PinholeCameraIntrinsic
from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.integration import TSDFVolumeColorType, UniformTSDFVolume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_wall_depth(
    intrinsic: PinholeCameraIntrinsic,
    wall_depth_m: float = 1.5,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """Create a synthetic depth image of a flat wall at *wall_depth_m*."""
    depth = np.full(
        (intrinsic.height, intrinsic.width),
        wall_depth_m * depth_scale,
        dtype=np.float32,
    )
    return depth


def _identity_extrinsic() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# Volume creation
# ---------------------------------------------------------------------------


class TestVolumeCreation:
    def test_default_shape(self):
        vol = UniformTSDFVolume(length=4.0, resolution=64)
        assert vol.tsdf.shape == (64, 64, 64)
        assert vol.weight.shape == (64, 64, 64)

    def test_custom_resolution(self):
        vol = UniformTSDFVolume(length=2.0, resolution=32)
        assert vol.tsdf.shape == (32, 32, 32)

    def test_voxel_size(self):
        vol = UniformTSDFVolume(length=4.0, resolution=128)
        assert abs(vol.voxel_size - 4.0 / 128) < 1e-8

    def test_initial_tsdf_is_one(self):
        vol = UniformTSDFVolume(length=2.0, resolution=16)
        np.testing.assert_allclose(np.array(vol.tsdf), 1.0)

    def test_initial_weight_is_zero(self):
        vol = UniformTSDFVolume(length=2.0, resolution=16)
        np.testing.assert_allclose(np.array(vol.weight), 0.0)


class TestReset:
    def test_reset_clears_data(self):
        vol = UniformTSDFVolume(length=2.0, resolution=16)
        # Mutate internals
        vol._tsdf[:] = 0.5
        vol._weight[:] = 10.0
        vol.reset()
        np.testing.assert_allclose(np.array(vol.tsdf), 1.0)
        np.testing.assert_allclose(np.array(vol.weight), 0.0)

    def test_reset_with_color(self):
        vol = UniformTSDFVolume(length=2.0, resolution=16, color=True)
        vol._color[:] = 128.0
        vol.reset()
        np.testing.assert_allclose(vol._color, 0.0)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_single_frame_changes_tsdf(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = UniformTSDFVolume(length=3.0, resolution=32, sdf_trunc=0.1)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        # Some weights must have been updated
        assert np.max(np.array(vol.weight)) > 0

    def test_multiple_frames_accumulate_weights(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = UniformTSDFVolume(length=3.0, resolution=32, sdf_trunc=0.1)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        w1 = np.max(np.array(vol.weight))
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        w2 = np.max(np.array(vol.weight))
        assert w2 > w1

    def test_depth_max_respected(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        # Wall at 5 m, but depth_max=3 m → should not integrate
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=5.0)
        vol = UniformTSDFVolume(length=6.0, resolution=32, sdf_trunc=0.2)
        vol.integrate(depth, intrinsic, _identity_extrinsic(), depth_max=3.0)
        np.testing.assert_allclose(np.array(vol.weight), 0.0)

    def test_sdf_trunc_respected(self):
        """Voxels far from the surface should remain untouched."""
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = UniformTSDFVolume(length=3.0, resolution=32, sdf_trunc=0.05)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        # Most voxels should still have weight 0 (only thin band updated)
        w = np.array(vol.weight)
        fraction_updated = np.count_nonzero(w) / w.size
        assert fraction_updated < 0.5  # majority untouched

    def test_weight_capped_at_255(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = UniformTSDFVolume(length=3.0, resolution=16, sdf_trunc=0.2)
        # Integrate many frames to push weights high
        for _ in range(300):
            vol.integrate(depth, intrinsic, _identity_extrinsic())
        assert np.max(np.array(vol.weight)) <= 255.0


# ---------------------------------------------------------------------------
# Point cloud extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_extract_nonempty_after_integration(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = UniformTSDFVolume(length=3.0, resolution=32, sdf_trunc=0.1)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        pc = vol.extract_point_cloud()
        assert isinstance(pc, PointCloud)
        assert pc.points is not None
        assert len(pc.points) > 0

    def test_flat_wall_points_near_expected_depth(self):
        """After integrating a flat wall, extracted points should cluster
        near the wall depth along the Z axis."""
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        wall_z = 1.5
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=wall_z)
        vol = UniformTSDFVolume(length=3.0, resolution=64, sdf_trunc=0.1)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        pc = vol.extract_point_cloud()
        pts = np.array(pc.points)
        # Z values of extracted points should be close to wall_z
        z_vals = pts[:, 2]
        assert np.abs(np.median(z_vals) - wall_z) < 0.15  # within a couple voxels

    def test_empty_volume_returns_empty_cloud(self):
        vol = UniformTSDFVolume(length=2.0, resolution=16)
        pc = vol.extract_point_cloud()
        assert pc.points is None or len(pc.points) == 0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class TestMemory:
    def test_64_volume_fits(self):
        vol = UniformTSDFVolume(length=4.0, resolution=64)
        assert vol.tsdf.shape == (64, 64, 64)

    def test_128_volume_integrates(self):
        """128^3 volume should integrate without error."""
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = UniformTSDFVolume(length=3.0, resolution=128, sdf_trunc=0.04)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        assert np.max(np.array(vol.weight)) > 0


# ---------------------------------------------------------------------------
# Color type enum
# ---------------------------------------------------------------------------


class TestColorType:
    def test_enum_members(self):
        assert TSDFVolumeColorType.NoColor.value == 0
        assert TSDFVolumeColorType.RGB8.value == 1
        assert TSDFVolumeColorType.Gray32.value == 2
