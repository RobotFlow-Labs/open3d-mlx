"""Tests for ScalableTSDFVolume."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.camera import PinholeCameraIntrinsic
from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.integration import ScalableTSDFVolume, UniformTSDFVolume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_wall_depth(
    intrinsic: PinholeCameraIntrinsic,
    wall_depth_m: float = 1.5,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """Create a synthetic depth image of a flat wall at *wall_depth_m*."""
    return np.full(
        (intrinsic.height, intrinsic.width),
        wall_depth_m * depth_scale,
        dtype=np.float32,
    )


def _identity_extrinsic() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScalableTSDFCreation:
    def test_default_params(self):
        vol = ScalableTSDFVolume()
        assert vol.voxel_size == 0.006
        assert vol.sdf_trunc == 0.04
        assert vol.block_resolution == 8
        assert vol.active_block_count == 0

    def test_custom_params(self):
        vol = ScalableTSDFVolume(voxel_size=0.01, sdf_trunc=0.05, block_resolution=16)
        assert vol.voxel_size == 0.01
        assert vol.block_size == 0.01 * 16


class TestScalableTSDFIntegration:
    def test_single_frame_allocates_blocks(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = ScalableTSDFVolume(voxel_size=0.02, sdf_trunc=0.1, block_resolution=8)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        assert vol.active_block_count > 0

    def test_memory_less_than_uniform_for_sparse_scene(self):
        """Scalable volume should use fewer voxels than a uniform volume
        covering the same bounding box for a sparse scene."""
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        voxel_size = 0.02

        vol_s = ScalableTSDFVolume(voxel_size=voxel_size, sdf_trunc=0.1, block_resolution=8)
        vol_s.integrate(depth, intrinsic, _identity_extrinsic())

        scalable_voxels = vol_s.active_block_count * (8 ** 3)

        # Equivalent uniform volume covering 4m cube
        uniform_res = int(4.0 / voxel_size)
        uniform_voxels = uniform_res ** 3

        assert scalable_voxels < uniform_voxels

    def test_extract_point_cloud_after_integration(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = ScalableTSDFVolume(voxel_size=0.02, sdf_trunc=0.1, block_resolution=8)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        pc = vol.extract_point_cloud()
        assert isinstance(pc, PointCloud)
        assert len(pc) > 0

    def test_reset_clears_blocks(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = ScalableTSDFVolume(voxel_size=0.02, sdf_trunc=0.1, block_resolution=8)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        assert vol.active_block_count > 0
        vol.reset()
        assert vol.active_block_count == 0

    def test_multiple_frames_accumulate(self):
        intrinsic = PinholeCameraIntrinsic.prime_sense_default()
        depth = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol = ScalableTSDFVolume(voxel_size=0.02, sdf_trunc=0.1, block_resolution=8)
        vol.integrate(depth, intrinsic, _identity_extrinsic())
        count_1 = vol.active_block_count

        # Second frame with shifted camera should allocate more (or same) blocks
        ext2 = np.eye(4, dtype=np.float64)
        ext2[0, 3] = 0.5  # shift camera sideways
        depth2 = _make_flat_wall_depth(intrinsic, wall_depth_m=1.5)
        vol.integrate(depth2, intrinsic, ext2)
        count_2 = vol.active_block_count

        # With a shifted camera, more blocks should be active
        assert count_2 >= count_1

    def test_empty_extraction(self):
        vol = ScalableTSDFVolume()
        pc = vol.extract_point_cloud()
        assert pc.is_empty()
