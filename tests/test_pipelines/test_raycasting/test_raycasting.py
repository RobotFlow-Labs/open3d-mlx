"""Tests for volume raycasting pipeline (PRD-10).

Tests cover ray generation, TSDF sampling, ray marching, and rendering.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.camera.intrinsics import PinholeCameraIntrinsic
from open3d_mlx.pipelines.integration.uniform_tsdf import UniformTSDFVolume
from open3d_mlx.pipelines.raycasting import RaycastingScene, generate_rays
from open3d_mlx.pipelines.raycasting.raycasting_scene import (
    _ray_aabb_intersect,
    _sample_tsdf_trilinear,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def simple_intrinsic():
    """Small 32x24 camera with known intrinsics."""
    return PinholeCameraIntrinsic(32, 24, 32.0, 32.0, 15.5, 11.5)


@pytest.fixture
def identity_extrinsic():
    """Identity 4x4 (camera at origin, looking along +Z)."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def flat_wall_volume():
    """32^3 TSDF volume with a flat wall perpendicular to Z at z=1.0.

    Volume spans [0, 2] in each axis, voxel_size = 2/32 = 0.0625.
    The wall is at z = 1.0 (voxel index 16 along z).
    TSDF: positive before wall (z < 1.0), negative after (z > 1.0).
    """
    R = 32
    length = 2.0
    voxel_size = length / R
    sdf_trunc = 0.2

    vol = UniformTSDFVolume(
        length=length, resolution=R, sdf_trunc=sdf_trunc, origin=np.array([0.0, 0.0, 0.0])
    )

    # Manually set TSDF: signed distance to plane z = 1.0
    coords = np.arange(R, dtype=np.float32)
    gi, gj, gk = np.meshgrid(coords, coords, coords, indexing="ij")
    z_centers = (gk + 0.5) * voxel_size  # z coordinate of each voxel center

    wall_z = 1.0
    sdf = wall_z - z_centers  # positive before wall, negative after
    tsdf = np.clip(sdf / sdf_trunc, -1.0, 1.0).astype(np.float32)

    vol._tsdf = tsdf
    vol._weight = np.ones((R, R, R), dtype=np.float32)

    return vol


@pytest.fixture
def sphere_volume():
    """32^3 TSDF volume with a sphere of radius 0.5 at center (1,1,1).

    Volume spans [0, 2] in each axis.
    """
    R = 32
    length = 2.0
    voxel_size = length / R
    sdf_trunc = 0.2

    vol = UniformTSDFVolume(
        length=length, resolution=R, sdf_trunc=sdf_trunc, origin=np.array([0.0, 0.0, 0.0])
    )

    coords = np.arange(R, dtype=np.float32)
    gi, gj, gk = np.meshgrid(coords, coords, coords, indexing="ij")
    x_c = (gi + 0.5) * voxel_size
    y_c = (gj + 0.5) * voxel_size
    z_c = (gk + 0.5) * voxel_size

    center = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    radius = 0.5
    dist = np.sqrt((x_c - center[0]) ** 2 + (y_c - center[1]) ** 2 + (z_c - center[2]) ** 2)
    sdf = dist - radius  # positive outside, negative inside
    tsdf = np.clip(sdf / sdf_trunc, -1.0, 1.0).astype(np.float32)

    vol._tsdf = tsdf
    vol._weight = np.ones((R, R, R), dtype=np.float32)

    return vol


@pytest.fixture
def empty_volume():
    """32^3 TSDF volume with all voxels at +1.0 (empty)."""
    R = 32
    vol = UniformTSDFVolume(
        length=2.0, resolution=R, sdf_trunc=0.2, origin=np.array([0.0, 0.0, 0.0])
    )
    # Default init is ones for tsdf, zeros for weight -- that's empty.
    return vol


# ------------------------------------------------------------------
# Ray generation tests
# ------------------------------------------------------------------


class TestGenerateRays:
    def test_shape(self, simple_intrinsic, identity_extrinsic):
        rays = generate_rays(simple_intrinsic, identity_extrinsic)
        assert rays.shape == (24, 32, 6), f"Expected (24, 32, 6), got {rays.shape}"

    def test_custom_size(self, simple_intrinsic, identity_extrinsic):
        rays = generate_rays(simple_intrinsic, identity_extrinsic, width=16, height=8)
        assert rays.shape == (8, 16, 6)

    def test_center_pixel_direction_identity(self, simple_intrinsic, identity_extrinsic):
        """Center pixel with identity extrinsic should point along +Z."""
        rays = generate_rays(simple_intrinsic, identity_extrinsic)
        rays_np = np.array(rays)

        # Center pixel: cx=15.5, cy=11.5 -> pixel (15, 11) or (16, 12) are close
        # At exactly (cx, cy) the direction should be (0, 0, 1).
        # Pixel (15, 11): u=15, dx=(15-15.5)/32 = -0.015625
        # Pixel (16, 12): u=16, dx=(16-15.5)/32 = 0.015625
        # Neither is exactly center, but the average should be close to (0,0,1).
        # Use the closest integer pixel.
        cv, cu = 12, 16  # closest to (cx=15.5, cy=11.5)
        d = rays_np[cv, cu, 3:]
        # Direction should be nearly (0, 0, 1)
        assert d[2] > 0.99, f"Expected z-direction ~1.0, got {d[2]}"
        assert abs(d[0]) < 0.05, f"Expected x-direction ~0, got {d[0]}"
        assert abs(d[1]) < 0.05, f"Expected y-direction ~0, got {d[1]}"

    def test_origins_are_camera_position(self, simple_intrinsic):
        """Ray origins should match camera position from extrinsic."""
        cam_pos = np.array([1.0, 2.0, 3.0])
        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = cam_pos

        rays = generate_rays(simple_intrinsic, ext)
        rays_np = np.array(rays)

        # All origins should be cam_pos
        origins = rays_np[:, :, :3]
        np.testing.assert_allclose(origins[0, 0], cam_pos, atol=1e-5)
        np.testing.assert_allclose(origins[10, 15], cam_pos, atol=1e-5)

    def test_directions_are_unit(self, simple_intrinsic, identity_extrinsic):
        """All ray directions should be unit vectors."""
        rays = generate_rays(simple_intrinsic, identity_extrinsic)
        rays_np = np.array(rays)
        dirs = rays_np[:, :, 3:]
        norms = np.linalg.norm(dirs, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ------------------------------------------------------------------
# TSDF sampling tests
# ------------------------------------------------------------------


class TestTSDFSampling:
    def test_exact_voxel_center(self, flat_wall_volume):
        """Sampling at a voxel center should return the exact TSDF value."""
        vol = flat_wall_volume
        R = vol.resolution
        voxel_size = vol.voxel_size

        # Voxel (8, 8, 8) center position
        pos = np.array([[(8 + 0.5) * voxel_size, (8 + 0.5) * voxel_size, (8 + 0.5) * voxel_size]])
        pos = pos + vol.origin

        val = _sample_tsdf_trilinear(
            pos, vol._tsdf, vol._weight, vol.origin, voxel_size, R
        )

        expected = vol._tsdf[8, 8, 8]
        np.testing.assert_allclose(val[0], expected, atol=1e-4)

    def test_interpolation_between_centers(self, flat_wall_volume):
        """Sampling between two voxel centers should interpolate."""
        vol = flat_wall_volume
        R = vol.resolution
        vs = vol.voxel_size

        # Position halfway between voxel (8,8,8) and (9,8,8)
        pos = np.array([[(8.5 + 0.5) * vs, (8 + 0.5) * vs, (8 + 0.5) * vs]])
        pos = pos + vol.origin

        val = _sample_tsdf_trilinear(
            pos, vol._tsdf, vol._weight, vol.origin, vs, R
        )

        # Should be average of tsdf[8,8,8] and tsdf[9,8,8]
        expected = (vol._tsdf[8, 8, 8] + vol._tsdf[9, 8, 8]) / 2.0
        np.testing.assert_allclose(val[0], expected, atol=1e-4)


# ------------------------------------------------------------------
# Ray-AABB intersection tests
# ------------------------------------------------------------------


class TestRayAABB:
    def test_hit(self):
        """Ray pointing at box should hit."""
        origins = np.array([[0.0, 0.5, 0.5]])
        directions = np.array([[1.0, 0.0, 0.0]])
        box_min = np.array([1.0, 0.0, 0.0])
        box_max = np.array([2.0, 1.0, 1.0])

        t_entry, t_exit = _ray_aabb_intersect(origins, directions, box_min, box_max)
        assert t_entry[0] < t_exit[0], "Ray should hit the box"
        np.testing.assert_allclose(t_entry[0], 1.0, atol=1e-6)

    def test_miss(self):
        """Ray going perpendicular and outside the box should miss."""
        origins = np.array([[0.0, 5.0, 0.5]])  # well above box
        directions = np.array([[1.0, 0.0, 0.0]])
        box_min = np.array([1.0, 0.0, 0.0])
        box_max = np.array([2.0, 1.0, 1.0])

        t_entry, t_exit = _ray_aabb_intersect(origins, directions, box_min, box_max)
        assert t_entry[0] > t_exit[0], "Ray should miss the box"


# ------------------------------------------------------------------
# Raycasting tests
# ------------------------------------------------------------------


class TestCastRays:
    def test_flat_wall_uniform_depth(self, flat_wall_volume, simple_intrinsic):
        """Rays hitting a flat wall should produce similar t_hit values."""
        vol = flat_wall_volume

        # Camera at (1.0, 1.0, -0.5), looking along +Z toward the wall at z=1.0
        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = [1.0, 1.0, -0.5]

        scene = RaycastingScene()
        scene.set_volume(vol)

        # Generate a small subset of rays near center
        intr = PinholeCameraIntrinsic(8, 8, 32.0, 32.0, 3.5, 3.5)
        rays = generate_rays(intr, ext)
        flat = rays.reshape(64, 6)

        result = scene.cast_rays(flat, max_steps=200)
        t_hit = np.array(result["t_hit"])

        # Filter hits
        hits = t_hit[np.isfinite(t_hit)]
        assert len(hits) > 0, "Should have some hits on the flat wall"

        # All hits should be approximately the same distance (1.5 from camera to wall)
        if len(hits) > 1:
            std = np.std(hits)
            assert std < 0.2, f"Flat wall depth should be uniform, std={std}"

    def test_empty_volume_no_hits(self, empty_volume):
        """Casting into an empty volume should produce all inf."""
        scene = RaycastingScene()
        scene.set_volume(empty_volume)

        # Simple rays along +Z
        rays = mx.array(
            [[1.0, 1.0, -1.0, 0.0, 0.0, 1.0],
             [0.5, 0.5, -1.0, 0.0, 0.0, 1.0]],
            dtype=mx.float32,
        )

        result = scene.cast_rays(rays, max_steps=100)
        t_hit = np.array(result["t_hit"])

        assert np.all(np.isinf(t_hit)), f"Empty volume should produce inf hits, got {t_hit}"

    def test_sphere_depth_varies(self, sphere_volume):
        """Rays hitting a sphere should produce varying depth."""
        scene = RaycastingScene()
        scene.set_volume(sphere_volume)

        # Camera at (1.0, 1.0, -0.5) looking at sphere center (1,1,1)
        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = [1.0, 1.0, -0.5]

        intr = PinholeCameraIntrinsic(16, 16, 32.0, 32.0, 7.5, 7.5)
        rays = generate_rays(intr, ext)
        flat = rays.reshape(256, 6)

        result = scene.cast_rays(flat, max_steps=200)
        t_hit = np.array(result["t_hit"])

        hits = t_hit[np.isfinite(t_hit)]
        assert len(hits) > 0, "Should hit the sphere"

        # Center ray should hit closer than edge rays
        if len(hits) > 2:
            assert np.max(hits) - np.min(hits) > 0.01, \
                "Sphere depth should vary with angle"


class TestRenderDepth:
    def test_shape(self, flat_wall_volume, simple_intrinsic):
        """render_depth should return (H, W) array."""
        scene = RaycastingScene()
        scene.set_volume(flat_wall_volume)

        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = [1.0, 1.0, -0.5]

        depth = scene.render_depth(simple_intrinsic, ext, max_steps=100)
        assert depth.shape == (simple_intrinsic.height, simple_intrinsic.width), \
            f"Expected ({simple_intrinsic.height}, {simple_intrinsic.width}), got {depth.shape}"


class TestRenderNormal:
    def test_shape(self, flat_wall_volume, simple_intrinsic):
        """render_normal should return (H, W, 3) array."""
        scene = RaycastingScene()
        scene.set_volume(flat_wall_volume)

        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = [1.0, 1.0, -0.5]

        normals = scene.render_normal(simple_intrinsic, ext, max_steps=100)
        assert normals.shape == (simple_intrinsic.height, simple_intrinsic.width, 3), \
            f"Expected ({simple_intrinsic.height}, {simple_intrinsic.width}, 3), got {normals.shape}"

    def test_flat_wall_normals(self, flat_wall_volume):
        """Normals of a flat wall perpendicular to Z should point along -Z."""
        scene = RaycastingScene()
        scene.set_volume(flat_wall_volume)

        ext = np.eye(4, dtype=np.float64)
        ext[:3, 3] = [1.0, 1.0, -0.5]

        intr = PinholeCameraIntrinsic(8, 8, 32.0, 32.0, 3.5, 3.5)
        rays = generate_rays(intr, ext)
        flat = rays.reshape(64, 6)

        result = scene.cast_rays(flat, max_steps=200)
        t_hit = np.array(result["t_hit"])
        normals = np.array(result["normals"])

        # For hit rays, normals should point roughly along -Z (toward camera)
        hit_mask = np.isfinite(t_hit)
        if np.any(hit_mask):
            hit_normals = normals[hit_mask]
            # z-component of normal should be negative (pointing toward camera)
            mean_z = np.mean(hit_normals[:, 2])
            assert mean_z < -0.5, \
                f"Flat wall normals should point along -Z, got mean z={mean_z}"
