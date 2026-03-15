"""Tests for marching cubes mesh extraction."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.pipelines.integration.marching_cubes import marching_cubes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sphere_sdf(resolution: int, center: float, radius: float) -> np.ndarray:
    """Create a sphere signed-distance field on a grid."""
    lin = np.linspace(0, resolution - 1, resolution)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.sqrt((x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2) - radius


def _plane_sdf(resolution: int, axis: int = 0, offset: float = 0.0) -> np.ndarray:
    """Create a plane SDF: linear gradient along one axis."""
    lin = np.linspace(-1.0, 1.0, resolution)
    grids = np.meshgrid(lin, lin, lin, indexing="ij")
    return grids[axis] - offset


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarchingCubesSphere:
    """Tests using a sphere SDF."""

    def test_sphere_produces_mesh(self):
        """A sphere SDF should produce vertices and triangles."""
        R = 20
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert vertices.shape[1] == 3
        assert triangles.shape[1] == 3
        assert vertices.shape[0] > 0
        assert triangles.shape[0] > 0

    def test_sphere_vertex_count_reasonable(self):
        """Vertex count should be reasonable for a sphere."""
        R = 20
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        vertices, triangles = marching_cubes(vol, level=0.0)

        # A sphere on a 20^3 grid with radius 5 should produce at least
        # a few dozen vertices (surface area ~ 4*pi*25 ~ 314 voxel faces)
        # but not more than the total number of possible edge crossings.
        assert vertices.shape[0] >= 20
        assert vertices.shape[0] < R ** 3  # way fewer than total voxels

    def test_sphere_vertices_near_surface(self):
        """All extracted vertices should lie approximately on the sphere."""
        R = 30
        center = R / 2
        radius = R / 4
        vol = _sphere_sdf(R, center=center, radius=radius)
        vertices, _ = marching_cubes(vol, level=0.0)

        # Compute distance of each vertex from the sphere center
        verts_np = np.array(vertices)
        dists = np.sqrt(np.sum((verts_np - center) ** 2, axis=1))
        # All distances should be close to the radius (within ~1 voxel)
        np.testing.assert_allclose(dists, radius, atol=1.5)


class TestMarchingCubesPlane:
    """Tests using a plane SDF."""

    def test_plane_produces_mesh(self):
        """A plane SDF should produce a flat surface."""
        R = 10
        vol = _plane_sdf(R, axis=0, offset=0.0)
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert vertices.shape[0] > 0
        assert triangles.shape[0] > 0

    def test_plane_vertices_on_plane(self):
        """Extracted plane vertices should have near-constant coordinate on the plane axis."""
        R = 10
        vol = _plane_sdf(R, axis=2, offset=0.0)
        spacing = (1.0, 1.0, 1.0)
        vertices, _ = marching_cubes(vol, level=0.0, spacing=spacing)

        verts_np = np.array(vertices)
        # The plane is at z = offset (0.0) in the SDF's coordinate system.
        # In grid coordinates, the zero-crossing is at the midpoint of the z-axis.
        z_vals = verts_np[:, 2]
        # All z values should be approximately the same
        assert np.std(z_vals) < 0.5, f"z std = {np.std(z_vals)}"


class TestMarchingCubesEdgeCases:
    """Edge case tests."""

    def test_empty_volume_all_positive(self):
        """All-positive volume (all outside) should produce no mesh."""
        vol = np.ones((5, 5, 5), dtype=np.float32) * 10.0
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert vertices.shape[0] == 0
        assert triangles.shape[0] == 0

    def test_full_volume_all_negative(self):
        """All-negative volume (all inside) should produce no mesh."""
        vol = np.ones((5, 5, 5), dtype=np.float32) * -10.0
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert vertices.shape[0] == 0
        assert triangles.shape[0] == 0

    def test_single_cube_crossing(self):
        """A minimal volume with one sign change should produce at least 1 triangle."""
        # 2x2x2 volume = 1 cube
        vol = np.ones((2, 2, 2), dtype=np.float32)
        vol[0, 0, 0] = -1.0  # single corner inside
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert triangles.shape[0] >= 1
        assert vertices.shape[0] >= 3  # at least 3 vertices for 1 triangle

    def test_tiny_volume(self):
        """Volume smaller than 2x2x2 should return empty."""
        vol = np.zeros((1, 5, 5), dtype=np.float32)
        vertices, triangles = marching_cubes(vol, level=0.0)
        assert vertices.shape[0] == 0
        assert triangles.shape[0] == 0


class TestMarchingCubesTriangleValidity:
    """Verify triangle index validity."""

    def test_triangle_indices_valid(self):
        """All triangle indices should be < number of vertices."""
        R = 15
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        vertices, triangles = marching_cubes(vol, level=0.0)

        tri_np = np.array(triangles)
        assert np.all(tri_np >= 0), "Negative triangle indices found"
        assert np.all(tri_np < vertices.shape[0]), (
            f"Triangle index {tri_np.max()} >= vertex count {vertices.shape[0]}"
        )

    def test_no_degenerate_triangles(self):
        """No triangle should have repeated vertex indices."""
        R = 15
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        vertices, triangles = marching_cubes(vol, level=0.0)

        tri_np = np.array(triangles)
        for i in range(tri_np.shape[0]):
            assert len(set(tri_np[i])) == 3, (
                f"Degenerate triangle at index {i}: {tri_np[i]}"
            )


class TestMarchingCubesSpacing:
    """Test spacing and origin parameters."""

    def test_spacing_scales_output(self):
        """Spacing parameter should scale vertex positions."""
        R = 10
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)

        verts_1x, _ = marching_cubes(vol, level=0.0, spacing=(1.0, 1.0, 1.0))
        verts_2x, _ = marching_cubes(vol, level=0.0, spacing=(2.0, 2.0, 2.0))

        v1 = np.array(verts_1x)
        v2 = np.array(verts_2x)

        # With 2x spacing, the bounding box should be roughly 2x larger
        extent_1 = v1.max(axis=0) - v1.min(axis=0)
        extent_2 = v2.max(axis=0) - v2.min(axis=0)

        np.testing.assert_allclose(extent_2, extent_1 * 2.0, rtol=0.01)

    def test_origin_offsets_output(self):
        """Origin parameter should offset all vertex positions."""
        R = 10
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        origin = np.array([10.0, 20.0, 30.0])

        verts_no_origin, _ = marching_cubes(vol, level=0.0)
        verts_with_origin, _ = marching_cubes(vol, level=0.0, origin=origin)

        v_no = np.array(verts_no_origin)
        v_yes = np.array(verts_with_origin)

        # The center of the mesh should be offset by the origin
        center_no = (v_no.max(axis=0) + v_no.min(axis=0)) / 2
        center_yes = (v_yes.max(axis=0) + v_yes.min(axis=0)) / 2

        np.testing.assert_allclose(center_yes - center_no, origin, atol=0.1)


class TestMarchingCubesWeightMask:
    """Test weight masking functionality."""

    def test_weight_masking_removes_mesh(self):
        """Zero-weight regions should not produce mesh vertices."""
        R = 20
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        weight = np.zeros((R, R, R), dtype=np.float32)  # all zero weight

        vertices, triangles = marching_cubes(
            vol, level=0.0, weight=weight, weight_threshold=0.0
        )

        assert vertices.shape[0] == 0
        assert triangles.shape[0] == 0

    def test_high_weight_produces_mesh(self):
        """High-weight regions should produce mesh normally."""
        R = 20
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        weight = np.ones((R, R, R), dtype=np.float32) * 10.0  # all high weight

        vertices, triangles = marching_cubes(
            vol, level=0.0, weight=weight, weight_threshold=0.0
        )

        assert vertices.shape[0] > 0
        assert triangles.shape[0] > 0

    def test_partial_weight_mask(self):
        """Only the high-weight half should produce mesh."""
        R = 20
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        weight = np.zeros((R, R, R), dtype=np.float32)
        # Only give weight to the upper half (i >= R/2)
        weight[R // 2 :, :, :] = 10.0

        vertices, triangles = marching_cubes(
            vol, level=0.0, weight=weight, weight_threshold=0.0
        )

        verts_np = np.array(vertices)
        if len(verts_np) > 0:
            # All vertices should be in the upper half (axis 0)
            # With some tolerance for interpolation at the boundary
            assert verts_np[:, 0].min() >= R / 2 - 1.5


class TestMarchingCubesOutputTypes:
    """Verify output types are correct MLX arrays."""

    def test_output_types(self):
        """Output should be MLX arrays with correct dtypes."""
        R = 10
        vol = _sphere_sdf(R, center=R / 2, radius=R / 4)
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert isinstance(vertices, mx.array)
        assert isinstance(triangles, mx.array)
        assert vertices.dtype == mx.float32
        assert triangles.dtype == mx.int32

    def test_empty_output_types(self):
        """Empty outputs should still be correct MLX arrays."""
        vol = np.ones((5, 5, 5)) * 10.0
        vertices, triangles = marching_cubes(vol, level=0.0)

        assert isinstance(vertices, mx.array)
        assert isinstance(triangles, mx.array)
        assert vertices.dtype == mx.float32
        assert triangles.dtype == mx.int32
        assert vertices.shape == (0, 3)
        assert triangles.shape == (0, 3)
