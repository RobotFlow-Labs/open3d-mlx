"""Tests for normal estimation (PCA-based)."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.ops.normals import (
    estimate_normals_pca,
    orient_normals_towards_viewpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plane_points():
    """100 points on the XY plane (z=0) with KNN indices."""
    rng = np.random.default_rng(42)
    N = 100
    pts = np.zeros((N, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-5, 5, N)
    pts[:, 1] = rng.uniform(-5, 5, N)
    pts[:, 2] = 0.0  # all on z=0

    # Compute KNN indices (k=10) using brute force
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=10)
    return mx.array(pts), mx.array(indices.astype(np.int32))


@pytest.fixture
def sphere_points():
    """200 points on a unit sphere with KNN indices."""
    rng = np.random.default_rng(123)
    N = 200
    # Random directions
    pts = rng.standard_normal((N, 3)).astype(np.float32)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms  # project to unit sphere

    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    _, indices = tree.query(pts, k=15)
    return mx.array(pts), mx.array(indices.astype(np.int32))


# ---------------------------------------------------------------------------
# Plane Normal Tests
# ---------------------------------------------------------------------------

class TestPlaneNormals:
    def test_plane_normals_z_axis(self, plane_points):
        """Normals of XY plane should be approximately +/- z-axis."""
        points, indices = plane_points
        normals = estimate_normals_pca(points, indices)

        normals_np = np.array(normals)
        assert normals_np.shape == (100, 3)

        # The z-component should dominate (|nz| close to 1)
        z_components = np.abs(normals_np[:, 2])
        assert np.all(z_components > 0.99), (
            f"Expected z-component near 1, got min={z_components.min():.4f}"
        )

        # xy components should be near 0
        xy_magnitude = np.sqrt(normals_np[:, 0] ** 2 + normals_np[:, 1] ** 2)
        assert np.all(xy_magnitude < 0.1)

    def test_tilted_plane(self):
        """Normals of a 45-degree tilted plane."""
        rng = np.random.default_rng(99)
        N = 80
        # Plane: z = x (normal should be [-1/sqrt2, 0, 1/sqrt2])
        x = rng.uniform(-3, 3, N).astype(np.float32)
        y = rng.uniform(-3, 3, N).astype(np.float32)
        z = x.copy()
        pts = np.stack([x, y, z], axis=1)

        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        _, indices = tree.query(pts, k=10)

        normals = estimate_normals_pca(mx.array(pts), mx.array(indices.astype(np.int32)))
        normals_np = np.array(normals)

        # Expected normal direction: [-1, 0, 1] / sqrt(2)
        expected = np.array([-1, 0, 1]) / np.sqrt(2)

        for i in range(N):
            # Normal could be +/- expected direction
            dot = abs(np.dot(normals_np[i], expected))
            assert dot > 0.95, (
                f"Point {i}: dot with expected = {dot:.4f}"
            )


# ---------------------------------------------------------------------------
# Sphere Normal Tests
# ---------------------------------------------------------------------------

class TestSphereNormals:
    def test_sphere_normals_radial(self, sphere_points):
        """Normals of a sphere should be approximately radial."""
        points, indices = sphere_points
        normals = estimate_normals_pca(points, indices)

        points_np = np.array(points)
        normals_np = np.array(normals)

        # For unit sphere, the radial direction is the point position itself
        # The normal should be parallel (or anti-parallel) to the point
        for i in range(len(points_np)):
            dot = abs(np.dot(normals_np[i], points_np[i]))
            # Some tolerance since the local PCA on a curved surface
            # is an approximation
            assert dot > 0.85, (
                f"Point {i}: |dot(normal, radial)| = {dot:.4f}"
            )


# ---------------------------------------------------------------------------
# Unit Length Tests
# ---------------------------------------------------------------------------

class TestUnitLength:
    def test_normals_are_unit_length(self, plane_points):
        """All normals should have unit length."""
        points, indices = plane_points
        normals = estimate_normals_pca(points, indices)
        norms = np.linalg.norm(np.array(normals), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_sphere_normals_unit_length(self, sphere_points):
        """Sphere normals should have unit length."""
        points, indices = sphere_points
        normals = estimate_normals_pca(points, indices)
        norms = np.linalg.norm(np.array(normals), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Orient Normals Tests
# ---------------------------------------------------------------------------

class TestOrientNormals:
    def test_orient_towards_origin(self, sphere_points):
        """Orienting sphere normals towards origin should make them
        point inward (anti-radial)."""
        points, indices = sphere_points
        normals = estimate_normals_pca(points, indices)

        # Orient towards origin
        oriented = orient_normals_towards_viewpoint(
            points, normals, viewpoint=mx.zeros(3)
        )
        oriented_np = np.array(oriented)
        points_np = np.array(points)

        # dot(normal, point) should be <= 0 (inward) for most points
        dots = np.sum(oriented_np * points_np, axis=1)
        inward_frac = np.mean(dots <= 0.1)
        assert inward_frac > 0.8, (
            f"Expected most normals inward, got {inward_frac:.2%}"
        )

    def test_orient_towards_far_viewpoint(self):
        """Normals oriented towards a far viewpoint should point that way."""
        pts = mx.array([[0, 0, 0], [1, 0, 0]], dtype=mx.float32)
        normals = mx.array([[0, 0, 1], [0, 0, -1]], dtype=mx.float32)
        viewpoint = mx.array([0, 0, 100], dtype=mx.float32)

        oriented = orient_normals_towards_viewpoint(pts, normals, viewpoint)
        oriented_np = np.array(oriented)

        # Both should point upward (positive z)
        assert oriented_np[0, 2] > 0
        assert oriented_np[1, 2] > 0

    def test_default_viewpoint_is_origin(self):
        """Default viewpoint should be [0,0,0]."""
        pts = mx.array([[1, 0, 0]], dtype=mx.float32)
        normals = mx.array([[1, 0, 0]], dtype=mx.float32)  # pointing away

        oriented = orient_normals_towards_viewpoint(pts, normals)
        # Should flip to point towards origin (negative x)
        assert float(oriented[0, 0]) < 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_few_neighbors(self):
        """Points with < 3 valid neighbors get default normal."""
        pts = mx.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            dtype=mx.float32,
        )
        # Indices with heavy -1 padding (only 2 valid per point)
        indices = mx.array(
            [[0, 1, -1, -1], [0, 1, -1, -1], [1, 2, -1, -1]],
            dtype=mx.int32,
        )
        normals = estimate_normals_pca(pts, indices)
        normals_np = np.array(normals)

        # Should still produce unit normals (possibly default z-axis)
        norms = np.linalg.norm(normals_np, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_collinear_points(self):
        """Collinear points have degenerate normal (should not crash)."""
        pts = mx.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
            dtype=mx.float32,
        )
        indices = mx.array(
            [[0, 1, 2, 3, 4]] * 5,
            dtype=mx.int32,
        )
        normals = estimate_normals_pca(pts, indices)
        normals_np = np.array(normals)

        # Should produce unit normals (even if direction is arbitrary)
        norms = np.linalg.norm(normals_np, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_shape_mismatch_raises(self):
        """Mismatched points and indices should raise."""
        pts = mx.array([[0, 0, 0]], dtype=mx.float32)
        indices = mx.array([[0, 0], [0, 0]], dtype=mx.int32)

        with pytest.raises(ValueError, match="Mismatch"):
            estimate_normals_pca(pts, indices)

    def test_wrong_points_shape_raises(self):
        """Non (N,3) points should raise."""
        with pytest.raises(ValueError, match="Expected points"):
            estimate_normals_pca(
                mx.zeros((5, 2)),
                mx.zeros((5, 3), dtype=mx.int32),
            )
