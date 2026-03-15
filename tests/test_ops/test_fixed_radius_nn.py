"""Tests for FixedRadiusIndex (spatial hashing nearest neighbor)."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.ops.fixed_radius_nn import FixedRadiusIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_points():
    """8 points on a unit cube corners."""
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return mx.array(pts)


@pytest.fixture
def random_points():
    """500 random 3D points in [-5, 5]^3."""
    rng = np.random.default_rng(42)
    return mx.array(rng.uniform(-5, 5, (500, 3)).astype(np.float32))


# ---------------------------------------------------------------------------
# Build Tests
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_build_from_cube(self, grid_points):
        """Should build index without error."""
        idx = FixedRadiusIndex(grid_points, radius=1.5)
        assert idx.num_points == 8
        assert idx.radius == 1.5

    def test_build_random(self, random_points):
        """Should build index from random points."""
        idx = FixedRadiusIndex(random_points, radius=0.5)
        assert idx.num_points == 500

    def test_invalid_radius_raises(self, grid_points):
        """Negative or zero radius should raise."""
        with pytest.raises(ValueError, match="radius must be > 0"):
            FixedRadiusIndex(grid_points, radius=0.0)
        with pytest.raises(ValueError, match="radius must be > 0"):
            FixedRadiusIndex(grid_points, radius=-1.0)

    def test_invalid_shape_raises(self):
        """Non (N,3) shape should raise."""
        with pytest.raises(ValueError, match="shape"):
            FixedRadiusIndex(mx.zeros((10, 2)), radius=1.0)

    def test_empty_points(self):
        """Empty point cloud should build without error."""
        idx = FixedRadiusIndex(mx.zeros((0, 3)), radius=1.0)
        assert idx.num_points == 0
        result_idx, result_dist = idx.search_nearest(mx.array([[0, 0, 0]], dtype=mx.float32))
        assert int(result_idx[0]) == -1


# ---------------------------------------------------------------------------
# search_nearest Tests
# ---------------------------------------------------------------------------

class TestSearchNearest:
    def test_self_search(self, grid_points):
        """Each point should find itself as nearest neighbor."""
        idx = FixedRadiusIndex(grid_points, radius=1.5)
        result_idx, result_dist = idx.search_nearest(grid_points)

        for i in range(8):
            assert int(result_idx[i]) == i, (
                f"Point {i} found {int(result_idx[i])} instead of itself"
            )
            assert float(result_dist[i]) == pytest.approx(0.0, abs=1e-6)

    def test_known_pair(self):
        """Test with two known points."""
        pts = mx.array([[0, 0, 0], [0.5, 0, 0]], dtype=mx.float32)
        query = mx.array([[0.1, 0, 0]], dtype=mx.float32)

        idx = FixedRadiusIndex(pts, radius=1.0)
        result_idx, result_dist = idx.search_nearest(query)

        assert int(result_idx[0]) == 0  # closer to origin
        assert float(result_dist[0]) == pytest.approx(0.01, abs=1e-5)

    def test_no_neighbor_returns_neg1(self, grid_points):
        """Query far from all points returns -1 and inf."""
        idx = FixedRadiusIndex(grid_points, radius=0.1)
        query = mx.array([[100, 100, 100]], dtype=mx.float32)
        result_idx, result_dist = idx.search_nearest(query)

        assert int(result_idx[0]) == -1
        assert np.isinf(float(result_dist[0]))

    def test_respects_radius(self):
        """Should only find neighbors within radius."""
        pts = mx.array(
            [[0, 0, 0], [2, 0, 0], [5, 0, 0]],
            dtype=mx.float32,
        )
        query = mx.array([[0, 0, 0]], dtype=mx.float32)

        # Radius 1.0: should not find point at (2,0,0)
        idx = FixedRadiusIndex(pts, radius=1.0)
        result_idx, _ = idx.search_nearest(query)
        assert int(result_idx[0]) == 0  # only self

        # Radius 2.5: should find point at (2,0,0) or self
        idx2 = FixedRadiusIndex(pts, radius=2.5)
        result_idx2, _ = idx2.search_nearest(query)
        assert int(result_idx2[0]) in (0, 1)  # self or (2,0,0)

    def test_matches_brute_force(self, random_points):
        """Results should match brute-force nearest neighbor."""
        radius = 1.0
        pts_np = np.array(random_points, dtype=np.float32)
        query_pts = random_points[:50]
        query_np = np.array(query_pts, dtype=np.float32)

        idx = FixedRadiusIndex(random_points, radius=radius)
        result_idx, result_dist = idx.search_nearest(query_pts)

        result_idx_np = np.array(result_idx)
        result_dist_np = np.array(result_dist)

        # Brute force
        for i in range(50):
            diffs = pts_np - query_np[i]
            sq_dists = np.sum(diffs ** 2, axis=1)

            # Find true nearest within radius
            mask = sq_dists <= radius ** 2
            if not np.any(mask):
                assert result_idx_np[i] == -1, (
                    f"Query {i}: expected -1 but got {result_idx_np[i]}"
                )
            else:
                true_min_dist = np.min(sq_dists[mask])
                true_min_idx = np.where(mask)[0][np.argmin(sq_dists[mask])]

                assert result_dist_np[i] == pytest.approx(
                    true_min_dist, abs=1e-4
                ), f"Query {i}: distance mismatch"
                assert result_idx_np[i] == true_min_idx, (
                    f"Query {i}: index mismatch "
                    f"(got {result_idx_np[i]}, expected {true_min_idx})"
                )

    def test_many_queries(self, random_points):
        """Should handle querying all 500 points."""
        idx = FixedRadiusIndex(random_points, radius=2.0)
        result_idx, result_dist = idx.search_nearest(random_points)

        # Every point should find itself
        result_idx_np = np.array(result_idx)
        result_dist_np = np.array(result_dist)

        for i in range(500):
            assert result_idx_np[i] == i
            assert result_dist_np[i] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# search (multi-neighbor) Tests
# ---------------------------------------------------------------------------

class TestSearchMulti:
    def test_max_nn_1_matches_search_nearest(self, grid_points):
        """search(max_nn=1) should match search_nearest."""
        idx = FixedRadiusIndex(grid_points, radius=1.5)

        idx1, dist1 = idx.search_nearest(grid_points)
        idx2, dist2 = idx.search(grid_points, max_nn=1)

        np.testing.assert_array_equal(
            np.array(idx1), np.array(idx2).flatten()
        )
        np.testing.assert_allclose(
            np.array(dist1), np.array(dist2).flatten(), atol=1e-6
        )

    def test_max_nn_3(self, grid_points):
        """search with max_nn=3 returns correct shape."""
        idx = FixedRadiusIndex(grid_points, radius=1.5)
        result_idx, result_dist = idx.search(grid_points, max_nn=3)

        assert result_idx.shape == (8, 3)
        assert result_dist.shape == (8, 3)

    def test_padding(self):
        """If fewer than max_nn neighbors, pad with -1/inf."""
        pts = mx.array([[0, 0, 0], [10, 10, 10]], dtype=mx.float32)
        idx = FixedRadiusIndex(pts, radius=1.0)
        result_idx, result_dist = idx.search(pts[:1], max_nn=5)

        idx_np = np.array(result_idx[0])
        dist_np = np.array(result_dist[0])

        # Only 1 neighbor (self) within radius
        assert idx_np[0] == 0
        assert np.all(idx_np[1:] == -1)
        assert np.all(np.isinf(dist_np[1:]))


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_large_radius(self, grid_points):
        """Very large radius should find everything."""
        idx = FixedRadiusIndex(grid_points, radius=100.0)
        result_idx, result_dist = idx.search(grid_points[:1], max_nn=8)
        valid = np.array(result_idx[0])
        valid = valid[valid >= 0]
        assert len(valid) == 8

    def test_tiny_radius(self, grid_points):
        """Very small radius: only self match for exact queries."""
        idx = FixedRadiusIndex(grid_points, radius=0.001)
        result_idx, result_dist = idx.search_nearest(grid_points)

        for i in range(8):
            assert int(result_idx[i]) == i

    def test_negative_coordinates(self):
        """Should handle negative coordinates."""
        pts = mx.array(
            [[-1, -1, -1], [1, 1, 1], [-5, -5, -5]],
            dtype=mx.float32,
        )
        idx = FixedRadiusIndex(pts, radius=5.0)
        result_idx, _ = idx.search_nearest(pts)

        for i in range(3):
            assert int(result_idx[i]) == i

    def test_single_point(self):
        """Single point should find itself."""
        pts = mx.array([[42, -7, 3.14]], dtype=mx.float32)
        idx = FixedRadiusIndex(pts, radius=1.0)
        result_idx, result_dist = idx.search_nearest(pts)

        assert int(result_idx[0]) == 0
        assert float(result_dist[0]) == pytest.approx(0.0, abs=1e-6)

    def test_dtype_output(self, grid_points):
        """Output dtypes should be int32 and float32."""
        idx = FixedRadiusIndex(grid_points, radius=1.5)
        result_idx, result_dist = idx.search_nearest(grid_points)

        assert result_idx.dtype == mx.int32
        assert result_dist.dtype == mx.float32
