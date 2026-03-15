"""Tests for NearestNeighborSearch (scipy cKDTree wrapper)."""

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch

scipy = pytest.importorskip("scipy", reason="scipy required for NearestNeighborSearch")


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
    """1000 random 3D points."""
    rng = np.random.default_rng(12345)
    return mx.array(rng.standard_normal((1000, 3)).astype(np.float32))


# ---------------------------------------------------------------------------
# KNN Search Tests
# ---------------------------------------------------------------------------

class TestKNNSearch:
    def test_k1_finds_self(self, grid_points):
        """Searching source points for k=1 should find themselves."""
        nns = NearestNeighborSearch(grid_points)
        indices, distances = nns.knn_search(grid_points, k=1)

        assert indices.shape == (8, 1)
        assert distances.shape == (8, 1)

        # Each point is its own nearest neighbor with distance 0
        for i in range(8):
            assert int(indices[i, 0]) == i
            assert float(distances[i, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_k5_correct_count(self, grid_points):
        """k=5 search returns exactly 5 neighbors per query."""
        nns = NearestNeighborSearch(grid_points)
        indices, distances = nns.knn_search(grid_points, k=5)

        assert indices.shape == (8, 5)
        assert distances.shape == (8, 5)

    def test_distances_sorted(self, random_points):
        """Returned distances should be non-decreasing per query."""
        nns = NearestNeighborSearch(random_points)
        _, distances = nns.knn_search(random_points[:50], k=10)

        dist_np = np.array(distances)
        for i in range(dist_np.shape[0]):
            assert np.all(np.diff(dist_np[i]) >= -1e-7), (
                f"Distances not sorted for query {i}"
            )

    def test_known_nearest_neighbor(self):
        """Test with a known configuration."""
        # Point at origin, nearest should be (0.1, 0, 0)
        pts = mx.array(
            [[0, 0, 0], [0.1, 0, 0], [1, 0, 0], [2, 0, 0]],
            dtype=mx.float32,
        )
        query = mx.array([[0, 0, 0]], dtype=mx.float32)
        nns = NearestNeighborSearch(pts)
        indices, distances = nns.knn_search(query, k=2)

        assert int(indices[0, 0]) == 0  # self
        assert int(indices[0, 1]) == 1  # (0.1, 0, 0)
        assert float(distances[0, 1]) == pytest.approx(0.01, abs=1e-5)

    def test_k_equals_n(self, grid_points):
        """k=N should work and return all points."""
        nns = NearestNeighborSearch(grid_points)
        indices, distances = nns.knn_search(grid_points[:1], k=8)
        assert indices.shape == (1, 8)
        # All indices should be present
        assert set(np.array(indices[0]).tolist()) == set(range(8))

    def test_k_too_large_raises(self, grid_points):
        """k > N should raise ValueError."""
        nns = NearestNeighborSearch(grid_points)
        with pytest.raises(ValueError, match="exceeds"):
            nns.knn_search(grid_points, k=100)

    def test_k_zero_raises(self, grid_points):
        """k=0 should raise ValueError."""
        nns = NearestNeighborSearch(grid_points)
        with pytest.raises(ValueError, match="k must be >= 1"):
            nns.knn_search(grid_points, k=0)

    def test_matches_scipy_directly(self, random_points):
        """Results should match raw scipy cKDTree.query."""
        from scipy.spatial import cKDTree

        pts_np = np.array(random_points, dtype=np.float64)
        tree = cKDTree(pts_np)
        query_np = pts_np[:20]
        k = 7

        scipy_dists, scipy_idx = tree.query(query_np, k=k)

        nns = NearestNeighborSearch(random_points)
        mlx_idx, mlx_sq_dists = nns.knn_search(random_points[:20], k=k)

        np.testing.assert_array_equal(
            np.array(mlx_idx, dtype=np.int32),
            scipy_idx.astype(np.int32),
        )
        np.testing.assert_allclose(
            np.array(mlx_sq_dists),
            (scipy_dists ** 2).astype(np.float32),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Radius Search Tests
# ---------------------------------------------------------------------------

class TestRadiusSearch:
    def test_finds_expected_neighbors(self, grid_points):
        """Points within radius should be found."""
        nns = NearestNeighborSearch(grid_points)
        # Origin's neighbors within radius 1.01: itself + 3 axis-aligned neighbors
        query = mx.array([[0, 0, 0]], dtype=mx.float32)
        idx_list, dist_list = nns.radius_search(query, radius=1.01)

        assert len(idx_list) == 1
        found = set(np.array(idx_list[0]).tolist())
        # Should find: 0 (self), 1 (1,0,0), 2 (0,1,0), 3 (0,0,1)
        assert {0, 1, 2, 3} == found

    def test_respects_max_nn(self, grid_points):
        """max_nn should limit number of returned neighbors."""
        nns = NearestNeighborSearch(grid_points)
        query = mx.array([[0.5, 0.5, 0.5]], dtype=mx.float32)
        idx_list, _ = nns.radius_search(query, radius=2.0, max_nn=3)

        assert len(np.array(idx_list[0])) <= 3

    def test_empty_result(self, grid_points):
        """Query far from all points returns empty."""
        nns = NearestNeighborSearch(grid_points)
        query = mx.array([[100, 100, 100]], dtype=mx.float32)
        idx_list, dist_list = nns.radius_search(query, radius=0.1)

        assert len(np.array(idx_list[0])) == 0

    def test_radius_zero_raises(self, grid_points):
        """radius <= 0 should raise ValueError."""
        nns = NearestNeighborSearch(grid_points)
        with pytest.raises(ValueError, match="radius must be > 0"):
            nns.radius_search(grid_points, radius=0.0)

    def test_sorted_by_distance(self, random_points):
        """Results should be sorted by distance."""
        nns = NearestNeighborSearch(random_points)
        _, dist_list = nns.radius_search(random_points[:10], radius=1.0)

        for dists in dist_list:
            d = np.array(dists)
            if len(d) > 1:
                assert np.all(np.diff(d) >= -1e-7)


# ---------------------------------------------------------------------------
# Hybrid Search Tests
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_combines_knn_and_radius(self, grid_points):
        """Hybrid search respects both radius and max_nn."""
        nns = NearestNeighborSearch(grid_points)
        query = mx.array([[0, 0, 0]], dtype=mx.float32)
        indices, distances, counts = nns.hybrid_search(
            query, radius=1.01, max_nn=10
        )

        assert indices.shape == (1, 10)
        assert distances.shape == (1, 10)
        assert counts.shape == (1,)

        n = int(counts[0])
        assert n == 4  # self + 3 axis neighbors

    def test_padding(self, grid_points):
        """Unfilled slots should be padded with -1/inf."""
        nns = NearestNeighborSearch(grid_points)
        query = mx.array([[0, 0, 0]], dtype=mx.float32)
        indices, distances, counts = nns.hybrid_search(
            query, radius=1.01, max_nn=10
        )

        idx_np = np.array(indices[0])
        dist_np = np.array(distances[0])
        n = int(counts[0])

        # Valid entries
        assert np.all(idx_np[:n] >= 0)
        assert np.all(np.isfinite(dist_np[:n]))

        # Padded entries
        assert np.all(idx_np[n:] == -1)
        assert np.all(np.isinf(dist_np[n:]))

    def test_max_nn_limits_results(self, grid_points):
        """max_nn should cap neighbors even if more exist in radius."""
        nns = NearestNeighborSearch(grid_points)
        query = mx.array([[0.5, 0.5, 0.5]], dtype=mx.float32)
        _, _, counts = nns.hybrid_search(query, radius=10.0, max_nn=3)

        assert int(counts[0]) <= 3

    def test_no_neighbors_in_radius(self, grid_points):
        """If nothing within radius, count should be 0."""
        nns = NearestNeighborSearch(grid_points)
        query = mx.array([[100, 100, 100]], dtype=mx.float32)
        indices, distances, counts = nns.hybrid_search(
            query, radius=0.1, max_nn=5
        )

        assert int(counts[0]) == 0
        assert np.all(np.array(indices[0]) == -1)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_point(self):
        """Single-point cloud should work."""
        pts = mx.array([[1, 2, 3]], dtype=mx.float32)
        nns = NearestNeighborSearch(pts)
        idx, dist = nns.knn_search(pts, k=1)
        assert int(idx[0, 0]) == 0
        assert float(dist[0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_empty_raises(self):
        """Empty point cloud should raise."""
        with pytest.raises(ValueError, match="empty"):
            NearestNeighborSearch(mx.zeros((0, 3)))

    def test_wrong_shape_raises(self):
        """Non (N,3) shape should raise."""
        with pytest.raises(ValueError, match="shape"):
            NearestNeighborSearch(mx.zeros((10, 2)))

    def test_dtype_preserved(self, grid_points):
        """Output should be int32 indices and float32 distances."""
        nns = NearestNeighborSearch(grid_points)
        idx, dist = nns.knn_search(grid_points, k=3)
        assert idx.dtype == mx.int32
        assert dist.dtype == mx.float32
