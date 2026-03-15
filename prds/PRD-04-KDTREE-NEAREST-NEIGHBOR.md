# PRD-04: KDTree & Nearest Neighbor Search

## Status: P0 — Foundation
## Priority: P0
## Phase: 1 — Foundation
## Estimated Effort: 2–3 days
## Depends On: PRD-00, PRD-01, PRD-02
## Blocks: PRD-05, PRD-06, PRD-07, PRD-08, PRD-10

---

## 1. Objective

Implement nearest neighbor search with two backends:

1. **CPU path**: `scipy.spatial.KDTree` (or `cKDTree`) — for KNN and radius search. Reliable, battle-tested, works for normal estimation and outlier removal.
2. **GPU path**: MLX fixed-radius spatial hashing — for the ICP inner loop where GPU throughput matters.

This mirrors Open3D's dual approach: NanoFlann (CPU) + FixedRadiusIndex (GPU).

---

## 2. Upstream Reference

| Our File | Upstream File | Notes |
|----------|--------------|-------|
| `ops/nearest_neighbor.py` | `cpp/open3d/core/nns/NanoFlannIndex.h/cpp` | CPU KNN |
| `ops/nearest_neighbor.py` | `cpp/open3d/core/nns/NearestNeighborSearch.h/cpp` | Interface |
| `ops/fixed_radius_nn.py` | `cpp/open3d/core/nns/FixedRadiusIndex.h/cpp` | GPU spatial hash |
| `ops/fixed_radius_nn.py` | `cpp/open3d/core/nns/FixedRadiusSearchImpl.h` (586 lines) | Algorithm |

---

## 3. API Design

### 3.1 CPU Nearest Neighbor (scipy-backed)

```python
class NearestNeighborSearch:
    """CPU-based nearest neighbor search using scipy KDTree.

    Used for: normal estimation, outlier removal, feature computation.
    """

    def __init__(self, points: mx.array):
        """Build KDTree index from points.

        Args:
            points: (N, 3) float32 array.
        """

    def knn_search(self, query: mx.array, k: int) -> tuple[mx.array, mx.array]:
        """K-nearest neighbor search.

        Args:
            query: (M, 3) query points.
            k: Number of neighbors.

        Returns:
            indices: (M, k) int32 — neighbor indices into source points.
            distances: (M, k) float32 — squared distances.
        """

    def radius_search(
        self, query: mx.array, radius: float, max_nn: int = 0
    ) -> tuple[list[mx.array], list[mx.array]]:
        """Fixed-radius search.

        Args:
            query: (M, 3) query points.
            radius: Search radius.
            max_nn: Maximum neighbors per query (0 = unlimited).

        Returns:
            indices: List of M arrays, each with variable-length neighbor indices.
            distances: List of M arrays, each with variable-length squared distances.
        """

    def hybrid_search(
        self, query: mx.array, radius: float, max_nn: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Hybrid KNN + radius: find up to max_nn neighbors within radius.

        Returns:
            indices: (M, max_nn) int32 — padded with -1.
            distances: (M, max_nn) float32 — padded with inf.
            counts: (M,) int32 — actual neighbor count per query.
        """
```

### 3.2 GPU Fixed-Radius Nearest Neighbor (MLX-native)

```python
class FixedRadiusIndex:
    """GPU-accelerated fixed-radius NN using spatial hashing.

    Used in ICP inner loop for correspondence search.
    Much faster than KDTree for large point clouds when radius is known.

    Algorithm:
    1. Compute voxel indices: floor(points / radius)
    2. Hash voxel indices to cell keys
    3. Sort points by cell key
    4. For each query, check 27 neighboring cells
    5. Filter by distance ≤ radius
    """

    def __init__(self, points: mx.array, radius: float):
        """Build spatial hash grid.

        Args:
            points: (N, 3) float32 array.
            radius: Search radius (determines cell size).
        """

    def search(self, query: mx.array, max_nn: int = 1) -> tuple[mx.array, mx.array]:
        """Find nearest neighbor(s) within radius.

        Args:
            query: (M, 3) query points.
            max_nn: Maximum neighbors per query.

        Returns:
            indices: (M, max_nn) int32 — -1 if no neighbor found.
            distances: (M, max_nn) float32 — inf if no neighbor found.
        """

    def search_nearest(self, query: mx.array) -> tuple[mx.array, mx.array]:
        """Find single nearest neighbor within radius (optimized path).

        Returns:
            indices: (M,) int32 — -1 if no neighbor.
            distances: (M,) float32 — inf if no neighbor.
        """
```

---

## 4. Implementation Details

### 4.1 CPU Path (scipy)

```python
from scipy.spatial import cKDTree

class NearestNeighborSearch:
    def __init__(self, points: mx.array):
        self._points_np = np.array(points)
        self._tree = cKDTree(self._points_np)

    def knn_search(self, query, k):
        query_np = np.array(query)
        distances, indices = self._tree.query(query_np, k=k)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        return mx.array(indices.astype(np.int32)), mx.array((distances ** 2).astype(np.float32))
```

### 4.2 GPU Path (MLX Spatial Hash)

```python
class FixedRadiusIndex:
    def __init__(self, points: mx.array, radius: float):
        self._points = points
        self._radius = radius
        self._cell_size = radius  # cell size = search radius

        # Compute cell indices for each point
        cell_idx = mx.floor(points / self._cell_size).astype(mx.int32)  # (N, 3)

        # Hash cell indices
        self._hash_keys = (
            cell_idx[:, 0] * 73856093 +
            cell_idx[:, 1] * 19349663 +
            cell_idx[:, 2] * 83492791
        )  # (N,)

        # Sort points by hash key for cache-coherent access
        self._sort_order = mx.argsort(self._hash_keys)
        self._sorted_keys = self._hash_keys[self._sort_order]
        self._sorted_points = points[self._sort_order]

        # Build hash table: key → (start, count) in sorted arrays
        # This uses numpy for the unique/searchsorted ops
        self._build_hash_table()

    def _build_hash_table(self):
        """Build lookup table from hash keys to sorted array ranges."""
        sorted_keys_np = np.array(self._sorted_keys)
        unique_keys, starts, counts = np.unique(
            sorted_keys_np, return_index=True, return_counts=True
        )
        self._table_keys = unique_keys
        self._table_starts = starts
        self._table_counts = counts

    def search_nearest(self, query: mx.array) -> tuple[mx.array, mx.array]:
        """Find nearest neighbor within radius using 27-cell search."""
        # For each query point:
        # 1. Compute its cell index
        # 2. Enumerate 27 neighboring cells (3x3x3)
        # 3. For each cell, look up points in hash table
        # 4. Compute distances, keep minimum within radius

        # This is the hot path for ICP — optimize with MLX vectorized ops
        query_cells = mx.floor(query / self._cell_size).astype(mx.int32)

        # Generate 27 neighbor offsets
        offsets = mx.array(np.array(np.meshgrid(
            [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]
        )).T.reshape(-1, 3), dtype=mx.int32)  # (27, 3)

        # For each query, check all 27 neighboring cells
        # Vectorize over queries, loop over 27 cells (small constant)
        best_idx = mx.full((query.shape[0],), -1, dtype=mx.int32)
        best_dist = mx.full((query.shape[0],), float('inf'), dtype=mx.float32)

        for offset in range(27):
            neighbor_cell = query_cells + offsets[offset]
            neighbor_hash = (
                neighbor_cell[:, 0] * 73856093 +
                neighbor_cell[:, 1] * 19349663 +
                neighbor_cell[:, 2] * 83492791
            )
            # Look up points in this cell and compute distances
            # (batched implementation using sorted array ranges)
            ...

        return best_idx, best_dist
```

### 4.3 Performance Consideration

The 27-cell loop is a constant (not scaling with N). The expensive part is distance computation within each cell. For the ICP use case (finding single nearest neighbor), this is typically 10-50 distance computations per query point.

For Phase 1, a Python loop over 27 cells with MLX-vectorized distance computation within each cell is sufficient. If benchmarks show this is slow, Phase 2 can use a Metal kernel.

---

## 5. Integration with PointCloud (from PRD-02)

After PRD-04 is complete, we backfill PRD-02's deferred methods:

```python
# In geometry/pointcloud.py

def estimate_normals(self, max_nn=30, radius=None):
    from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch
    from open3d_mlx.ops.normals import estimate_normals_pca

    nns = NearestNeighborSearch(self.points)
    if radius is not None:
        indices, _ = nns.hybrid_search(self.points, radius=radius, max_nn=max_nn)
    else:
        indices, _ = nns.knn_search(self.points, k=max_nn)

    self.normals = estimate_normals_pca(self.points, indices)

def remove_statistical_outliers(self, nb_neighbors=20, std_ratio=2.0):
    from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch

    nns = NearestNeighborSearch(self.points)
    _, distances = nns.knn_search(self.points, k=nb_neighbors)
    mean_dists = mx.mean(mx.sqrt(distances), axis=1)
    global_mean = mx.mean(mean_dists)
    global_std = mx.sqrt(mx.mean((mean_dists - global_mean) ** 2))
    mask = mean_dists < (global_mean + std_ratio * global_std)
    return self.select_by_mask(mask), mask
```

---

## 6. Module: `ops/normals.py`

```python
def estimate_normals_pca(
    points: mx.array,
    neighbor_indices: mx.array,
) -> mx.array:
    """Estimate normals using PCA on local neighborhoods.

    Args:
        points: (N, 3) point positions.
        neighbor_indices: (N, K) indices of K neighbors per point.

    Returns:
        normals: (N, 3) unit normals.

    Algorithm:
        For each point:
        1. Gather K neighbors
        2. Center neighborhood (subtract mean)
        3. Compute covariance: C = (1/K) * P^T @ P
        4. Smallest eigenvector of C = normal
        We batch this using MLX matmul + SVD.
    """
    N, K = neighbor_indices.shape

    # Gather neighbor points: (N, K, 3)
    neighbors = points[neighbor_indices]

    # Center each neighborhood
    centroids = mx.mean(neighbors, axis=1, keepdims=True)  # (N, 1, 3)
    centered = neighbors - centroids  # (N, K, 3)

    # Covariance matrices: (N, 3, 3)
    # C_i = centered_i^T @ centered_i / K
    cov = mx.matmul(
        mx.transpose(centered, axes=(0, 2, 1)),  # (N, 3, K)
        centered                                    # (N, K, 3)
    ) / K  # (N, 3, 3)

    # SVD: smallest singular vector = normal
    U, S, Vt = mx.linalg.svd(cov)  # U: (N, 3, 3), S: (N, 3)
    normals = Vt[:, -1, :]  # (N, 3) — last row of Vt = smallest eigenvector

    return normals
```

---

## 7. Tests

```
# CPU NN tests
test_knn_search_k1_finds_self
test_knn_search_k5_correct_count
test_knn_search_distances_sorted
test_radius_search_finds_expected_neighbors
test_radius_search_respects_max_nn
test_hybrid_search_combines_knn_and_radius
test_hybrid_search_padding

# GPU FixedRadius tests
test_fixed_radius_build_index
test_fixed_radius_search_nearest_self
test_fixed_radius_search_nearest_known_pair
test_fixed_radius_no_neighbor_returns_neg1
test_fixed_radius_respects_radius

# Normal estimation
test_normals_plane_points
test_normals_unit_length
test_normals_sphere_points_radial

# Cross-validation
test_knn_matches_scipy_directly
test_fixed_radius_matches_brute_force
test_normals_match_open3d  # if open3d installed
```

---

## 8. Acceptance Criteria

- [ ] KNN search returns correct k neighbors for simple known cases
- [ ] Radius search finds all points within radius
- [ ] Hybrid search returns at most max_nn within radius
- [ ] FixedRadiusIndex finds nearest neighbor matching brute-force result
- [ ] Normal estimation on a plane returns consistent normals
- [ ] Normal estimation produces unit-length vectors
- [ ] PointCloud.estimate_normals() works end-to-end
- [ ] PointCloud.remove_statistical_outliers() removes known outliers
- [ ] scipy is required only for CPU path (optional dependency)
- [ ] All tests pass
