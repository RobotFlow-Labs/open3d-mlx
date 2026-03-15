"""CPU-based nearest neighbor search using scipy cKDTree.

Used for: normal estimation, outlier removal, feature computation.
Wraps scipy.spatial.cKDTree for reliable KNN, radius, and hybrid search.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None  # type: ignore[assignment, misc]


class NearestNeighborSearch:
    """CPU-based nearest neighbor search using scipy KDTree.

    Args:
        points: (N, 3) float32 MLX array of source points.

    Raises:
        ImportError: If scipy is not installed.
        ValueError: If points is not 2-D with 3 columns or is empty.
    """

    def __init__(self, points: mx.array) -> None:
        if cKDTree is None:
            raise ImportError(
                "scipy is required for NearestNeighborSearch. "
                "Install with: pip install scipy"
            )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected points with shape (N, 3), got {points.shape}"
            )
        if points.shape[0] == 0:
            raise ValueError("Cannot build KDTree from empty point set")

        self._points_mx = points
        self._points_np = np.array(points, dtype=np.float64)
        self._tree = cKDTree(self._points_np)
        self._n = points.shape[0]

    @property
    def num_points(self) -> int:
        """Number of indexed source points."""
        return self._n

    def knn_search(
        self, query: mx.array, k: int
    ) -> tuple[mx.array, mx.array]:
        """K-nearest neighbor search.

        Args:
            query: (M, 3) query points.
            k: Number of neighbors (must be >= 1).

        Returns:
            indices: (M, k) int32 neighbor indices into source points.
            squared_distances: (M, k) float32 squared Euclidean distances.

        Raises:
            ValueError: If k < 1 or k > N.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > self._n:
            raise ValueError(
                f"k={k} exceeds number of source points ({self._n})"
            )

        query_np = np.array(query, dtype=np.float64)
        distances, indices = self._tree.query(query_np, k=k)

        # scipy returns 1-D arrays when k=1; ensure 2-D
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        # scipy returns Euclidean distances; convert to squared
        sq_distances = (distances ** 2).astype(np.float32)
        indices = indices.astype(np.int32)

        return mx.array(indices), mx.array(sq_distances)

    def radius_search(
        self, query: mx.array, radius: float, max_nn: int = 0
    ) -> tuple[list[mx.array], list[mx.array]]:
        """Fixed-radius search returning variable-length results.

        Args:
            query: (M, 3) query points.
            radius: Search radius (Euclidean distance).
            max_nn: Maximum neighbors per query (0 = unlimited).

        Returns:
            indices: List of M arrays, each with variable-length int32
                neighbor indices.
            distances: List of M arrays, each with variable-length float32
                squared distances.
        """
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        query_np = np.array(query, dtype=np.float64)
        # query_ball_point returns list of arrays of indices
        results = self._tree.query_ball_point(query_np, r=radius)

        all_indices: list[mx.array] = []
        all_distances: list[mx.array] = []

        for i, idx_list in enumerate(results):
            if len(idx_list) == 0:
                all_indices.append(mx.array(np.array([], dtype=np.int32)))
                all_distances.append(mx.array(np.array([], dtype=np.float32)))
                continue

            idx_arr = np.array(idx_list, dtype=np.int32)
            # Compute squared distances
            diff = self._points_np[idx_arr] - query_np[i]
            sq_dist = np.sum(diff ** 2, axis=1).astype(np.float32)

            # Sort by distance
            sort_order = np.argsort(sq_dist)
            idx_arr = idx_arr[sort_order]
            sq_dist = sq_dist[sort_order]

            # Apply max_nn limit
            if max_nn > 0 and len(idx_arr) > max_nn:
                idx_arr = idx_arr[:max_nn]
                sq_dist = sq_dist[:max_nn]

            all_indices.append(mx.array(idx_arr))
            all_distances.append(mx.array(sq_dist))

        return all_indices, all_distances

    def hybrid_search(
        self, query: mx.array, radius: float, max_nn: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Hybrid KNN + radius: find up to max_nn neighbors within radius.

        Combines radius constraint with a max neighbor count, returning
        fixed-size padded arrays suitable for batched processing.

        Args:
            query: (M, 3) query points.
            radius: Search radius.
            max_nn: Maximum neighbors per query.

        Returns:
            indices: (M, max_nn) int32 padded with -1.
            squared_distances: (M, max_nn) float32 padded with inf.
            counts: (M,) int32 actual neighbor count per query.
        """
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if max_nn < 1:
            raise ValueError(f"max_nn must be >= 1, got {max_nn}")

        idx_lists, dist_lists = self.radius_search(query, radius, max_nn=max_nn)

        M = query.shape[0]
        out_indices = np.full((M, max_nn), -1, dtype=np.int32)
        out_distances = np.full((M, max_nn), np.inf, dtype=np.float32)
        out_counts = np.zeros(M, dtype=np.int32)

        for i in range(M):
            idx_np = np.array(idx_lists[i], dtype=np.int32)
            dist_np = np.array(dist_lists[i], dtype=np.float32)
            n = len(idx_np)
            if n > 0:
                out_indices[i, :n] = idx_np
                out_distances[i, :n] = dist_np
            out_counts[i] = n

        return (
            mx.array(out_indices),
            mx.array(out_distances),
            mx.array(out_counts),
        )
