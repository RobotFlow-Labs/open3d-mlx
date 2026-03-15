"""GPU-friendly fixed-radius nearest neighbor search via spatial hashing.

Used in the ICP inner loop for correspondence search.
Much faster than KDTree for large point clouds when radius is known.

Algorithm:
    1. Compute voxel indices: floor(points / cell_size)
    2. Hash voxel indices to cell keys using prime hashing
    3. Sort points by cell key
    4. For each query, check 27 neighboring cells (3x3x3 cube)
    5. Filter by distance <= radius, keep nearest
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

# Prime numbers for spatial hashing (same as Open3D upstream)
_P1 = np.int64(73856093)
_P2 = np.int64(19349663)
_P3 = np.int64(83492791)
_HASH_MOD = np.int64(2**31 - 1)  # large prime modulus

# Pre-computed 27-cell neighbor offsets (3x3x3 cube)
_OFFSETS_NP = np.array(
    [
        [dx, dy, dz]
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ],
    dtype=np.int64,
)


def _hash_cells(cell_idx: np.ndarray) -> np.ndarray:
    """Compute spatial hash keys from integer cell indices.

    Args:
        cell_idx: (N, 3) int64 cell indices.

    Returns:
        keys: (N,) int32 hash keys (non-negative).
    """
    # Use int64 arithmetic to avoid overflow, then mod to int32 range
    h = (
        cell_idx[:, 0].astype(np.int64) * _P1
        + cell_idx[:, 1].astype(np.int64) * _P2
        + cell_idx[:, 2].astype(np.int64) * _P3
    )
    h = h % _HASH_MOD
    return h.astype(np.int32)


class FixedRadiusIndex:
    """Fixed-radius nearest neighbor search using spatial hashing.

    Args:
        points: (N, 3) float32 MLX array.
        radius: Search radius (determines cell size).

    Raises:
        ValueError: If points is not (N, 3) or radius <= 0.
    """

    def __init__(self, points: mx.array, radius: float) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Expected points with shape (N, 3), got {points.shape}"
            )
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")

        self._radius = float(radius)
        self._cell_size = float(radius)
        self._n = points.shape[0]

        # Convert to numpy for index building
        pts_np = np.array(points, dtype=np.float32)
        self._points_np = pts_np

        if self._n == 0:
            # Empty index
            self._sorted_points = np.empty((0, 3), dtype=np.float32)
            self._sorted_keys = np.empty((0,), dtype=np.int32)
            self._sort_order = np.empty((0,), dtype=np.int32)
            self._table_keys = np.empty((0,), dtype=np.int32)
            self._table_starts = np.empty((0,), dtype=np.int64)
            self._table_counts = np.empty((0,), dtype=np.int64)
            return

        # Compute cell indices: floor(points / cell_size)
        cell_idx = np.floor(pts_np / self._cell_size).astype(np.int64)

        # Hash cell indices
        hash_keys = _hash_cells(cell_idx)

        # Sort points by hash key for cache-coherent access
        sort_order = np.argsort(hash_keys)
        self._sort_order = sort_order.astype(np.int32)
        self._sorted_keys = hash_keys[sort_order]
        self._sorted_points = pts_np[sort_order]

        # Build lookup table: unique key -> (start, count)
        unique_keys, starts, counts = np.unique(
            self._sorted_keys, return_index=True, return_counts=True
        )
        self._table_keys = unique_keys
        self._table_starts = starts.astype(np.int64)
        self._table_counts = counts.astype(np.int64)

    @property
    def num_points(self) -> int:
        """Number of indexed source points."""
        return self._n

    @property
    def radius(self) -> float:
        """Search radius."""
        return self._radius

    def search_nearest(
        self, query: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Find single nearest neighbor within radius using 27-cell search.

        Args:
            query: (M, 3) query points.

        Returns:
            indices: (M,) int32 original point indices. -1 if no neighbor
                found within radius.
            distances: (M,) float32 squared Euclidean distances. inf if no
                neighbor found.
        """
        query_np = np.array(query, dtype=np.float32)
        M = query_np.shape[0]

        best_idx = np.full(M, -1, dtype=np.int32)
        best_dist = np.full(M, np.inf, dtype=np.float32)

        if self._n == 0 or M == 0:
            return mx.array(best_idx), mx.array(best_dist)

        radius_sq = self._radius ** 2

        # Compute query cell indices
        query_cells = np.floor(query_np / self._cell_size).astype(np.int64)

        # For each of the 27 neighbor offsets, batch-process all queries
        for offset in _OFFSETS_NP:
            neighbor_cells = query_cells + offset  # (M, 3)
            neighbor_hash = _hash_cells(neighbor_cells)  # (M,) int32

            # For each unique hash key in this batch, look up the bucket
            unique_query_keys = np.unique(neighbor_hash)

            for qk in unique_query_keys:
                # Find which queries map to this key
                query_mask = neighbor_hash == qk

                # Look up bucket in hash table
                table_pos = np.searchsorted(self._table_keys, qk)
                if table_pos >= len(self._table_keys) or self._table_keys[table_pos] != qk:
                    continue  # No points in this cell

                start = int(self._table_starts[table_pos])
                count = int(self._table_counts[table_pos])
                bucket_points = self._sorted_points[start : start + count]  # (C, 3)
                bucket_original_idx = self._sort_order[start : start + count]

                # Compute distances from masked queries to bucket points
                query_subset_indices = np.where(query_mask)[0]
                query_subset = query_np[query_subset_indices]  # (Q, 3)

                # Pairwise squared distances: (Q, C)
                diff = query_subset[:, None, :] - bucket_points[None, :, :]  # (Q, C, 3)
                sq_dists = np.sum(diff ** 2, axis=2)  # (Q, C)

                # Find minimum per query
                min_idx_in_bucket = np.argmin(sq_dists, axis=1)  # (Q,)
                min_dists = sq_dists[np.arange(len(query_subset_indices)), min_idx_in_bucket]

                # Update best if within radius and closer than current best
                for j, qi in enumerate(query_subset_indices):
                    d = min_dists[j]
                    if d <= radius_sq and d < best_dist[qi]:
                        best_dist[qi] = d
                        best_idx[qi] = bucket_original_idx[min_idx_in_bucket[j]]

        return mx.array(best_idx), mx.array(best_dist)

    def search(
        self, query: mx.array, max_nn: int = 1
    ) -> tuple[mx.array, mx.array]:
        """Find nearest neighbor(s) within radius.

        Args:
            query: (M, 3) query points.
            max_nn: Maximum neighbors per query (default 1).

        Returns:
            indices: (M, max_nn) int32 padded with -1 if fewer found.
            distances: (M, max_nn) float32 padded with inf.
        """
        if max_nn < 1:
            raise ValueError(f"max_nn must be >= 1, got {max_nn}")

        query_np = np.array(query, dtype=np.float32)
        M = query_np.shape[0]

        out_idx = np.full((M, max_nn), -1, dtype=np.int32)
        out_dist = np.full((M, max_nn), np.inf, dtype=np.float32)

        if self._n == 0 or M == 0:
            return mx.array(out_idx), mx.array(out_dist)

        radius_sq = self._radius ** 2
        query_cells = np.floor(query_np / self._cell_size).astype(np.int64)

        # Collect all candidate (query_idx, point_idx, sq_dist) tuples
        # then pick top-max_nn per query.
        # For efficiency, gather per-query candidates.
        candidates: list[list[tuple[int, float]]] = [[] for _ in range(M)]

        for offset in _OFFSETS_NP:
            neighbor_cells = query_cells + offset
            neighbor_hash = _hash_cells(neighbor_cells)

            unique_query_keys = np.unique(neighbor_hash)

            for qk in unique_query_keys:
                query_mask = neighbor_hash == qk
                table_pos = np.searchsorted(self._table_keys, qk)
                if table_pos >= len(self._table_keys) or self._table_keys[table_pos] != qk:
                    continue

                start = int(self._table_starts[table_pos])
                count = int(self._table_counts[table_pos])
                bucket_points = self._sorted_points[start : start + count]
                bucket_original_idx = self._sort_order[start : start + count]

                query_subset_indices = np.where(query_mask)[0]
                query_subset = query_np[query_subset_indices]

                diff = query_subset[:, None, :] - bucket_points[None, :, :]
                sq_dists = np.sum(diff ** 2, axis=2)

                for j, qi in enumerate(query_subset_indices):
                    for c in range(count):
                        d = float(sq_dists[j, c])
                        if d <= radius_sq:
                            candidates[qi].append(
                                (int(bucket_original_idx[c]), d)
                            )

        # Deduplicate and pick top-max_nn per query
        for qi in range(M):
            if not candidates[qi]:
                continue
            # Deduplicate by point index, keep min distance
            seen: dict[int, float] = {}
            for pidx, d in candidates[qi]:
                if pidx not in seen or d < seen[pidx]:
                    seen[pidx] = d
            sorted_cands = sorted(seen.items(), key=lambda x: x[1])
            n = min(len(sorted_cands), max_nn)
            for j in range(n):
                out_idx[qi, j] = sorted_cands[j][0]
                out_dist[qi, j] = sorted_cands[j][1]

        return mx.array(out_idx), mx.array(out_dist)
