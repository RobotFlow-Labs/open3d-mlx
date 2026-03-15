"""GPU-friendly fixed-radius nearest neighbor search via spatial hashing.

Used in the ICP inner loop for correspondence search.
Much faster than KDTree for large point clouds when radius is known.

Algorithm:
    1. Compute voxel indices: floor(points / cell_size)
    2. Hash voxel indices to cell keys using prime hashing
    3. Sort points by cell key (on GPU via MLX)
    4. For each query, check 27 neighboring cells (3x3x3 cube)
    5. Filter by distance <= radius, keep nearest

This implementation keeps data on GPU (MLX arrays) as long as possible
and uses vectorized numpy only for the hash-table lookup (since MLX
lacks np.unique / searchsorted). All distance computations are batched.
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


def _hash_cells_np(cell_idx: np.ndarray) -> np.ndarray:
    """Compute spatial hash keys from integer cell indices (numpy path).

    Args:
        cell_idx: (N, 3) int64 cell indices.

    Returns:
        keys: (N,) int32 hash keys (non-negative).
    """
    h = (
        cell_idx[:, 0].astype(np.int64) * _P1
        + cell_idx[:, 1].astype(np.int64) * _P2
        + cell_idx[:, 2].astype(np.int64) * _P3
    )
    h = h % _HASH_MOD
    return h.astype(np.int32)


class FixedRadiusIndex:
    """Fixed-radius nearest neighbor search using spatial hashing.

    Index construction hashes and sorts points on-GPU via MLX.
    The hash table (unique keys, bucket starts/counts) is built with
    numpy since MLX lacks ``np.unique``.  Query processing is fully
    vectorized with no Python loops over individual query points.

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
        self._radius_sq = self._radius * self._radius
        self._cell_size = float(radius)
        self._n = points.shape[0]

        if self._n == 0:
            # Empty index — store minimal arrays
            self._sorted_points_np = np.empty((0, 3), dtype=np.float32)
            self._sort_order_np = np.empty((0,), dtype=np.int32)
            self._table_keys = np.empty((0,), dtype=np.int32)
            self._table_starts = np.empty((0,), dtype=np.int64)
            self._table_counts = np.empty((0,), dtype=np.int64)
            return

        # --- GPU: hash computation and sort via MLX ---
        cell_idx = mx.floor(points / self._cell_size).astype(mx.int32)

        # Spatial hash on GPU (use int32 arithmetic, matching numpy path)
        hash_keys = (
            cell_idx[:, 0].astype(mx.int64) * 73856093
            + cell_idx[:, 1].astype(mx.int64) * 19349663
            + cell_idx[:, 2].astype(mx.int64) * 83492791
        )
        # Mod to keep in positive int32 range
        hash_keys = (hash_keys % (2**31 - 1)).astype(mx.int32)

        # Sort on GPU
        sort_order = mx.argsort(hash_keys)
        sorted_keys = hash_keys[sort_order]
        sorted_points = points[sort_order]
        mx.eval(sorted_keys, sorted_points, sort_order)

        # --- CPU: build hash table (unique/searchsorted not in MLX) ---
        self._sort_order_np = np.array(sort_order, dtype=np.int32)
        self._sorted_points_np = np.array(sorted_points, dtype=np.float32)
        sorted_keys_np = np.array(sorted_keys, dtype=np.int32)

        unique_keys, starts, counts = np.unique(
            sorted_keys_np, return_index=True, return_counts=True
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

    # ------------------------------------------------------------------
    # search_nearest — fully vectorized, no per-query Python loops
    # ------------------------------------------------------------------

    def search_nearest(
        self, query: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Find single nearest neighbor within radius using 27-cell search.

        Fully vectorized: no Python loops over individual query points.
        Hash computation done on GPU, distance computation batched via
        numpy broadcasting.

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
        best_dist_sq = np.full(M, np.inf, dtype=np.float32)

        if self._n == 0 or M == 0:
            return mx.array(best_idx), mx.array(best_dist_sq)

        radius_sq = self._radius_sq

        # Compute query cell indices
        query_cells = np.floor(query_np / self._cell_size).astype(np.int64)

        sorted_pts = self._sorted_points_np
        sort_order = self._sort_order_np
        table_keys = self._table_keys
        table_starts = self._table_starts
        table_counts = self._table_counts
        n_buckets = len(table_keys)

        # For each of the 27 neighbor offsets, batch-process ALL queries
        for offset in _OFFSETS_NP:
            nb_cells = query_cells + offset  # (M, 3)
            nb_hash = _hash_cells_np(nb_cells)  # (M,) int32

            # Vectorized bucket lookup via searchsorted
            positions = np.searchsorted(table_keys, nb_hash)
            valid_mask = (
                (positions < n_buckets)
                & (table_keys[np.clip(positions, 0, n_buckets - 1)] == nb_hash)
            )

            if not valid_mask.any():
                continue

            valid_qi = np.where(valid_mask)[0]
            valid_positions = positions[valid_qi]

            # Group queries by bucket for efficient batched distance calc
            unique_bpos = np.unique(valid_positions)
            for bp in unique_bpos:
                qi_mask = valid_positions == bp
                qi_in_bucket = valid_qi[qi_mask]

                start = int(table_starts[bp])
                count = int(table_counts[bp])
                bucket_pts = sorted_pts[start : start + count]  # (C, 3)

                # Vectorized distances: (Q, 1, 3) - (1, C, 3) -> (Q, C)
                diffs = query_np[qi_in_bucket, None, :] - bucket_pts[None, :, :]
                sq_dists = np.sum(diffs * diffs, axis=2)  # (Q, C)

                # Min per query
                min_idx_in_bucket = np.argmin(sq_dists, axis=1)  # (Q,)
                min_dists = sq_dists[
                    np.arange(len(qi_in_bucket)), min_idx_in_bucket
                ]

                # Vectorized update (no per-query Python loop)
                update_mask = (min_dists <= radius_sq) & (
                    min_dists < best_dist_sq[qi_in_bucket]
                )
                update_qi = qi_in_bucket[update_mask]
                best_dist_sq[update_qi] = min_dists[update_mask]
                best_idx[update_qi] = sort_order[
                    start + min_idx_in_bucket[update_mask]
                ]

        # Return squared distances (inf where no match)
        return mx.array(best_idx), mx.array(best_dist_sq)

    # ------------------------------------------------------------------
    # search — multi-neighbor, vectorized where possible
    # ------------------------------------------------------------------

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

        radius_sq = self._radius_sq
        query_cells = np.floor(query_np / self._cell_size).astype(np.int64)

        sorted_pts = self._sorted_points_np
        sort_order = self._sort_order_np
        table_keys = self._table_keys
        table_starts = self._table_starts
        table_counts = self._table_counts
        n_buckets = len(table_keys)

        # Collect per-query candidate lists: (original_idx, sq_dist)
        # We use a dict keyed by point index to deduplicate
        candidates: list[dict[int, float]] = [{} for _ in range(M)]

        for offset in _OFFSETS_NP:
            nb_cells = query_cells + offset
            nb_hash = _hash_cells_np(nb_cells)

            positions = np.searchsorted(table_keys, nb_hash)
            valid_mask = (
                (positions < n_buckets)
                & (table_keys[np.clip(positions, 0, n_buckets - 1)] == nb_hash)
            )

            if not valid_mask.any():
                continue

            valid_qi = np.where(valid_mask)[0]
            valid_positions = positions[valid_qi]

            unique_bpos = np.unique(valid_positions)
            for bp in unique_bpos:
                qi_mask = valid_positions == bp
                qi_in_bucket = valid_qi[qi_mask]

                start = int(table_starts[bp])
                count = int(table_counts[bp])
                bucket_pts = sorted_pts[start : start + count]  # (C, 3)
                bucket_orig = sort_order[start : start + count]  # (C,)

                # Vectorized distances
                diffs = query_np[qi_in_bucket, None, :] - bucket_pts[None, :, :]
                sq_dists = np.sum(diffs * diffs, axis=2)  # (Q, C)

                # Mask by radius
                within = sq_dists <= radius_sq  # (Q, C)

                # Collect candidates per query
                for j, qi in enumerate(qi_in_bucket):
                    for c in range(count):
                        if within[j, c]:
                            pidx = int(bucket_orig[c])
                            d = float(sq_dists[j, c])
                            if pidx not in candidates[qi] or d < candidates[qi][pidx]:
                                candidates[qi][pidx] = d

        # Pick top-max_nn per query
        for qi in range(M):
            if not candidates[qi]:
                continue
            sorted_cands = sorted(candidates[qi].items(), key=lambda x: x[1])
            n = min(len(sorted_cands), max_nn)
            for j in range(n):
                out_idx[qi, j] = sorted_cands[j][0]
                out_dist[qi, j] = sorted_cands[j][1]

        return mx.array(out_idx), mx.array(out_dist)
