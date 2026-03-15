"""Fast Point Feature Histogram (FPFH) descriptor computation.

FPFH captures local geometric properties via a histogram of angular
relationships between point normals in a neighborhood.  The 33-dimensional
descriptor is widely used for coarse registration and feature matching.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def compute_fpfh_feature(pcd, search_param):
    """Compute Fast Point Feature Histogram (FPFH) for each point.

    Args:
        pcd: PointCloud with normals estimated.
        search_param: ``KDTreeSearchParamHybrid`` or
            ``KDTreeSearchParamRadius`` specifying the neighborhood.

    Returns:
        (N, 33) float32 MLX array -- 33-dimensional FPFH descriptor per point.

    Raises:
        ValueError: If the point cloud has no normals or *search_param*
            does not specify a radius.

    Algorithm
    ---------
    1. For each point, find neighbours within *radius* using hybrid search.
    2. Compute SPFH (Simplified Point Feature Histogram) via vectorized
       per-neighbor-slot passes:
       - Establish a Darboux frame: u = n1, v = (u x d) / ||u x d||, w = u x v
       - Compute angles: alpha = v . n2, phi = u . d, theta = atan2(w . n2, u . n2)
       - Bin each angle into 11 bins -> 33-dim histogram.
    3. Weight neighbour SPFHs by inverse distance -> FPFH.
    """
    if not pcd.has_normals():
        raise ValueError("FPFH requires normals. Call estimate_normals() first.")

    from open3d_mlx.ops import NearestNeighborSearch

    pts = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    N = len(pts)

    # Get search radius / max_nn from param
    if hasattr(search_param, "radius"):
        radius = search_param.radius
        max_nn = getattr(search_param, "max_nn", 30)
    else:
        raise ValueError("FPFH requires a radius-based search parameter")

    nns = NearestNeighborSearch(pcd.points)

    n_bins = 11  # bins per angle feature
    n_features = n_bins * 3  # 33

    # Use hybrid search for fixed-size padded output: (N, max_nn)
    indices, dists, counts = nns.hybrid_search(pcd.points, radius, max_nn)
    idx_np = np.array(indices)    # (N, max_nn) int32, padded with -1
    dist_np = np.array(dists)     # (N, max_nn) float32, padded with inf
    count_np = np.array(counts)   # (N,) int32

    # ── VECTORIZED SPFH computation ──────────────────────────────────────
    # Instead of O(N*K) Python loops, iterate over K neighbor slots
    # with fully vectorized NumPy operations over all N points per slot.

    spfh = np.zeros((N, n_features), dtype=np.float32)

    for k in range(max_nn):
        j_indices = idx_np[:, k]  # (N,) — neighbor index for slot k
        valid = (j_indices >= 0) & (k < count_np)  # (N,) bool mask

        if not valid.any():
            continue

        # Gather neighbor positions and normals (use 0 for invalid to avoid OOB)
        j_safe = np.where(valid, j_indices, 0)
        p2 = pts[j_safe]       # (N, 3)
        n2 = normals[j_safe]   # (N, 3)

        diff = p2 - pts  # (N, 3)
        d = np.linalg.norm(diff, axis=1, keepdims=True)  # (N, 1)
        d = np.maximum(d, 1e-10)
        diff_norm = diff / d  # (N, 3)

        # Darboux frame
        u = normals  # (N, 3)
        v = np.cross(u, diff_norm)  # (N, 3)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        v_valid = (v_norm > 1e-10).squeeze()
        valid = valid & v_valid
        v_norm = np.maximum(v_norm, 1e-10)
        v = v / v_norm
        w = np.cross(u, v)  # (N, 3)

        # Angles
        alpha = np.clip(np.sum(v * n2, axis=1), -1, 1)  # (N,)
        phi = np.clip(np.sum(u * diff_norm, axis=1), -1, 1)  # (N,)
        theta = np.arctan2(
            np.sum(w * n2, axis=1), np.sum(u * n2, axis=1)
        )  # (N,)

        # Bin into [0, n_bins - 1]
        a_bin = np.clip(((alpha + 1) / 2 * n_bins).astype(np.int32), 0, n_bins - 1)
        p_bin = np.clip(((phi + 1) / 2 * n_bins).astype(np.int32), 0, n_bins - 1)
        t_bin = np.clip(
            ((theta / np.pi + 1) / 2 * n_bins).astype(np.int32), 0, n_bins - 1
        )

        # Accumulate into histogram (vectorized scatter via np.add.at)
        valid_idx = np.where(valid)[0]
        np.add.at(spfh, (valid_idx, a_bin[valid_idx]), 1)
        np.add.at(spfh, (valid_idx, n_bins + p_bin[valid_idx]), 1)
        np.add.at(spfh, (valid_idx, 2 * n_bins + t_bin[valid_idx]), 1)

    # Normalize SPFH
    total = np.maximum(count_np[:, None], 1).astype(np.float32)
    spfh /= total

    # ── FPFH = SPFH + weighted sum of neighbor SPFHs ────────────────────
    fpfh = spfh.copy()
    for k in range(max_nn):
        j_indices = idx_np[:, k]
        valid = (j_indices >= 0) & (k < count_np)
        j_safe = np.where(valid, j_indices, 0)

        d = np.sqrt(np.maximum(dist_np[:, k], 1e-10))
        w = 1.0 / d
        w = np.where(valid, w, 0.0)

        fpfh += w[:, None] * spfh[j_safe]

    return mx.array(fpfh)
