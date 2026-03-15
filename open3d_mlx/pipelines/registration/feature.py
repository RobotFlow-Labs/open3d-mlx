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
    1. For each point, find neighbours within *radius*.
    2. Compute SPFH (Simplified Point Feature Histogram):
       - Establish a Darboux frame: u = n1, v = (u x d) / ||u x d||, w = u x v
       - Compute angles: alpha = v . n2, phi = u . d, theta = atan2(w . n2, u . n2)
       - Bin each angle into 11 bins -> 33-dim histogram.
    3. Weight neighbour SPFHs by inverse distance -> FPFH.
    """
    if not pcd.has_normals():
        raise ValueError("FPFH requires normals. Call estimate_normals() first.")

    from open3d_mlx.ops import NearestNeighborSearch

    pts_np = np.asarray(pcd.points, dtype=np.float64)
    normals_np = np.asarray(pcd.normals, dtype=np.float64)
    N = len(pts_np)

    # Get search radius / max_nn from param
    if hasattr(search_param, "radius"):
        radius = search_param.radius
        max_nn = getattr(search_param, "max_nn", 30)
    else:
        raise ValueError("FPFH requires a radius-based search parameter")

    nns = NearestNeighborSearch(pcd.points)

    n_bins = 11  # bins per angle feature
    n_features = n_bins * 3  # 33

    # Step 1: radius search for all points
    indices_list, dists_list = nns.radius_search(pcd.points, radius, max_nn=max_nn)

    # Step 2: Compute SPFH for every point
    spfh = np.zeros((N, n_features), dtype=np.float32)

    for i in range(N):
        idx = np.asarray(indices_list[i], dtype=np.int64)
        if len(idx) < 2:
            continue

        p1 = pts_np[i]
        n1 = normals_np[i]

        hist = np.zeros(n_features, dtype=np.float32)
        count = 0

        for j in idx:
            j = int(j)
            if j == i:
                continue

            p2 = pts_np[j]
            n2 = normals_np[j]

            diff = p2 - p1
            d = np.linalg.norm(diff)
            if d < 1e-10:
                continue

            # Darboux frame
            u = n1
            v = np.cross(u, diff / d)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-10:
                continue
            v = v / v_norm
            w = np.cross(u, v)

            # Angles
            alpha = float(np.clip(np.dot(v, n2), -1.0, 1.0))
            phi = float(np.clip(np.dot(u, diff / d), -1.0, 1.0))
            theta = float(np.arctan2(np.dot(w, n2), np.dot(u, n2)))

            # Bin into [0, n_bins - 1]
            a_bin = int(np.clip((alpha + 1.0) / 2.0 * n_bins, 0, n_bins - 1))
            p_bin = int(np.clip((phi + 1.0) / 2.0 * n_bins, 0, n_bins - 1))
            t_bin = int(
                np.clip((theta / np.pi + 1.0) / 2.0 * n_bins, 0, n_bins - 1)
            )

            hist[a_bin] += 1
            hist[n_bins + p_bin] += 1
            hist[2 * n_bins + t_bin] += 1
            count += 1

        if count > 0:
            spfh[i] = hist / count

    # Step 3: Weight neighbour SPFHs -> FPFH
    fpfh = np.zeros_like(spfh)
    for i in range(N):
        idx = np.asarray(indices_list[i], dtype=np.int64)
        dists = np.asarray(dists_list[i], dtype=np.float32)

        if len(idx) == 0:
            fpfh[i] = spfh[i]
            continue

        # Inverse-distance weights
        weights = np.zeros(len(idx), dtype=np.float64)
        for k in range(len(idx)):
            d = float(np.sqrt(dists[k]))
            weights[k] = 1.0 / max(d, 1e-10)

        weighted_sum = np.zeros(n_features, dtype=np.float64)
        for k in range(len(idx)):
            weighted_sum += weights[k] * spfh[int(idx[k])]

        total_weight = weights.sum()
        if total_weight > 0:
            fpfh[i] = spfh[i] + (weighted_sum / total_weight).astype(np.float32)

    return mx.array(fpfh)
