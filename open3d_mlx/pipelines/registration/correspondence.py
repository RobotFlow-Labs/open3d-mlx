"""Correspondence search for ICP registration.

Finds nearest-neighbor correspondences between source and target point clouds
using the GPU-friendly FixedRadiusIndex.
"""

from __future__ import annotations

import mlx.core as mx

from open3d_mlx.ops.fixed_radius_nn import FixedRadiusIndex


def find_correspondences(
    source_points: mx.array,
    target_index: FixedRadiusIndex,
    max_distance: float,
) -> tuple[mx.array, mx.array]:
    """Find nearest-neighbor correspondences within a distance threshold.

    For each source point, finds the nearest target point using the
    pre-built spatial hash index.  Points without a neighbor within
    ``max_distance`` receive a correspondence of ``-1``.

    Parameters
    ----------
    source_points : mx.array
        ``(N, 3)`` source point positions.
    target_index : FixedRadiusIndex
        Pre-built spatial hash index on target points.
    max_distance : float
        Maximum correspondence distance.

    Returns
    -------
    correspondences : mx.array
        ``(N,)`` int32 indices into target.  ``-1`` means no match.
    distances : mx.array
        ``(N,)`` float32 *squared* Euclidean distances to correspondences.
        ``inf`` where no match was found.
    """
    indices, sq_dists = target_index.search_nearest(source_points)
    return indices, sq_dists
