"""Normal estimation for 3D point clouds via PCA.

Computes surface normals by analyzing local neighborhoods using
singular value decomposition (SVD) of the local covariance matrix.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def estimate_normals_pca(
    points: mx.array,
    neighbor_indices: mx.array,
) -> mx.array:
    """Estimate normals using PCA on local neighborhoods.

    For each point, gathers its K neighbors, centers the neighborhood,
    computes the 3x3 covariance matrix, and takes the smallest singular
    vector as the surface normal.

    Args:
        points: (N, 3) point positions.
        neighbor_indices: (N, K) int32 indices of K neighbors per point.
            Entries of -1 are treated as padding and excluded.

    Returns:
        normals: (N, 3) unit normals (float32).

    Raises:
        ValueError: If shapes are incompatible.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points (N, 3), got {points.shape}")
    if neighbor_indices.ndim != 2:
        raise ValueError(
            f"Expected neighbor_indices (N, K), got {neighbor_indices.shape}"
        )
    if neighbor_indices.shape[0] != points.shape[0]:
        raise ValueError(
            f"Mismatch: points has {points.shape[0]} rows, "
            f"neighbor_indices has {neighbor_indices.shape[0]} rows"
        )

    N, K = neighbor_indices.shape

    # Convert to numpy for SVD (MLX batched SVD may not be available)
    points_np = np.array(points, dtype=np.float64)
    indices_np = np.array(neighbor_indices, dtype=np.int32)

    normals_np = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        # Get valid neighbor indices (exclude -1 padding)
        idx = indices_np[i]
        valid_mask = idx >= 0
        valid_idx = idx[valid_mask]

        if len(valid_idx) < 3:
            # Not enough neighbors for a reliable normal; default to z-axis
            normals_np[i] = [0.0, 0.0, 1.0]
            continue

        # Gather neighbors and center
        neighbors = points_np[valid_idx]  # (Kv, 3)
        centroid = np.mean(neighbors, axis=0, keepdims=True)  # (1, 3)
        centered = neighbors - centroid  # (Kv, 3)

        # Covariance matrix
        cov = (centered.T @ centered) / len(valid_idx)  # (3, 3)

        # SVD: smallest singular vector = normal
        try:
            _, S, Vt = np.linalg.svd(cov)
            normals_np[i] = Vt[-1]  # Last row = smallest eigenvector
        except np.linalg.LinAlgError:
            normals_np[i] = [0.0, 0.0, 1.0]

    # Normalize to unit length
    norms = np.linalg.norm(normals_np, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    normals_np = normals_np / norms

    return mx.array(normals_np.astype(np.float32))


def estimate_normals_pca_batched(
    points: mx.array,
    neighbor_indices: mx.array,
) -> mx.array:
    """Estimate normals using batched MLX operations (faster for large N).

    Same as estimate_normals_pca but uses MLX matmul and SVD for the
    batch. Requires all neighbor slots to be valid (no -1 padding).

    Args:
        points: (N, 3) point positions.
        neighbor_indices: (N, K) int32 indices, all valid (no -1).

    Returns:
        normals: (N, 3) unit normals (float32).
    """
    N, K = neighbor_indices.shape

    # Clamp indices to valid range for gathering
    safe_indices = mx.clip(neighbor_indices, 0, points.shape[0] - 1)

    # Gather neighbor points: (N, K, 3)
    neighbors = points[safe_indices]

    # Center each neighborhood
    centroids = mx.mean(neighbors, axis=1, keepdims=True)  # (N, 1, 3)
    centered = neighbors - centroids  # (N, K, 3)

    # Covariance matrices: (N, 3, 3)
    centered_t = mx.transpose(centered, axes=(0, 2, 1))  # (N, 3, K)
    cov = mx.matmul(centered_t, centered) / K  # (N, 3, 3)

    # SVD: smallest singular vector = normal
    # mx.linalg.svd returns (U, S, Vt) with shapes (N,3,3), (N,3), (N,3,3)
    _U, _S, Vt = mx.linalg.svd(cov)
    normals = Vt[:, -1, :]  # (N, 3) — last row of Vt

    # Normalize
    norms = mx.sqrt(mx.sum(normals ** 2, axis=1, keepdims=True))
    norms = mx.maximum(norms, mx.array(1e-12))
    normals = normals / norms

    return normals


def orient_normals_towards_viewpoint(
    points: mx.array,
    normals: mx.array,
    viewpoint: mx.array | None = None,
) -> mx.array:
    """Orient normals to consistently point towards a viewpoint.

    Args:
        points: (N, 3) point positions.
        normals: (N, 3) unit normals.
        viewpoint: (3,) viewpoint position. Defaults to origin [0,0,0].

    Returns:
        oriented_normals: (N, 3) normals flipped to face the viewpoint.
    """
    if viewpoint is None:
        viewpoint = mx.zeros(3, dtype=mx.float32)

    # Direction from point to viewpoint
    direction = viewpoint[None, :] - points  # (N, 3)

    # Dot product with normals
    dots = mx.sum(normals * direction, axis=1)  # (N,)

    # Flip normals where dot product is negative
    sign = mx.where(dots < 0, mx.array(-1.0), mx.array(1.0))  # (N,)
    return normals * sign[:, None]
