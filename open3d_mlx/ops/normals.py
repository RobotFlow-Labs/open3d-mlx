"""Normal estimation for 3D point clouds via PCA.

Computes surface normals by analyzing local neighborhoods using
singular value decomposition (SVD) of the local covariance matrix.

The batched path (``estimate_normals_pca_batched``) is fully MLX-native
and runs on GPU. The general path (``estimate_normals_pca``) handles
-1 padding by using numpy batched SVD (no per-point Python loops).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def estimate_normals_pca(
    points: mx.array,
    neighbor_indices: mx.array,
) -> mx.array:
    """Estimate normals using PCA on local neighborhoods.

    Fully vectorized: gathers all neighbors at once, computes batch
    covariance matrices, and runs a single batched SVD. No Python
    loops over individual points.

    Handles -1 padding in ``neighbor_indices`` by replacing invalid
    entries with the point's own index (which contributes zero to the
    centered covariance).

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

    # Move to numpy for batched SVD (MLX SVD works but numpy is more
    # robust for degenerate covariance matrices)
    mx.eval(points, neighbor_indices)
    pts_np = np.array(points, dtype=np.float64)
    idx_np = np.array(neighbor_indices, dtype=np.int32)

    # Valid-neighbor mask: (N, K)
    valid_mask = idx_np >= 0

    # Replace -1 with the point's own index (safe gather, contributes
    # zero after centering)
    own_idx = np.arange(N, dtype=np.int32)[:, None]  # (N, 1)
    idx_safe = np.where(valid_mask, idx_np, own_idx)

    # Gather ALL neighbors at once: (N, K, 3)
    neighbors = pts_np[idx_safe]

    # Count valid neighbors per point: (N,)
    valid_counts = valid_mask.sum(axis=1)

    # Compute centroids: (N, 3)
    # Zero out invalid slots before summing
    valid_mask_3d = valid_mask[:, :, None]  # (N, K, 1)
    neighbor_sum = np.where(valid_mask_3d, neighbors, 0.0).sum(axis=1)  # (N, 3)
    centroids = neighbor_sum / np.maximum(valid_counts[:, None], 1)

    # Center neighborhoods: (N, K, 3)
    centered = neighbors - centroids[:, None, :]
    centered = np.where(valid_mask_3d, centered, 0.0)

    # Batch covariance: (N, 3, 3) via einsum
    cov = np.einsum("nki,nkj->nij", centered, centered)
    cov /= np.maximum(valid_counts[:, None, None], 1)

    # Batch SVD — numpy handles (N, 3, 3) natively
    _, _, Vt = np.linalg.svd(cov)  # Vt: (N, 3, 3)

    # Normal = last row of Vt (smallest singular vector)
    normals = Vt[:, 2, :]  # (N, 3)

    # For points with < 3 valid neighbors, use default z-axis normal
    degenerate = valid_counts < 3
    normals[degenerate] = [0.0, 0.0, 1.0]

    # Normalize to unit length
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(norms > 1e-12, normals / norms, 0.0)

    # Final safety: any zero-norm rows get z-axis
    zero_mask = np.linalg.norm(normals, axis=1) < 1e-8
    normals[zero_mask] = [0.0, 0.0, 1.0]

    return mx.array(normals.astype(np.float32))


def estimate_normals_pca_batched(
    points: mx.array,
    neighbor_indices: mx.array,
) -> mx.array:
    """Estimate normals using fully MLX-native batched operations (GPU).

    Faster than ``estimate_normals_pca`` when all neighbor slots are
    valid (no -1 padding). Runs entirely on Apple Silicon GPU via MLX.

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

    Runs entirely on GPU via MLX.

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
