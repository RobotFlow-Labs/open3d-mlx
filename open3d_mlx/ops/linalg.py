"""Linear algebra utilities for 3D geometry operations.

Provides cross product, batched SVD, and batched linear solve.
Uses MLX-native operations where available, with numpy fallbacks
for operations that benefit from its robustness (e.g. singular
matrix handling in batched solve).
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


def compute_cross_product(a: mx.array, b: mx.array) -> mx.array:
    """Compute element-wise cross product of two (N, 3) arrays.

    Runs entirely on GPU via MLX.

    Args:
        a: (N, 3) or (3,) first vectors.
        b: (N, 3) or (3,) second vectors.

    Returns:
        cross: same shape as input, cross product a x b.
    """
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
        squeeze = True
    else:
        squeeze = False

    cx = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    cy = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    cz = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    result = mx.stack([cx, cy, cz], axis=1)

    if squeeze:
        result = result[0]
    return result


def batched_svd(
    matrices: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Batched SVD decomposition.

    Tries MLX-native SVD first (GPU). Falls back to numpy batched SVD
    which handles (N, M, K) natively without per-matrix loops.

    Args:
        matrices: (B, M, N) batch of matrices.

    Returns:
        U: (B, M, min(M,N)) left singular vectors.
        S: (B, min(M,N)) singular values.
        Vt: (B, min(M,N), N) right singular vectors (transposed).
    """
    # Try MLX native first (runs on GPU)
    try:
        U, S, Vt = mx.linalg.svd(matrices)
        return U, S, Vt
    except Exception:
        pass

    # Fallback: numpy batched SVD (handles (B, M, N) natively, no loops)
    np_matrices = np.array(matrices, dtype=np.float64)
    U_np, S_np, Vt_np = np.linalg.svd(np_matrices, full_matrices=False)
    return (
        mx.array(U_np.astype(np.float32)),
        mx.array(S_np.astype(np.float32)),
        mx.array(Vt_np.astype(np.float32)),
    )


def batched_solve(
    A: mx.array, b: mx.array
) -> mx.array:
    """Solve batched linear systems A @ x = b.

    Uses numpy's vectorized solve which handles (B, N, N) natively.
    Falls back to lstsq for singular matrices.

    Args:
        A: (B, N, N) batch of square matrices.
        b: (B, N) or (B, N, K) right-hand sides.

    Returns:
        x: same shape as b, solutions.
    """
    A_np = np.array(A, dtype=np.float64)
    b_np = np.array(b, dtype=np.float64)

    try:
        x_np = np.linalg.solve(A_np, b_np)
    except np.linalg.LinAlgError:
        # Some matrices are singular — fall back to per-matrix lstsq
        B = A_np.shape[0]
        x_np = np.zeros_like(b_np)
        for i in range(B):
            try:
                x_np[i] = np.linalg.solve(A_np[i], b_np[i])
            except np.linalg.LinAlgError:
                x_np[i] = np.linalg.lstsq(A_np[i], b_np[i], rcond=None)[0]

    return mx.array(x_np.astype(np.float32))


def symmetric_eigendecomposition(
    matrices: mx.array,
) -> tuple[mx.array, mx.array]:
    """Eigendecomposition of symmetric matrices.

    Uses numpy's vectorized eigh which handles (B, N, N) natively
    without per-matrix loops.

    Args:
        matrices: (B, N, N) batch of symmetric matrices.

    Returns:
        eigenvalues: (B, N) sorted ascending.
        eigenvectors: (B, N, N) columns are eigenvectors.
    """
    mat_np = np.array(matrices, dtype=np.float64)

    # numpy.linalg.eigh handles batched (B, N, N) natively
    eigenvalues, eigenvectors = np.linalg.eigh(mat_np)

    return (
        mx.array(eigenvalues.astype(np.float32)),
        mx.array(eigenvectors.astype(np.float32)),
    )
