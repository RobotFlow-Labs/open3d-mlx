"""Transformation estimation methods for ICP registration.

Provides SVD-based point-to-point and linearized point-to-plane estimators.
All internal computation uses numpy float64 for numerical stability; results
are returned as MLX float32 arrays for GPU compatibility.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


class TransformationEstimationPointToPoint:
    """Point-to-point transformation estimation via SVD.

    Minimizes: sum_i ||R @ s_i + t - t_i||^2

    Uses the closed-form SVD solution (Arun et al. 1987).
    """

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> mx.array:
        """Estimate rigid transformation from correspondences.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` source points (already transformed).
        target_points : mx.array
            ``(M, 3)`` target points.
        correspondences : mx.array
            ``(N,)`` indices into target. ``-1`` = no match.

        Returns
        -------
        mx.array
            ``(4, 4)`` float32 transformation matrix.
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0

        src_np = np.array(source_points, dtype=np.float64)[inlier_mask]
        tgt_np = np.array(target_points, dtype=np.float64)[corr_np[inlier_mask]]

        if len(src_np) < 3:
            # Not enough correspondences for a meaningful transform
            return mx.eye(4, dtype=mx.float32)

        # Centroids
        s_mean = src_np.mean(axis=0)
        t_mean = tgt_np.mean(axis=0)

        # Cross-covariance matrix H = (S - s_mean)^T @ (T - t_mean)
        H = (src_np - s_mean).T @ (tgt_np - t_mean)

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Rotation (handle reflection via determinant check)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

        # Translation
        t = t_mean - R @ s_mean

        # Build 4x4 homogeneous transformation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

        return mx.array(T.astype(np.float32))

    def compute_rmse(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> tuple[float, float]:
        """Compute fitness and RMSE for given correspondences.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` source points.
        target_points : mx.array
            ``(M, 3)`` target points.
        correspondences : mx.array
            ``(N,)`` int32 indices into target. ``-1`` = no match.

        Returns
        -------
        tuple[float, float]
            ``(fitness, inlier_rmse)``
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0
        num_inliers = int(inlier_mask.sum())
        n_source = len(corr_np)

        if n_source == 0 or num_inliers == 0:
            return 0.0, float("inf")

        fitness = num_inliers / n_source

        src_np = np.array(source_points, dtype=np.float64)[inlier_mask]
        tgt_np = np.array(target_points, dtype=np.float64)[corr_np[inlier_mask]]

        sq_dists = np.sum((src_np - tgt_np) ** 2, axis=1)
        rmse = float(np.sqrt(sq_dists.mean()))

        return fitness, rmse


class TransformationEstimationPointToPlane:
    """Point-to-plane transformation estimation via linearized least-squares.

    Minimizes: sum_i ((R @ s_i + t - t_i) . n_i)^2

    Uses the small-angle approximation to linearize the rotation, then solves
    a 6x6 linear system.  The resulting small-angle rotation parameters are
    converted to a proper rotation matrix via the Rodrigues formula.

    Requires the target point cloud to have normals.
    """

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        target_normals: mx.array,
        correspondences: mx.array,
    ) -> mx.array:
        """Estimate transformation using point-to-plane metric.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` transformed source points.
        target_points : mx.array
            ``(M, 3)`` target points.
        target_normals : mx.array
            ``(M, 3)`` target normals.
        correspondences : mx.array
            ``(N,)`` indices into target. ``-1`` = no match.

        Returns
        -------
        mx.array
            ``(4, 4)`` float32 transformation matrix.
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0
        inlier_indices = corr_np[inlier_mask]

        src_np = np.array(source_points, dtype=np.float64)[inlier_mask]
        tgt_np = np.array(target_points, dtype=np.float64)[inlier_indices]
        nrm_np = np.array(target_normals, dtype=np.float64)[inlier_indices]

        if len(src_np) < 6:
            # Not enough correspondences for a 6-DOF solve
            return mx.eye(4, dtype=mx.float32)

        # Residuals: (t - s) . n
        diff = tgt_np - src_np  # (K, 3)
        b = np.sum(diff * nrm_np, axis=1)  # (K,)

        # Jacobian: J_i = [cross(s_i, n_i), n_i]  (1x6)
        cross = np.column_stack([
            src_np[:, 1] * nrm_np[:, 2] - src_np[:, 2] * nrm_np[:, 1],
            src_np[:, 2] * nrm_np[:, 0] - src_np[:, 0] * nrm_np[:, 2],
            src_np[:, 0] * nrm_np[:, 1] - src_np[:, 1] * nrm_np[:, 0],
        ])  # (K, 3)

        J = np.concatenate([cross, nrm_np], axis=1)  # (K, 6)

        # Normal equations: (J^T J) x = J^T b
        JtJ = J.T @ J  # (6, 6)
        Jtb = J.T @ b  # (6,)

        # Solve with regularization for numerical stability
        try:
            x = np.linalg.solve(JtJ, Jtb)  # (6,)
        except np.linalg.LinAlgError:
            # Singular system — add small regularization
            x = np.linalg.solve(JtJ + 1e-10 * np.eye(6), Jtb)

        alpha, beta, gamma = x[0], x[1], x[2]
        tx, ty, tz = x[3], x[4], x[5]

        # Build rotation from small-angle parameters via Rodrigues
        R = _rotation_from_euler_small(alpha, beta, gamma)

        # Build 4x4 transformation
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        return mx.array(T.astype(np.float32))

    def compute_rmse(
        self,
        source_points: mx.array,
        target_points: mx.array,
        target_normals: mx.array,
        correspondences: mx.array,
    ) -> tuple[float, float]:
        """Compute fitness and point-to-plane RMSE for given correspondences.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` source points.
        target_points : mx.array
            ``(M, 3)`` target points.
        target_normals : mx.array
            ``(M, 3)`` target normals.
        correspondences : mx.array
            ``(N,)`` int32 indices into target. ``-1`` = no match.

        Returns
        -------
        tuple[float, float]
            ``(fitness, inlier_rmse)``
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0
        num_inliers = int(inlier_mask.sum())
        n_source = len(corr_np)

        if n_source == 0 or num_inliers == 0:
            return 0.0, float("inf")

        fitness = num_inliers / n_source

        src_np = np.array(source_points, dtype=np.float64)[inlier_mask]
        tgt_np = np.array(target_points, dtype=np.float64)[corr_np[inlier_mask]]

        sq_dists = np.sum((src_np - tgt_np) ** 2, axis=1)
        rmse = float(np.sqrt(sq_dists.mean()))

        return fitness, rmse


def _rotation_from_euler_small(
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Build a rotation matrix from small-angle Euler parameters.

    Uses the Rodrigues formula for accuracy even at larger angles.

    Parameters
    ----------
    alpha, beta, gamma : float
        Rotation parameters (radians) about x, y, z axes respectively.

    Returns
    -------
    np.ndarray
        ``(3, 3)`` float64 rotation matrix.
    """
    angle = math.sqrt(alpha**2 + beta**2 + gamma**2)
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)

    axis = np.array([alpha, beta, gamma], dtype=np.float64) / angle
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R
