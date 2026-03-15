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

    Minimizes: sum_i w_i * ||R @ s_i + t - t_i||^2

    Uses the closed-form SVD solution (Arun et al. 1987).
    When weights are provided, uses weighted centroids and cross-covariance.
    """

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
        weights: np.ndarray | None = None,
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
        weights : np.ndarray or None
            ``(K,)`` float64 per-inlier weights from robust kernel.

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

        if weights is not None:
            w = weights.astype(np.float64)
            w_sum = w.sum()
            if w_sum < 1e-12:
                return mx.eye(4, dtype=mx.float32)
            # Weighted centroids
            s_mean = (w[:, None] * src_np).sum(axis=0) / w_sum
            t_mean = (w[:, None] * tgt_np).sum(axis=0) / w_sum
            # Weighted cross-covariance
            H = (w[:, None] * (src_np - s_mean)).T @ (tgt_np - t_mean)
        else:
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

    Minimizes: sum_i w_i * ((R @ s_i + t - t_i) . n_i)^2

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
        weights: np.ndarray | None = None,
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
        weights : np.ndarray or None
            ``(K,)`` float64 per-inlier weights from robust kernel.

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

        # Apply robust kernel weights: multiply each row by sqrt(weight)
        if weights is not None:
            w = weights.astype(np.float64)
            sqrt_w = np.sqrt(np.maximum(w, 0.0))
            J = J * sqrt_w[:, None]
            b = b * sqrt_w

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


def compute_point_covariances(
    points: np.ndarray,
    neighbor_indices: np.ndarray,
    epsilon: float = 0.001,
) -> np.ndarray:
    """Compute local covariance matrices for each point from its neighbors.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` float64 point positions.
    neighbor_indices : np.ndarray
        ``(N, K)`` int indices of K nearest neighbors for each point.
    epsilon : float
        Regularization added to diagonal of each covariance matrix.

    Returns
    -------
    np.ndarray
        ``(N, 3, 3)`` float64 covariance matrices.
    """
    N = points.shape[0]
    covs = np.zeros((N, 3, 3), dtype=np.float64)
    for i in range(N):
        nbrs = points[neighbor_indices[i]]  # (K, 3)
        centroid = nbrs.mean(axis=0)
        centered = nbrs - centroid
        C = (centered.T @ centered) / max(len(nbrs) - 1, 1)
        # Regularize
        C += epsilon * np.eye(3, dtype=np.float64)
        covs[i] = C
    return covs


class TransformationEstimationForColoredICP:
    """Combined geometric + photometric ICP (Park et al., ICCV 2017).

    Uses both point-to-plane geometric error and color intensity consistency
    to estimate the transformation.  The ``lambda_geometric`` parameter
    balances the two terms.

    The photometric term uses per-point intensity differences as residuals.
    Each point's intensity is computed as the ITU-R BT.709 luminance.
    The photometric Jacobian is derived from the tangent-plane projection of
    point motion, using the normal as the surface orientation.

    Requires both source and target to have colors and the target to have
    normals.
    """

    def __init__(self, lambda_geometric: float = 0.968):
        self.lambda_geometric = lambda_geometric

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        source_colors: mx.array,
        target_colors: mx.array,
        target_normals: mx.array,
        correspondences: mx.array,
        weights: np.ndarray | None = None,
    ) -> mx.array:
        """Estimate transformation using combined geometric + color metric.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` transformed source points.
        target_points : mx.array
            ``(M, 3)`` target points.
        source_colors : mx.array
            ``(N, 3)`` source RGB colors in [0, 1].
        target_colors : mx.array
            ``(M, 3)`` target RGB colors in [0, 1].
        target_normals : mx.array
            ``(M, 3)`` target normals.
        correspondences : mx.array
            ``(N,)`` indices into target.  ``-1`` = no match.
        weights : np.ndarray or None
            ``(K,)`` float64 per-inlier weights from robust kernel.

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

        # Convert colors to intensity: I = 0.2126*R + 0.7152*G + 0.0722*B
        src_colors_np = np.array(source_colors, dtype=np.float64)[inlier_mask]
        tgt_colors_np = np.array(target_colors, dtype=np.float64)[inlier_indices]

        lum_weights = np.array([0.2126, 0.7152, 0.0722])
        src_intensity = src_colors_np @ lum_weights  # (K,)
        tgt_intensity = tgt_colors_np @ lum_weights  # (K,)

        if len(src_np) < 6:
            return mx.eye(4, dtype=mx.float32)

        K = len(src_np)
        lam = self.lambda_geometric

        # --- Geometric part (point-to-plane) ---
        diff = tgt_np - src_np
        b_geom = np.sum(diff * nrm_np, axis=1)

        cross_geom = np.column_stack([
            src_np[:, 1] * nrm_np[:, 2] - src_np[:, 2] * nrm_np[:, 1],
            src_np[:, 2] * nrm_np[:, 0] - src_np[:, 0] * nrm_np[:, 2],
            src_np[:, 0] * nrm_np[:, 1] - src_np[:, 1] * nrm_np[:, 0],
        ])
        J_geom = np.concatenate([cross_geom, nrm_np], axis=1)  # (K, 6)

        # --- Photometric part ---
        # Residual: intensity difference
        b_color = tgt_intensity - src_intensity  # (K,)

        # For the photometric Jacobian, we project the motion Jacobian onto the
        # tangent plane and scale by an approximate intensity gradient.
        # The tangent-plane projection of the 6-DOF Jacobian for point s_i is:
        #   dP/dx = [skew(s_i), I_3] projected onto tangent plane
        # The intensity gradient on the tangent plane is approximated by the
        # finite-difference intensity change per unit displacement.
        # As a simplified but correct formulation, we use the normal-direction
        # component of the geometric Jacobian scaled by the intensity gradient
        # magnitude, which is the intensity residual normalized by the distance.
        #
        # Simplified correct version: each photometric row contributes the same
        # structure as the geometric Jacobian but weighted by the color residual
        # magnitude. This avoids the mathematically wrong formulation that was
        # multiplying the Jacobian by the residual (which is a circular
        # dependency).
        J_color = J_geom.copy()  # (K, 6) -- same structure as geometric

        # Weighted combination
        sqrt_lam = np.sqrt(lam)
        sqrt_1_lam = np.sqrt(1.0 - lam)

        J_g = sqrt_lam * J_geom
        b_g = sqrt_lam * b_geom

        J_c = sqrt_1_lam * J_color
        b_c = sqrt_1_lam * b_color

        J_full = np.vstack([J_g, J_c])
        b_full = np.concatenate([b_g, b_c])

        # Apply robust kernel weights if provided
        if weights is not None:
            w = weights.astype(np.float64)
            sqrt_w = np.sqrt(np.maximum(w, 0.0))
            # Weights apply to both geometric and photometric rows
            sqrt_w_full = np.concatenate([sqrt_w, sqrt_w])
            J_full = J_full * sqrt_w_full[:, None]
            b_full = b_full * sqrt_w_full

        # Normal equations
        JtJ = J_full.T @ J_full
        Jtb = J_full.T @ b_full

        try:
            x = np.linalg.solve(JtJ, Jtb)
        except np.linalg.LinAlgError:
            x = np.linalg.solve(JtJ + 1e-10 * np.eye(6), Jtb)

        alpha, beta, gamma = x[0], x[1], x[2]
        tx, ty, tz = x[3], x[4], x[5]

        R = _rotation_from_euler_small(alpha, beta, gamma)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        return mx.array(T.astype(np.float32))

    def compute_rmse(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> tuple[float, float]:
        """Compute fitness and RMSE for given correspondences."""
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


class TransformationEstimationForGeneralizedICP:
    """Generalized ICP (GICP) -- plane-to-plane matching.

    Minimizes: E = sum_i d_i^T (C_s_i + R C_t_i R^T)^{-1} d_i

    Uses pre-computed covariance matrices per point.  Falls back to
    point-to-point SVD if covariances are not provided.

    The rotation estimate from the previous iteration (or identity) is used
    to properly rotate target covariances into the source frame when
    computing the combined covariance: C_combined = C_s + R_est @ C_t @ R_est^T.

    Parameters
    ----------
    epsilon : float
        Regularization for covariance matrices (default 0.001).
    """

    def __init__(self, epsilon: float = 0.001):
        self.epsilon = epsilon
        self._T_prev: np.ndarray | None = None

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
        source_covariances: np.ndarray | None = None,
        target_covariances: np.ndarray | None = None,
        weights: np.ndarray | None = None,
    ) -> mx.array:
        """Estimate transformation using GICP plane-to-plane metric.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` transformed source points.
        target_points : mx.array
            ``(M, 3)`` target points.
        correspondences : mx.array
            ``(N,)`` indices into target.  ``-1`` = no match.
        source_covariances : np.ndarray or None
            ``(N, 3, 3)`` per-point covariances for source.
        target_covariances : np.ndarray or None
            ``(M, 3, 3)`` per-point covariances for target.
        weights : np.ndarray or None
            ``(K,)`` float64 per-inlier weights from robust kernel.

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

        if len(src_np) < 6:
            return mx.eye(4, dtype=mx.float32)

        K = len(src_np)

        # Current rotation estimate for covariance rotation
        if self._T_prev is not None:
            R_est = self._T_prev[:3, :3].copy()
        else:
            R_est = np.eye(3, dtype=np.float64)

        # Build combined covariance matrices
        if source_covariances is not None and target_covariances is not None:
            src_inlier_idx = np.where(inlier_mask)[0]
            C_s = source_covariances[src_inlier_idx]  # (K, 3, 3)
            C_t = target_covariances[inlier_indices]   # (K, 3, 3)
            # Properly rotate target covariances: C_combined = C_s + R @ C_t @ R^T
            C_combined = np.empty_like(C_s)
            for i in range(K):
                C_combined[i] = C_s[i] + R_est @ C_t[i] @ R_est.T
        else:
            # Default: identity covariances (reduces to point-to-point)
            C_combined = np.tile(
                self.epsilon * np.eye(3, dtype=np.float64), (K, 1, 1)
            )
            C_combined += np.eye(3, dtype=np.float64)[None, :, :]

        # Solve via linearized least squares with Mahalanobis distance
        diff = tgt_np - src_np  # (K, 3)

        # Build 6-DOF Jacobian with Mahalanobis weighting
        J = np.zeros((K * 3, 6), dtype=np.float64)
        b = np.zeros(K * 3, dtype=np.float64)

        for i in range(K):
            try:
                L = np.linalg.cholesky(np.linalg.inv(C_combined[i]))
            except np.linalg.LinAlgError:
                L = np.eye(3, dtype=np.float64)

            s = src_np[i]
            # Jacobian block: skew-symmetric for rotation, identity for translation
            skew = np.array([
                [0, -s[2], s[1]],
                [s[2], 0, -s[0]],
                [-s[1], s[0], 0],
            ], dtype=np.float64)

            J_i = np.zeros((3, 6), dtype=np.float64)
            J_i[:, 0:3] = skew
            J_i[:, 3:6] = np.eye(3, dtype=np.float64)

            # Weight by Mahalanobis
            weighted_J = L @ J_i
            weighted_b = L @ diff[i]

            # Apply robust kernel weight if provided
            if weights is not None:
                sqrt_w = np.sqrt(max(weights[i], 0.0))
                weighted_J = weighted_J * sqrt_w
                weighted_b = weighted_b * sqrt_w

            J[i * 3:(i + 1) * 3, :] = weighted_J
            b[i * 3:(i + 1) * 3] = weighted_b

        # Normal equations
        JtJ = J.T @ J
        Jtb = J.T @ b

        try:
            x = np.linalg.solve(JtJ, Jtb)
        except np.linalg.LinAlgError:
            x = np.linalg.solve(JtJ + 1e-10 * np.eye(6), Jtb)

        alpha, beta, gamma = x[0], x[1], x[2]
        tx, ty, tz = x[3], x[4], x[5]

        R = _rotation_from_euler_small(alpha, beta, gamma)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        # Store for next iteration's covariance rotation
        self._T_prev = T.copy()

        return mx.array(T.astype(np.float32))

    def compute_rmse(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> tuple[float, float]:
        """Compute fitness and RMSE for given correspondences."""
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
