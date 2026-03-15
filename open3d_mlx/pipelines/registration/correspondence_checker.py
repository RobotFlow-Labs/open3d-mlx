"""Correspondence checkers for registration.

Provides filters that validate correspondences based on geometric criteria
such as distance, edge length consistency, and normal angle agreement.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np


class CorrespondenceCheckerBasedOnDistance:
    """Filter correspondences by Euclidean distance threshold.

    Parameters
    ----------
    distance_threshold : float
        Maximum allowed distance between corresponding points.
    """

    def __init__(self, distance_threshold: float):
        if distance_threshold <= 0:
            raise ValueError(
                f"distance_threshold must be > 0, got {distance_threshold}"
            )
        self.distance_threshold = distance_threshold

    def check(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
        distances: mx.array,
    ) -> np.ndarray:
        """Return boolean mask of valid correspondences.

        Parameters
        ----------
        source_points : mx.array
            ``(N, 3)`` source points.
        target_points : mx.array
            ``(M, 3)`` target points.
        correspondences : mx.array
            ``(N,)`` int32 indices into target. ``-1`` = no match.
        distances : mx.array
            ``(N,)`` squared Euclidean distances.

        Returns
        -------
        np.ndarray
            ``(N,)`` boolean mask. ``True`` = valid correspondence.
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        dist_np = np.array(distances, dtype=np.float64)

        valid = corr_np >= 0
        # Compare squared distances against squared threshold
        thresh_sq = self.distance_threshold ** 2
        valid &= dist_np < thresh_sq

        return valid


class CorrespondenceCheckerBasedOnEdgeLength:
    """Filter correspondences by edge length similarity.

    For pairs of correspondences (i, j), checks that the ratio of edge
    lengths ||s_i - s_j|| / ||t_ci - t_cj|| is close to 1.

    Parameters
    ----------
    similarity_threshold : float
        Minimum ratio (must be in (0, 1]). An edge pair passes if
        ``min(len_s, len_t) / max(len_s, len_t) >= similarity_threshold``.
    """

    def __init__(self, similarity_threshold: float = 0.9):
        if not 0 < similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
            )
        self.similarity_threshold = similarity_threshold

    def check(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> np.ndarray:
        """Return boolean mask of correspondences that pass the edge length check.

        Checks a random sample of edge pairs for efficiency. A correspondence
        is marked invalid if more than half of its sampled edges fail.

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
        np.ndarray
            ``(N,)`` boolean mask.
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        src_np = np.array(source_points, dtype=np.float64)
        tgt_np = np.array(target_points, dtype=np.float64)

        N = len(corr_np)
        valid = corr_np >= 0
        inlier_indices = np.where(valid)[0]

        if len(inlier_indices) < 2:
            return valid

        # Check edge lengths between consecutive inlier pairs
        num_checks = min(len(inlier_indices) - 1, 100)
        fail_count = np.zeros(N, dtype=np.int32)
        check_count = np.zeros(N, dtype=np.int32)

        for k in range(num_checks):
            i = inlier_indices[k]
            j = inlier_indices[k + 1]

            src_edge = np.linalg.norm(src_np[i] - src_np[j])
            tgt_edge = np.linalg.norm(tgt_np[corr_np[i]] - tgt_np[corr_np[j]])

            max_edge = max(src_edge, tgt_edge)
            if max_edge < 1e-10:
                continue

            min_edge = min(src_edge, tgt_edge)
            ratio = min_edge / max_edge

            check_count[i] += 1
            check_count[j] += 1
            if ratio < self.similarity_threshold:
                fail_count[i] += 1
                fail_count[j] += 1

        # Mark as invalid if majority of checks fail
        for idx in inlier_indices:
            if check_count[idx] > 0 and fail_count[idx] > check_count[idx] / 2:
                valid[idx] = False

        return valid


class CorrespondenceCheckerBasedOnNormal:
    """Filter correspondences by normal angle agreement.

    Parameters
    ----------
    normal_angle_threshold : float
        Maximum allowed angle (in radians) between corresponding normals.
    """

    def __init__(self, normal_angle_threshold: float):
        if normal_angle_threshold <= 0:
            raise ValueError(
                f"normal_angle_threshold must be > 0, got {normal_angle_threshold}"
            )
        self.normal_angle_threshold = normal_angle_threshold

    def check(
        self,
        source_normals: mx.array,
        target_normals: mx.array,
        correspondences: mx.array,
    ) -> np.ndarray:
        """Return boolean mask of correspondences with compatible normals.

        Parameters
        ----------
        source_normals : mx.array
            ``(N, 3)`` source normals.
        target_normals : mx.array
            ``(M, 3)`` target normals.
        correspondences : mx.array
            ``(N,)`` int32 indices into target. ``-1`` = no match.

        Returns
        -------
        np.ndarray
            ``(N,)`` boolean mask.
        """
        corr_np = np.array(correspondences, dtype=np.int32)
        src_n = np.array(source_normals, dtype=np.float64)
        tgt_n = np.array(target_normals, dtype=np.float64)

        N = len(corr_np)
        valid = corr_np >= 0
        inlier_mask = valid.copy()

        if not inlier_mask.any():
            return valid

        # Compute dot products for inlier correspondences
        inlier_src = src_n[inlier_mask]
        inlier_tgt = tgt_n[corr_np[inlier_mask]]

        dots = np.sum(inlier_src * inlier_tgt, axis=1)
        # Clamp to [-1, 1] for numerical safety
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(np.abs(dots))  # Use abs for direction-agnostic normals

        # Update valid mask
        angle_valid = angles < self.normal_angle_threshold
        inlier_indices = np.where(inlier_mask)[0]
        for k, idx in enumerate(inlier_indices):
            if not angle_valid[k]:
                valid[idx] = False

        return valid
