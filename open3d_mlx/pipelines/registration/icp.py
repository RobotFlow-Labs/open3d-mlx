"""ICP (Iterative Closest Point) registration.

Implements the main ICP loop for both point-to-point and point-to-plane
variants.  The correspondence search uses FixedRadiusIndex for GPU-friendly
spatial hashing, and transformation accumulation uses numpy float64 for
numerical stability.

Matches Open3D: o3d.t.pipelines.registration.icp()
"""

from __future__ import annotations

import math
from typing import Optional, Union

import mlx.core as mx
import numpy as np

from open3d_mlx.geometry import PointCloud
from open3d_mlx.ops.fixed_radius_nn import FixedRadiusIndex
from open3d_mlx.pipelines.registration.convergence import ICPConvergenceCriteria
from open3d_mlx.pipelines.registration.correspondence import find_correspondences
from open3d_mlx.pipelines.registration.result import RegistrationResult
from open3d_mlx.pipelines.registration.transformation import (
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
)


def registration_icp(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    init_source_to_target: Optional[mx.array] = None,
    estimation_method: Optional[
        Union[
            TransformationEstimationPointToPoint,
            TransformationEstimationPointToPlane,
        ]
    ] = None,
    criteria: Optional[ICPConvergenceCriteria] = None,
    voxel_size: float = -1.0,
) -> RegistrationResult:
    """ICP registration: align source to target via iterative closest point.

    Parameters
    ----------
    source : PointCloud
        Source point cloud to be aligned.
    target : PointCloud
        Target (reference) point cloud.
    max_correspondence_distance : float
        Maximum distance for a correspondence to be considered valid.
    init_source_to_target : mx.array or None
        Initial ``(4, 4)`` transformation guess.  Defaults to identity.
    estimation_method : estimation class or None
        Transformation estimation method.  Defaults to
        ``TransformationEstimationPointToPoint()``.
    criteria : ICPConvergenceCriteria or None
        Convergence criteria.  Defaults to ``ICPConvergenceCriteria()``.
    voxel_size : float
        If > 0, downsample both clouds before registration.

    Returns
    -------
    RegistrationResult
        Contains the final transformation, fitness, RMSE, and metadata.

    Raises
    ------
    ValueError
        If point-to-plane estimation is used but target has no normals.
    """
    if criteria is None:
        criteria = ICPConvergenceCriteria()
    if estimation_method is None:
        estimation_method = TransformationEstimationPointToPoint()

    is_point_to_plane = isinstance(
        estimation_method, TransformationEstimationPointToPlane
    )

    if is_point_to_plane and not target.has_normals():
        raise ValueError(
            "Point-to-plane ICP requires target normals. "
            "Call target.estimate_normals() first."
        )

    # Handle empty clouds
    if source.is_empty() or target.is_empty():
        init_T = mx.eye(4, dtype=mx.float32)
        if init_source_to_target is not None:
            init_T = init_source_to_target
        return RegistrationResult(
            transformation=init_T,
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondences=None,
            num_iterations=0,
            converged=False,
        )

    # Cumulative transformation in float64 for precision
    if init_source_to_target is not None:
        T_cumulative = np.array(init_source_to_target, dtype=np.float64)
    else:
        T_cumulative = np.eye(4, dtype=np.float64)

    # Optional downsampling
    src = source
    tgt = target
    if voxel_size > 0:
        src = source.voxel_down_sample(voxel_size)
        tgt = target.voxel_down_sample(voxel_size)

    n_source = len(src)
    if n_source == 0:
        return RegistrationResult(
            transformation=mx.array(T_cumulative.astype(np.float32)),
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondences=None,
            num_iterations=0,
            converged=False,
        )

    # Apply initial transformation to source
    source_transformed = src.clone().transform(
        mx.array(T_cumulative.astype(np.float32))
    )

    # Build target index once (target does not move)
    target_index = FixedRadiusIndex(tgt.points, max_correspondence_distance)

    prev_fitness = 0.0
    prev_rmse = float("inf")
    converged = False
    fitness = 0.0
    rmse = float("inf")
    correspondences = None

    for i in range(criteria.max_iteration):
        # 1. Find correspondences
        correspondences, sq_dists = find_correspondences(
            source_transformed.points, target_index, max_correspondence_distance
        )

        # 2. Compute metrics
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0
        num_inliers = int(inlier_mask.sum())
        fitness = num_inliers / n_source

        if num_inliers == 0:
            break

        sq_dists_np = np.array(sq_dists, dtype=np.float64)
        inlier_sq_dists = sq_dists_np[inlier_mask]
        rmse = float(np.sqrt(inlier_sq_dists.mean()))

        # 3. Check convergence (skip first iteration — no previous to compare)
        if i > 0:
            delta_fitness = abs(fitness - prev_fitness)
            delta_rmse = abs(rmse - prev_rmse)
            if (
                delta_fitness < criteria.relative_fitness
                and delta_rmse < criteria.relative_rmse
            ):
                converged = True
                break

        prev_fitness = fitness
        prev_rmse = rmse

        # 4. Estimate incremental transformation
        if is_point_to_plane:
            T_step = estimation_method.compute_transformation(
                source_transformed.points,
                tgt.points,
                tgt.normals,
                correspondences,
            )
        else:
            T_step = estimation_method.compute_transformation(
                source_transformed.points,
                tgt.points,
                correspondences,
            )

        # 5. Apply step and accumulate in float64
        T_step_np = np.array(T_step, dtype=np.float64)
        T_cumulative = T_step_np @ T_cumulative

        # Re-transform original source by cumulative transform
        source_transformed = src.clone().transform(
            mx.array(T_cumulative.astype(np.float32))
        )

    # Final evaluation after the loop
    if correspondences is not None and num_inliers > 0:
        # Re-evaluate correspondences with the final transformed source
        correspondences, sq_dists = find_correspondences(
            source_transformed.points, target_index, max_correspondence_distance
        )
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0
        num_inliers = int(inlier_mask.sum())
        fitness = num_inliers / n_source
        if num_inliers > 0:
            sq_dists_np = np.array(sq_dists, dtype=np.float64)
            inlier_sq_dists = sq_dists_np[inlier_mask]
            rmse = float(np.sqrt(inlier_sq_dists.mean()))
        else:
            rmse = float("inf")

    result_T = mx.array(T_cumulative.astype(np.float32))
    mx.eval(result_T)

    return RegistrationResult(
        transformation=result_T,
        fitness=fitness,
        inlier_rmse=rmse,
        correspondences=correspondences,
        num_iterations=i + 1 if criteria.max_iteration > 0 else 0,
        converged=converged,
    )


def evaluate_registration(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    transformation: Optional[mx.array] = None,
) -> RegistrationResult:
    """Evaluate registration quality without iterating.

    Applies the transformation (if given), finds correspondences, and
    computes fitness and RMSE metrics.

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.
    max_correspondence_distance : float
        Maximum correspondence distance.
    transformation : mx.array or None
        ``(4, 4)`` transformation to apply to source.  Defaults to identity.

    Returns
    -------
    RegistrationResult
        Metrics (no iteration is performed).
    """
    if transformation is None:
        T = mx.eye(4, dtype=mx.float32)
    else:
        T = transformation

    # Handle empty clouds
    if source.is_empty() or target.is_empty():
        return RegistrationResult(
            transformation=T,
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondences=None,
            num_iterations=0,
            converged=False,
        )

    # Transform source
    source_transformed = source.clone().transform(T)
    n_source = len(source)

    # Build target index and find correspondences
    target_index = FixedRadiusIndex(target.points, max_correspondence_distance)
    correspondences, sq_dists = find_correspondences(
        source_transformed.points, target_index, max_correspondence_distance
    )

    # Compute metrics
    corr_np = np.array(correspondences, dtype=np.int32)
    inlier_mask = corr_np >= 0
    num_inliers = int(inlier_mask.sum())

    if num_inliers == 0:
        return RegistrationResult(
            transformation=T,
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondences=correspondences,
            num_iterations=0,
            converged=False,
        )

    fitness = num_inliers / n_source
    sq_dists_np = np.array(sq_dists, dtype=np.float64)
    inlier_sq_dists = sq_dists_np[inlier_mask]
    rmse = float(np.sqrt(inlier_sq_dists.mean()))

    return RegistrationResult(
        transformation=T,
        fitness=fitness,
        inlier_rmse=rmse,
        correspondences=correspondences,
        num_iterations=0,
        converged=False,
    )
