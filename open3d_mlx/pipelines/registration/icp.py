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
from open3d_mlx.pipelines.registration.robust_kernel import RobustKernel
from open3d_mlx.pipelines.registration.transformation import (
    TransformationEstimationForColoredICP,
    TransformationEstimationForGeneralizedICP,
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
            TransformationEstimationForColoredICP,
            TransformationEstimationForGeneralizedICP,
        ]
    ] = None,
    criteria: Optional[ICPConvergenceCriteria] = None,
    voxel_size: float = -1.0,
    kernel: Optional[RobustKernel] = None,
    correspondence_checkers: Optional[list] = None,
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
    kernel : RobustKernel or None
        Robust kernel for outlier down-weighting in IRLS.  If ``None``,
        standard (unweighted) least-squares is used.
    correspondence_checkers : list or None
        List of correspondence checker instances.  After finding
        correspondences, each checker filters invalid ones.

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
    is_colored = isinstance(
        estimation_method, TransformationEstimationForColoredICP
    )
    is_gicp = isinstance(
        estimation_method, TransformationEstimationForGeneralizedICP
    )

    if is_colored:
        if not source.has_colors() or not target.has_colors():
            raise ValueError(
                "Colored ICP requires both source and target to have colors."
            )
        if not target.has_normals():
            raise ValueError(
                "Colored ICP requires target normals. "
                "Call target.estimate_normals() first."
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

    # Cache source points in numpy for incremental transformation
    source_pts_np = np.array(src.points, dtype=np.float64)

    # Build target index once (target does not move)
    target_index = FixedRadiusIndex(tgt.points, max_correspondence_distance)

    prev_fitness = 0.0
    prev_rmse = float("inf")
    converged = False
    num_inliers = 0
    fitness = 0.0
    rmse = float("inf")
    correspondences = None

    for i in range(criteria.max_iteration):
        # 1. Compute transformed source points incrementally
        R_cum = T_cumulative[:3, :3]
        t_cum = T_cumulative[:3, 3]
        transformed_pts = (source_pts_np @ R_cum.T + t_cum).astype(np.float32)
        transformed_pts_mx = mx.array(transformed_pts)

        # 2. Find correspondences
        correspondences, sq_dists = find_correspondences(
            transformed_pts_mx, target_index, max_correspondence_distance
        )

        # 3. Apply correspondence checkers
        if correspondence_checkers:
            corr_np_check = np.array(correspondences, dtype=np.int32)
            valid_mask = corr_np_check >= 0  # Start with inlier mask

            from open3d_mlx.pipelines.registration.correspondence_checker import (
                CorrespondenceCheckerBasedOnDistance,
                CorrespondenceCheckerBasedOnEdgeLength,
                CorrespondenceCheckerBasedOnNormal,
            )

            for checker in correspondence_checkers:
                if isinstance(checker, CorrespondenceCheckerBasedOnDistance):
                    checker_valid = checker.check(
                        transformed_pts_mx, tgt.points, correspondences, sq_dists
                    )
                elif isinstance(checker, CorrespondenceCheckerBasedOnEdgeLength):
                    checker_valid = checker.check(
                        transformed_pts_mx, tgt.points, correspondences
                    )
                elif isinstance(checker, CorrespondenceCheckerBasedOnNormal):
                    if src.has_normals() and tgt.has_normals():
                        # Rotate source normals by cumulative rotation
                        src_normals_np = np.array(src.normals, dtype=np.float64)
                        rotated_normals = (src_normals_np @ R_cum.T).astype(np.float32)
                        checker_valid = checker.check(
                            mx.array(rotated_normals), tgt.normals, correspondences
                        )
                    else:
                        # Skip normal check if normals not available
                        checker_valid = valid_mask
                else:
                    continue

                valid_mask &= checker_valid

            # Update correspondences: set filtered ones to -1
            new_corr = corr_np_check.copy()
            new_corr[~valid_mask] = -1
            correspondences = mx.array(new_corr)

            # Update sq_dists for filtered correspondences
            sq_dists_np = np.array(sq_dists, dtype=np.float32)
            sq_dists_np[~valid_mask] = float("inf")
            sq_dists = mx.array(sq_dists_np)

        # 4. Compute metrics
        corr_np = np.array(correspondences, dtype=np.int32)
        inlier_mask = corr_np >= 0
        num_inliers = int(inlier_mask.sum())
        fitness = num_inliers / n_source

        if num_inliers == 0:
            break

        sq_dists_np = np.array(sq_dists, dtype=np.float64)
        inlier_sq_dists = sq_dists_np[inlier_mask]
        rmse = float(np.sqrt(inlier_sq_dists.mean()))

        # 5. Check convergence (skip first iteration -- no previous to compare)
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

        # 6. Compute robust kernel weights if provided
        weights_np = None
        if kernel is not None:
            # Compute distances (not squared) for the residuals
            distances_inlier = np.sqrt(inlier_sq_dists)
            weights_mx = kernel.weight(mx.array(distances_inlier.astype(np.float32)))
            mx.eval(weights_mx)
            weights_np = np.array(weights_mx, dtype=np.float64)

        # 7. Estimate incremental transformation
        if is_colored:
            # For colored ICP, need to get colors for transformed source
            T_step = estimation_method.compute_transformation(
                transformed_pts_mx,
                tgt.points,
                src.colors,
                tgt.colors,
                tgt.normals,
                correspondences,
                weights=weights_np,
            )
        elif is_point_to_plane:
            T_step = estimation_method.compute_transformation(
                transformed_pts_mx,
                tgt.points,
                tgt.normals,
                correspondences,
                weights=weights_np,
            )
        elif is_gicp:
            T_step = estimation_method.compute_transformation(
                transformed_pts_mx,
                tgt.points,
                correspondences,
                weights=weights_np,
            )
        else:
            T_step = estimation_method.compute_transformation(
                transformed_pts_mx,
                tgt.points,
                correspondences,
                weights=weights_np,
            )

        # 8. Accumulate in float64
        T_step_np = np.array(T_step, dtype=np.float64)
        T_cumulative = T_step_np @ T_cumulative

    # Final evaluation after the loop
    if correspondences is not None and num_inliers > 0:
        # Compute final transformed points
        R_cum = T_cumulative[:3, :3]
        t_cum = T_cumulative[:3, 3]
        final_pts = (source_pts_np @ R_cum.T + t_cum).astype(np.float32)
        final_pts_mx = mx.array(final_pts)

        # Re-evaluate correspondences with the final transformed source
        correspondences, sq_dists = find_correspondences(
            final_pts_mx, target_index, max_correspondence_distance
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


def multi_scale_icp(
    source: PointCloud,
    target: PointCloud,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    criteria_list: Optional[list[ICPConvergenceCriteria]] = None,
    init_source_to_target: Optional[mx.array] = None,
    estimation_method: Optional[
        Union[
            TransformationEstimationPointToPoint,
            TransformationEstimationPointToPlane,
            TransformationEstimationForColoredICP,
            TransformationEstimationForGeneralizedICP,
        ]
    ] = None,
    kernel: Optional[RobustKernel] = None,
) -> RegistrationResult:
    """Coarse-to-fine multi-scale ICP registration.

    Runs ICP at each scale (from coarse to fine), using the result of each
    scale as the initial transformation for the next finer scale.

    Parameters
    ----------
    source : PointCloud
        Source point cloud.
    target : PointCloud
        Target point cloud.
    voxel_sizes : list[float]
        Voxel sizes for each scale level (coarse to fine).
    max_correspondence_distances : list[float]
        Max correspondence distance at each scale.
    criteria_list : list[ICPConvergenceCriteria] or None
        Convergence criteria per scale.  If ``None``, defaults are used.
    init_source_to_target : mx.array or None
        Initial ``(4, 4)`` transformation.  Defaults to identity.
    estimation_method : estimation class or None
        Transformation estimation method.  Defaults to point-to-point.
    kernel : RobustKernel or None
        Robust kernel for outlier down-weighting.

    Returns
    -------
    RegistrationResult
        Result from the finest scale.

    Raises
    ------
    ValueError
        If ``voxel_sizes`` and ``max_correspondence_distances`` have
        different lengths.
    """
    if len(voxel_sizes) != len(max_correspondence_distances):
        raise ValueError(
            f"voxel_sizes ({len(voxel_sizes)}) and "
            f"max_correspondence_distances ({len(max_correspondence_distances)}) "
            f"must have the same length."
        )

    if estimation_method is None:
        estimation_method = TransformationEstimationPointToPoint()

    needs_normals = isinstance(
        estimation_method,
        (TransformationEstimationPointToPlane, TransformationEstimationForColoredICP),
    )

    T = init_source_to_target if init_source_to_target is not None else mx.eye(4, dtype=mx.float32)
    result = None

    for i in range(len(voxel_sizes)):
        vs = voxel_sizes[i]
        max_dist = max_correspondence_distances[i]
        crit = criteria_list[i] if criteria_list is not None else None

        # Downsample
        src_down = source.voxel_down_sample(vs)
        tgt_down = target.voxel_down_sample(vs)

        # Estimate normals if needed
        if needs_normals:
            if not tgt_down.has_normals():
                tgt_down.estimate_normals()
            if not src_down.has_normals():
                src_down.estimate_normals()

        result = registration_icp(
            src_down,
            tgt_down,
            max_correspondence_distance=max_dist,
            init_source_to_target=T,
            estimation_method=estimation_method,
            criteria=crit,
            kernel=kernel,
        )
        T = result.transformation

    # If no scales were provided, return identity result
    if result is None:
        return RegistrationResult(
            transformation=T,
            fitness=0.0,
            inlier_rmse=float("inf"),
            correspondences=None,
            num_iterations=0,
            converged=False,
        )

    return result
