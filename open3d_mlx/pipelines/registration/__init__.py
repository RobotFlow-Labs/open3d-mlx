"""ICP registration pipeline.

Provides point-to-point, point-to-plane, colored, and generalized ICP
registration, multi-scale ICP, convergence criteria, result types,
transformation estimation methods, correspondence checkers, and robust kernels.
"""

from open3d_mlx.pipelines.registration.convergence import ICPConvergenceCriteria
from open3d_mlx.pipelines.registration.correspondence import find_correspondences
from open3d_mlx.pipelines.registration.correspondence_checker import (
    CorrespondenceCheckerBasedOnDistance,
    CorrespondenceCheckerBasedOnEdgeLength,
    CorrespondenceCheckerBasedOnNormal,
)
from open3d_mlx.pipelines.registration.icp import (
    evaluate_registration,
    multi_scale_icp,
    registration_icp,
)
from open3d_mlx.pipelines.registration.result import RegistrationResult
from open3d_mlx.pipelines.registration.robust_kernel import (
    CauchyLoss,
    GMLoss,
    HuberLoss,
    L2Loss,
    RobustKernel,
    TukeyLoss,
)
from open3d_mlx.pipelines.registration.transformation import (
    TransformationEstimationForColoredICP,
    TransformationEstimationForGeneralizedICP,
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
    compute_point_covariances,
)
from open3d_mlx.pipelines.registration.feature import compute_fpfh_feature

__all__ = [
    "CauchyLoss",
    "CorrespondenceCheckerBasedOnDistance",
    "CorrespondenceCheckerBasedOnEdgeLength",
    "CorrespondenceCheckerBasedOnNormal",
    "GMLoss",
    "HuberLoss",
    "ICPConvergenceCriteria",
    "L2Loss",
    "RegistrationResult",
    "RobustKernel",
    "TransformationEstimationForColoredICP",
    "TransformationEstimationForGeneralizedICP",
    "TransformationEstimationPointToPlane",
    "TransformationEstimationPointToPoint",
    "TukeyLoss",
    "compute_fpfh_feature",
    "compute_point_covariances",
    "evaluate_registration",
    "find_correspondences",
    "multi_scale_icp",
    "registration_icp",
]
