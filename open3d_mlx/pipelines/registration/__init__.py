"""ICP registration pipeline.

Provides point-to-point and point-to-plane ICP registration, convergence
criteria, result types, transformation estimation methods, and robust kernels.
"""

from open3d_mlx.pipelines.registration.convergence import ICPConvergenceCriteria
from open3d_mlx.pipelines.registration.correspondence import find_correspondences
from open3d_mlx.pipelines.registration.icp import (
    evaluate_registration,
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
    TransformationEstimationPointToPlane,
    TransformationEstimationPointToPoint,
)

__all__ = [
    "CauchyLoss",
    "GMLoss",
    "HuberLoss",
    "ICPConvergenceCriteria",
    "L2Loss",
    "RegistrationResult",
    "RobustKernel",
    "TransformationEstimationPointToPlane",
    "TransformationEstimationPointToPoint",
    "TukeyLoss",
    "evaluate_registration",
    "find_correspondences",
    "registration_icp",
]
