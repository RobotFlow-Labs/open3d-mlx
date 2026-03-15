"""ICP convergence criteria.

Matches Open3D: o3d.t.pipelines.registration.ICPConvergenceCriteria
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ICPConvergenceCriteria:
    """Convergence criteria for the ICP loop.

    The ICP algorithm stops when *both* the change in fitness and change in
    RMSE fall below their respective thresholds, or when ``max_iteration``
    iterations have been completed.

    Parameters
    ----------
    relative_fitness : float
        Minimum change in fitness between consecutive iterations.
    relative_rmse : float
        Minimum change in inlier RMSE between consecutive iterations.
    max_iteration : int
        Maximum number of ICP iterations.
    """

    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    max_iteration: int = 30

    def __post_init__(self) -> None:
        if self.relative_fitness < 0:
            raise ValueError(
                f"relative_fitness must be >= 0, got {self.relative_fitness}"
            )
        if self.relative_rmse < 0:
            raise ValueError(
                f"relative_rmse must be >= 0, got {self.relative_rmse}"
            )
        if self.max_iteration < 1:
            raise ValueError(
                f"max_iteration must be >= 1, got {self.max_iteration}"
            )
