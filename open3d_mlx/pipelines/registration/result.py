"""ICP registration result type.

Matches Open3D: o3d.t.pipelines.registration.RegistrationResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class RegistrationResult:
    """Result of an ICP registration.

    Attributes
    ----------
    transformation : mx.array
        ``(4, 4)`` float32 rigid transformation (source -> target).
    fitness : float
        Fraction of source points that have a valid correspondence
        (``|inliers| / |source|``).
    inlier_rmse : float
        Root mean square error of inlier correspondences.
    correspondences : mx.array or None
        ``(N,)`` int32 array mapping each source point to a target index.
        ``-1`` indicates no correspondence within the maximum distance.
    num_iterations : int
        Number of ICP iterations actually executed.
    converged : bool
        Whether the algorithm converged before reaching ``max_iteration``.
    """

    transformation: mx.array = field(default_factory=lambda: mx.eye(4))
    fitness: float = 0.0
    inlier_rmse: float = float("inf")
    correspondences: Optional[mx.array] = None
    num_iterations: int = 0
    converged: bool = False

    def is_better_than(self, other: RegistrationResult) -> bool:
        """Compare two results: higher fitness wins; if tied, lower RMSE wins.

        Parameters
        ----------
        other : RegistrationResult
            The result to compare against.

        Returns
        -------
        bool
        """
        if self.fitness != other.fitness:
            return self.fitness > other.fitness
        return self.inlier_rmse < other.inlier_rmse

    def __repr__(self) -> str:
        return (
            f"RegistrationResult("
            f"fitness={self.fitness:.6f}, "
            f"inlier_rmse={self.inlier_rmse:.6f}, "
            f"num_iterations={self.num_iterations}, "
            f"converged={self.converged})"
        )
