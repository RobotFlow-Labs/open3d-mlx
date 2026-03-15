"""Robust loss kernels for weighted ICP.

Each kernel computes per-residual weights that down-weight outliers in the
iteratively re-weighted least-squares (IRLS) loop used by ICP.

Matches Open3D: ``open3d.t.pipelines.registration.robust_kernel``

Upstream reference
------------------
- ``cpp/open3d/pipelines/registration/RobustKernel.h``
- ``cpp/open3d/pipelines/registration/RobustKernelImpl.h``
"""

from __future__ import annotations

import mlx.core as mx


class RobustKernel:
    """Base class for robust loss functions used in ICP.

    Subclasses must implement :meth:`weight` which returns a per-element
    weight for each residual.  A weight of 1 means "keep as-is" (L2);
    a weight of 0 means "fully reject this correspondence".
    """

    def weight(self, residual: mx.array) -> mx.array:
        """Compute IRLS weight for each residual value.

        Parameters
        ----------
        residual : mx.array
            Residual values, any shape.

        Returns
        -------
        mx.array
            Weights with the same shape as *residual*, values in [0, 1].
        """
        raise NotImplementedError


class L2Loss(RobustKernel):
    """Standard L2 (least-squares) loss.

    Weight is 1 everywhere — no outlier rejection.
    """

    def weight(self, residual: mx.array) -> mx.array:
        return mx.ones_like(residual)


class HuberLoss(RobustKernel):
    """Huber loss: L2 inside ``[-k, k]``, L1 outside.

    Parameters
    ----------
    k : float
        Transition threshold between L2 and L1 regions.
        Default is 1.345 (95 % efficiency under Gaussian noise).
    """

    def __init__(self, k: float = 1.345) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def weight(self, residual: mx.array) -> mx.array:
        abs_r = mx.abs(residual)
        # Avoid division by zero: clamp denominator
        safe_abs = mx.maximum(abs_r, mx.array(1e-10))
        return mx.where(abs_r <= self.k, mx.ones_like(abs_r), self.k / safe_abs)


class TukeyLoss(RobustKernel):
    """Tukey's biweight loss: zero weight for ``|r| > k``.

    Parameters
    ----------
    k : float
        Rejection threshold.  Default is 4.685
        (95 % efficiency under Gaussian noise).
    """

    def __init__(self, k: float = 4.685) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def weight(self, residual: mx.array) -> mx.array:
        abs_r = mx.abs(residual)
        ratio = abs_r / self.k
        inside = (1.0 - ratio * ratio) ** 2
        return mx.where(abs_r <= self.k, inside, mx.zeros_like(abs_r))


class CauchyLoss(RobustKernel):
    """Cauchy (Lorentzian) loss.

    Weight decreases smoothly but never reaches zero.

    Parameters
    ----------
    k : float
        Scale parameter.  Default is 1.0.
    """

    def __init__(self, k: float = 1.0) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def weight(self, residual: mx.array) -> mx.array:
        return 1.0 / (1.0 + (residual / self.k) ** 2)


class GMLoss(RobustKernel):
    """Geman-McClure loss.

    Strongly suppresses large residuals; weight decreases faster than Cauchy.

    Parameters
    ----------
    k : float
        Scale parameter.  Default is 1.0.
    """

    def __init__(self, k: float = 1.0) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def weight(self, residual: mx.array) -> mx.array:
        denom = self.k + residual ** 2
        return (self.k / denom) ** 2
