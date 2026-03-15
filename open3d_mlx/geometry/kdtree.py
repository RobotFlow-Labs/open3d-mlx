"""KDTree search parameter classes for Open3D-MLX.

These are lightweight parameter containers used to specify search strategies
(KNN, radius, hybrid) throughout the library. They mirror the upstream
Open3D API for compatibility.

Actual KDTree search implementation is deferred to PRD-04 (ops).
"""

from __future__ import annotations


class KDTreeSearchParamKNN:
    """K-nearest-neighbour search parameter.

    Parameters
    ----------
    knn : int
        Number of neighbours to search for. Default 30.
    """

    def __init__(self, knn: int = 30) -> None:
        if knn < 1:
            raise ValueError(f"knn must be >= 1, got {knn}")
        self.knn = knn

    def __repr__(self) -> str:
        return f"KDTreeSearchParamKNN(knn={self.knn})"


class KDTreeSearchParamRadius:
    """Radius search parameter.

    Parameters
    ----------
    radius : float
        Search radius. Must be positive.
    """

    def __init__(self, radius: float) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        self.radius = radius

    def __repr__(self) -> str:
        return f"KDTreeSearchParamRadius(radius={self.radius})"


class KDTreeSearchParamHybrid:
    """Hybrid KNN + radius search parameter.

    Searches for up to *max_nn* neighbours within *radius*.

    Parameters
    ----------
    radius : float
        Search radius. Must be positive.
    max_nn : int
        Maximum number of neighbours. Default 30.
    """

    def __init__(self, radius: float, max_nn: int = 30) -> None:
        if radius <= 0:
            raise ValueError(f"radius must be > 0, got {radius}")
        if max_nn < 1:
            raise ValueError(f"max_nn must be >= 1, got {max_nn}")
        self.radius = radius
        self.max_nn = max_nn

    def __repr__(self) -> str:
        return (
            f"KDTreeSearchParamHybrid(radius={self.radius}, "
            f"max_nn={self.max_nn})"
        )
