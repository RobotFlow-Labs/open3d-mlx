"""MLX operations for 3D processing.

Provides nearest neighbor search, normal estimation, and linear algebra
utilities for point cloud processing on Apple Silicon.
"""

from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch
from open3d_mlx.ops.fixed_radius_nn import FixedRadiusIndex
from open3d_mlx.ops.normals import (
    estimate_normals_pca,
    estimate_normals_pca_batched,
    orient_normals_towards_viewpoint,
)
from open3d_mlx.ops.linalg import (
    compute_cross_product,
    batched_svd,
    batched_solve,
    symmetric_eigendecomposition,
)

__all__ = [
    "NearestNeighborSearch",
    "FixedRadiusIndex",
    "estimate_normals_pca",
    "estimate_normals_pca_batched",
    "orient_normals_towards_viewpoint",
    "compute_cross_product",
    "batched_svd",
    "batched_solve",
    "symmetric_eigendecomposition",
]
