"""Open3D-MLX: Apple Silicon-native 3D perception pipelines.

Modules
-------
core
    MLX tensor utilities, device abstraction, dtype helpers.
geometry
    PointCloud, KDTree search parameter types.
io
    Point cloud file I/O (PLY, PCD).
pipelines
    Registration (ICP), TSDF integration, raycasting.
camera
    Pinhole camera intrinsics.
interop
    Conversion helpers between Open3D-MLX and vanilla Open3D.
"""

from open3d_mlx._version import __version__
from open3d_mlx import core, geometry, io, pipelines, camera, interop

__all__ = [
    "__version__",
    "core",
    "geometry",
    "io",
    "pipelines",
    "camera",
    "interop",
]
