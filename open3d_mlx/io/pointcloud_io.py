"""Top-level point cloud I/O dispatcher.

Auto-detects format from file extension and delegates to the
appropriate format-specific reader/writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from open3d_mlx.io.ply import read_ply, write_ply
from open3d_mlx.io.pcd import read_pcd, write_pcd

# ── Format detection ────────────────────────────────────────────────────────

_FORMAT_MAP = {
    ".ply": "ply",
    ".pcd": "pcd",
}


def _detect_format(filename: str | Path) -> str:
    """Detect point cloud format from file extension."""
    ext = Path(filename).suffix.lower()
    if ext not in _FORMAT_MAP:
        raise ValueError(
            f"Unsupported point cloud format: '{ext}'. "
            f"Supported extensions: {list(_FORMAT_MAP.keys())}"
        )
    return _FORMAT_MAP[ext]


# ── PointCloud integration helpers ──────────────────────────────────────────


def _try_import_pointcloud():
    """Try to import PointCloud class; return None if not available."""
    try:
        from open3d_mlx.geometry import PointCloud
        return PointCloud
    except (ImportError, AttributeError):
        return None


def _try_import_mlx():
    """Try to import mlx.core; return None if not available."""
    try:
        import mlx.core as mx
        return mx
    except ImportError:
        return None


def _dict_to_pointcloud(data: dict[str, np.ndarray]):
    """Convert data dict to PointCloud if the class is available, else return dict."""
    PointCloud = _try_import_pointcloud()
    mx = _try_import_mlx()

    if PointCloud is None or mx is None:
        # Normalize colors to float32 in [0,1] if uint8
        if "colors" in data and data["colors"].dtype == np.uint8:
            data["colors"] = data["colors"].astype(np.float32) / 255.0
        return data

    points = mx.array(data["points"].astype(np.float32))
    pcd = PointCloud(points)

    if "normals" in data:
        pcd.normals = mx.array(data["normals"].astype(np.float32))
    if "colors" in data:
        colors = data["colors"]
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = mx.array(colors.astype(np.float32))

    return pcd


def _pointcloud_to_dict(pointcloud) -> dict[str, np.ndarray]:
    """Convert a PointCloud (or dict) to a plain numpy dict."""
    if isinstance(pointcloud, dict):
        return pointcloud

    data: dict[str, np.ndarray] = {}
    data["points"] = np.array(pointcloud.points)

    if hasattr(pointcloud, "has_normals") and pointcloud.has_normals():
        data["normals"] = np.array(pointcloud.normals)
    elif hasattr(pointcloud, "normals") and pointcloud.normals is not None:
        data["normals"] = np.array(pointcloud.normals)

    if hasattr(pointcloud, "has_colors") and pointcloud.has_colors():
        data["colors"] = np.array(pointcloud.colors)
    elif hasattr(pointcloud, "colors") and pointcloud.colors is not None:
        data["colors"] = np.array(pointcloud.colors)

    return data


def _remove_non_finite(data: dict[str, np.ndarray], remove_nan: bool, remove_inf: bool) -> dict[str, np.ndarray]:
    """Remove points with NaN or Inf coordinates."""
    points = data["points"]
    mask = np.ones(len(points), dtype=bool)

    if remove_nan:
        mask &= ~np.any(np.isnan(points), axis=1)
    if remove_inf:
        mask &= ~np.any(np.isinf(points), axis=1)

    if np.all(mask):
        return data

    result: dict[str, np.ndarray] = {}
    result["points"] = points[mask]
    if "normals" in data:
        result["normals"] = data["normals"][mask]
    if "colors" in data:
        result["colors"] = data["colors"][mask]
    return result


# ── Public API ──────────────────────────────────────────────────────────────


def read_point_cloud(
    filename: str | Path,
    format: str = "auto",
    remove_nan_points: bool = False,
    remove_infinite_points: bool = False,
):
    """Read a point cloud from file.

    Args:
        filename: Path to .ply or .pcd file.
        format: ``"auto"`` (detect from extension), ``"ply"``, or ``"pcd"``.
        remove_nan_points: Remove vertices with NaN coordinates.
        remove_infinite_points: Remove vertices with Inf coordinates.

    Returns:
        A ``PointCloud`` object if the geometry module is available,
        otherwise a dict with ``"points"`` and optional ``"normals"``/``"colors"``.
    """
    filename = str(filename)
    fmt = format if format != "auto" else _detect_format(filename)

    if fmt == "ply":
        data = read_ply(filename)
    elif fmt == "pcd":
        data = read_pcd(filename)
    else:
        raise ValueError(f"Unknown point cloud format: {fmt}")

    if remove_nan_points or remove_infinite_points:
        data = _remove_non_finite(data, remove_nan_points, remove_infinite_points)

    return _dict_to_pointcloud(data)


def write_point_cloud(
    filename: str | Path,
    pointcloud,
    write_ascii: bool = False,
    compressed: bool = False,
) -> bool:
    """Write a point cloud to file.

    Args:
        filename: Output path. Format detected from extension.
        pointcloud: A ``PointCloud`` object or a dict with ``"points"``.
        write_ascii: If True, write ASCII format. Default is binary.
        compressed: Reserved for future use (compression support).

    Returns:
        True on success.
    """
    filename = str(filename)
    fmt = _detect_format(filename)
    data = _pointcloud_to_dict(pointcloud)

    if fmt == "ply":
        write_ply(filename, data, ascii=write_ascii)
    elif fmt == "pcd":
        write_pcd(filename, data, ascii=write_ascii)
    else:
        raise ValueError(f"Unknown point cloud format: {fmt}")

    return True
