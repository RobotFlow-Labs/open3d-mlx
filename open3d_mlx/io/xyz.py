"""XYZ point cloud format reader and writer.

Supports plain-text XYZ files where each line contains:
  x y z [nx ny nz] [r g b]
Space or tab delimited, no header.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_xyz(filepath: str | Path) -> dict[str, np.ndarray]:
    """Read an XYZ point cloud file.

    Each line: ``x y z [nx ny nz] [r g b]``.
    Space or tab delimited. No header.

    Args:
        filepath: Path to the ``.xyz`` file.

    Returns:
        Dictionary with keys:
        - ``"points"``: (N, 3) float32 array of xyz positions.
        - ``"normals"``: (N, 3) float32 array, if 6+ columns present.
        - ``"colors"``: (N, 3) float32 array in [0, 1], if 9 columns present.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XYZ file not found: {filepath}")

    data = np.loadtxt(filepath, dtype=np.float32)

    if data.ndim == 1:
        # Single point
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            f"XYZ file must have at least 3 columns (x y z), got {data.shape[1]}"
        )

    result: dict[str, np.ndarray] = {"points": data[:, :3]}

    if data.shape[1] >= 6:
        result["normals"] = data[:, 3:6]

    if data.shape[1] >= 9:
        colors = data[:, 6:9]
        if colors.max() > 2.0:  # Clearly uint8 range, not edge-case floats
            colors = colors / 255.0
        result["colors"] = colors.astype(np.float32)

    return result


def write_xyz(
    filepath: str | Path,
    data: dict[str, np.ndarray],
    include_normals: bool = True,
    include_colors: bool = True,
) -> None:
    """Write point cloud data to an XYZ file.

    One point per line: ``x y z [nx ny nz] [r g b]``.

    Args:
        filepath: Output file path.
        data: Dictionary with ``"points"`` (required), optionally
              ``"normals"`` and ``"colors"``.
        include_normals: If True and normals are present, write them.
        include_colors: If True and colors are present, write them.
    """
    filepath = Path(filepath)
    points = data["points"]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")

    cols = [points.astype(np.float32)]

    if include_normals and "normals" in data:
        cols.append(data["normals"].astype(np.float32))

    if include_colors and "colors" in data:
        cols.append(data["colors"].astype(np.float32))

    arr = np.hstack(cols)
    np.savetxt(filepath, arr, fmt="%.6f")
