"""PTS point cloud format reader and writer.

PTS files start with a point count on the first line, followed by
point data lines: ``x y z [intensity] [r g b]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_pts(filepath: str | Path) -> dict[str, np.ndarray]:
    """Read a PTS point cloud file.

    First line: point count.  Then ``x y z [intensity] [r g b]`` per line.

    Args:
        filepath: Path to the ``.pts`` file.

    Returns:
        Dictionary with keys:
        - ``"points"``: (N, 3) float32 array of xyz positions.
        - ``"colors"``: (N, 3) float32 array in [0, 1], if 7 columns present.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PTS file not found: {filepath}")

    with open(filepath) as f:
        first_line = f.readline().strip()
        n = int(first_line)

    data = np.loadtxt(filepath, skiprows=1, dtype=np.float32, max_rows=n)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            f"PTS file must have at least 3 columns (x y z), got {data.shape[1]}"
        )

    result: dict[str, np.ndarray] = {"points": data[:, :3]}

    # Format: x y z intensity r g b  (7 columns)
    if data.shape[1] >= 7:
        colors = data[:, 4:7]
        if colors.max() > 1.0:
            colors = colors / 255.0
        result["colors"] = colors.astype(np.float32)

    return result


def write_pts(
    filepath: str | Path,
    data: dict[str, np.ndarray],
) -> None:
    """Write point cloud data to a PTS file.

    First line: point count.  Then ``x y z [intensity r g b]`` per line.

    Args:
        filepath: Output file path.
        data: Dictionary with ``"points"`` (required), optionally ``"colors"``.
    """
    filepath = Path(filepath)
    pts = data["points"]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")

    pts = pts.astype(np.float32)
    has_colors = "colors" in data

    with open(filepath, "w") as f:
        f.write(f"{len(pts)}\n")
        for i in range(len(pts)):
            line = f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f}"
            if has_colors:
                c = (data["colors"][i] * 255).astype(np.uint8)
                line += f" 0 {c[0]} {c[1]} {c[2]}"
            f.write(line + "\n")
