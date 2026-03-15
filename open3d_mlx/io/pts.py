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
        if colors.max() > 2.0:  # Clearly uint8 range, not edge-case floats
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
        if has_colors:
            colors_u8 = np.clip(data["colors"] * 255, 0, 255).astype(np.uint8)
            intensity = np.zeros((len(pts), 1), dtype=np.uint8)
            # Build: x y z intensity r g b
            lines = []
            float_fmt = "%.6f %.6f %.6f"
            for i in range(len(pts)):
                fpart = float_fmt % (pts[i, 0], pts[i, 1], pts[i, 2])
                lines.append(
                    f"{fpart} 0 {colors_u8[i, 0]} {colors_u8[i, 1]} {colors_u8[i, 2]}"
                )
            f.write("\n".join(lines) + "\n")
        else:
            import io as _io
            buf = _io.StringIO()
            np.savetxt(buf, pts, fmt="%.6f", delimiter=" ")
            f.write(buf.getvalue())
