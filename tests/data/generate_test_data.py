"""Generate static test data files for I/O tests.

Run this script once to populate tests/data/ with sample PLY and PCD files.
The test suite uses pytest fixtures (tmp_path) instead, so these files are
provided for manual inspection and cross-validation only.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

CUBE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)

CUBE_COLORS = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [128, 128, 128],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)


def write_cube_ascii_ply():
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for p in CUBE_POINTS:
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    (HERE / "cube_ascii.ply").write_text("\n".join(lines) + "\n")


def write_cube_binary_ply():
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]) + "\n"
    with open(HERE / "cube_binary.ply", "wb") as f:
        f.write(header.encode("ascii"))
        for p in CUBE_POINTS:
            f.write(struct.pack("<fff", p[0], p[1], p[2]))


def write_cube_colors_ply():
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for p, c in zip(CUBE_POINTS, CUBE_COLORS):
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}")
    (HERE / "cube_colors.ply").write_text("\n".join(lines) + "\n")


def write_cube_pcd():
    lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 8",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 8",
        "DATA ascii",
    ]
    for p in CUBE_POINTS:
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    (HERE / "cube.pcd").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_cube_ascii_ply()
    write_cube_binary_ply()
    write_cube_colors_ply()
    write_cube_pcd()
    print(f"Generated test data in {HERE}")
