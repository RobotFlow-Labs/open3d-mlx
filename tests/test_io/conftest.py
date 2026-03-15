"""Fixtures that generate test PLY and PCD files for I/O tests."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

# ── Shared cube data ────────────────────────────────────────────────────────

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

CUBE_NORMALS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)
# Normalise for realism
CUBE_NORMALS = CUBE_NORMALS / np.linalg.norm(CUBE_NORMALS, axis=1, keepdims=True)

CUBE_COLORS_UINT8 = np.array(
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


# ── PLY file generators ────────────────────────────────────────────────────


def _write_cube_ascii_ply(path: Path) -> Path:
    """Write an 8-vertex ASCII PLY with positions only."""
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
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_cube_binary_ply(path: Path) -> Path:
    """Write an 8-vertex binary_little_endian PLY with positions only."""
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]) + "\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p in CUBE_POINTS:
            f.write(struct.pack("<fff", p[0], p[1], p[2]))
    return path


def _write_cube_colors_ply(path: Path) -> Path:
    """Write an 8-vertex ASCII PLY with uint8 RGB colors."""
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
    for p, c in zip(CUBE_POINTS, CUBE_COLORS_UINT8):
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_cube_normals_ply(path: Path) -> Path:
    """Write an 8-vertex ASCII PLY with normals."""
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "end_header",
    ]
    for p, n in zip(CUBE_POINTS, CUBE_NORMALS):
        lines.append(
            f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
            f"{n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_cube_binary_big_endian_ply(path: Path) -> Path:
    """Write an 8-vertex binary_big_endian PLY with positions only."""
    header = "\n".join([
        "ply",
        "format binary_big_endian 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]) + "\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p in CUBE_POINTS:
            f.write(struct.pack(">fff", p[0], p[1], p[2]))
    return path


def _write_cube_float_colors_ply(path: Path) -> Path:
    """Write an 8-vertex ASCII PLY with float RGB colors."""
    colors_float = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 8",
        "property float x",
        "property float y",
        "property float z",
        "property float red",
        "property float green",
        "property float blue",
        "end_header",
    ]
    for p, c in zip(CUBE_POINTS, colors_float):
        lines.append(
            f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
            f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


# ── PCD file generators ────────────────────────────────────────────────────


def _write_cube_ascii_pcd(path: Path) -> Path:
    """Write an 8-vertex ASCII PCD with positions only."""
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
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_cube_binary_pcd(path: Path) -> Path:
    """Write an 8-vertex binary PCD with positions only."""
    header = "\n".join([
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
        "DATA binary",
    ]) + "\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for p in CUBE_POINTS:
            f.write(struct.pack("<fff", p[0], p[1], p[2]))
    return path


def _write_cube_rgb_pcd(path: Path) -> Path:
    """Write an 8-vertex binary PCD with float-packed RGB."""
    from open3d_mlx.io.pcd import _pack_pcd_rgb

    rgb_packed = _pack_pcd_rgb(CUBE_COLORS_UINT8)

    header = "\n".join([
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z rgb",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        "WIDTH 8",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 8",
        "DATA binary",
    ]) + "\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(8):
            p = CUBE_POINTS[i]
            f.write(struct.pack("<fff", p[0], p[1], p[2]))
            f.write(struct.pack("<f", rgb_packed[i]))
    return path


# ── Pytest fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def data_dir(tmp_path):
    """Temporary directory for test data files."""
    return tmp_path


@pytest.fixture
def cube_ascii_ply(data_dir):
    return _write_cube_ascii_ply(data_dir / "cube_ascii.ply")


@pytest.fixture
def cube_binary_ply(data_dir):
    return _write_cube_binary_ply(data_dir / "cube_binary.ply")


@pytest.fixture
def cube_colors_ply(data_dir):
    return _write_cube_colors_ply(data_dir / "cube_colors.ply")


@pytest.fixture
def cube_normals_ply(data_dir):
    return _write_cube_normals_ply(data_dir / "cube_normals.ply")


@pytest.fixture
def cube_big_endian_ply(data_dir):
    return _write_cube_binary_big_endian_ply(data_dir / "cube_big_endian.ply")


@pytest.fixture
def cube_float_colors_ply(data_dir):
    return _write_cube_float_colors_ply(data_dir / "cube_float_colors.ply")


@pytest.fixture
def cube_ascii_pcd(data_dir):
    return _write_cube_ascii_pcd(data_dir / "cube.pcd")


@pytest.fixture
def cube_binary_pcd(data_dir):
    return _write_cube_binary_pcd(data_dir / "cube_binary.pcd")


@pytest.fixture
def cube_rgb_pcd(data_dir):
    return _write_cube_rgb_pcd(data_dir / "cube_rgb.pcd")
