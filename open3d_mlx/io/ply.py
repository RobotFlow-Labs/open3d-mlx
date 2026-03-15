"""PLY (Polygon File Format) reader and writer.

Supports ASCII, binary_little_endian, and binary_big_endian formats.
Reads vertex properties: positions (x,y,z), normals (nx,ny,nz), colors (red,green,blue).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

# ── PLY type → numpy dtype / struct char mapping ────────────────────────────

_PLY_TYPE_TO_NUMPY = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
    # Aliases used by some PLY writers
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "float32": np.float32,
    "float64": np.float64,
}

_NUMPY_DTYPE_TO_PLY = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
}

_PLY_TYPE_TO_STRUCT = {
    "char": "b",
    "uchar": "B",
    "short": "h",
    "ushort": "H",
    "int": "i",
    "uint": "I",
    "float": "f",
    "double": "d",
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "float32": "f",
    "float64": "d",
}

# ── Property → semantic mapping ─────────────────────────────────────────────

_POSITION_NAMES = {"x", "y", "z"}
_NORMAL_NAMES = {"nx", "ny", "nz"}
_COLOR_NAMES = {"red", "green", "blue"}
_ALT_COLOR_NAMES = {"diffuse_red", "diffuse_green", "diffuse_blue"}


def _parse_header(f: BinaryIO) -> tuple[str, int, list[tuple[str, str]]]:
    """Parse PLY header.

    Returns:
        (format_str, vertex_count, [(property_name, ply_type), ...])
    """
    magic = f.readline().strip()
    if magic != b"ply":
        raise ValueError(f"Not a PLY file: magic line is {magic!r}")

    fmt = None
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex_element = False

    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        line_str = line.decode("ascii", errors="replace").strip()

        if line_str == "end_header":
            break

        parts = line_str.split()
        if not parts:
            continue

        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            if parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex_element = True
            else:
                in_vertex_element = False
        elif parts[0] == "property" and in_vertex_element:
            if parts[1] == "list":
                # Skip list properties (e.g. face vertex_indices)
                continue
            prop_type = parts[1]
            prop_name = parts[2]
            properties.append((prop_name, prop_type))
        elif parts[0] == "comment":
            continue

    if fmt is None:
        raise ValueError("PLY header missing format declaration")

    return fmt, vertex_count, properties


def _read_ascii(
    f: BinaryIO,
    vertex_count: int,
    properties: list[tuple[str, str]],
) -> dict[str, np.ndarray]:
    """Read ASCII PLY vertex data."""
    import io as _io

    n_props = len(properties)
    remaining = f.read().decode("ascii")
    lines = remaining.splitlines()[:vertex_count]
    if len(lines) < vertex_count:
        raise ValueError(
            f"PLY ASCII: expected {vertex_count} vertex lines, got {len(lines)}"
        )
    raw = np.loadtxt(_io.StringIO("\n".join(lines)), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < n_props:
        raise ValueError(
            f"PLY ASCII: expected {n_props} columns, got {raw.shape[1]}"
        )
    raw = raw[:, :n_props]

    return _extract_fields(raw, properties)


def _read_binary(
    f: BinaryIO,
    vertex_count: int,
    properties: list[tuple[str, str]],
    byte_order: str,
) -> dict[str, np.ndarray]:
    """Read binary PLY vertex data."""
    bo_char = "<" if byte_order == "binary_little_endian" else ">"
    struct_chars = [_PLY_TYPE_TO_STRUCT[pt] for _, pt in properties]
    fmt_str = bo_char + "".join(struct_chars)
    vertex_size = struct.calcsize(fmt_str)

    raw_bytes = f.read(vertex_count * vertex_size)
    if len(raw_bytes) < vertex_count * vertex_size:
        raise ValueError(
            f"PLY binary: expected {vertex_count * vertex_size} bytes, "
            f"got {len(raw_bytes)}"
        )

    # Build a numpy structured dtype for efficient parsing
    np_bo = "<" if byte_order == "binary_little_endian" else ">"
    np_dtype = np.dtype(
        [(name, np.dtype(f"{np_bo}{_PLY_TYPE_TO_STRUCT[pt]}").str) for name, pt in properties]
    )
    structured = np.frombuffer(raw_bytes, dtype=np_dtype, count=vertex_count)

    # Convert to a float64 2D array for uniform handling
    n_props = len(properties)
    raw = np.empty((vertex_count, n_props), dtype=np.float64)
    for j, (name, _pt) in enumerate(properties):
        raw[:, j] = structured[name].astype(np.float64)

    return _extract_fields(raw, properties)


def _extract_fields(
    raw: np.ndarray,
    properties: list[tuple[str, str]],
) -> dict[str, np.ndarray]:
    """Extract semantic fields (points, normals, colors) from raw data array."""
    prop_index = {name: i for i, (name, _) in enumerate(properties)}
    prop_types = {name: ptype for name, ptype in properties}
    result: dict[str, np.ndarray] = {}

    # Points (required)
    if "x" in prop_index and "y" in prop_index and "z" in prop_index:
        result["points"] = np.column_stack([
            raw[:, prop_index["x"]],
            raw[:, prop_index["y"]],
            raw[:, prop_index["z"]],
        ]).astype(np.float32)

    # Normals
    if "nx" in prop_index and "ny" in prop_index and "nz" in prop_index:
        result["normals"] = np.column_stack([
            raw[:, prop_index["nx"]],
            raw[:, prop_index["ny"]],
            raw[:, prop_index["nz"]],
        ]).astype(np.float32)

    # Colors — try standard names first, then diffuse_* aliases
    color_keys = None
    for r_name, g_name, b_name in [
        ("red", "green", "blue"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ]:
        if r_name in prop_index and g_name in prop_index and b_name in prop_index:
            color_keys = (r_name, g_name, b_name)
            break

    if color_keys is not None:
        r_name, g_name, b_name = color_keys
        colors_raw = np.column_stack([
            raw[:, prop_index[r_name]],
            raw[:, prop_index[g_name]],
            raw[:, prop_index[b_name]],
        ])
        # Determine if colors are uint8 or float based on property type
        r_type = prop_types[r_name]
        if r_type in ("uchar", "uint8"):
            # Return as uint8 — caller decides normalization
            result["colors"] = colors_raw.astype(np.uint8)
        else:
            result["colors"] = colors_raw.astype(np.float32)

    return result


# ── Public API ──────────────────────────────────────────────────────────────


def read_ply(filepath: str | Path) -> dict[str, np.ndarray]:
    """Read a PLY file and return a dict of numpy arrays.

    Supports ASCII, binary_little_endian, and binary_big_endian formats.

    Args:
        filepath: Path to the .ply file.

    Returns:
        Dictionary with keys:
        - ``"points"``: (N, 3) float32 array of xyz positions.
        - ``"normals"``: (N, 3) float32 array, if present.
        - ``"colors"``: (N, 3) array. uint8 if source is uint8, else float32.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PLY file not found: {filepath}")

    with open(filepath, "rb") as f:
        fmt, vertex_count, properties = _parse_header(f)

        if vertex_count == 0:
            return {"points": np.empty((0, 3), dtype=np.float32)}

        if fmt == "ascii":
            data = _read_ascii(f, vertex_count, properties)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            data = _read_binary(f, vertex_count, properties, fmt)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")

    if "points" not in data:
        raise ValueError("PLY file has no x/y/z vertex properties")

    return data


def write_ply(
    filepath: str | Path,
    data: dict[str, np.ndarray],
    ascii: bool = False,
) -> None:
    """Write point cloud data to a PLY file.

    Args:
        filepath: Output file path.
        data: Dictionary with ``"points"`` (required), optionally
              ``"normals"`` and ``"colors"``.
        ascii: If True, write ASCII format; otherwise binary_little_endian.
    """
    filepath = Path(filepath)
    points = data["points"]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")

    n = len(points)
    normals = data.get("normals")
    colors = data.get("colors")

    # Ensure proper dtypes
    points = points.astype(np.float32)
    if normals is not None:
        normals = normals.astype(np.float32)
    if colors is not None:
        if colors.dtype == np.float32 or colors.dtype == np.float64:
            # Store as uint8 in PLY
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Build header
    fmt_str = "ascii 1.0" if ascii else "binary_little_endian 1.0"
    header_lines = [
        "ply",
        f"format {fmt_str}",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if normals is not None:
        header_lines += [
            "property float nx",
            "property float ny",
            "property float nz",
        ]
    if colors is not None:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))

        if n == 0:
            return

        if ascii:
            # Build all columns and write in bulk
            float_cols = [points]
            if normals is not None:
                float_cols.append(normals)
            float_data = np.hstack(float_cols)

            if colors is not None:
                # Write float columns + uint8 color columns
                lines = []
                float_fmt = " ".join(["%.6f"] * float_data.shape[1])
                for i in range(n):
                    fpart = float_fmt % tuple(float_data[i])
                    cpart = f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
                    lines.append(fpart + " " + cpart)
                f.write(("\n".join(lines) + "\n").encode("ascii"))
            else:
                import io as _io
                buf = _io.BytesIO()
                np.savetxt(buf, float_data, fmt="%.6f", delimiter=" ")
                f.write(buf.getvalue())
        else:
            # Binary little-endian — build structured array and write in bulk
            dtype_fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
            if normals is not None:
                dtype_fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
            if colors is not None:
                dtype_fields += [('red', '<u1'), ('green', '<u1'), ('blue', '<u1')]

            vertex_data = np.empty(n, dtype=np.dtype(dtype_fields))
            vertex_data['x'] = points[:, 0]
            vertex_data['y'] = points[:, 1]
            vertex_data['z'] = points[:, 2]
            if normals is not None:
                vertex_data['nx'] = normals[:, 0]
                vertex_data['ny'] = normals[:, 1]
                vertex_data['nz'] = normals[:, 2]
            if colors is not None:
                vertex_data['red'] = colors[:, 0]
                vertex_data['green'] = colors[:, 1]
                vertex_data['blue'] = colors[:, 2]
            f.write(vertex_data.tobytes())
