"""PCD (Point Cloud Data) reader and writer.

Supports DATA ascii and DATA binary formats.
Handles PCL's float-packed RGB encoding.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

# ── PCD field-type mapping ──────────────────────────────────────────────────

_PCD_TYPE_SIZE_TO_NUMPY = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
}

_NUMPY_TO_PCD_TYPE_SIZE = {
    np.dtype(np.float32): ("F", 4),
    np.dtype(np.float64): ("F", 8),
    np.dtype(np.uint8): ("U", 1),
    np.dtype(np.uint32): ("U", 4),
    np.dtype(np.int32): ("I", 4),
}


# ── RGB packing helpers ─────────────────────────────────────────────────────


def _unpack_pcd_rgb(rgb_float: np.ndarray) -> np.ndarray:
    """Unpack PCD float-encoded RGB to (N, 3) uint8.

    In PCD format, RGB is stored as a single float32 whose bit pattern
    encodes R in bits 16-23, G in bits 8-15, B in bits 0-7.
    """
    rgb_int = rgb_float.view(np.uint32)
    r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
    b = (rgb_int & 0xFF).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _pack_pcd_rgb(colors_uint8: np.ndarray) -> np.ndarray:
    """Pack (N, 3) uint8 RGB into PCD float encoding."""
    rgb_int = (
        colors_uint8[:, 0].astype(np.uint32) << 16
        | colors_uint8[:, 1].astype(np.uint32) << 8
        | colors_uint8[:, 2].astype(np.uint32)
    )
    return rgb_int.view(np.float32)


# ── Header parsing ──────────────────────────────────────────────────────────


def _parse_pcd_header(f: BinaryIO) -> dict:
    """Parse PCD header fields.

    Returns dict with keys: version, fields, sizes, types, counts,
    width, height, viewpoint, points, data_format.
    """
    header: dict = {}
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PCD header")
        line_str = line.decode("ascii", errors="replace").strip()
        if not line_str or line_str.startswith("#"):
            continue

        parts = line_str.split()
        key = parts[0].upper()

        if key == "VERSION":
            header["version"] = parts[1]
        elif key == "FIELDS":
            header["fields"] = parts[1:]
        elif key == "SIZE":
            header["sizes"] = [int(x) for x in parts[1:]]
        elif key == "TYPE":
            header["types"] = parts[1:]
        elif key == "COUNT":
            header["counts"] = [int(x) for x in parts[1:]]
        elif key == "WIDTH":
            header["width"] = int(parts[1])
        elif key == "HEIGHT":
            header["height"] = int(parts[1])
        elif key == "VIEWPOINT":
            header["viewpoint"] = [float(x) for x in parts[1:]]
        elif key == "POINTS":
            header["points"] = int(parts[1])
        elif key == "DATA":
            header["data_format"] = parts[1].lower()
            break  # DATA is always last header line

    # Validate
    if "fields" not in header:
        raise ValueError("PCD header missing FIELDS")
    if "data_format" not in header:
        raise ValueError("PCD header missing DATA")

    # Default counts to 1 if not specified
    n_fields = len(header["fields"])
    if "counts" not in header:
        header["counts"] = [1] * n_fields
    if "points" not in header:
        header["points"] = header.get("width", 0) * header.get("height", 1)

    return header


# ── Readers ─────────────────────────────────────────────────────────────────


def _read_pcd_ascii(
    f: BinaryIO,
    header: dict,
) -> dict[str, np.ndarray]:
    """Read PCD ASCII data."""
    import io as _io

    n_points = header["points"]
    fields = header["fields"]
    n_fields = len(fields)

    remaining = f.read().decode("ascii")
    lines = remaining.splitlines()[:n_points]
    if len(lines) < n_points:
        raise ValueError(
            f"PCD ASCII: expected {n_points} data lines, got {len(lines)}"
        )
    raw = np.loadtxt(_io.StringIO("\n".join(lines)), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < n_fields:
        raise ValueError(
            f"PCD ASCII: expected {n_fields} columns, got {raw.shape[1]}"
        )
    raw = raw[:, :n_fields]

    return _extract_pcd_fields(raw, header)


def _read_pcd_binary(
    f: BinaryIO,
    header: dict,
) -> dict[str, np.ndarray]:
    """Read PCD binary (uncompressed) data."""
    n_points = header["points"]
    fields = header["fields"]
    sizes = header["sizes"]
    types = header["types"]
    counts = header["counts"]

    # Build structured dtype
    dtype_list = []
    for field_name, size, type_char, count in zip(fields, sizes, types, counts):
        np_type = _PCD_TYPE_SIZE_TO_NUMPY.get((type_char, size))
        if np_type is None:
            raise ValueError(
                f"Unsupported PCD type: TYPE={type_char} SIZE={size}"
            )
        if count == 1:
            dtype_list.append((field_name, np_type))
        else:
            dtype_list.append((field_name, np_type, (count,)))

    np_dtype = np.dtype(dtype_list)
    point_size = np_dtype.itemsize
    raw_bytes = f.read(n_points * point_size)
    if len(raw_bytes) < n_points * point_size:
        raise ValueError(
            f"PCD binary: expected {n_points * point_size} bytes, "
            f"got {len(raw_bytes)}"
        )

    structured = np.frombuffer(raw_bytes, dtype=np_dtype, count=n_points)

    # Convert to float64 2D
    n_fields = len(fields)
    raw = np.empty((n_points, n_fields), dtype=np.float64)
    for j, field_name in enumerate(fields):
        col = structured[field_name]
        if col.ndim > 1:
            col = col[:, 0]
        raw[:, j] = col.astype(np.float64)

    return _extract_pcd_fields(raw, header, structured=structured)


def _extract_pcd_fields(
    raw: np.ndarray,
    header: dict,
    structured: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Extract semantic fields from PCD data."""
    fields = header["fields"]
    field_index = {name: i for i, name in enumerate(fields)}
    result: dict[str, np.ndarray] = {}

    # Points
    if "x" in field_index and "y" in field_index and "z" in field_index:
        result["points"] = np.column_stack([
            raw[:, field_index["x"]],
            raw[:, field_index["y"]],
            raw[:, field_index["z"]],
        ]).astype(np.float32)

    # Normals
    for nx_name, ny_name, nz_name in [
        ("normal_x", "normal_y", "normal_z"),
        ("nx", "ny", "nz"),
    ]:
        if (
            nx_name in field_index
            and ny_name in field_index
            and nz_name in field_index
        ):
            result["normals"] = np.column_stack([
                raw[:, field_index[nx_name]],
                raw[:, field_index[ny_name]],
                raw[:, field_index[nz_name]],
            ]).astype(np.float32)
            break

    # RGB — PCD uses float-packed encoding
    if "rgb" in field_index:
        rgb_col_idx = field_index["rgb"]
        if structured is not None:
            # Binary: the raw bytes already have the correct bit pattern
            rgb_float = structured["rgb"].astype(np.float32)
        else:
            # ASCII: PCL writes packed RGB as an integer value (e.g. 16711680
            # for pure red). Convert the parsed float64 to uint32, then view
            # as float32 to get the bit-pattern representation that
            # _unpack_pcd_rgb expects.
            rgb_int = raw[:, rgb_col_idx].astype(np.uint32)
            rgb_float = rgb_int.view(np.float32)
        result["colors"] = _unpack_pcd_rgb(rgb_float)

    return result


# ── Public API ──────────────────────────────────────────────────────────────


def read_pcd(filepath: str | Path) -> dict[str, np.ndarray]:
    """Read a PCD file and return a dict of numpy arrays.

    Supports DATA ascii and DATA binary formats.

    Args:
        filepath: Path to the .pcd file.

    Returns:
        Dictionary with keys:
        - ``"points"``: (N, 3) float32 array.
        - ``"normals"``: (N, 3) float32 array, if present.
        - ``"colors"``: (N, 3) uint8 array, if RGB field present.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PCD file not found: {filepath}")

    with open(filepath, "rb") as f:
        header = _parse_pcd_header(f)
        n_points = header["points"]

        if n_points == 0:
            return {"points": np.empty((0, 3), dtype=np.float32)}

        data_fmt = header["data_format"]
        if data_fmt == "ascii":
            data = _read_pcd_ascii(f, header)
        elif data_fmt == "binary":
            data = _read_pcd_binary(f, header)
        else:
            raise ValueError(f"Unsupported PCD data format: {data_fmt}")

    if "points" not in data:
        raise ValueError("PCD file has no x/y/z fields")

    return data


def write_pcd(
    filepath: str | Path,
    data: dict[str, np.ndarray],
    ascii: bool = False,
) -> None:
    """Write point cloud data to a PCD file.

    Args:
        filepath: Output file path.
        data: Dictionary with ``"points"`` (required), optionally
              ``"normals"`` and ``"colors"``.
        ascii: If True, write DATA ascii; otherwise DATA binary.
    """
    filepath = Path(filepath)
    points = data["points"].astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")

    n = len(points)
    normals = data.get("normals")
    colors = data.get("colors")

    if normals is not None:
        normals = normals.astype(np.float32)
    if colors is not None:
        if colors.dtype in (np.float32, np.float64):
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Build header
    fields = ["x", "y", "z"]
    sizes = [4, 4, 4]
    types = ["F", "F", "F"]
    counts = [1, 1, 1]

    if normals is not None:
        fields += ["normal_x", "normal_y", "normal_z"]
        sizes += [4, 4, 4]
        types += ["F", "F", "F"]
        counts += [1, 1, 1]

    if colors is not None:
        fields.append("rgb")
        sizes.append(4)
        types.append("F")
        counts.append(1)

    data_tag = "ascii" if ascii else "binary"
    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        f"SIZE {' '.join(str(s) for s in sizes)}",
        f"TYPE {' '.join(types)}",
        f"COUNT {' '.join(str(c) for c in counts)}",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        f"DATA {data_tag}",
    ]
    header = "\n".join(header_lines) + "\n"

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))

        if n == 0:
            return

        # Pack RGB if present
        rgb_packed = None
        if colors is not None:
            rgb_packed = _pack_pcd_rgb(colors)

        if ascii:
            # Build float columns and write in bulk
            float_cols = [points]
            if normals is not None:
                float_cols.append(normals)
            float_data = np.hstack(float_cols)

            if rgb_packed is not None:
                # PCD ASCII: write packed RGB as its uint32 integer value
                rgb_ints = rgb_packed.view(np.uint32)
                lines = []
                float_fmt = " ".join(["%.6f"] * float_data.shape[1])
                for i in range(n):
                    fpart = float_fmt % tuple(float_data[i])
                    lines.append(fpart + " " + str(int(rgb_ints[i])))
                f.write(("\n".join(lines) + "\n").encode("ascii"))
            else:
                import io as _io
                buf = _io.BytesIO()
                np.savetxt(buf, float_data, fmt="%.6f", delimiter=" ")
                f.write(buf.getvalue())
        else:
            # Binary — build structured array and write in bulk
            dtype_fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
            if normals is not None:
                dtype_fields += [('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]
            if rgb_packed is not None:
                dtype_fields.append(('rgb', '<f4'))

            vertex_data = np.empty(n, dtype=np.dtype(dtype_fields))
            vertex_data['x'] = points[:, 0]
            vertex_data['y'] = points[:, 1]
            vertex_data['z'] = points[:, 2]
            if normals is not None:
                vertex_data['nx'] = normals[:, 0]
                vertex_data['ny'] = normals[:, 1]
                vertex_data['nz'] = normals[:, 2]
            if rgb_packed is not None:
                vertex_data['rgb'] = rgb_packed
            f.write(vertex_data.tobytes())
