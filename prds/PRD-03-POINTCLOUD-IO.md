# PRD-03: Point Cloud I/O (PLY & PCD)

## Status: P0 — Foundation
## Priority: P0
## Phase: 1 — Foundation
## Estimated Effort: 2 days
## Depends On: PRD-00, PRD-01, PRD-02
## Blocks: PRD-11 (benchmarks need test data loading)

---

## 1. Objective

Implement PLY and PCD file format readers/writers. These two formats cover ~90% of robotics point cloud use cases. Both ASCII and binary variants supported. Loading returns a `PointCloud` with MLX arrays; saving accepts `PointCloud` and writes to file.

---

## 2. Upstream Reference

| Our File | Upstream File | Lines | Notes |
|----------|--------------|-------|-------|
| `io/ply.py` | `cpp/open3d/io/FilePLY.cpp` | 1,038 | PLY reader/writer |
| `io/pcd.py` | `cpp/open3d/io/FilePCD.cpp` | 804 | PCD reader/writer |
| `io/pointcloud_io.py` | `cpp/open3d/io/PointCloudIO.cpp` | 238 | Dispatcher |

---

## 3. API Design

### 3.1 Top-Level Functions

```python
# open3d_mlx/io/__init__.py

def read_point_cloud(
    filename: str,
    format: str = "auto",
    remove_nan_points: bool = False,
    remove_infinite_points: bool = False,
) -> PointCloud:
    """Read point cloud from file.

    Args:
        filename: Path to .ply or .pcd file.
        format: "auto" (detect from extension), "ply", or "pcd".
        remove_nan_points: Remove NaN coordinate points after loading.
        remove_infinite_points: Remove Inf coordinate points after loading.

    Returns:
        PointCloud with points, and optionally normals/colors.
    """


def write_point_cloud(
    filename: str,
    pointcloud: PointCloud,
    write_ascii: bool = False,
    compressed: bool = False,
) -> bool:
    """Write point cloud to file.

    Args:
        filename: Output path. Format detected from extension.
        pointcloud: PointCloud to write.
        write_ascii: If True, write ASCII format. Default binary.
        compressed: If True and format supports it, use compression.

    Returns:
        True on success.
    """
```

### 3.2 Format Dispatch

```python
def _detect_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    FORMAT_MAP = {
        ".ply": "ply",
        ".pcd": "pcd",
    }
    if ext not in FORMAT_MAP:
        raise ValueError(f"Unsupported format: {ext}. Supported: {list(FORMAT_MAP.keys())}")
    return FORMAT_MAP[ext]
```

---

## 4. PLY Format Implementation

### 4.1 PLY Overview

PLY (Polygon File Format / Stanford Triangle Format):
- Header in ASCII (always), data in ASCII or binary (little/big endian)
- Properties are named and typed
- We read: `x y z` (positions), `nx ny nz` (normals), `red green blue` (colors)

### 4.2 PLY Reader

```python
def read_ply(filepath: str) -> dict[str, np.ndarray]:
    """Parse PLY file, return dict of numpy arrays.

    Supports:
    - ASCII format
    - binary_little_endian format
    - binary_big_endian format

    Returns dict with keys: "points" (N,3), optionally "normals" (N,3), "colors" (N,3).
    """
    # 1. Parse header
    #    - Read lines until "end_header"
    #    - Extract format (ascii/binary_little/binary_big)
    #    - Extract element count
    #    - Extract property names and types
    #
    # 2. Read data
    #    - ASCII: np.loadtxt or line-by-line parsing
    #    - Binary: np.frombuffer with correct dtype
    #
    # 3. Map property names to our convention:
    #    x,y,z → points
    #    nx,ny,nz → normals
    #    red,green,blue → colors (normalize uint8 to [0,1] float32)
    #    diffuse_red, etc. → also colors
```

### 4.3 PLY Writer

```python
def write_ply(filepath: str, data: dict[str, np.ndarray], ascii: bool = False) -> None:
    """Write PLY file.

    Args:
        filepath: Output path.
        data: Dict with "points" (required), optionally "normals", "colors".
        ascii: Write ASCII format if True, else binary_little_endian.
    """
    # Header
    #   ply
    #   format {ascii|binary_little_endian} 1.0
    #   element vertex {N}
    #   property float x
    #   property float y
    #   property float z
    #   [property float nx ...]
    #   [property uchar red ...]
    #   end_header
    #
    # Data
    #   ASCII: one vertex per line
    #   Binary: packed struct per vertex
```

---

## 5. PCD Format Implementation

### 5.1 PCD Overview

PCD (Point Cloud Data) — PCL's native format:
- Header fields: VERSION, FIELDS, SIZE, TYPE, COUNT, WIDTH, HEIGHT, VIEWPOINT, POINTS, DATA
- Data: ascii or binary
- Fields are named: `x y z normal_x normal_y normal_z rgb`

### 5.2 PCD Reader

```python
def read_pcd(filepath: str) -> dict[str, np.ndarray]:
    """Parse PCD file, return dict of numpy arrays.

    Supports:
    - DATA ascii
    - DATA binary
    - DATA binary_compressed (lzf decompression)

    PCD rgb encoding: single float32 packing 3 bytes (R<<16 | G<<8 | B).
    We unpack to (N, 3) float32 in [0, 1].
    """
```

### 5.3 PCD Writer

```python
def write_pcd(filepath: str, data: dict[str, np.ndarray], ascii: bool = False) -> None:
    """Write PCD file."""
```

### 5.4 PCD RGB Encoding

PCD encodes RGB as a single float32 containing packed bytes:

```python
def _unpack_pcd_rgb(rgb_float: np.ndarray) -> np.ndarray:
    """Unpack PCD float-encoded RGB to (N, 3) uint8."""
    rgb_int = rgb_float.view(np.uint32)
    r = ((rgb_int >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_int >> 8) & 0xFF).astype(np.uint8)
    b = (rgb_int & 0xFF).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)

def _pack_pcd_rgb(colors_uint8: np.ndarray) -> np.ndarray:
    """Pack (N, 3) uint8 RGB to PCD float encoding."""
    rgb_int = (colors_uint8[:, 0].astype(np.uint32) << 16 |
               colors_uint8[:, 1].astype(np.uint32) << 8 |
               colors_uint8[:, 2].astype(np.uint32))
    return rgb_int.view(np.float32)
```

---

## 6. Integration with PointCloud

```python
def read_point_cloud(filename, format="auto", remove_nan_points=False, remove_infinite_points=False):
    fmt = format if format != "auto" else _detect_format(filename)

    if fmt == "ply":
        data = read_ply(filename)
    elif fmt == "pcd":
        data = read_pcd(filename)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    pcd = PointCloud(mx.array(data["points"].astype(np.float32)))

    if "normals" in data:
        pcd.normals = mx.array(data["normals"].astype(np.float32))
    if "colors" in data:
        colors = data["colors"]
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float32) / 255.0
        pcd.colors = mx.array(colors.astype(np.float32))

    if remove_nan_points or remove_infinite_points:
        pcd = pcd.remove_non_finite_points()

    return pcd


def write_point_cloud(filename, pointcloud, write_ascii=False, compressed=False):
    fmt = _detect_format(filename)
    data = {"points": np.array(pointcloud.points)}
    if pointcloud.has_normals():
        data["normals"] = np.array(pointcloud.normals)
    if pointcloud.has_colors():
        data["colors"] = np.array(pointcloud.colors)

    if fmt == "ply":
        write_ply(filename, data, ascii=write_ascii)
    elif fmt == "pcd":
        write_pcd(filename, data, ascii=write_ascii)
    return True
```

---

## 7. Tests

### Test Data

Create minimal test PLY/PCD files in `tests/data/`:

- `cube_ascii.ply` — 8 vertices of a unit cube, ASCII format
- `cube_binary.ply` — same data, binary_little_endian
- `cube_with_normals.ply` — 8 vertices with normals
- `cube_with_colors.ply` — 8 vertices with RGB colors
- `cube.pcd` — PCD binary format

### Test Cases

```
# PLY tests
test_read_ply_ascii_positions_only
test_read_ply_binary_positions_only
test_read_ply_with_normals
test_read_ply_with_colors_uint8
test_read_ply_with_colors_float
test_write_ply_ascii_roundtrip
test_write_ply_binary_roundtrip
test_ply_preserves_precision_float32

# PCD tests
test_read_pcd_ascii
test_read_pcd_binary
test_read_pcd_rgb_unpacking
test_write_pcd_ascii_roundtrip
test_write_pcd_binary_roundtrip

# Integration
test_read_point_cloud_auto_detect_ply
test_read_point_cloud_auto_detect_pcd
test_read_point_cloud_remove_nan
test_write_point_cloud_auto_detect
test_roundtrip_ply_preserves_all_attributes
test_roundtrip_pcd_preserves_all_attributes
test_empty_pointcloud_write_read

# Cross-validation
test_ply_matches_open3d_reader   # if open3d installed
test_pcd_matches_open3d_reader   # if open3d installed
```

---

## 8. Acceptance Criteria

- [ ] `read_point_cloud("test.ply")` returns PointCloud with correct point count
- [ ] ASCII and binary PLY variants both read correctly
- [ ] PCD ASCII and binary variants both read correctly
- [ ] PCD RGB float packing/unpacking works correctly
- [ ] Write → Read roundtrip preserves points within float32 precision
- [ ] Normals and colors survive roundtrip
- [ ] Auto-detection works for .ply and .pcd extensions
- [ ] Unsupported extensions raise clear ValueError
- [ ] `remove_nan_points=True` removes NaN entries
- [ ] All tests pass
- [ ] No dependencies beyond numpy (file I/O is CPU-side)
