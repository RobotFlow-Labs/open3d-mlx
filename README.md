# Open3D-MLX

Apple Silicon-native 3D perception pipelines via [MLX](https://github.com/ml-explore/mlx).

A focused port of [Open3D](https://github.com/isl-org/Open3D)'s GPU-accelerated pipelines to Apple Silicon, providing native Metal GPU acceleration for ICP registration, TSDF integration, raycasting, and point cloud I/O.

## Quick Start

```bash
# Install with uv
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

```python
import open3d_mlx as o3m

# Load point clouds
source = o3m.io.read_point_cloud("scan_001.ply")
target = o3m.io.read_point_cloud("scan_002.ply")

# ICP registration (GPU-accelerated on Apple Silicon)
result = o3m.pipelines.registration.registration_icp(
    source, target,
    max_correspondence_distance=0.05,
    estimation_method=o3m.pipelines.registration.TransformationEstimationPointToPlane(),
)

print(f"Fitness: {result.fitness:.4f}")
print(f"RMSE:    {result.inlier_rmse:.4f}")
aligned = source.transform(result.transformation)
```

## API Overview

| Module | Description |
|--------|-------------|
| `open3d_mlx.core` | MLX tensor utilities, device abstraction, dtype helpers |
| `open3d_mlx.geometry` | `PointCloud`, KDTree search parameters |
| `open3d_mlx.io` | Point cloud file I/O (PLY, PCD -- ASCII and binary) |
| `open3d_mlx.pipelines.registration` | ICP (point-to-point, point-to-plane), robust kernels |
| `open3d_mlx.pipelines.integration` | TSDF volume integration from depth frames |
| `open3d_mlx.pipelines.raycasting` | Volume raycasting (TSDF to depth/normals) |
| `open3d_mlx.camera` | Pinhole camera intrinsics |
| `open3d_mlx.interop` | Conversion to/from vanilla Open3D |

## Installation

### Requirements

- Python >= 3.10
- macOS with Apple Silicon (M1/M2/M3/M4)
- MLX >= 0.22.0

### From Source

```bash
git clone https://github.com/RobotFlow-Labs/open3d-mlx.git
cd open3d-mlx
uv pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

## Benchmarks

Performance comparisons against Open3D CUDA are tracked in `benchmarks/`. Run with:

```bash
pytest benchmarks/ -v
```

Benchmark results will be published after v0.2.0.

## Migration from Open3D

### Imports

```python
# Open3D
import open3d as o3d

# Open3D-MLX
import open3d_mlx as o3m
```

### Point Clouds

```python
# Open3D
pcd = o3d.geometry.PointCloud()
pcd = o3d.io.read_point_cloud("file.ply")

# Open3D-MLX (same API)
pcd = o3m.geometry.PointCloud()
pcd = o3m.io.read_point_cloud("file.ply")

# Points are MLX arrays, not numpy -- convert if needed:
import numpy as np
points_np = np.array(pcd.points)
```

### Registration

```python
# Open3D
result = o3d.pipelines.registration.registration_icp(source, target, 0.05, ...)

# Open3D-MLX (same parameters, same result structure)
result = o3m.pipelines.registration.registration_icp(source, target, 0.05, ...)

# Transformation is an MLX array:
T_np = np.array(result.transformation)
```

### TSDF Integration

```python
# Same API: UniformTSDFVolume, .integrate(), .extract_point_cloud()
volume = o3m.pipelines.integration.UniformTSDFVolume(
    length=4.0, resolution=512, sdf_trunc=0.04
)
```

### Key Differences

1. All tensors are `mlx.core.array` (not `numpy` or `o3d.core.Tensor`)
2. No `.to(device)` needed -- unified memory on Apple Silicon
3. No visualization -- use Open3D or polyscope for that (see interop below)
4. `float32` is the primary dtype (not `float64`)

### Interop with Open3D

For visualization or features not yet ported, convert between the two:

```python
from open3d_mlx.interop import to_open3d, from_open3d

# Use Open3D-MLX for computation
result = o3m.pipelines.registration.registration_icp(source, target, 0.05)
aligned = source.transform(result.transformation)

# Convert to Open3D for visualization
import open3d as o3d
o3d.visualization.draw_geometries([to_open3d(aligned), to_open3d(target)])
```

## Architecture

```
open3d_mlx/
  core/          # MLX tensor wrapper, device, dtypes
  geometry/      # PointCloud, KDTree search params
  io/            # PLY/PCD file I/O
  camera/        # Pinhole camera intrinsics
  ops/           # MLX custom kernels
  pipelines/
    registration/   # ICP variants, robust kernels
    integration/    # TSDF volume integration
    raycasting/     # Volume raycasting
  interop.py     # Open3D conversion helpers
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
