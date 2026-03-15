# PRD-12: API Polish, Documentation & Interop

## Status: P1 — Polish
## Priority: P1
## Phase: 5 — Polish
## Estimated Effort: 2 days
## Depends On: All previous PRDs
## Blocks: Nothing (final PRD)

---

## 1. Objective

Final polish pass: ensure the API is clean and consistent, write documentation and tutorials, create a migration guide for Open3D users, and implement interop with vanilla Open3D for hybrid workflows.

---

## 2. API Consistency Audit

### 2.1 Naming Conventions

Verify all public API follows these rules:

| Convention | Example | Notes |
|-----------|---------|-------|
| Module names | `snake_case` | `pipelines.registration` |
| Class names | `PascalCase` | `PointCloud`, `ICPConvergenceCriteria` |
| Function names | `snake_case` | `registration_icp`, `read_point_cloud` |
| Constants | `UPPER_SNAKE` | `DTYPE_MAP` |
| Private | `_prefix` | `_find_correspondences` |

### 2.2 Function Signature Audit

Verify parameters match Open3D naming where possible:

```python
# Open3D:
o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance, ...)

# Ours (should match):
open3d_mlx.pipelines.registration.registration_icp(source, target, max_correspondence_distance, ...)
```

### 2.3 Return Type Consistency

- All transforms: `mx.array` of shape `(4, 4)`
- All points: `mx.array` of shape `(N, 3)`
- All metrics: Python `float`
- All indices: `mx.array` of dtype `int32`

---

## 3. Open3D Interop Layer

### 3.1 Conversion Functions

```python
# open3d_mlx/interop.py

def to_open3d(pcd: PointCloud) -> "o3d.geometry.PointCloud":
    """Convert Open3D-MLX PointCloud to Open3D legacy PointCloud.

    Requires: open3d package installed.
    """
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points))
    if pcd.has_normals():
        o3d_pcd.normals = o3d.utility.Vector3dVector(np.array(pcd.normals))
    if pcd.has_colors():
        o3d_pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors))
    return o3d_pcd


def from_open3d(o3d_pcd: "o3d.geometry.PointCloud") -> PointCloud:
    """Convert Open3D legacy PointCloud to Open3D-MLX PointCloud."""
    pcd = PointCloud(mx.array(np.asarray(o3d_pcd.points).astype(np.float32)))
    if o3d_pcd.has_normals():
        pcd.normals = mx.array(np.asarray(o3d_pcd.normals).astype(np.float32))
    if o3d_pcd.has_colors():
        pcd.colors = mx.array(np.asarray(o3d_pcd.colors).astype(np.float32))
    return pcd


def to_open3d_tensor(pcd: PointCloud) -> "o3d.t.geometry.PointCloud":
    """Convert to Open3D tensor-based PointCloud."""
    import open3d as o3d
    o3d_pcd = o3d.t.geometry.PointCloud()
    o3d_pcd.point.positions = o3d.core.Tensor(np.array(pcd.points))
    if pcd.has_normals():
        o3d_pcd.point.normals = o3d.core.Tensor(np.array(pcd.normals))
    if pcd.has_colors():
        o3d_pcd.point.colors = o3d.core.Tensor(np.array(pcd.colors))
    return o3d_pcd
```

### 3.2 Hybrid Workflow Example

```python
import open3d as o3d
import open3d_mlx as o3m
from open3d_mlx.interop import to_open3d, from_open3d

# Load with Open3D-MLX (MLX-native)
source = o3m.io.read_point_cloud("scan_001.ply")
target = o3m.io.read_point_cloud("scan_002.ply")

# Register with Open3D-MLX (GPU-accelerated on Apple Silicon)
result = o3m.pipelines.registration.registration_icp(
    source, target, max_correspondence_distance=0.05,
    estimation_method=o3m.pipelines.registration.TransformationEstimationPointToPlane()
)

# Visualize with Open3D (use its visualization)
source_aligned = source.transform(result.transformation)
o3d.visualization.draw_geometries([
    to_open3d(source_aligned),
    to_open3d(target),
])
```

---

## 4. Documentation

### 4.1 README.md Structure

```markdown
# Open3D-MLX

Apple Silicon-native 3D perception pipelines via MLX.

## What This Is
Port of Open3D's GPU-accelerated pipelines to Apple Silicon using MLX.
Native Metal GPU acceleration for ICP, TSDF, raycasting.

## Quick Start
## Installation
## API Reference
## Benchmarks
## Migration from Open3D
## Architecture
## Contributing
## License
```

### 4.2 Migration Guide

```markdown
# Migrating from Open3D

## Imports
- `import open3d as o3d` → `import open3d_mlx as o3m`

## Point Clouds
- `o3d.geometry.PointCloud()` → `o3m.geometry.PointCloud()`
- `o3d.io.read_point_cloud()` → `o3m.io.read_point_cloud()`
- Points are MLX arrays, not numpy (use `np.array(pcd.points)` to convert)

## Registration
- `o3d.pipelines.registration.registration_icp()` →
  `o3m.pipelines.registration.registration_icp()`
- Same parameters, same result structure
- Transformation is MLX array (use `np.array(result.transformation)` for numpy)

## TSDF
- Same API: `UniformTSDFVolume`, `.integrate()`, `.extract_point_cloud()`

## Key Differences
1. All tensors are `mlx.core.array` (not `numpy` or `o3d.core.Tensor`)
2. No `.to(device)` needed — unified memory
3. No visualization — use Open3D or polyscope for that
4. `float32` is primary dtype (not float64)
```

### 4.3 Tutorials (as Python scripts)

```
examples/
├── 01_load_and_visualize.py       # Load PLY, basic operations
├── 02_icp_registration.py         # Align two scans
├── 03_tsdf_reconstruction.py      # Integrate depth frames
├── 04_raycasting.py               # Render from TSDF
├── 05_multiscale_icp.py           # Coarse-to-fine registration
└── 06_interop_with_open3d.py      # Use with Open3D visualization
```

---

## 5. `__all__` Exports

Ensure every public module has `__all__` properly defined:

```python
# open3d_mlx/pipelines/registration/__init__.py
from open3d_mlx.pipelines.registration.icp import registration_icp, evaluate_registration, multi_scale_icp
from open3d_mlx.pipelines.registration.convergence import ICPConvergenceCriteria
from open3d_mlx.pipelines.registration.result import RegistrationResult
from open3d_mlx.pipelines.registration.transformation import (
    TransformationEstimationPointToPoint,
    TransformationEstimationPointToPlane,
    TransformationEstimationForColoredICP,
    TransformationEstimationForGeneralizedICP,
)
from open3d_mlx.pipelines.registration.robust_kernel import (
    L2Loss, HuberLoss, TukeyLoss, CauchyLoss, GMLoss,
)

__all__ = [
    "registration_icp", "evaluate_registration", "multi_scale_icp",
    "ICPConvergenceCriteria", "RegistrationResult",
    "TransformationEstimationPointToPoint", "TransformationEstimationPointToPlane",
    "TransformationEstimationForColoredICP", "TransformationEstimationForGeneralizedICP",
    "L2Loss", "HuberLoss", "TukeyLoss", "CauchyLoss", "GMLoss",
]
```

---

## 6. Tests

```
# Interop
test_to_open3d_preserves_points
test_to_open3d_preserves_normals
test_from_open3d_preserves_points
test_roundtrip_open3d_interop

# Import completeness
test_all_public_modules_importable
test_all_dunder_all_defined
test_no_private_names_in_dunder_all

# Examples
test_example_01_runs_without_error
test_example_02_runs_without_error
```

---

## 7. Acceptance Criteria

- [ ] All public APIs follow naming conventions
- [ ] `to_open3d()` / `from_open3d()` conversions work
- [ ] README.md covers installation, quick start, benchmarks
- [ ] Migration guide covers all major API differences
- [ ] Example scripts run without errors
- [ ] All `__init__.py` have proper `__all__` exports
- [ ] `help(o3m.pipelines.registration.registration_icp)` shows useful docstring
- [ ] No circular imports
- [ ] All tests pass
