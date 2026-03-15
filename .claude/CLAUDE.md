# Open3D-MLX - Claude Code Project Config

## Project Overview
Open3D-MLX: Apple Silicon-native port of Open3D's GPU-accelerated pipelines using MLX.
Focused subset: ICP registration, TSDF integration, raycasting, KDTree, point cloud I/O.
We do NOT port the full C++ core, visualization, or TensorFlow backend.

## Key Architecture
- `open3d_mlx/core/` - MLX tensor wrapper, device abstraction, dtypes
- `open3d_mlx/geometry/` - PointCloud, TSDF volume, voxel grid (MLX-backed)
- `open3d_mlx/pipelines/registration/` - ICP (point-to-point, point-to-plane, colored, generalized)
- `open3d_mlx/pipelines/integration/` - TSDF volume integration from depth frames
- `open3d_mlx/pipelines/raycasting/` - Volume raycasting (TSDF -> depth/normals)
- `open3d_mlx/ops/` - MLX custom kernels (nearest neighbor, voxel hashing, reduction)
- `open3d_mlx/io/` - Point cloud file I/O (.ply, .pcd)
- `tests/` - Unit tests mirroring Open3D's test structure
- `benchmarks/` - Performance comparisons vs Open3D CUDA
- `repositories/open3d-upstream/` - Reference upstream (git-ignored)

## Critical Dependencies
- Python >= 3.10
- MLX >= 0.22.0
- NumPy >= 1.24
- mlx-graphs (optional, for graph-based ops)

## Upstream Reference
- Source: `repositories/open3d-upstream/` (isl-org/Open3D main branch)
- Key upstream paths:
  - `cpp/open3d/pipelines/registration/` - ICP, FPFH, correspondence
  - `cpp/open3d/pipelines/integration/` - TSDF volumes
  - `cpp/open3d/t/pipelines/registration/` - Tensor-based ICP (GPU path)
  - `cpp/open3d/t/geometry/RaycastingScene.{cpp,h}` - Raycasting
  - `cpp/open3d/core/` - Tensor, Device, hashmap
  - `cpp/open3d/geometry/KDTreeFlann.{cpp,h}` - KDTree

## Dev Commands
```bash
uv venv .venv --python 3.12         # Create venv
uv pip install -e ".[dev]"          # Install in dev mode
uv pip install -e ".[all]"          # Install with all extras (incl. open3d interop)
uv run pytest tests/                # Run tests
uv run pytest benchmarks/           # Run benchmarks
./scripts/sync_upstream.sh          # Pull latest upstream reference
```

## PRDs
All build plans are in `prds/` — see `prds/README.md` for the dependency graph and build order.
13 modular PRDs covering foundation → registration → integration → raycasting → polish.

## Conventions
- Python package manager: **`uv`** (never pip directly)
- Follow Open3D's API naming where possible for compatibility
- All GPU ops go through MLX -- no CUDA, no Metal directly
- Use `rg` (ripgrep) instead of `grep`
- `mlx.core.array` IS the tensor type -- no custom wrapper
- Primary dtype: `float32` (MLX float64/int64 GPU support is limited)
- Upstream is read-only reference in `repositories/open3d-upstream/` -- never modify it
- Every module header comments which upstream file it references

# currentDate
Today's date is 2026-03-15.
