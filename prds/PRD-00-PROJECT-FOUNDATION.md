# PRD-00: Project Foundation & Upstream Sync Architecture

## Status: BLOCKING (all other PRDs depend on this)
## Priority: P0
## Phase: 1 — Foundation
## Estimated Effort: 1 day

---

## 1. Objective

Bootstrap the `open3d-mlx` Python package with modern tooling (`uv`, `pyproject.toml`), establish the upstream sync strategy so we can pull Open3D updates without merge conflicts, and lay out the package skeleton that all subsequent PRDs build into.

---

## 2. Upstream Sync Strategy

### 2.1 Core Principle: Reference, Don't Fork

We do **NOT** fork the Open3D C++ code. We write a **new Python package** that reimplements the GPU-accelerated pipelines in pure Python + MLX. The upstream repo is kept as a **read-only reference**.

```
open3d-mlx/
├── open3d_mlx/          ← OUR CODE (new Python package)
├── repositories/
│   └── open3d-upstream/  ← READ-ONLY upstream (git-ignored)
├── prds/                 ← These PRD documents
├── tests/                ← Our test suite
├── benchmarks/           ← Performance comparisons
└── UPSTREAM_VERSION.md   ← Tracks which upstream commit we're aligned to
```

### 2.2 Upstream Tracking

- `repositories/open3d-upstream/` is a **shallow clone** of `isl-org/Open3D`, git-ignored from our repo
- `UPSTREAM_VERSION.md` records the upstream commit hash we last referenced
- A script `scripts/sync_upstream.sh` handles pulling the latest upstream:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

UPSTREAM_DIR="repositories/open3d-upstream"
UPSTREAM_REPO="https://github.com/isl-org/Open3D.git"

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
    echo "Cloning upstream..."
    git clone --depth=1 "$UPSTREAM_REPO" "$UPSTREAM_DIR"
else
    echo "Updating upstream..."
    cd "$UPSTREAM_DIR"
    git fetch --depth=1 origin main
    git checkout FETCH_HEAD
    cd ../..
fi

HASH=$(cd "$UPSTREAM_DIR" && git rev-parse HEAD)
DATE=$(date -u +%Y-%m-%d)
cat > UPSTREAM_VERSION.md << EOF
# Upstream Reference

Synced to: **isl-org/Open3D** \`${HASH:0:12}\` (${DATE})

Repository: https://github.com/isl-org/Open3D
Commit: https://github.com/isl-org/Open3D/commit/${HASH}
EOF

echo "Synced to upstream: ${HASH:0:12}"
```

### 2.3 Why This Works for Future Updates

When Open3D updates their ICP algorithm or adds a new feature:

1. Run `scripts/sync_upstream.sh` to pull latest
2. Diff the specific upstream files our PRDs reference (e.g., `Registration.cpp`)
3. Apply relevant changes to our MLX implementation
4. Update `UPSTREAM_VERSION.md`
5. No merge conflicts because we never modify upstream files

### 2.4 Upstream Reference Map

Each module in our codebase has a documented mapping to upstream files in its PRD. Example:

```python
# open3d_mlx/pipelines/registration/icp.py
#
# Upstream reference:
#   cpp/open3d/t/pipelines/registration/Registration.cpp
#   cpp/open3d/t/pipelines/registration/TransformationEstimation.cpp
#
# Last synced: 2026-03-15 (see UPSTREAM_VERSION.md)
```

---

## 3. Package Structure

```
open3d-mlx/
├── pyproject.toml
├── README.md
├── LICENSE                          # Apache 2.0
├── UPSTREAM_VERSION.md
├── PROMPT.md                        # Master build prompt
├── .claude/
│   ├── CLAUDE.md
│   ├── settings.json
│   └── settings.local.json          # gitignored
├── .gitignore
├── scripts/
│   ├── sync_upstream.sh             # Upstream sync
│   └── benchmark.py                 # Benchmark runner
├── prds/                            # PRD documents
│   ├── PRD-00-PROJECT-FOUNDATION.md
│   ├── PRD-01-CORE-TYPES.md
│   ├── ...
│   └── PRD-12-DOCS-POLISH.md
├── open3d_mlx/
│   ├── __init__.py                  # Package root, version, top-level imports
│   ├── _version.py                  # Single source of version
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dtype.py                 # Dtype mapping (Open3D ↔ MLX)
│   │   ├── device.py                # Device abstraction
│   │   └── tensor_utils.py          # MLX array utilities
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── pointcloud.py            # PointCloud class
│   │   ├── boundingbox.py           # AABB, OBB
│   │   ├── kdtree.py                # KDTree search params
│   │   └── voxel_grid.py            # VoxelGrid (for downsampling)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── registration/
│   │   │   ├── __init__.py
│   │   │   ├── icp.py               # ICP algorithm
│   │   │   ├── transformation.py    # Transformation estimation
│   │   │   ├── convergence.py       # Convergence criteria
│   │   │   ├── correspondence.py    # Correspondence search
│   │   │   ├── robust_kernel.py     # Robust loss functions
│   │   │   └── result.py            # RegistrationResult
│   │   ├── integration/
│   │   │   ├── __init__.py
│   │   │   ├── tsdf_volume.py       # TSDF base
│   │   │   ├── uniform_tsdf.py      # Uniform grid TSDF
│   │   │   ├── scalable_tsdf.py     # Hash-based TSDF
│   │   │   └── marching_cubes.py    # Mesh extraction
│   │   └── raycasting/
│   │       ├── __init__.py
│   │       ├── raycasting_scene.py  # Ray-volume intersection
│   │       └── ray_utils.py         # Ray generation from camera
│   ├── ops/
│   │   ├── __init__.py
│   │   ├── nearest_neighbor.py      # KNN, radius, hybrid search
│   │   ├── fixed_radius_nn.py       # GPU-friendly spatial hash NN
│   │   ├── voxel_ops.py             # Voxel hashing, downsampling
│   │   ├── linalg.py                # SVD, solve wrappers
│   │   └── normals.py               # Normal estimation
│   ├── io/
│   │   ├── __init__.py
│   │   ├── pointcloud_io.py         # read/write dispatcher
│   │   ├── ply.py                   # PLY format
│   │   └── pcd.py                   # PCD format
│   └── camera/
│       ├── __init__.py
│       └── intrinsics.py            # PinholeCameraIntrinsic
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── test_core/
│   │   ├── test_dtype.py
│   │   ├── test_device.py
│   │   └── test_tensor_utils.py
│   ├── test_geometry/
│   │   ├── test_pointcloud.py
│   │   └── test_kdtree.py
│   ├── test_pipelines/
│   │   ├── test_registration/
│   │   │   ├── test_icp.py
│   │   │   ├── test_transformation.py
│   │   │   └── test_robust_kernel.py
│   │   ├── test_integration/
│   │   │   └── test_tsdf.py
│   │   └── test_raycasting/
│   │       └── test_raycasting.py
│   ├── test_ops/
│   │   ├── test_nearest_neighbor.py
│   │   └── test_fixed_radius_nn.py
│   └── test_io/
│       ├── test_ply.py
│       └── test_pcd.py
├── benchmarks/
│   ├── conftest.py
│   ├── bench_icp.py
│   ├── bench_tsdf.py
│   ├── bench_nn.py
│   └── bench_pointcloud.py
└── repositories/
    └── open3d-upstream/              # gitignored
```

---

## 4. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "open3d-mlx"
version = "0.1.0"
description = "Apple Silicon-native 3D perception pipelines via MLX"
readme = "README.md"
license = {text = "Apache-2.0"}
requires-python = ">=3.10"
authors = [
    {name = "AIFLOW LABS", email = "ilessio@aiflowlabs.io"}
]
keywords = ["3d", "point-cloud", "icp", "mlx", "apple-silicon", "robotics"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
]

dependencies = [
    "mlx>=0.22.0",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-benchmark>=4.0",
    "ruff>=0.4.0",
]
interop = [
    "open3d>=0.18.0",           # For cross-validation and legacy interop
    "scipy>=1.10.0",            # For KDTree CPU fallback
]
all = [
    "open3d-mlx[dev,interop]",
]

[project.urls]
Homepage = "https://github.com/RobotFlow-Labs/open3d-mlx"
Repository = "https://github.com/RobotFlow-Labs/open3d-mlx"

[tool.setuptools.packages.find]
include = ["open3d_mlx*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --tb=short"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

---

## 5. Key Files Content

### 5.1 `open3d_mlx/__init__.py`

```python
"""Open3D-MLX: Apple Silicon-native 3D perception pipelines."""

from open3d_mlx._version import __version__

from open3d_mlx import core
from open3d_mlx import geometry
from open3d_mlx import io
from open3d_mlx import pipelines

__all__ = ["core", "geometry", "io", "pipelines", "__version__"]
```

### 5.2 `open3d_mlx/_version.py`

```python
__version__ = "0.1.0"
```

### 5.3 `tests/conftest.py`

```python
"""Shared test fixtures for open3d-mlx."""

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic numpy RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_points(rng):
    """100 random 3D points as MLX array."""
    return mx.array(rng.standard_normal((100, 3)).astype(np.float32))


@pytest.fixture
def sample_pointcloud(rng):
    """PointCloud with 100 random points, normals, colors."""
    from open3d_mlx.geometry import PointCloud

    points = rng.standard_normal((100, 3)).astype(np.float32)
    normals = rng.standard_normal((100, 3)).astype(np.float32)
    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    colors = rng.uniform(0, 1, (100, 3)).astype(np.float32)

    pcd = PointCloud(mx.array(points))
    pcd.normals = mx.array(normals)
    pcd.colors = mx.array(colors)
    return pcd


@pytest.fixture
def identity_4x4():
    """4x4 identity matrix as MLX array."""
    return mx.eye(4)
```

---

## 6. Environment Setup (uv)

```bash
# Create virtual environment
uv venv .venv --python 3.12

# Install in dev mode with all extras
uv pip install -e ".[all]"

# Or minimal install (just MLX + numpy)
uv pip install -e .

# Run tests
uv run pytest tests/

# Run benchmarks
uv run pytest benchmarks/ --benchmark-only
```

---

## 7. Acceptance Criteria

- [ ] `uv venv .venv --python 3.12 && uv pip install -e ".[dev]"` succeeds
- [ ] `python -c "import open3d_mlx; print(open3d_mlx.__version__)"` prints `0.1.0`
- [ ] `python -c "import mlx.core as mx; print(mx.default_device())"` works
- [ ] `pytest tests/ --co` discovers test structure (0 tests pass yet, but collection works)
- [ ] `scripts/sync_upstream.sh` clones or updates upstream
- [ ] All directories from section 3 exist
- [ ] `.gitignore` excludes `repositories/`, `.venv/`, `__pycache__/`, etc.
- [ ] Every `__init__.py` is importable without errors

---

## 8. Dependencies on Other PRDs

None — this is the foundation. All other PRDs depend on this.

## 9. Blocked By

Nothing.

## 10. Blocks

All other PRDs (01–12).
