# Open3D-MLX — PRD Index

## Build Order & Dependency Graph

```
Phase 1: FOUNDATION (Weeks 1-2)
─────────────────────────────────

  PRD-00  Project Foundation ──────┐
     │                             │
     ▼                             │
  PRD-01  Core Types ──────────────┤
     │                             │
     ├─────────────┐               │
     ▼             ▼               ▼
  PRD-02        PRD-03          PRD-04
  PointCloud    I/O (PLY/PCD)   KDTree & NN
     │                             │
     └──────────┬──────────────────┘

Phase 2: REGISTRATION (Weeks 3-5)
──────────────────────────────────

                │
                ▼
  PRD-05  ICP Point-to-Point
     │
     ▼
  PRD-06  ICP Point-to-Plane
     │
     ▼
  PRD-07  Advanced ICP (Colored, GICP, Multi-Scale, Robust)

Phase 3: INTEGRATION (Weeks 6-8)
─────────────────────────────────

  PRD-02 ──────┐
               ▼
  PRD-08  TSDF Uniform Volume
     │
     ▼
  PRD-09  Scalable TSDF + Marching Cubes

Phase 4: RAYCASTING (Weeks 9-10)
─────────────────────────────────

  PRD-08 ──────┐
               ▼
  PRD-10  Volume Raycasting

Phase 5: POLISH (Weeks 11-12)
──────────────────────────────

  All ─────────┐
               ▼
  PRD-11  Benchmarks & CI
     │
     ▼
  PRD-12  Docs, Interop & Polish
```

## Parallelization Strategy

Phases 1-2 and Phase 3 can overlap:

```
Sequential: PRD-00 → PRD-01 (foundation must be first)

Parallel batch 1:  PRD-02 | PRD-03 | PRD-04  (all depend only on PRD-01)

Sequential: PRD-05 → PRD-06 → PRD-07 (registration chain)

Parallel batch 2:  PRD-08 (can start as soon as PRD-02 is done)

Sequential: PRD-09 depends on PRD-08
            PRD-10 depends on PRD-08

Parallel batch 3:  PRD-09 | PRD-10 (both depend on PRD-08, independent of each other)

Final:  PRD-11 → PRD-12
```

## PRD Summary Table

| PRD | Name | Priority | Phase | Depends On | Est. Days |
|-----|------|----------|-------|------------|-----------|
| **00** | Project Foundation | P0 | 1 | — | 1 |
| **01** | Core Types & Device | P0 | 1 | 00 | 1 |
| **02** | PointCloud Geometry | P0 | 1 | 01 | 2 |
| **03** | Point Cloud I/O | P0 | 1 | 01, 02 | 2 |
| **04** | KDTree & Nearest Neighbor | P0 | 1 | 01, 02 | 2–3 |
| **05** | ICP Point-to-Point | P0 | 2 | 02, 04 | 2–3 |
| **06** | ICP Point-to-Plane | P0 | 2 | 05 | 1–2 |
| **07** | Advanced ICP | P1 | 2 | 05, 06 | 3–4 |
| **08** | TSDF Volume (Uniform) | P0 | 3 | 02 | 3–4 |
| **09** | Scalable TSDF + Marching Cubes | P1 | 3 | 08 | 3–4 |
| **10** | Volume Raycasting | P1 | 4 | 08 | 2–3 |
| **11** | Benchmarks & CI | P1 | 5 | 05, 08 | 2 |
| **12** | Docs, Interop & Polish | P1 | 5 | All | 2 |

**Total estimated: ~26-33 days**

## Key Architecture Decisions

1. **No Tensor wrapper** — `mlx.core.array` IS the tensor. Zero overhead.
2. **Upstream as read-only reference** — We never modify `repositories/open3d-upstream/`. Our code is a new Python package that reimplements algorithms using MLX.
3. **`uv` for everything** — Virtual environments, package install, running tests.
4. **scipy for CPU NN** — KDTree is CPU-native. GPU path uses spatial hashing.
5. **Float32 primary** — MLX float64 GPU support is limited. Keep float32 for computation, Python float for scalar results.
6. **Modular and composable** — Each module works independently. Users can mix with vanilla Open3D.

## Upstream Sync Strategy

```bash
# Pull latest upstream (non-destructive)
./scripts/sync_upstream.sh

# Check what changed in a specific area
cd repositories/open3d-upstream
git diff HEAD~1 -- cpp/open3d/t/pipelines/registration/

# Apply relevant changes to our MLX implementation
# No merge conflicts — our code lives in open3d_mlx/, upstream in repositories/
```

## Quick Start (after PRD-00)

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest tests/ -q
```
