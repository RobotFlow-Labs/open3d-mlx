# Session Restart Guide

## How This Project Was Built

Open3D-MLX was built in a single Claude Code session using **multi-agent parallelization**. The entire project — 42 source files, 572 tests, 8 examples — was created from scratch in one conversation.

## Session Architecture

### Phase 1: Planning
1. Read `PROMPT.md` (the master build spec)
2. Loaded `/port-to-mlx` skill (MLX porting patterns from prior projects)
3. Explored upstream Open3D repo (`repositories/open3d-upstream/`) with 2 parallel Explore agents
4. Created 13 modular PRDs in `prds/` with full dependency graph

### Phase 2: Foundation Build (4 parallel agents)
- Agent 1: PRD-01 Core Types (dtype, device, tensor_utils)
- Agent 2: PRD-02 PointCloud Geometry
- Agent 3: PRD-03 Point Cloud I/O (PLY, PCD)
- Agent 4: PRD-04 KDTree & Nearest Neighbor Search

### Phase 3: Pipeline Build (4 parallel agents)
- Agent 1: PRD-05+06 ICP Registration (P2P + P2Plane)
- Agent 2: PRD-08 TSDF Volume + Camera
- Agent 3: PRD-07 Robust Kernels
- Agent 4: PRD-09 Marching Cubes

### Phase 4: Polish (3 parallel agents)
- Agent 1: PRD-10 Volume Raycasting
- Agent 2: PRD-12 Docs, Interop, README
- Agent 3: PRD-11 Benchmarks & CI

### Phase 5: Code Review + Gap Fill
- 1 code review agent → found 4 critical + 16 high issues
- 3 parallel fix agents for code review issues
- 3 parallel agents to fill PROMPT.md scope gaps:
  - Colored ICP, GICP, Multi-scale ICP, Correspondence Checkers
  - Scalable TSDF, AxisAlignedBoundingBox, crop, farthest_point_down_sample
  - XYZ/PTS formats, FPFH feature descriptors

### Phase 6: MLX GPU Acceleration (4 parallel agents)
- Agent 1: Vectorized FixedRadiusIndex + batched normal estimation
- Agent 2: ICP loop fixes (robust kernels integration, incremental transforms)
- Agent 3: MLX-native TSDF integration + raycasting
- Agent 4: Vectorized FPFH + bulk I/O operations

### Phase 7: Documentation
- README with Mermaid diagrams
- 8 working examples (all tested)
- SESSION_RESTART.md (this file)

## Total Agent Invocations

~25 parallel agent sessions across 7 phases, plus ~5 code review agents.

---

## How to Resume Development

### Prerequisites

```bash
cd /Users/ilessio/Development/AIFLOWLABS/R&D/open3d-mlx
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
uv pip install scipy  # for CPU KDTree fallback
```

### Verify Everything Works

```bash
# Run all tests
.venv/bin/python -m pytest tests/ --tb=short

# Run all examples
for f in examples/0*.py; do .venv/bin/python "$f"; done

# Collect benchmarks
.venv/bin/python -m pytest benchmarks/ --co
```

### Key Files to Read First

| File | Purpose |
|------|---------|
| `PROMPT.md` | Master build specification — what we port and why |
| `prds/README.md` | PRD index with dependency graph and build order |
| `.claude/CLAUDE.md` | Claude Code project config (architecture, conventions, commands) |
| `UPSTREAM_VERSION.md` | Which Open3D commit we're aligned to |
| `README.md` | Public-facing docs with architecture diagrams |

### Key Architectural Decisions

1. **`mlx.core.array` IS the tensor** — no custom wrapper class
2. **Upstream is read-only reference** — `repositories/open3d-upstream/` is git-ignored, never modified
3. **`uv` for everything** — venv, install, run
4. **Float32 primary** — MLX float64/int64 GPU support is limited; auto-downcast
5. **MLX for GPU, numpy for what MLX lacks** — unique, searchsorted, SVD batch, 2D gather stay numpy
6. **API-compatible with Open3D** — same function names, same parameter conventions

### Upstream Sync

When Open3D updates algorithms:

```bash
./scripts/sync_upstream.sh          # Pull latest Open3D
cd repositories/open3d-upstream
git diff HEAD~1 -- cpp/open3d/t/pipelines/registration/  # See what changed
# Apply relevant changes to our MLX implementation
```

### Adding a New Feature

1. Check if there's a PRD in `prds/` — if not, create one
2. Identify the upstream reference files in `repositories/open3d-upstream/`
3. Implement in the matching `open3d_mlx/` module
4. Add tests in `tests/` mirroring the source structure
5. Run full suite: `pytest tests/`
6. Add an example if the feature is user-facing

### Running Code Review

```bash
# Use Claude Code's /code-review skill
/code-review
```

The project has been through 2 full code reviews. Key findings were:
- Hash collision risk in spatial hash → mitigated by vectorized bucket processing
- Python loops in hot paths → replaced with vectorized numpy/MLX ops
- Colored ICP Jacobian was mathematically wrong → fixed
- GICP missing rotation in covariance → fixed
- Robust kernels were dead code → integrated into ICP loop
- clone() was wasteful → uses mx.array copy constructor

### Git History

```
8350d68 perf: MLX GPU acceleration + code review fixes (572 tests)
4e07cc4 feat: complete PROMPT.md scope — all P1+P2 gaps filled (572 tests)
4439c18 docs: polished README with Mermaid diagrams + 8 working examples
80ab077 feat: Phase 3 — raycasting, benchmarks, CI, docs, interop (477 tests)
3615042 feat: marching cubes mesh extraction from TSDF volumes (433 tests)
49ed99c feat: Phase 2 — ICP registration, TSDF integration, robust kernels (415 tests)
06c5429 fix: address code review — performance, correctness, wire up normals
c8aa392 feat: Phase 1 foundation — core types, PointCloud, I/O, KDTree/NN (261 tests)
```

---

## What's Left for Future Sessions

### Not Yet Implemented (explicitly out of scope per PROMPT.md)
- Odometry (`t/pipelines/odometry/`) — depends on ICP + TSDF being production-solid
- SLAM (`t/pipelines/slam/`) — requires odometry + loop closure
- Mesh reconstruction (Poisson, ball pivoting)
- Visualization — use Open3D or polyscope via `interop`
- Additional file formats (GLTF, OBJ, STL, E57, LAS)

### Performance Optimization Opportunities
- Metal kernel for FixedRadiusIndex (currently vectorized numpy, GPU hash only)
- MLX-native marching cubes (currently numpy with lookup tables)
- Ray marching with MLX (currently numpy for trilinear sampling)
- Async double-buffer for TSDF integration (overlap CPU/GPU)

### Quality Improvements
- Run actual benchmarks and publish numbers in README
- Add property-based tests (hypothesis)
- CI benchmark regression tracking
- API docstrings for every public function (many exist, some missing)

---

## Claude Code Configuration

The project uses these Claude Code settings:

- `.claude/CLAUDE.md` — project-level instructions
- `.claude/settings.json` — shared permissions (committed)
- `.claude/settings.local.json` — local permissions with MCP tools (gitignored)

### Skills Used
- `/port-to-mlx` — MLX porting patterns (FixedRadiusIndex, batched SVD, blocked scatter)
- `/config-claude` — Project bootstrap
- `/code-review` — Comprehensive code quality review
- `/markdown-mermaid-writing` — README diagrams

### MCP Servers Used
- `context7` — Library documentation lookup
- `github` — Repository creation and management
