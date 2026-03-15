# PRD-11: Benchmarks & CI

## Status: P1 — Polish
## Priority: P1
## Phase: 5 — Polish
## Estimated Effort: 2 days
## Depends On: PRD-05, PRD-08 (needs working pipelines to benchmark)
## Blocks: PRD-12

---

## 1. Objective

Establish a benchmark suite comparing Open3D-MLX against Open3D (CUDA and CPU) on key workloads. Set up GitHub Actions CI for automated testing on Apple Silicon.

---

## 2. Benchmark Suite

### 2.1 Benchmark Matrix

| Benchmark | Metric | Parameters |
|-----------|--------|------------|
| ICP Point-to-Point | Time (ms), iterations, RMSE | N = 1K, 10K, 100K, 1M points |
| ICP Point-to-Plane | Time (ms), iterations, RMSE | N = 1K, 10K, 100K, 1M points |
| KNN Search (k=30) | Time (ms) | N = 10K, 100K, 1M points |
| Fixed-Radius NN | Time (ms) | N = 10K, 100K, 1M points |
| Voxel Downsampling | Time (ms), output count | N = 100K, 1M points |
| Normal Estimation | Time (ms) | N = 10K, 100K points |
| TSDF Integration | Time per frame (ms) | Resolution: 64³, 128³, 256³ |
| TSDF Point Extraction | Time (ms) | Resolution: 64³, 128³, 256³ |
| Raycasting Depth | Time per frame (ms) | 640×480, 128³ volume |
| PLY Read/Write | Time (ms) | N = 100K, 1M points |

### 2.2 Comparison Targets

| Backend | Hardware | Notes |
|---------|----------|-------|
| **Open3D-MLX** | Apple M1/M2/M3/M4 | Our implementation |
| **Open3D CPU** | Same Mac (via Rosetta or native) | `o3d.Device("CPU:0")` |
| **Open3D CUDA** | NVIDIA RTX 3090/4090 | `o3d.Device("CUDA:0")` (if available) |

### 2.3 Benchmark Files

```python
# benchmarks/bench_icp.py

import pytest
import mlx.core as mx
import numpy as np


@pytest.fixture(params=[1000, 10000, 100000])
def point_count(request):
    return request.param


@pytest.fixture
def source_target_pair(point_count, rng):
    """Generate source/target with known transformation."""
    points = rng.standard_normal((point_count, 3)).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [0.1, 0.05, -0.02]  # small translation

    source = mx.array(points)
    target = mx.array((points @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
    return source, target, T


def test_bench_icp_point_to_point(benchmark, source_target_pair):
    source, target, _ = source_target_pair
    from open3d_mlx.geometry import PointCloud
    from open3d_mlx.pipelines.registration import registration_icp

    src_pcd = PointCloud(source)
    tgt_pcd = PointCloud(target)

    result = benchmark(
        registration_icp, src_pcd, tgt_pcd,
        max_correspondence_distance=0.5,
    )
    assert result.fitness > 0.9


# benchmarks/bench_tsdf.py

def test_bench_tsdf_integrate_128(benchmark):
    """Benchmark single frame TSDF integration at 128³."""
    from open3d_mlx.pipelines.integration import UniformTSDFVolume
    from open3d_mlx.camera import PinholeCameraIntrinsic

    volume = UniformTSDFVolume(length=4.0, resolution=128, sdf_trunc=0.04)
    intrinsic = PinholeCameraIntrinsic.prime_sense_default()
    depth = mx.array(np.random.randint(500, 3000, (480, 640), dtype=np.uint16))
    extrinsic = mx.eye(4)

    benchmark(volume.integrate, depth, intrinsic, extrinsic)
```

### 2.4 Running Benchmarks

```bash
# Full benchmark suite
uv run pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Single benchmark
uv run pytest benchmarks/bench_icp.py -k "10000" --benchmark-only

# Compare against saved baseline
uv run pytest benchmarks/ --benchmark-compare=baseline.json
```

---

## 3. CI: GitHub Actions

### 3.1 Workflow: `.github/workflows/test.yml`

```yaml
name: Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: macos-14  # M1 runner
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          uv venv .venv --python ${{ matrix.python-version }}
          uv pip install -e ".[dev]"

      - name: Run tests
        run: uv run pytest tests/ -q --tb=short

      - name: Run import smoke test
        run: |
          uv run python -c "
          import open3d_mlx
          print(f'Version: {open3d_mlx.__version__}')
          import mlx.core as mx
          print(f'MLX device: {mx.default_device()}')
          print('All imports OK')
          "

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv venv .venv && uv pip install ruff
      - run: uv run ruff check open3d_mlx/ tests/

  benchmark:
    runs-on: macos-14
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: test

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv venv .venv && uv pip install -e ".[dev]"
      - name: Run benchmarks
        run: uv run pytest benchmarks/ --benchmark-only --benchmark-json=benchmark.json
      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark.json
```

---

## 4. Performance Reporting

### 4.1 Benchmark Results Table (Target)

After initial implementation, we publish this table in README:

```markdown
| Operation | Points | Open3D CPU | Open3D-MLX (M1) | Speedup |
|-----------|--------|-----------|-----------------|---------|
| ICP P2P | 10K | TBD ms | TBD ms | TBDx |
| ICP P2P | 100K | TBD ms | TBD ms | TBDx |
| KNN (k=30) | 100K | TBD ms | TBD ms | TBDx |
| Voxel Down | 1M | TBD ms | TBD ms | TBDx |
| TSDF Integrate | 128³ | TBD ms | TBD ms | TBDx |
```

### 4.2 Memory Reporting

```python
def report_memory():
    """Report MLX Metal memory usage."""
    active = mx.get_active_memory() / 1e6
    peak = mx.get_peak_memory() / 1e6
    print(f"Active: {active:.1f} MB, Peak: {peak:.1f} MB")
```

---

## 5. Tests for CI

```
test_import_open3d_mlx
test_import_all_submodules
test_mlx_device_available
test_basic_pointcloud_creation
test_basic_icp_runs
test_basic_tsdf_runs
```

---

## 6. Acceptance Criteria

- [ ] Benchmark suite runs with `uv run pytest benchmarks/`
- [ ] Benchmarks cover ICP, TSDF, KNN, downsampling, I/O
- [ ] Results are saved as JSON for comparison
- [ ] GitHub Actions workflow passes on macos-14 (Apple Silicon)
- [ ] CI tests both Python 3.11 and 3.12
- [ ] Lint check passes with ruff
- [ ] Import smoke test validates all modules load correctly
- [ ] Benchmark results are uploaded as CI artifacts
