# Open3D-MLX Examples

Self-contained example scripts demonstrating each major feature of the Open3D-MLX library.

## Running

```bash
# Run all examples
cd open3d-mlx
for f in examples/0*.py; do python "$f"; done

# Run a single example
python examples/01_pointcloud_basics.py
```

## Examples

| # | Script | What it demonstrates |
|---|--------|---------------------|
| 01 | `01_pointcloud_basics.py` | Create point clouds, transform, voxel downsample, estimate normals, paint, filter |
| 02 | `02_io_read_write.py` | PLY and PCD file I/O roundtrip with colors |
| 03 | `03_icp_registration.py` | Point-to-point and point-to-plane ICP registration on a sphere |
| 04 | `04_tsdf_reconstruction.py` | TSDF volume integration from synthetic depth frames, surface extraction |
| 05 | `05_raycasting.py` | Render depth and normal maps from a TSDF volume via ray marching |
| 06 | `06_robust_kernels.py` | L2, Huber, Tukey, Cauchy, and Geman-McClure robust loss weight functions |
| 07 | `07_nearest_neighbor.py` | CPU KDTree (KNN), GPU spatial hash, and hybrid neighbor search |
| 08 | `08_full_pipeline.py` | End-to-end: generate scans, I/O roundtrip, downsample, ICP register, merge |

## Requirements

- Python >= 3.10
- MLX >= 0.22.0
- NumPy >= 1.24
- SciPy (for KDTree-based neighbor search)

Install the library in dev mode:

```bash
pip install -e ".[dev]"
```
