# Open3D-MLX: Master Build Prompt

## Mission

Port Open3D's GPU-accelerated Python pipelines to Apple MLX, giving every roboticist with a Mac native GPU performance for 3D perception workloads.

Open3D (12k+ GitHub stars) is the standard library for 3D data processing in robotics and computer vision. Its GPU pipelines -- ICP registration, TSDF integration, raycasting -- currently require CUDA. This project brings those pipelines to Apple Silicon via MLX, maintaining API compatibility where practical.

**Built by AIFLOW LABS / RobotFlow Labs.**

---

## Scope: What We Port

We port the **GPU-accelerated computational pipelines** that roboticists use daily. These are the operations that benefit most from Metal/MLX acceleration and currently have no Apple Silicon GPU path.

### 1. ICP Registration (Point Cloud Alignment)

Upstream: `cpp/open3d/pipelines/registration/` and `cpp/open3d/t/pipelines/registration/`

| Algorithm | Upstream Files | Priority |
|-----------|---------------|----------|
| Point-to-Point ICP | `Registration.cpp`, `TransformationEstimation.cpp` | P0 |
| Point-to-Plane ICP | `Registration.cpp`, `TransformationEstimation.cpp` | P0 |
| Colored ICP | `ColoredICP.cpp` | P1 |
| Generalized ICP | `GeneralizedICP.cpp` | P1 |
| Correspondence checking | `CorrespondenceChecker.cpp` | P0 |
| Robust kernels (Huber, Tukey, etc.) | `RobustKernel.cpp` | P1 |
| FPFH feature extraction | `Feature.cpp` | P2 |
| Fast Global Registration | `FastGlobalRegistration.cpp` | P2 |

**What this enables:** Scan-to-scan alignment, localization, mapping. The core loop of most 3D SLAM systems.

### 2. TSDF Integration (Depth to 3D Volume)

Upstream: `cpp/open3d/pipelines/integration/`

| Component | Upstream Files | Priority |
|-----------|---------------|----------|
| Uniform TSDF Volume | `UniformTSDFVolume.cpp` | P0 |
| Scalable TSDF Volume | `ScalableTSDFVolume.cpp` | P1 |
| Depth frame integration | `TSDFVolume.h` (virtual interface) | P0 |
| Marching cubes extraction | `MarchingCubesConst.h` | P1 |
| Voxel hashing | `cpp/open3d/core/hashmap/` | P1 |

**What this enables:** Real-time 3D reconstruction from depth cameras (RealSense, ZED, Azure Kinect).

### 3. Raycasting (Volume to Depth/Normals)

Upstream: `cpp/open3d/t/geometry/RaycastingScene.cpp`

| Component | Upstream Files | Priority |
|-----------|---------------|----------|
| Ray-TSDF intersection | `RaycastingScene.cpp` | P0 |
| Depth map rendering | `RaycastingScene.cpp` | P0 |
| Normal map rendering | `RaycastingScene.cpp` | P1 |
| Ray-triangle intersection | `RaycastingScene.cpp` | P2 |

**What this enables:** Synthetic depth rendering, ICP data association, occlusion checking.

### 4. KDTree / Nearest Neighbor Search

Upstream: `cpp/open3d/geometry/KDTreeFlann.cpp`, `cpp/open3d/core/nns/`

| Component | Priority |
|-----------|----------|
| KNN search (k nearest) | P0 |
| Radius search | P0 |
| Hybrid search (KNN + radius) | P1 |
| Fixed-radius NN (GPU-friendly) | P1 |

**What this enables:** Correspondence finding for ICP, feature matching, outlier removal.

Note: KDTree is inherently CPU-friendly. We provide a numpy/scipy fallback and an MLX fixed-radius NN for the GPU path (grid-based spatial hashing).

### 5. Point Cloud I/O

Upstream: `cpp/open3d/io/PointCloudIO.cpp` and format-specific readers

| Format | Priority |
|--------|----------|
| PLY (ASCII + binary) | P0 |
| PCD (ASCII + binary) | P0 |
| XYZ | P2 |
| PTS | P2 |

**What this enables:** Loading and saving point clouds. Boring but essential.

### 6. Core Geometry Types

Upstream: `cpp/open3d/geometry/PointCloud.cpp`, `cpp/open3d/t/geometry/PointCloud.cpp`

| Component | Priority |
|-----------|----------|
| PointCloud (points, normals, colors) | P0 |
| Normal estimation | P0 |
| Downsampling (voxel, uniform, random) | P0 |
| Statistical outlier removal | P1 |
| Radius outlier removal | P1 |
| Crop / bounding box | P1 |
| Transform (rigid, affine) | P0 |

---

## Scope: What We Do NOT Port

This is a focused port. The following are explicitly out of scope:

| Category | Why Not |
|----------|---------|
| **Visualization / rendering** (`visualization/`) | Use Open3D's existing viz, polyscope, or rerun. Not a GPU compute problem. |
| **TensorFlow ML backend** (`python/open3d/ml/tf/`) | We are MLX-native, not a TF bridge. |
| **PyTorch ML backend** (`python/open3d/ml/torch/`) | Same -- this project is MLX ops, not PyTorch wrappers. |
| **Full C++ core** (`cpp/open3d/core/`) | We reimplement only what we need (Tensor wrapper, hashmap) in Python+MLX. |
| **Legacy geometry** (non-tensor path) | We implement the tensor-based (t::geometry) API patterns. |
| **Odometry** (`t/pipelines/odometry/`) | Future scope. Depends on ICP + TSDF being solid first. |
| **SLAM** (`t/pipelines/slam/`) | Future scope. Requires odometry + loop closure. |
| **SLAC** (`t/pipelines/slac/`) | Niche. Not priority for initial release. |
| **Color map optimization** (`pipelines/color_map/`) | Low demand. |
| **Mesh reconstruction** (Poisson, ball pivoting) | Future scope. |
| **Image / RGBD processing** | Use existing libraries (PIL, cv2). |
| **Web visualizer** | Out of scope entirely. |
| **File formats** (GLTF, OBJ, STL, E57, LAS) | Future scope. PLY and PCD cover 90% of robotics use cases. |

---

## Build Order

### Phase 1: Foundation (Weeks 1-2)

1. **Core types** -- MLX-backed tensor wrapper, Device abstraction, PointCloud class
2. **I/O** -- PLY and PCD readers/writers
3. **Basic ops** -- Transform, downsample, normal estimation
4. **KDTree** -- CPU KNN/radius search (scipy.spatial.KDTree or FLANN wrapper)

Milestone: Load a PLY file, downsample it, estimate normals, save it back.

### Phase 2: Registration (Weeks 3-5)

5. **Correspondence search** -- KNN-based, fixed-radius grid
6. **Point-to-point ICP** -- SVD-based transformation estimation on MLX
7. **Point-to-plane ICP** -- Linearized least-squares on MLX
8. **Convergence criteria** -- RMSE, fitness, max iterations
9. **Robust kernels** -- Huber, Tukey, GM for outlier rejection

Milestone: Align two point cloud scans end-to-end on GPU.

### Phase 3: Integration (Weeks 6-8)

10. **Uniform TSDF volume** -- Voxel grid with SDF + weight storage
11. **Depth integration** -- Project depth frames into TSDF volume (MLX kernel)
12. **Marching cubes** -- Extract mesh from TSDF
13. **Scalable TSDF** -- Voxel block hashing for large scenes

Milestone: Integrate 50 depth frames into a 3D reconstruction.

### Phase 4: Raycasting (Weeks 9-10)

14. **Ray generation** -- Camera intrinsics to ray origins/directions
15. **Ray-TSDF marching** -- Step through volume, find zero-crossing
16. **Depth + normal rendering** -- Output depth map and normal map from volume

Milestone: Render synthetic depth from TSDF volume for ICP loop closure.

### Phase 5: Polish (Weeks 11-12)

17. **Benchmarks** -- Comparative benchmarks vs Open3D CUDA (RTX 3090) and Open3D CPU
18. **API cleanup** -- Ensure API feels natural alongside Open3D
19. **Documentation** -- API docs, tutorials, migration guide from Open3D
20. **CI** -- GitHub Actions on M1/M2/M3 runners

---

## Technical Decisions

### Why MLX, Not Metal Directly?

MLX provides a high-level array framework with automatic differentiation, lazy evaluation, and unified memory. Writing raw Metal shaders gains marginal performance at massive engineering cost. MLX is the right abstraction level for this project.

### Tensor Design

Open3D's tensor API (`o3d.core.Tensor`) wraps DLPack-compatible tensors across devices. We simplify:

```python
# Our tensors ARE mlx.core.array
# No wrapper needed -- mlx.core.array is the tensor
points = mx.array(numpy_points)  # float32, unified memory
```

We use `mlx.core.array` directly as the tensor type. No wrapper class. This gives us zero-copy interop with NumPy and PyTorch (via DLPack) for free.

### GPU Nearest Neighbor

KDTree is CPU-bound by nature. For the GPU path (used in ICP inner loop), we implement fixed-radius nearest neighbor via spatial hashing:

1. Voxelize space into grid cells
2. Hash point positions to cells (MLX kernel)
3. For each query point, check 27 neighboring cells
4. Filter by distance threshold

This is the same approach Open3D uses in `core/nns/FixedRadiusIndex`.

### Memory Model

MLX uses unified memory -- no CPU-GPU copies. This is a major advantage over CUDA Open3D where `Tensor.cuda()` / `Tensor.cpu()` transfers dominate small-batch latency.

---

## Upstream Reference

The upstream Open3D source is cloned at `repositories/open3d-upstream/` for reference. Key paths:

```
cpp/open3d/
  core/                          # Tensor, Device, Hashmap, NNS
  geometry/                      # Legacy PointCloud, KDTreeFlann
  pipelines/
    registration/                # ICP, FPFH, GlobalOptimization
    integration/                 # TSDF volumes
  t/
    geometry/                    # Tensor-based PointCloud, RaycastingScene
    pipelines/
      registration/              # Tensor-based ICP (GPU path)
      slam/                      # SLAM pipeline
  io/                            # File readers/writers
  visualization/                 # NOT PORTED

python/open3d/
  ml/torch/                      # NOT PORTED
  ml/tf/                         # NOT PORTED
```

**Always refer to the tensor-based (`t/`) implementations when they exist** -- those are the GPU-optimized paths we are porting.

---

## API Design Goals

1. **Familiar to Open3D users** -- Same function names and parameters where possible
2. **MLX-native** -- Use `mlx.core.array` directly, not custom tensor wrappers
3. **Zero-copy NumPy interop** -- `np.array(mlx_result)` and `mx.array(np_data)` just work
4. **Composable** -- Each module works independently; users can mix with vanilla Open3D

Example target API:

```python
import open3d_mlx as o3m

# Load point clouds
source = o3m.io.read_point_cloud("scan_001.ply")
target = o3m.io.read_point_cloud("scan_002.ply")

# Downsample
source_down = source.voxel_down_sample(0.02)
target_down = target.voxel_down_sample(0.02)

# Estimate normals
source_down.estimate_normals(search_param=o3m.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(search_param=o3m.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# ICP registration
result = o3m.pipelines.registration.registration_icp(
    source_down, target_down,
    max_correspondence_distance=0.05,
    estimation_method=o3m.pipelines.registration.TransformationEstimationPointToPlane()
)

print(result.transformation)  # 4x4 MLX array
print(result.fitness)         # fraction of correspondences
print(result.inlier_rmse)     # RMSE of inlier correspondences
```

---

## Project Info

- **Organization:** AIFLOW LABS / RobotFlow Labs
- **License:** Apache 2.0 (matching Open3D upstream)
- **Python:** >= 3.10
- **Platform:** macOS (Apple Silicon M1/M2/M3/M4)
