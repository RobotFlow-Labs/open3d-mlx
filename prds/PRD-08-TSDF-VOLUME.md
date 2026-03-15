# PRD-08: TSDF Volume Integration (Uniform)

## Status: P0 — Core Pipeline
## Priority: P0
## Phase: 3 — Integration
## Estimated Effort: 3–4 days
## Depends On: PRD-00, PRD-01, PRD-02
## Blocks: PRD-09, PRD-10, PRD-11

---

## 1. Objective

Implement Uniform TSDF (Truncated Signed Distance Function) volume — the core data structure for real-time 3D reconstruction from depth cameras. A TSDF volume is a 3D grid where each voxel stores the signed distance to the nearest surface and a weight. Depth frames are integrated into the volume one at a time, and a mesh or point cloud can be extracted from the accumulated volume.

---

## 2. Upstream Reference

| Our File | Upstream File | Lines | Notes |
|----------|--------------|-------|-------|
| `pipelines/integration/tsdf_volume.py` | `cpp/open3d/pipelines/integration/TSDFVolume.h` | 84 | Base interface |
| `pipelines/integration/uniform_tsdf.py` | `cpp/open3d/pipelines/integration/UniformTSDFVolume.cpp` | 531 | Implementation |
| `camera/intrinsics.py` | `cpp/open3d/camera/PinholeCameraIntrinsic.h` | ~100 | Camera model |

Note: There is no tensor-based TSDF in upstream (it's legacy-only). This is a **gap we fill** with MLX-native implementation.

---

## 3. Algorithm

### 3.1 TSDF Integration (per depth frame)

```
For each voxel (i, j, k) in the volume:
    1. Compute voxel center in world coordinates: p_world = origin + (i,j,k) * voxel_size
    2. Transform to camera coordinates: p_cam = extrinsic @ p_world
    3. Project to pixel: (u, v) = project(p_cam, intrinsics)
    4. If (u, v) is within image bounds:
        a. Read depth at (u, v): d = depth[v, u]
        b. Compute signed distance: sdf = d - p_cam.z
        c. If |sdf| < truncation_distance:
            - Weighted average update:
              tsdf_new = (tsdf_old * w_old + sdf * w_frame) / (w_old + w_frame)
              w_new = min(w_old + w_frame, w_max)
            - If color available, same weighted average for RGB
```

### 3.2 Key Insight for MLX

The naive per-voxel loop is O(N³) — too slow in Python. MLX vectorization:

1. Generate ALL voxel centers as (N³, 3) array
2. Transform all voxel centers to camera frame (batch matmul)
3. Project all to pixel coordinates (batch projection)
4. Gather depth values at projected pixels (index into depth image)
5. Compute SDF for all voxels simultaneously
6. Apply truncation mask
7. Weighted average update (vectorized)

This is a single MLX computation graph — no Python loops over voxels.

---

## 4. API Design

### 4.1 Camera Intrinsics

```python
# camera/intrinsics.py

@dataclass
class PinholeCameraIntrinsic:
    """Pinhole camera intrinsic parameters.

    Matches Open3D: o3d.camera.PinholeCameraIntrinsic
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def intrinsic_matrix(self) -> mx.array:
        """3x3 intrinsic matrix K."""
        return mx.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=mx.float32)

    @classmethod
    def prime_sense_default(cls) -> "PinholeCameraIntrinsic":
        """PrimeSense / Kinect default intrinsics."""
        return cls(640, 480, 525.0, 525.0, 319.5, 239.5)
```

### 4.2 TSDF Volume

```python
# pipelines/integration/uniform_tsdf.py

class UniformTSDFVolume:
    """Uniform voxel grid TSDF volume.

    Matches Open3D: o3d.pipelines.integration.UniformTSDFVolume
    """

    def __init__(
        self,
        length: float = 4.0,
        resolution: int = 128,
        sdf_trunc: float = 0.04,
        color: bool = False,
        origin: mx.array | None = None,
    ):
        """Create TSDF volume.

        Args:
            length: Total side length of the cubic volume (meters).
            resolution: Number of voxels per side (total voxels = resolution³).
            sdf_trunc: Truncation distance for SDF values.
            color: If True, also store per-voxel color.
            origin: (3,) volume origin in world coordinates. Default: (0,0,0).
        """
        self.length = length
        self.resolution = resolution
        self.voxel_size = length / resolution
        self.sdf_trunc = sdf_trunc

        self.origin = origin if origin is not None else mx.zeros(3)

        # Volume storage: (R, R, R) for TSDF values and weights
        self._tsdf = mx.zeros((resolution, resolution, resolution), dtype=mx.float32)
        self._weight = mx.zeros((resolution, resolution, resolution), dtype=mx.float32)
        if color:
            self._color = mx.zeros((resolution, resolution, resolution, 3), dtype=mx.float32)
        else:
            self._color = None

    def reset(self) -> None:
        """Clear the volume."""

    def integrate(
        self,
        depth: mx.array,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic: mx.array,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        color: mx.array | None = None,
    ) -> None:
        """Integrate a depth frame into the volume.

        Args:
            depth: (H, W) uint16 or float32 depth image.
            intrinsic: Camera intrinsics.
            extrinsic: (4, 4) camera-to-world (or world-to-camera) transformation.
            depth_scale: Depth values are divided by this to get meters.
            depth_max: Maximum depth to integrate (meters).
            color: (H, W, 3) optional color image.
        """

    def extract_point_cloud(self) -> "PointCloud":
        """Extract surface points from TSDF zero-crossings.

        Finds voxels where TSDF transitions through zero and
        interpolates the surface position.
        """

    def extract_triangle_mesh(self) -> dict:
        """Extract mesh via marching cubes.

        Returns dict with 'vertices' (V, 3) and 'triangles' (F, 3).
        (Full TriangleMesh class is future scope.)
        """

    @property
    def tsdf(self) -> mx.array:
        """Raw TSDF values (R, R, R)."""

    @property
    def weight(self) -> mx.array:
        """Integration weights (R, R, R)."""
```

---

## 5. Implementation Details

### 5.1 Vectorized Integration

```python
def integrate(self, depth, intrinsic, extrinsic, depth_scale=1000.0, depth_max=3.0, color=None):
    R = self.resolution

    # 1. Generate voxel centers (R³, 3) — precompute once and cache
    if not hasattr(self, '_voxel_centers'):
        grid = mx.meshgrid(
            mx.arange(R, dtype=mx.float32),
            mx.arange(R, dtype=mx.float32),
            mx.arange(R, dtype=mx.float32),
            indexing='ij'
        )
        self._voxel_centers = mx.stack(grid, axis=-1).reshape(-1, 3)  # (R³, 3)
        self._voxel_centers = self._voxel_centers * self.voxel_size + self.origin

    voxel_world = self._voxel_centers  # (R³, 3)

    # 2. Transform to camera frame
    # extrinsic is world_to_camera (4x4)
    R_mat = extrinsic[:3, :3]
    t_vec = extrinsic[:3, 3]
    voxel_cam = voxel_world @ R_mat.T + t_vec  # (R³, 3)

    # 3. Project to pixel coordinates
    z = voxel_cam[:, 2]
    u = (voxel_cam[:, 0] * intrinsic.fx / z + intrinsic.cx)
    v = (voxel_cam[:, 1] * intrinsic.fy / z + intrinsic.cy)

    # 4. Validity mask
    valid = (
        (z > 0) &
        (u >= 0) & (u < intrinsic.width - 1) &
        (v >= 0) & (v < intrinsic.height - 1)
    )

    # 5. Sample depth at projected pixels (nearest neighbor)
    u_int = mx.clip(u.astype(mx.int32), 0, intrinsic.width - 1)
    v_int = mx.clip(v.astype(mx.int32), 0, intrinsic.height - 1)

    depth_float = depth.astype(mx.float32) / depth_scale
    sampled_depth = depth_float[v_int, u_int]  # (R³,)

    # 6. Compute SDF
    sdf = sampled_depth - z  # positive = in front of surface

    # 7. Truncation mask
    valid = valid & (sampled_depth > 0) & (sampled_depth < depth_max)
    valid = valid & (mx.abs(sdf) < self.sdf_trunc)

    # 8. Normalize SDF to [-1, 1]
    tsdf_val = mx.clip(sdf / self.sdf_trunc, -1.0, 1.0)

    # 9. Weighted average update
    tsdf_flat = self._tsdf.reshape(-1)
    weight_flat = self._weight.reshape(-1)

    w_new = mx.where(valid, weight_flat + 1.0, weight_flat)
    tsdf_new = mx.where(
        valid,
        (tsdf_flat * weight_flat + tsdf_val) / mx.maximum(w_new, 1.0),
        tsdf_flat
    )
    w_new = mx.minimum(w_new, 255.0)  # cap weight

    self._tsdf = tsdf_new.reshape(R, R, R)
    self._weight = w_new.reshape(R, R, R)
    mx.eval(self._tsdf, self._weight)  # Materialize
```

### 5.2 Point Cloud Extraction

```python
def extract_point_cloud(self):
    """Extract surface points at TSDF zero-crossings."""
    R = self.resolution
    w_mask = self._weight > 0

    # Find zero-crossings along each axis
    # A zero-crossing exists between voxels where TSDF changes sign
    points_list = []

    for axis in range(3):
        tsdf_shifted = mx.roll(self._tsdf, -1, axis=axis)
        weight_shifted = mx.roll(self._weight, -1, axis=axis)

        # Sign change between adjacent voxels
        crossing = (self._tsdf * tsdf_shifted < 0) & w_mask & (weight_shifted > 0)

        # Interpolate position along the axis
        # t = tsdf[i] / (tsdf[i] - tsdf[i+1])
        t = self._tsdf / (self._tsdf - tsdf_shifted + 1e-10)
        t = mx.clip(t, 0.0, 1.0)

        # ... compute interpolated 3D positions at crossings
        # Add to points_list

    # Concatenate and return PointCloud
    ...
```

---

## 6. Memory Considerations

| Resolution | Voxels | TSDF + Weight Memory | With Color |
|-----------|--------|---------------------|------------|
| 64³ | 262K | 2 MB | 5 MB |
| 128³ | 2M | 16 MB | 40 MB |
| 256³ | 16M | 128 MB | 320 MB |
| 512³ | 134M | 1 GB | 2.5 GB |

For Apple Silicon with 16–128 GB unified memory, 256³ is comfortable, 512³ possible on higher-end machines.

The voxel centers array (R³, 3) adds 3× the single-channel cost. Cache it and reuse across frames.

---

## 7. Tests

```
# Volume creation
test_create_volume_default
test_create_volume_custom_resolution
test_volume_reset_clears_data

# Integration
test_integrate_single_frame
test_integrate_multiple_frames
test_integrate_respects_depth_max
test_integrate_respects_truncation
test_integrate_weighted_average

# Extraction
test_extract_pointcloud_from_single_plane
test_extract_pointcloud_nonempty_after_integration
test_extract_pointcloud_density_increases_with_frames

# Camera
test_pinhole_intrinsic_matrix
test_pinhole_prime_sense_default

# Memory
test_volume_64_fits_in_memory
test_volume_128_fits_in_memory

# Cross-validation
test_integration_matches_open3d  # if open3d installed
```

---

## 8. Acceptance Criteria

- [ ] Volume creates with correct shape (R, R, R)
- [ ] Single depth frame integration updates TSDF values
- [ ] Multiple frame integration accumulates correctly
- [ ] Zero-crossing point extraction returns surface points
- [ ] Depth values beyond `depth_max` are ignored
- [ ] SDF values beyond `sdf_trunc` are ignored
- [ ] Weight capping prevents unbounded growth
- [ ] 128³ volume integrates 1 frame in < 500ms on M1
- [ ] Camera intrinsics produce correct projection
- [ ] All tests pass
