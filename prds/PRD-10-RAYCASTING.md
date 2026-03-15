# PRD-10: Volume Raycasting

## Status: P1 — Core Pipeline
## Priority: P1
## Phase: 4 — Raycasting
## Estimated Effort: 2–3 days
## Depends On: PRD-08 (TSDF Volume)
## Blocks: PRD-11

---

## 1. Objective

Implement ray-TSDF volume marching — casting rays from a virtual camera through the TSDF volume to render synthetic depth maps and normal maps. This is critical for:

1. **ICP loop closure**: Generate synthetic depth from the reconstruction to align against new scans
2. **Occlusion checking**: Determine which parts of the volume are visible from a given viewpoint
3. **Visualization**: Render the reconstruction from any viewpoint without mesh extraction

---

## 2. Upstream Reference

| Our File | Upstream File | Lines | Notes |
|----------|--------------|-------|-------|
| `pipelines/raycasting/raycasting_scene.py` | `cpp/open3d/t/geometry/RaycastingScene.cpp` | 1,746 | Full raycasting |
| `pipelines/raycasting/ray_utils.py` | (internal to RaycastingScene) | — | Ray generation |

Note: Upstream RaycastingScene supports ray-triangle intersection (BVH). We implement only **ray-TSDF marching** for now — the use case that needs GPU acceleration.

---

## 3. Algorithm: Ray-TSDF Marching

```
For each pixel (u, v) in output image:
    1. Generate ray: origin = camera_pos, direction = unproject(u, v, intrinsics)
    2. Transform ray to volume frame
    3. Find ray-AABB intersection (entry/exit t values)
    4. March along ray from t_entry to t_exit:
       a. Sample TSDF at current position (trilinear interpolation)
       b. If TSDF sign changes (positive → negative): found surface
          - Bisect to find zero-crossing
          - Record depth = t at zero-crossing
          - Compute normal from TSDF gradient
       c. Step size = max(|tsdf_value| * voxel_size, min_step)
    5. Output: depth[v, u] = t_hit, normal[v, u] = gradient
```

### 3.1 Adaptive Step Size

The key optimization: step size adapts to the distance field value. Far from surfaces (large |TSDF|), take big steps. Near surfaces (small |TSDF|), take small steps. This typically requires 50-100 steps per ray vs. 512+ for fixed stepping.

---

## 4. API Design

### 4.1 Ray Generation

```python
# pipelines/raycasting/ray_utils.py

def generate_rays(
    intrinsic: PinholeCameraIntrinsic,
    extrinsic: mx.array,
    width: int | None = None,
    height: int | None = None,
) -> mx.array:
    """Generate rays for all pixels in a camera view.

    Args:
        intrinsic: Camera intrinsics.
        extrinsic: (4, 4) camera-to-world transformation.
        width: Image width (default: intrinsic.width).
        height: Image height (default: intrinsic.height).

    Returns:
        rays: (H, W, 6) array where rays[v, u] = [ox, oy, oz, dx, dy, dz].
              Origins are in world space, directions are unit vectors.
    """


def generate_rays_flat(
    intrinsic: PinholeCameraIntrinsic,
    extrinsic: mx.array,
) -> mx.array:
    """Generate rays as flat (H*W, 6) array for batch processing."""
```

### 4.2 Raycasting

```python
# pipelines/raycasting/raycasting_scene.py

class RaycastingScene:
    """Ray-TSDF volume intersection for rendering.

    Renders synthetic depth and normal maps from a TSDF volume.
    """

    def __init__(self):
        self._volume = None

    def set_volume(self, volume: UniformTSDFVolume | ScalableTSDFVolume) -> None:
        """Set the TSDF volume to raycast against."""
        self._volume = volume

    def cast_rays(
        self,
        rays: mx.array,
        max_steps: int = 200,
        min_step_size: float = 0.001,
    ) -> dict[str, mx.array]:
        """Cast rays against the TSDF volume.

        Args:
            rays: (N, 6) ray origins and directions.
            max_steps: Maximum steps per ray.
            min_step_size: Minimum step size (meters).

        Returns:
            dict with:
                "t_hit": (N,) float32 — distance along ray to surface (inf = miss)
                "normals": (N, 3) float32 — surface normals at hit points
                "positions": (N, 3) float32 — world-space hit positions
        """

    def render_depth(
        self,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic: mx.array,
        max_steps: int = 200,
    ) -> mx.array:
        """Render a depth image from the TSDF volume.

        Args:
            intrinsic: Camera intrinsics.
            extrinsic: (4, 4) camera-to-world transformation.

        Returns:
            depth: (H, W) float32 depth image in meters.
        """

    def render_normal(
        self,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic: mx.array,
        max_steps: int = 200,
    ) -> mx.array:
        """Render a normal map from the TSDF volume.

        Returns:
            normals: (H, W, 3) float32 normal map.
        """
```

---

## 5. Implementation Details

### 5.1 Ray Generation (MLX-vectorized)

```python
def generate_rays(intrinsic, extrinsic, width=None, height=None):
    W = width or intrinsic.width
    H = height or intrinsic.height

    # Pixel coordinates
    u = mx.arange(W, dtype=mx.float32)
    v = mx.arange(H, dtype=mx.float32)
    uu, vv = mx.meshgrid(u, v, indexing='xy')  # (H, W)

    # Unproject to camera-frame directions
    dx = (uu - intrinsic.cx) / intrinsic.fx
    dy = (vv - intrinsic.cy) / intrinsic.fy
    dz = mx.ones_like(dx)

    # Stack and normalize
    dirs_cam = mx.stack([dx, dy, dz], axis=-1)  # (H, W, 3)
    dirs_cam = dirs_cam / mx.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # Rotate to world frame
    R = extrinsic[:3, :3]  # camera-to-world rotation
    t = extrinsic[:3, 3]   # camera position in world

    dirs_world = mx.einsum('ij,hwj->hwi', R, dirs_cam)  # (H, W, 3)

    # Origin is camera position (same for all rays)
    origins = mx.broadcast_to(t, dirs_world.shape)  # (H, W, 3)

    return mx.concatenate([origins, dirs_world], axis=-1)  # (H, W, 6)
```

### 5.2 TSDF Ray Marching (Core Loop)

```python
def _march_rays(
    rays: mx.array,          # (N, 6)
    tsdf: mx.array,          # (R, R, R)
    weight: mx.array,        # (R, R, R)
    volume_origin: mx.array, # (3,)
    voxel_size: float,
    resolution: int,
    max_steps: int = 200,
    min_step: float = 0.001,
) -> tuple[mx.array, mx.array]:
    """March rays through TSDF volume.

    Returns:
        t_hit: (N,) distance to surface (inf = no hit)
        hit_normals: (N, 3) surface normals
    """
    N = rays.shape[0]
    origins = rays[:, :3]     # (N, 3)
    directions = rays[:, 3:]  # (N, 3)

    # Initialize
    t_hit = mx.full((N,), float('inf'), dtype=mx.float32)
    hit_normals = mx.zeros((N, 3), dtype=mx.float32)

    # Ray-AABB intersection to find entry/exit
    volume_max = volume_origin + resolution * voxel_size
    t_entry, t_exit = _ray_aabb_intersect(origins, directions, volume_origin, volume_max)

    # Active mask
    active = t_entry < t_exit
    t_current = mx.where(active, t_entry + min_step, mx.zeros_like(t_entry))

    prev_tsdf = mx.ones((N,), dtype=mx.float32)  # start positive (outside)

    for step in range(max_steps):
        # Current position along ray
        pos = origins + t_current[:, None] * directions  # (N, 3)

        # Sample TSDF (trilinear interpolation)
        tsdf_val = _sample_tsdf_trilinear(pos, tsdf, volume_origin, voxel_size, resolution)

        # Check for zero-crossing (sign change)
        crossing = active & (prev_tsdf > 0) & (tsdf_val < 0)

        if mx.any(crossing).item():
            # Bisect for accurate surface position
            t_surface = _bisect_zero_crossing(
                origins, directions, t_current - min_step, t_current,
                tsdf, volume_origin, voxel_size, resolution, crossing
            )
            t_hit = mx.where(crossing, t_surface, t_hit)

            # Compute normal from TSDF gradient
            hit_pos = origins + t_surface[:, None] * directions
            normals = _compute_tsdf_gradient(hit_pos, tsdf, volume_origin, voxel_size, resolution)
            hit_normals = mx.where(crossing[:, None], normals, hit_normals)

            # Deactivate rays that hit
            active = active & ~crossing

        if not mx.any(active).item():
            break

        # Adaptive step
        step_size = mx.maximum(mx.abs(tsdf_val) * voxel_size, min_step)
        t_current = t_current + step_size
        active = active & (t_current < t_exit)
        prev_tsdf = tsdf_val

    return t_hit, hit_normals
```

### 5.3 TSDF Gradient (for normals)

```python
def _compute_tsdf_gradient(pos, tsdf, origin, voxel_size, resolution):
    """Central difference gradient of TSDF at given positions."""
    eps = voxel_size * 0.5
    dx = (_sample_tsdf(pos + mx.array([eps, 0, 0]), ...) -
          _sample_tsdf(pos - mx.array([eps, 0, 0]), ...))
    dy = (_sample_tsdf(pos + mx.array([0, eps, 0]), ...) -
          _sample_tsdf(pos - mx.array([0, eps, 0]), ...))
    dz = (_sample_tsdf(pos + mx.array([0, 0, eps]), ...) -
          _sample_tsdf(pos - mx.array([0, 0, eps]), ...))
    grad = mx.stack([dx, dy, dz], axis=-1)
    return grad / (mx.linalg.norm(grad, axis=-1, keepdims=True) + 1e-10)
```

---

## 6. Performance Considerations

- **Vectorization**: All N rays processed in parallel (single MLX graph)
- **Adaptive stepping**: Reduces iterations from 512 to ~50-100
- **Trilinear interpolation**: 8 voxel reads per sample (cache-friendly in MLX)
- **Early termination**: Rays that hit are deactivated (reduces work per step)
- **For max speed**: Consider Metal kernel for the march loop (future optimization)

---

## 7. Tests

```
# Ray generation
test_generate_rays_shape
test_generate_rays_center_pixel_direction
test_generate_rays_corner_directions
test_generate_rays_identity_extrinsic

# Ray-AABB intersection
test_ray_aabb_hit
test_ray_aabb_miss
test_ray_aabb_inside

# TSDF sampling
test_sample_tsdf_exact_voxel_center
test_sample_tsdf_interpolation
test_sample_tsdf_outside_volume

# Raycasting
test_raycast_flat_wall
test_raycast_sphere_volume
test_raycast_empty_volume_no_hits
test_raycast_depth_image_shape
test_raycast_normal_image_shape

# Render
test_render_depth_from_integrated_volume
test_render_normal_from_integrated_volume
test_depth_matches_original_depth_frame

# Cross-validation
test_depth_render_matches_open3d  # if open3d installed
```

---

## 8. Acceptance Criteria

- [ ] Ray generation produces correct directions for known intrinsics
- [ ] TSDF trilinear interpolation returns correct values at voxel centers
- [ ] Raycasting a flat wall produces uniform depth
- [ ] Raycasting a sphere TSDF produces circular depth pattern
- [ ] Normal computation gives correct surface normals
- [ ] `render_depth()` returns (H, W) depth image
- [ ] `render_normal()` returns (H, W, 3) normal map
- [ ] Rendered depth from integrated volume approximately matches original depth frames
- [ ] 640×480 render completes in < 5 seconds on M1 (128³ volume)
- [ ] All tests pass
