# PRD-09: Scalable TSDF, Voxel Hashing & Marching Cubes

## Status: P1 — Extended Integration
## Priority: P1
## Phase: 3 — Integration
## Estimated Effort: 3–4 days
## Depends On: PRD-08
## Blocks: PRD-10, PRD-11

---

## 1. Objective

Extend TSDF integration with:
1. **Scalable TSDF Volume** — hash-based sparse voxel storage for large scenes (rooms, buildings) where a uniform grid would waste memory on empty space
2. **Marching Cubes** — extract a triangle mesh from the TSDF zero-isosurface
3. **Voxel Block Hashing** — the data structure that enables scalable TSDF

---

## 2. Upstream Reference

| Our File | Upstream File | Lines | Notes |
|----------|--------------|-------|-------|
| `pipelines/integration/scalable_tsdf.py` | `ScalableTSDFVolume.cpp` | 442 | Scalable TSDF |
| `pipelines/integration/marching_cubes.py` | `MarchingCubesConst.h` | 328 | Lookup tables |
| `ops/voxel_ops.py` | `cpp/open3d/core/hashmap/HashMap.h/cpp` | ~500 | Hashmap |

---

## 3. Voxel Block Hashing

### 3.1 Concept

Instead of allocating a full R³ grid, allocate **blocks** (e.g., 8³ voxels) only where depth data exists. A hash table maps block coordinates to block storage.

```
Block key: (bx, by, bz) = floor(world_pos / (block_size * voxel_size))
Block storage: (8, 8, 8) TSDF + weight arrays per active block
```

### 3.2 Data Structure

```python
class VoxelBlockGrid:
    """Sparse voxel block grid using hash-based spatial indexing.

    Each active region is stored as a block of block_resolution³ voxels.
    Only blocks near observed surfaces are allocated.
    """

    def __init__(
        self,
        voxel_size: float = 0.006,
        block_resolution: int = 8,
        block_count: int = 10000,
    ):
        self.voxel_size = voxel_size
        self.block_resolution = block_resolution  # 8 = 8³ voxels per block
        self.block_size = voxel_size * block_resolution

        # Hash table: block_key (int64→int32) → block_index
        self._block_keys: dict[tuple[int,int,int], int] = {}
        self._blocks_tsdf: list[mx.array] = []   # each (8,8,8)
        self._blocks_weight: list[mx.array] = []  # each (8,8,8)
        self._blocks_color: list[mx.array] | None = None

    def _get_or_create_block(self, bx: int, by: int, bz: int) -> int:
        """Get block index, creating new block if needed."""

    def integrate(self, depth, intrinsic, extrinsic, depth_scale=1000.0, depth_max=3.0):
        """Integrate depth frame into sparse voxel grid.

        1. Determine which blocks the depth frustum intersects
        2. Allocate new blocks as needed
        3. For each active block, run the TSDF integration kernel
        """

    def extract_point_cloud(self) -> "PointCloud":
        """Extract surface points from all active blocks."""

    def extract_triangle_mesh(self) -> dict:
        """Extract mesh via marching cubes on all active blocks."""
```

### 3.3 Block Allocation Strategy

```python
def _allocate_blocks_from_depth(self, depth, intrinsic, extrinsic, depth_scale, depth_max):
    """Determine which blocks to allocate based on depth frustum.

    1. For a sparse set of depth pixels (stride=4):
       a. Unproject to 3D world point
       b. Compute block key from world position
       c. Also add neighboring blocks within truncation distance
    2. Allocate any new blocks not yet in hash table
    """
```

---

## 4. Marching Cubes

### 4.1 Algorithm

Marching cubes extracts a triangle mesh from a scalar field's zero-isosurface:

1. For each 2³ cube of voxels, classify vertices as inside/outside (TSDF sign)
2. Look up edge pattern in 256-entry table
3. Interpolate vertex positions along edges where sign changes
4. Emit triangles

### 4.2 Implementation

```python
# pipelines/integration/marching_cubes.py

# The 256-entry lookup tables (from upstream MarchingCubesConst.h)
EDGE_TABLE = [...]   # 256 entries, each a 12-bit mask of intersected edges
TRI_TABLE = [...]    # 256 × 16 entries, triangle vertex indices (-1 terminated)

def marching_cubes(
    tsdf: mx.array,
    weight: mx.array,
    voxel_size: float,
    origin: mx.array,
    weight_threshold: float = 0.0,
) -> tuple[mx.array, mx.array]:
    """Extract triangle mesh from TSDF volume via marching cubes.

    Args:
        tsdf: (R, R, R) signed distance values.
        weight: (R, R, R) integration weights.
        voxel_size: Size of each voxel.
        origin: (3,) world-space origin of the volume.
        weight_threshold: Minimum weight for a voxel to be considered valid.

    Returns:
        vertices: (V, 3) float32 mesh vertices.
        triangles: (F, 3) int32 triangle indices.
    """
```

### 4.3 MLX-Friendly Implementation Strategy

Full vectorization of marching cubes is complex. Strategy:

**Phase 1 (this PRD)**: NumPy-based marching cubes on CPU.
- Transfer TSDF to numpy: `np.array(tsdf)`
- Run marching cubes in numpy (or use `scikit-image.measure.marching_cubes` if available)
- Return mesh vertices/triangles as MLX arrays

**Phase 2 (future)**: MLX-native marching cubes.
- Vectorize cube classification across all cubes
- Use lookup tables as MLX index operations
- Edge interpolation as batch operation

```python
def marching_cubes(tsdf, weight, voxel_size, origin, weight_threshold=0.0):
    tsdf_np = np.array(tsdf)
    weight_np = np.array(weight)

    # Mask out low-weight voxels
    tsdf_np[weight_np <= weight_threshold] = 1.0  # outside

    try:
        from skimage.measure import marching_cubes as sk_marching_cubes
        verts, faces, _, _ = sk_marching_cubes(tsdf_np, level=0.0, spacing=(voxel_size,)*3)
        verts += np.array(origin)
        return mx.array(verts.astype(np.float32)), mx.array(faces.astype(np.int32))
    except ImportError:
        # Fallback: our own implementation
        return _marching_cubes_numpy(tsdf_np, weight_np, voxel_size, np.array(origin))
```

---

## 5. Scalable TSDF Volume API

```python
class ScalableTSDFVolume:
    """Scalable hash-based TSDF volume for large scenes.

    Matches Open3D: o3d.pipelines.integration.ScalableTSDFVolume

    Unlike UniformTSDFVolume, this only allocates memory for observed regions.
    Suitable for room-scale and building-scale reconstruction.
    """

    def __init__(
        self,
        voxel_size: float = 0.006,
        sdf_trunc: float = 0.04,
        color: bool = False,
        block_resolution: int = 8,
        block_count: int = 50000,
    ):
        """
        Args:
            voxel_size: Individual voxel size (meters).
            sdf_trunc: SDF truncation distance.
            color: Store per-voxel color.
            block_resolution: Voxels per block side (8 = 8³ voxels per block).
            block_count: Initial block allocation capacity.
        """

    def integrate(
        self,
        depth: mx.array,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic: mx.array,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        color: mx.array | None = None,
    ) -> None:
        """Integrate a depth frame."""

    def extract_point_cloud(self) -> "PointCloud": ...
    def extract_triangle_mesh(self) -> dict: ...

    @property
    def active_block_count(self) -> int:
        """Number of allocated blocks."""

    def reset(self) -> None: ...
```

---

## 6. Tests

```
# Voxel block hashing
test_block_allocation_from_depth
test_block_deduplication
test_block_neighbor_allocation
test_block_count_grows_with_frames

# Scalable TSDF
test_scalable_integrate_single_frame
test_scalable_integrate_multiple_frames
test_scalable_memory_less_than_uniform
test_scalable_extract_pointcloud
test_scalable_extract_mesh

# Marching cubes
test_marching_cubes_sphere_tsdf
test_marching_cubes_plane_tsdf
test_marching_cubes_empty_volume
test_marching_cubes_single_cube_crossing
test_marching_cubes_respects_weight_threshold

# Integration with UniformTSDFVolume
test_uniform_extract_mesh
test_uniform_mesh_vertex_count_reasonable

# Cross-validation
test_scalable_matches_uniform_for_small_scene  # same results on small data
```

---

## 7. Acceptance Criteria

- [ ] ScalableTSDFVolume allocates blocks only near observed surfaces
- [ ] Memory usage is significantly less than equivalent UniformTSDFVolume for sparse scenes
- [ ] Block allocation handles multiple frames incrementally
- [ ] Marching cubes extracts valid mesh from sphere TSDF
- [ ] Marching cubes handles weight thresholding
- [ ] Point cloud extraction works on both uniform and scalable volumes
- [ ] Mesh extraction works on both uniform and scalable volumes
- [ ] `scikit-image` fallback for marching cubes works when installed
- [ ] Native marching cubes works without scikit-image
- [ ] All tests pass
