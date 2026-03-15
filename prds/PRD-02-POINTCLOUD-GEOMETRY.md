# PRD-02: Point Cloud Geometry

## Status: P0 — Foundation
## Priority: P0
## Phase: 1 — Foundation
## Estimated Effort: 2 days
## Depends On: PRD-00, PRD-01
## Blocks: PRD-03, PRD-04, PRD-05, PRD-06, PRD-07, PRD-08, PRD-10

---

## 1. Objective

Implement the `PointCloud` class — the central data structure for all 3D operations. Stores points, normals, and colors as MLX arrays. Provides transforms, downsampling, normal estimation, and outlier removal. Follows the tensor-based (`t::geometry::PointCloud`) API patterns from upstream.

---

## 2. Upstream Reference

| Our File | Upstream File | Lines | Notes |
|----------|--------------|-------|-------|
| `geometry/pointcloud.py` | `cpp/open3d/t/geometry/PointCloud.h` | 762 | Class interface |
| `geometry/pointcloud.py` | `cpp/open3d/t/geometry/PointCloud.cpp` | 1,407 | Implementation |
| `ops/normals.py` | `cpp/open3d/t/geometry/PointCloud.cpp` | (subset) | Normal estimation |
| `ops/voxel_ops.py` | `cpp/open3d/t/geometry/PointCloud.cpp` | (subset) | Voxel downsampling |

---

## 3. API Design

### 3.1 PointCloud Class

```python
class PointCloud:
    """MLX-backed point cloud with attributes.

    Attributes are stored as MLX arrays in a dict-like container.
    Required: 'positions' (N, 3). Optional: 'normals', 'colors', custom.
    """

    def __init__(self, points: mx.array | np.ndarray | None = None):
        """Create PointCloud.

        Args:
            points: (N, 3) array of 3D coordinates, or None for empty cloud.
        """

    # --- Properties ---
    @property
    def points(self) -> mx.array:
        """(N, 3) point positions."""

    @points.setter
    def points(self, value: mx.array): ...

    @property
    def normals(self) -> mx.array | None:
        """(N, 3) point normals, or None."""

    @normals.setter
    def normals(self, value: mx.array): ...

    @property
    def colors(self) -> mx.array | None:
        """(N, 3) point colors [0, 1] float32, or None."""

    @colors.setter
    def colors(self, value: mx.array): ...

    def __len__(self) -> int:
        """Number of points."""

    def is_empty(self) -> bool: ...
    def has_normals(self) -> bool: ...
    def has_colors(self) -> bool: ...

    # --- Geometric Queries ---
    def get_min_bound(self) -> mx.array:
        """(3,) minimum coordinates."""

    def get_max_bound(self) -> mx.array:
        """(3,) maximum coordinates."""

    def get_center(self) -> mx.array:
        """(3,) centroid."""

    def get_axis_aligned_bounding_box(self) -> tuple[mx.array, mx.array]:
        """Returns (min_bound, max_bound)."""

    # --- Transforms ---
    def transform(self, transformation: mx.array) -> "PointCloud":
        """Apply 4x4 rigid transformation. Returns new PointCloud."""

    def translate(self, translation: mx.array, relative: bool = True) -> "PointCloud": ...
    def rotate(self, rotation: mx.array, center: mx.array | None = None) -> "PointCloud": ...
    def scale(self, factor: float, center: mx.array | None = None) -> "PointCloud": ...

    # --- Filtering ---
    def select_by_index(self, indices: mx.array, invert: bool = False) -> "PointCloud": ...
    def select_by_mask(self, mask: mx.array, invert: bool = False) -> "PointCloud": ...

    def remove_non_finite_points(self) -> "PointCloud":
        """Remove NaN and Inf points."""

    def remove_duplicated_points(self) -> "PointCloud": ...

    # --- Downsampling ---
    def voxel_down_sample(self, voxel_size: float) -> "PointCloud":
        """Voxel grid downsampling. Returns new PointCloud."""

    def uniform_down_sample(self, every_k_points: int) -> "PointCloud": ...
    def random_down_sample(self, sampling_ratio: float) -> "PointCloud": ...

    # --- Normals ---
    def estimate_normals(
        self,
        max_nn: int = 30,
        radius: float | None = None,
    ) -> None:
        """Estimate normals using PCA on local neighborhoods. Modifies in-place."""

    def normalize_normals(self) -> None:
        """Normalize all normals to unit length. Modifies in-place."""

    def orient_normals_towards_camera(
        self, camera_location: mx.array = None
    ) -> None:
        """Orient normals to face camera. Default camera at origin."""

    # --- Outlier Removal ---
    def remove_statistical_outliers(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> tuple["PointCloud", mx.array]:
        """Returns (filtered_cloud, inlier_mask)."""

    def remove_radius_outliers(
        self, nb_points: int = 2, search_radius: float = 1.0
    ) -> tuple["PointCloud", mx.array]:
        """Returns (filtered_cloud, inlier_mask)."""

    # --- Painting ---
    def paint_uniform_color(self, color: mx.array) -> "PointCloud": ...

    # --- Interop ---
    def to_numpy(self) -> dict[str, np.ndarray]:
        """Export all attributes as numpy dict."""

    @classmethod
    def from_numpy(cls, points: np.ndarray, **kwargs) -> "PointCloud":
        """Create from numpy arrays."""

    # --- Copy ---
    def clone(self) -> "PointCloud":
        """Deep copy."""

    # --- Append ---
    def __add__(self, other: "PointCloud") -> "PointCloud":
        """Concatenate two point clouds."""
```

### 3.2 KDTree Search Params (for API compatibility)

```python
class KDTreeSearchParamKNN:
    """KNN search parameter."""
    def __init__(self, knn: int = 30): ...

class KDTreeSearchParamRadius:
    """Radius search parameter."""
    def __init__(self, radius: float): ...

class KDTreeSearchParamHybrid:
    """Hybrid KNN + radius search parameter."""
    def __init__(self, radius: float, max_nn: int = 30): ...
```

---

## 4. Implementation Details

### 4.1 Internal Storage

```python
class PointCloud:
    def __init__(self, points=None):
        self._attributes: dict[str, mx.array] = {}
        if points is not None:
            if isinstance(points, np.ndarray):
                points = mx.array(points.astype(np.float32))
            check_points_shape(points, "points")
            self._attributes["positions"] = ensure_float32(points)
```

### 4.2 Transform Implementation

```python
def transform(self, transformation: mx.array) -> "PointCloud":
    """Apply 4x4 SE(3) transformation.

    Points: p' = R @ p + t
    Normals: n' = R @ n (rotation only, no translation)
    Colors: unchanged
    """
    R = transformation[:3, :3]  # (3, 3)
    t = transformation[:3, 3]   # (3,)

    new_pcd = self.clone()
    # Broadcast: (N, 3) @ (3, 3)^T + (3,)
    new_pcd.points = self.points @ R.T + t[None, :]

    if self.has_normals():
        new_pcd.normals = self.normals @ R.T

    return new_pcd
```

### 4.3 Voxel Downsampling (MLX-native)

```python
def voxel_down_sample(self, voxel_size: float) -> "PointCloud":
    """Voxel grid downsampling using spatial hashing.

    Algorithm:
    1. Compute voxel indices: idx = floor(points / voxel_size)
    2. Hash voxel indices to unique keys
    3. Group points by key, take mean per group
    """
    # Voxel coordinates
    voxel_idx = mx.floor(self.points / voxel_size).astype(mx.int32)  # (N, 3)

    # Hash: combine x, y, z into single key
    # Use large primes to minimize collisions
    keys = voxel_idx[:, 0] * 73856093 + voxel_idx[:, 1] * 19349663 + voxel_idx[:, 2] * 83492791

    # Sort by key, find unique boundaries, average within groups
    # (uses numpy for unique/scatter since MLX lacks these)
    ...
```

### 4.4 Normal Estimation

```python
def estimate_normals(self, max_nn=30, radius=None):
    """Estimate normals via PCA on local neighborhoods.

    For each point:
    1. Find k nearest neighbors
    2. Compute covariance matrix of neighborhood
    3. Smallest eigenvector of covariance = normal direction

    Uses MLX batched SVD for GPU acceleration.
    """
    # Requires KNN from PRD-04 (ops/nearest_neighbor.py)
    # Fall back to scipy.spatial.KDTree if available
    ...
```

---

## 5. Tests

### `tests/test_geometry/test_pointcloud.py`

```
test_create_empty_pointcloud
test_create_from_mlx_array
test_create_from_numpy
test_len_and_is_empty
test_points_shape_validation
test_has_normals_colors
test_get_min_max_bound
test_get_center
test_transform_identity
test_transform_translation
test_transform_rotation
test_transform_preserves_normals_direction
test_translate_relative
test_translate_absolute
test_rotate_around_center
test_scale_uniform
test_select_by_index
test_select_by_index_invert
test_select_by_mask
test_remove_non_finite_points
test_voxel_down_sample_reduces_count
test_voxel_down_sample_preserves_attributes
test_uniform_down_sample
test_random_down_sample_ratio
test_paint_uniform_color
test_clone_independence
test_add_concatenation
test_numpy_roundtrip
```

### Cross-Validation with Open3D (if installed)

```python
@pytest.mark.skipif(not has_open3d, reason="open3d not installed")
def test_transform_matches_open3d():
    """Our transform produces same result as Open3D."""
    ...

@pytest.mark.skipif(not has_open3d, reason="open3d not installed")
def test_voxel_downsample_matches_open3d():
    """Downsampled count is within ±5% of Open3D result."""
    ...
```

---

## 6. Acceptance Criteria

- [ ] `PointCloud(mx.array(...))` creates valid point cloud
- [ ] `.transform()` with 4x4 matrix correctly transforms points and normals
- [ ] `.voxel_down_sample(0.05)` reduces point count by >50% on dense data
- [ ] `.uniform_down_sample(5)` takes every 5th point
- [ ] `.random_down_sample(0.5)` gives ~50% of original points
- [ ] `.select_by_index()` and `.select_by_mask()` correctly filter
- [ ] `.remove_non_finite_points()` removes NaN/Inf entries
- [ ] `.clone()` creates independent deep copy
- [ ] `pcd1 + pcd2` concatenates points and attributes
- [ ] `.to_numpy()` returns dict with correct shapes
- [ ] All attributes (normals, colors) are carried through transform/filter/downsample ops
- [ ] All tests pass

---

## 7. Deferred to Later PRDs

- `.estimate_normals()` → implemented in PRD-04 (needs KNN)
- `.remove_statistical_outliers()` → implemented in PRD-04 (needs KNN)
- `.remove_radius_outliers()` → implemented in PRD-04 (needs radius search)
- `.cluster_dbscan()` → future scope
- `.segment_plane()` → future scope
