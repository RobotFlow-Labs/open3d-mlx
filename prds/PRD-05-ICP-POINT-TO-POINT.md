# PRD-05: ICP Registration — Point-to-Point

## Status: P0 — Core Pipeline
## Priority: P0
## Phase: 2 — Registration
## Estimated Effort: 2–3 days
## Depends On: PRD-00, PRD-01, PRD-02, PRD-04
## Blocks: PRD-06, PRD-07, PRD-11

---

## 1. Objective

Implement point-to-point ICP (Iterative Closest Point) registration — the most basic and widely used scan alignment algorithm. Given two point clouds (source, target), find the rigid transformation that aligns them.

This is the first pipeline module. It establishes the registration loop, convergence criteria, result types, and correspondence search patterns that PRD-06 and PRD-07 build on.

---

## 2. Upstream Reference

| Our File | Upstream File | Lines | Notes |
|----------|--------------|-------|-------|
| `pipelines/registration/icp.py` | `cpp/open3d/t/pipelines/registration/Registration.cpp` | 471 | ICP loop |
| `pipelines/registration/transformation.py` | `cpp/open3d/t/pipelines/registration/TransformationEstimation.cpp` | 397 | SVD-based estimation |
| `pipelines/registration/convergence.py` | `cpp/open3d/t/pipelines/registration/Registration.h` | (structs) | Criteria |
| `pipelines/registration/result.py` | `cpp/open3d/t/pipelines/registration/Registration.h` | (structs) | Result type |
| `pipelines/registration/correspondence.py` | `cpp/open3d/t/pipelines/kernel/` | (kernels) | Correspondence search |

---

## 3. Algorithm

### 3.1 ICP Loop (pseudocode)

```
Input: source (N, 3), target (M, 3), max_distance, init_transform, max_iter
Output: transformation (4, 4), fitness, inlier_rmse

1. Apply init_transform to source
2. Build NN index on target
3. Repeat until convergence:
   a. Find correspondences: for each source point, find nearest target within max_distance
   b. Reject outliers: distance > max_distance → no correspondence
   c. Estimate transformation: minimize Σ ||T(s_i) - t_c(i)||²
      - Point-to-point: SVD on correspondence pairs
   d. Apply transformation to source
   e. Compute fitness = |inliers| / |source|
   f. Compute RMSE = sqrt(mean(inlier_distances²))
   g. Check convergence: |Δfitness| < threshold AND |ΔRMSE| < threshold
4. Return cumulative transformation, fitness, RMSE
```

### 3.2 SVD-Based Transformation Estimation

For point-to-point ICP, the optimal rigid transformation is:

```
Given: source points S = {s_i}, target correspondences T = {t_i}

1. Compute centroids: s̄ = mean(S), t̄ = mean(T)
2. Center: S' = S - s̄, T' = T - t̄
3. Cross-covariance: H = S'^T @ T'  (3×3 matrix)
4. SVD: H = U @ Σ @ V^T
5. Rotation: R = V @ diag(1, 1, det(V @ U^T)) @ U^T
6. Translation: t = t̄ - R @ s̄
7. Transformation: T = [[R, t], [0, 0, 0, 1]]
```

The `diag(1, 1, det(V @ U^T))` handles reflection (ensures proper rotation).

---

## 4. API Design

### 4.1 Data Types

```python
# pipelines/registration/convergence.py

@dataclass
class ICPConvergenceCriteria:
    """ICP convergence criteria.

    Matches Open3D: o3d.t.pipelines.registration.ICPConvergenceCriteria
    """
    relative_fitness: float = 1e-6
    relative_rmse: float = 1e-6
    max_iteration: int = 30


# pipelines/registration/result.py

@dataclass
class RegistrationResult:
    """ICP registration result.

    Matches Open3D: o3d.t.pipelines.registration.RegistrationResult
    """
    transformation: mx.array  # (4, 4) float32
    fitness: float = 0.0       # |inliers| / |source|
    inlier_rmse: float = float('inf')  # sqrt(mean(inlier_dist²))
    correspondences: mx.array | None = None  # (N,) int32, -1 = no match
    num_iterations: int = 0
    converged: bool = False

    def is_better_than(self, other: "RegistrationResult") -> bool:
        """Compare: higher fitness wins; if tied, lower RMSE wins."""
        if self.fitness != other.fitness:
            return self.fitness > other.fitness
        return self.inlier_rmse < other.inlier_rmse
```

### 4.2 Transformation Estimation

```python
# pipelines/registration/transformation.py

class TransformationEstimationPointToPoint:
    """Point-to-point transformation estimation via SVD.

    Minimizes: Σ_i ||R @ s_i + t - t_i||²
    """

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> mx.array:
        """Estimate rigid transformation from correspondences.

        Args:
            source_points: (N, 3) source points.
            target_points: (M, 3) target points.
            correspondences: (N,) indices into target. -1 = no match.

        Returns:
            (4, 4) transformation matrix.
        """

    def compute_rmse(
        self,
        source_points: mx.array,
        target_points: mx.array,
        correspondences: mx.array,
    ) -> tuple[float, float]:
        """Compute fitness and RMSE for given correspondences.

        Returns:
            (fitness, inlier_rmse)
        """
```

### 4.3 Main ICP Function

```python
# pipelines/registration/icp.py

def registration_icp(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    init_source_to_target: mx.array | None = None,
    estimation_method: TransformationEstimationPointToPoint | None = None,
    criteria: ICPConvergenceCriteria | None = None,
    voxel_size: float = -1.0,
) -> RegistrationResult:
    """ICP registration.

    Matches Open3D: o3d.t.pipelines.registration.icp()

    Args:
        source: Source point cloud.
        target: Target point cloud.
        max_correspondence_distance: Maximum correspondence distance.
        init_source_to_target: Initial 4x4 transformation (default: identity).
        estimation_method: Transformation estimation (default: PointToPoint).
        criteria: Convergence criteria (default: ICPConvergenceCriteria()).
        voxel_size: If > 0, downsample both clouds first.

    Returns:
        RegistrationResult with transformation, fitness, RMSE.
    """


def evaluate_registration(
    source: PointCloud,
    target: PointCloud,
    max_correspondence_distance: float,
    transformation: mx.array | None = None,
) -> RegistrationResult:
    """Evaluate registration quality without iterating.

    Applies transformation, finds correspondences, computes metrics.
    """
```

---

## 5. Implementation Details

### 5.1 Correspondence Search

```python
def _find_correspondences(
    source_points: mx.array,
    target_index: FixedRadiusIndex | NearestNeighborSearch,
    max_distance: float,
) -> tuple[mx.array, mx.array]:
    """Find nearest neighbor correspondences.

    Returns:
        correspondences: (N,) int32, -1 = no match within max_distance.
        distances: (N,) float32, distance to correspondent.
    """
    # Use FixedRadiusIndex for GPU path (ICP inner loop)
    # Falls back to NearestNeighborSearch CPU path if needed
```

### 5.2 SVD on MLX

```python
def _estimate_point_to_point(
    source_corr: mx.array,  # (K, 3) source inlier points
    target_corr: mx.array,  # (K, 3) target correspondence points
) -> mx.array:
    """SVD-based rigid transformation estimation.

    Returns (4, 4) transformation matrix.
    """
    # Centroids
    s_mean = mx.mean(source_corr, axis=0)  # (3,)
    t_mean = mx.mean(target_corr, axis=0)  # (3,)

    # Center
    s_centered = source_corr - s_mean  # (K, 3)
    t_centered = target_corr - t_mean  # (K, 3)

    # Cross-covariance H = S^T @ T
    H = s_centered.T @ t_centered  # (3, 3)

    # SVD
    U, S, Vt = mx.linalg.svd(H)

    # Rotation (handle reflection)
    d = mx.linalg.det(Vt.T @ U.T)
    sign_matrix = mx.array([1.0, 1.0, d])
    R = Vt.T @ mx.diag(sign_matrix) @ U.T  # (3, 3)

    # Translation
    t = t_mean - R @ s_mean  # (3,)

    # Build 4x4
    T = mx.eye(4, dtype=mx.float32)
    T = T.at[:3, :3].add(R - mx.eye(3))  # set rotation
    T = T.at[:3, 3].add(t)                # set translation

    return T
```

### 5.3 ICP Loop

```python
def registration_icp(source, target, max_correspondence_distance, ...):
    criteria = criteria or ICPConvergenceCriteria()
    estimation = estimation_method or TransformationEstimationPointToPoint()
    T_cumulative = init_source_to_target if init_source_to_target is not None else mx.eye(4)

    # Optional downsampling
    if voxel_size > 0:
        source = source.voxel_down_sample(voxel_size)
        target = target.voxel_down_sample(voxel_size)

    # Transform source by initial guess
    source_transformed = source.transform(T_cumulative)

    # Build target index (once — target doesn't move)
    target_index = FixedRadiusIndex(target.points, max_correspondence_distance)

    prev_fitness = 0.0
    prev_rmse = float('inf')

    for i in range(criteria.max_iteration):
        # 1. Find correspondences
        correspondences, distances = _find_correspondences(
            source_transformed.points, target_index, max_correspondence_distance
        )

        # 2. Compute metrics
        inlier_mask = correspondences >= 0
        num_inliers = int(mx.sum(inlier_mask).item())
        fitness = num_inliers / len(source)

        if num_inliers == 0:
            break

        inlier_distances = distances[inlier_mask]
        rmse = float(mx.sqrt(mx.mean(inlier_distances ** 2)).item())

        # 3. Check convergence
        if (abs(fitness - prev_fitness) < criteria.relative_fitness and
            abs(rmse - prev_rmse) < criteria.relative_rmse):
            converged = True
            break

        prev_fitness = fitness
        prev_rmse = rmse

        # 4. Estimate transformation
        T_step = estimation.compute_transformation(
            source_transformed.points, target.points, correspondences
        )

        # 5. Apply and accumulate
        source_transformed = source_transformed.transform(T_step)
        T_cumulative = T_step @ T_cumulative

    mx.eval(T_cumulative)  # Materialize result

    return RegistrationResult(
        transformation=T_cumulative,
        fitness=fitness,
        inlier_rmse=rmse,
        correspondences=correspondences,
        num_iterations=i + 1,
        converged=converged if 'converged' in dir() else False,
    )
```

---

## 6. Tests

```
# SVD transformation estimation
test_svd_identity_when_identical_points
test_svd_recovers_known_translation
test_svd_recovers_known_rotation
test_svd_recovers_rotation_and_translation
test_svd_handles_reflection

# ICP convergence
test_icp_identical_clouds_zero_rmse
test_icp_known_translation
test_icp_known_rotation_45deg
test_icp_converges_within_max_iterations
test_icp_respects_max_correspondence_distance
test_icp_voxel_downsample

# Evaluate registration
test_evaluate_registration_identity
test_evaluate_registration_with_transform

# Edge cases
test_icp_empty_source_raises
test_icp_empty_target_raises
test_icp_no_overlap_zero_fitness
test_icp_partial_overlap

# Cross-validation
test_icp_matches_open3d_result  # if open3d installed
```

---

## 7. Acceptance Criteria

- [ ] SVD estimation recovers known rigid transformations (atol=1e-4)
- [ ] ICP aligns identical clouds with fitness ~1.0 and RMSE ~0.0
- [ ] ICP recovers known translation (5, 0, 0) within 1e-3
- [ ] ICP recovers known 45° rotation within 1e-3
- [ ] ICP converges (stops early) when clouds are aligned
- [ ] `max_correspondence_distance` correctly rejects distant pairs
- [ ] `RegistrationResult` contains valid transformation, fitness, RMSE
- [ ] `evaluate_registration()` computes metrics without iterating
- [ ] All tests pass
- [ ] Performance: 10K points aligns in < 2 seconds on M1

---

## 8. Performance Notes

- **Hot path**: correspondence search (NN query) dominates runtime
- **GPU benefit**: FixedRadiusIndex vectorizes distance computation across all queries
- **MLX lazy eval**: accumulate the graph through the loop, eval at end for best GPU utilization
- **Float32**: All computation in float32 for GPU efficiency
- **Target index built once**: target point cloud is static throughout ICP
