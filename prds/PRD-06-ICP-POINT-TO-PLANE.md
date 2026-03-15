# PRD-06: ICP Registration — Point-to-Plane

## Status: P0 — Core Pipeline
## Priority: P0
## Phase: 2 — Registration
## Estimated Effort: 1–2 days
## Depends On: PRD-05 (reuses ICP loop, correspondence, result types)
## Blocks: PRD-07, PRD-11

---

## 1. Objective

Implement point-to-plane ICP — the faster-converging variant that uses target surface normals. Instead of minimizing point-to-point distance, it minimizes the distance from source points to the tangent plane at target correspondences. Converges ~10x faster than point-to-point on smooth surfaces.

---

## 2. Upstream Reference

| Our File | Upstream File | Notes |
|----------|--------------|-------|
| `pipelines/registration/transformation.py` | `TransformationEstimation.cpp` lines for PointToPlane | Linearized least-squares |

---

## 3. Algorithm

### Point-to-Plane Error

```
E = Σ_i ((R @ s_i + t - t_i) · n_i)²
```

Where `n_i` is the target normal at correspondence `i`. This minimizes the projection of the residual onto the surface normal — allowing sliding along the surface.

### Linearized Solution

For small rotations, parameterize R ≈ I + [α]× where [α]× is the skew-symmetric matrix:

```
[α]× = [[0, -γ, β], [γ, 0, -α], [-β, α, 0]]
```

The optimization becomes a 6×6 linear system:

```
J^T J x = J^T b
```

Where:
- `x = [α, β, γ, tx, ty, tz]` — 6 DOF transformation parameters
- `J` is the Jacobian matrix (N×6)
- `b` is the residual vector (N,)

For each correspondence (s_i, t_i, n_i):
```
row_i of J = [cross(s_i, n_i), n_i]  (1×6)
b_i = (t_i - s_i) · n_i              (scalar)
```

So: `JtJ = Σ_i J_i^T @ J_i` (6×6) and `Jtb = Σ_i J_i^T * b_i` (6,)

---

## 4. API Design

```python
class TransformationEstimationPointToPlane:
    """Point-to-plane transformation estimation via linearized least-squares.

    Requires: target point cloud has normals.
    Minimizes: Σ ((R@s + t - t_corr) · n_corr)²
    """

    def compute_transformation(
        self,
        source_points: mx.array,
        target_points: mx.array,
        target_normals: mx.array,
        correspondences: mx.array,
    ) -> mx.array:
        """Estimate transformation using point-to-plane metric.

        Args:
            source_points: (N, 3) transformed source points.
            target_points: (M, 3) target points.
            target_normals: (M, 3) target normals.
            correspondences: (N,) indices into target. -1 = no match.

        Returns:
            (4, 4) transformation matrix.
        """
```

---

## 5. Implementation

```python
def _estimate_point_to_plane(
    source_corr: mx.array,   # (K, 3) source inlier points
    target_corr: mx.array,   # (K, 3) target correspondence points
    target_normals: mx.array, # (K, 3) target normals at correspondences
) -> mx.array:
    """Linearized point-to-plane ICP.

    Solves the 6x6 normal equation: (J^T J) x = J^T b
    """
    # Residuals: (t - s) · n
    diff = target_corr - source_corr  # (K, 3)
    b = mx.sum(diff * target_normals, axis=1)  # (K,)

    # Jacobian columns:
    # [s × n | n] where × is cross product
    cross = mx.stack([
        source_corr[:, 1] * target_normals[:, 2] - source_corr[:, 2] * target_normals[:, 1],
        source_corr[:, 2] * target_normals[:, 0] - source_corr[:, 0] * target_normals[:, 2],
        source_corr[:, 0] * target_normals[:, 1] - source_corr[:, 1] * target_normals[:, 0],
    ], axis=1)  # (K, 3)

    # J = [cross, normals]  shape (K, 6)
    J = mx.concatenate([cross, target_normals], axis=1)  # (K, 6)

    # Normal equations: (J^T J) x = J^T b
    JtJ = J.T @ J      # (6, 6)
    Jtb = J.T @ b       # (6,)

    # Solve
    x = mx.linalg.solve(JtJ, Jtb)  # (6,) = [α, β, γ, tx, ty, tz]

    # Build transformation matrix from parameters
    alpha, beta, gamma = float(x[0].item()), float(x[1].item()), float(x[2].item())
    tx, ty, tz = float(x[3].item()), float(x[4].item()), float(x[5].item())

    # Rotation matrix from small-angle approximation
    # For better accuracy, use Rodrigues or matrix exponential
    R = _rotation_from_euler_small(alpha, beta, gamma)
    t = mx.array([tx, ty, tz], dtype=mx.float32)

    T = mx.eye(4, dtype=mx.float32)
    T = T.at[:3, :3].add(R - mx.eye(3))
    T = T.at[:3, 3].add(t)
    return T


def _rotation_from_euler_small(alpha, beta, gamma):
    """Build rotation matrix. Uses Rodrigues formula for accuracy."""
    import math
    angle = math.sqrt(alpha**2 + beta**2 + gamma**2)
    if angle < 1e-10:
        return mx.eye(3, dtype=mx.float32)

    axis = mx.array([alpha, beta, gamma], dtype=mx.float32) / angle
    K = mx.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=mx.float32)
    R = mx.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R
```

---

## 6. Integration with ICP Loop

The ICP loop from PRD-05 is reused. The only change is passing `target.normals` to the estimation method:

```python
# In registration_icp():
if isinstance(estimation, TransformationEstimationPointToPlane):
    if not target.has_normals():
        raise ValueError("Point-to-plane ICP requires target normals. "
                         "Call target.estimate_normals() first.")
    T_step = estimation.compute_transformation(
        source_transformed.points, target.points, target.normals, correspondences
    )
```

---

## 7. Tests

```
# Point-to-plane estimation
test_plane_estimation_identity_aligned
test_plane_estimation_known_translation
test_plane_estimation_known_small_rotation
test_plane_estimation_requires_normals

# ICP with point-to-plane
test_icp_point_to_plane_converges_faster
test_icp_point_to_plane_on_planar_surface
test_icp_point_to_plane_known_transform
test_icp_point_to_plane_missing_normals_raises

# Convergence comparison
test_point_to_plane_fewer_iterations_than_point_to_point

# Cross-validation
test_point_to_plane_matches_open3d  # if open3d installed
```

---

## 8. Acceptance Criteria

- [ ] Recovers known translation within 1e-3
- [ ] Recovers known small rotation (< 30°) within 1e-3
- [ ] Raises ValueError when target has no normals
- [ ] Converges in fewer iterations than point-to-point on smooth surfaces
- [ ] 6x6 linear system solves correctly on MLX
- [ ] Rodrigues rotation formula produces valid rotation matrices
- [ ] All tests pass
