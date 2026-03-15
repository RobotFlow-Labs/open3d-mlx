# PRD-07: Advanced ICP — Colored, Generalized, Multi-Scale, Robust Kernels

## Status: P1 — Extended Registration
## Priority: P1
## Phase: 2 — Registration
## Estimated Effort: 3–4 days
## Depends On: PRD-05, PRD-06
## Blocks: PRD-11

---

## 1. Objective

Extend the ICP framework with:
1. **Colored ICP** — uses color consistency for better alignment on textured surfaces
2. **Generalized ICP (GICP)** — plane-to-plane metric using local covariances
3. **Multi-Scale ICP** — coarse-to-fine registration at multiple voxel resolutions
4. **Robust Kernels** — Huber, Tukey, Cauchy for outlier rejection

---

## 2. Upstream Reference

| Our File | Upstream File | Notes |
|----------|--------------|-------|
| `pipelines/registration/transformation.py` | `TransformationEstimation.cpp` | Colored + GICP classes |
| `pipelines/registration/robust_kernel.py` | `RobustKernel.h/RobustKernelImpl.h` | Loss functions |
| `pipelines/registration/icp.py` | `Registration.cpp` | MultiScaleICP |

---

## 3. Robust Kernels

### 3.1 API

```python
# pipelines/registration/robust_kernel.py

class RobustKernel:
    """Base class for robust loss functions."""
    def weight(self, residual: mx.array) -> mx.array:
        """Compute weight for each residual. Returns (N,) weights."""
        raise NotImplementedError

class L2Loss(RobustKernel):
    """Standard L2 loss (no robustness, weight = 1)."""
    def weight(self, residual): return mx.ones_like(residual)

class HuberLoss(RobustKernel):
    """Huber loss: L2 for |r| < k, L1 for |r| >= k."""
    def __init__(self, k: float = 1.345):
        self.k = k
    def weight(self, residual):
        abs_r = mx.abs(residual)
        return mx.where(abs_r <= self.k, mx.ones_like(abs_r), self.k / abs_r)

class TukeyLoss(RobustKernel):
    """Tukey's biweight: zero weight for |r| > k."""
    def __init__(self, k: float = 4.685):
        self.k = k
    def weight(self, residual):
        abs_r = mx.abs(residual)
        inside = (1.0 - (abs_r / self.k) ** 2) ** 2
        return mx.where(abs_r <= self.k, inside, mx.zeros_like(abs_r))

class CauchyLoss(RobustKernel):
    """Cauchy (Lorentzian) loss."""
    def __init__(self, k: float = 1.0):
        self.k = k
    def weight(self, residual):
        return 1.0 / (1.0 + (residual / self.k) ** 2)

class GMLoss(RobustKernel):
    """Geman-McClure loss."""
    def __init__(self, k: float = 1.0):
        self.k = k
    def weight(self, residual):
        return self.k / (self.k + residual ** 2) ** 2
```

### 3.2 Integration with ICP

Robust kernels modify the transformation estimation by weighting each correspondence:

```python
# In transformation estimation:
weights = kernel.weight(distances)
# Weighted least squares:
# Point-to-point: weight each pair in SVD
# Point-to-plane: weight each row of Jacobian
```

---

## 4. Colored ICP

### 4.1 Algorithm

Colored ICP adds a photometric term to the geometric term:

```
E = Σ_i [ (1-δ) * ||R@s_i + t - t_i||² + δ * (C_s(s_i) - C_t(t_i))² ]
```

Where `δ` is the color weight (default 0.968 from the paper), and C is a scalar intensity derived from RGB.

### 4.2 API

```python
class TransformationEstimationForColoredICP:
    """Colored ICP estimation using combined geometric + photometric error.

    Requires: both source and target have colors.
    Reference: Park et al., "Colored Point Cloud Registration Revisited", ICCV 2017.
    """

    def __init__(self, lambda_geometric: float = 0.968):
        self.lambda_geometric = lambda_geometric

    def compute_transformation(
        self,
        source_points, target_points,
        source_colors, target_colors,
        target_normals,
        correspondences,
    ) -> mx.array:
        """Estimate transformation with color + geometry term."""
```

### 4.3 Implementation Notes

1. Convert RGB to intensity: `I = 0.2126*R + 0.7152*G + 0.0722*B`
2. Compute color gradient on target surface (finite differences along tangent plane)
3. Build Jacobian with 6 DOF: geometric rows + photometric rows
4. Solve weighted least squares

---

## 5. Generalized ICP (GICP)

### 5.1 Algorithm

GICP models each point as a local planar patch using its covariance matrix:

```
E = Σ_i d_i^T (C_s + R C_t R^T)^{-1} d_i
```

Where `C_s, C_t` are local covariance matrices and `d_i = R@s_i + t - t_i`.

This is "plane-to-plane" matching — the most robust variant for structured environments.

### 5.2 API

```python
class TransformationEstimationForGeneralizedICP:
    """Generalized ICP (plane-to-plane).

    Reference: Segal et al., "Generalized-ICP", RSS 2009.
    """

    def __init__(self, epsilon: float = 0.001):
        """Args:
            epsilon: Regularization for covariance matrices.
        """
        self.epsilon = epsilon

    def compute_transformation(
        self,
        source_points, target_points,
        source_covariances, target_covariances,
        correspondences,
    ) -> mx.array:
        """Estimate transformation using GICP metric."""
```

### 5.3 Covariance Computation

```python
def compute_point_covariances(
    points: mx.array,
    neighbor_indices: mx.array,
    epsilon: float = 0.001,
) -> mx.array:
    """Compute local covariance matrix for each point.

    Returns: (N, 3, 3) covariance matrices.
    """
    # Same PCA as normal estimation (PRD-04),
    # but return full covariance instead of just smallest eigenvector
    # Regularize: C = C + epsilon * I (prevents singular matrices)
```

---

## 6. Multi-Scale ICP

### 6.1 API

```python
def multi_scale_icp(
    source: PointCloud,
    target: PointCloud,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    criteria_list: list[ICPConvergenceCriteria] | None = None,
    init_source_to_target: mx.array | None = None,
    estimation_method=None,
) -> RegistrationResult:
    """Coarse-to-fine multi-scale ICP.

    Runs ICP at each scale level, using the result as initialization
    for the next finer level.

    Example:
        result = multi_scale_icp(
            source, target,
            voxel_sizes=[0.1, 0.05, 0.02],
            max_correspondence_distances=[0.3, 0.15, 0.05],
        )
    """
```

### 6.2 Implementation

```python
def multi_scale_icp(source, target, voxel_sizes, max_correspondence_distances, ...):
    assert len(voxel_sizes) == len(max_correspondence_distances)
    n_scales = len(voxel_sizes)

    if criteria_list is None:
        criteria_list = [ICPConvergenceCriteria()] * n_scales

    T = init_source_to_target if init_source_to_target is not None else mx.eye(4)

    for i in range(n_scales):
        # Downsample at current scale
        src_down = source.voxel_down_sample(voxel_sizes[i])
        tgt_down = target.voxel_down_sample(voxel_sizes[i])

        # Estimate normals if needed
        if isinstance(estimation_method, TransformationEstimationPointToPlane):
            src_down.estimate_normals(max_nn=30)
            tgt_down.estimate_normals(max_nn=30)

        result = registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=max_correspondence_distances[i],
            init_source_to_target=T,
            estimation_method=estimation_method,
            criteria=criteria_list[i],
        )
        T = result.transformation

    return result
```

---

## 7. Tests

```
# Robust kernels
test_l2_weight_always_one
test_huber_weight_l2_region
test_huber_weight_l1_region
test_tukey_weight_zero_outside
test_cauchy_weight_decreasing
test_gm_weight_decreasing

# Colored ICP
test_colored_icp_requires_colors
test_colored_icp_improves_on_textured_surfaces
test_colored_icp_intensity_computation

# GICP
test_gicp_covariance_computation
test_gicp_planar_surface
test_gicp_improves_on_structured_scenes

# Multi-scale
test_multi_scale_icp_coarse_to_fine
test_multi_scale_icp_better_than_single_scale
test_multi_scale_icp_validates_list_lengths

# Robust ICP
test_icp_with_huber_kernel
test_icp_with_tukey_rejects_outliers
test_robust_icp_handles_50pct_outliers
```

---

## 8. Acceptance Criteria

- [ ] All 5 robust kernels produce correct weights for known inputs
- [ ] Robust ICP handles 30%+ outlier correspondences better than L2
- [ ] Colored ICP uses color information (verified by better fitness on textured data)
- [ ] GICP computes per-point covariances (N, 3, 3) correctly
- [ ] Multi-scale ICP runs at 3+ scale levels
- [ ] Multi-scale converges better than single-scale on large displacements
- [ ] All estimation methods plug into the same ICP loop (PRD-05 reuse)
- [ ] All tests pass
