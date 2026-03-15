"""Robust Kernels -- Visualize weight functions for outlier rejection."""
import numpy as np, mlx.core as mx
from open3d_mlx.pipelines.registration import (
    L2Loss,
    HuberLoss,
    TukeyLoss,
    CauchyLoss,
    GMLoss,
)

residuals = mx.array(np.linspace(-5, 5, 101).astype(np.float32))

kernels = {
    "L2": L2Loss(),
    "Huber": HuberLoss(k=1.345),
    "Tukey": TukeyLoss(k=4.685),
    "Cauchy": CauchyLoss(k=1.0),
    "GM": GMLoss(k=1.0),
}

print(f"{'Residual':>10} | " + " | ".join(f"{name:>8}" for name in kernels))
print("-" * 70)

for r_val in [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]:
    r = mx.array([r_val])
    weights = {name: float(np.array(k.weight(r))[0]) for name, k in kernels.items()}
    print(f"{r_val:>10.1f} | " + " | ".join(f"{w:>8.4f}" for w in weights.values()))

print("\nKey insight: Tukey gives ZERO weight to large outliers")
print("            Cauchy/GM reduce but never fully reject")
print("\nExample 06 complete")
