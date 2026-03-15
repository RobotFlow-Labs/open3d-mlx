"""ICP Registration -- Align two point cloud scans."""
import numpy as np, mlx.core as mx
from open3d_mlx.geometry import PointCloud
from open3d_mlx.pipelines.registration import (
    registration_icp,
    TransformationEstimationPointToPoint,
    TransformationEstimationPointToPlane,
    ICPConvergenceCriteria,
)

# Generate source point cloud (sphere surface)
rng = np.random.default_rng(42)
N = 5000
theta = rng.uniform(0, 2 * np.pi, N).astype(np.float32)
phi = rng.uniform(0, np.pi, N).astype(np.float32)
r = 1.0
points = np.stack(
    [r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)],
    axis=1,
)
source = PointCloud(mx.array(points))

# Create target by applying known transformation
angle = np.radians(15)
R_true = np.array(
    [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)
t_true = np.array([0.3, -0.1, 0.05], dtype=np.float32)
target_pts = points @ R_true.T + t_true
target = PointCloud(mx.array(target_pts))

# Point-to-Point ICP
result_p2p = registration_icp(
    source,
    target,
    max_correspondence_distance=0.5,
    estimation_method=TransformationEstimationPointToPoint(),
    criteria=ICPConvergenceCriteria(max_iteration=50),
)
print("=== Point-to-Point ICP ===")
print(f"Fitness:     {result_p2p.fitness:.4f}")
print(f"RMSE:        {result_p2p.inlier_rmse:.6f}")
print(f"Iterations:  {result_p2p.num_iterations}")
print(f"Converged:   {result_p2p.converged}")

# Point-to-Plane ICP (estimate normals first)
source.estimate_normals(max_nn=30)
target.estimate_normals(max_nn=30)
result_p2plane = registration_icp(
    source,
    target,
    max_correspondence_distance=0.5,
    estimation_method=TransformationEstimationPointToPlane(),
    criteria=ICPConvergenceCriteria(max_iteration=50),
)
print("\n=== Point-to-Plane ICP ===")
print(f"Fitness:     {result_p2plane.fitness:.4f}")
print(f"RMSE:        {result_p2plane.inlier_rmse:.6f}")
print(f"Iterations:  {result_p2plane.num_iterations}")
print(f"Converged:   {result_p2plane.converged}")

print(
    f"\nP2Plane used {result_p2plane.num_iterations} iters vs "
    f"P2P's {result_p2p.num_iterations}"
)
print("\nExample 03 complete")
