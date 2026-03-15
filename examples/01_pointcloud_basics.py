"""Point Cloud Basics -- Create, transform, downsample, and query point clouds."""
import numpy as np
import mlx.core as mx
from open3d_mlx.geometry import PointCloud

# Create from numpy
rng = np.random.default_rng(42)
points = rng.standard_normal((10000, 3)).astype(np.float32)
pcd = PointCloud(mx.array(points))
print(f"Created: {pcd}")
print(f"Center: {np.array(pcd.get_center())}")
print(f"Bounds: {np.array(pcd.get_min_bound())} to {np.array(pcd.get_max_bound())}")

# Transform
T = np.eye(4, dtype=np.float32)
T[:3, 3] = [1, 2, 3]
moved = pcd.transform(mx.array(T))
print(f"\nAfter translation [1,2,3]: center = {np.array(moved.get_center())}")

# Downsample
downsampled = pcd.voxel_down_sample(0.5)
print(f"\nVoxel downsample (0.5): {len(pcd)} -> {len(downsampled)} points")

# Estimate normals
pcd.estimate_normals(max_nn=30)
print(f"Normals estimated: {pcd.has_normals()}")

# Paint and filter
painted = pcd.paint_uniform_color([1.0, 0.0, 0.0])
clean = painted.remove_non_finite_points()
print(f"After cleaning: {len(clean)} points")

print("\nExample 01 complete")
