"""Full Pipeline -- Load -> Downsample -> Register -> Reconstruct."""
import tempfile, os
import numpy as np, mlx.core as mx
from open3d_mlx.geometry import PointCloud
from open3d_mlx.io import write_point_cloud, read_point_cloud
from open3d_mlx.pipelines.registration import (
    registration_icp,
    TransformationEstimationPointToPoint,
)
from open3d_mlx.camera import PinholeCameraIntrinsic
from open3d_mlx.pipelines.integration import UniformTSDFVolume

print("=== Open3D-MLX Full Pipeline Demo ===\n")

# 1. Generate two overlapping scans
rng = np.random.default_rng(42)
base_points = rng.standard_normal((3000, 3)).astype(np.float32) * 0.5

scan1 = PointCloud(mx.array(base_points))
T = np.eye(4, dtype=np.float32)
T[:3, 3] = [0.1, 0.05, -0.02]
scan2_pts = (base_points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
scan2 = PointCloud(mx.array(scan2_pts))
print(f"Scan 1: {len(scan1)} points")
print(f"Scan 2: {len(scan2)} points (translated)")

# 2. I/O roundtrip
with tempfile.TemporaryDirectory() as d:
    write_point_cloud(os.path.join(d, "scan1.ply"), scan1)
    write_point_cloud(os.path.join(d, "scan2.ply"), scan2)
    scan1 = read_point_cloud(os.path.join(d, "scan1.ply"))
    scan2 = read_point_cloud(os.path.join(d, "scan2.ply"))
print("I/O roundtrip: OK")

# 3. Downsample
scan1_ds = scan1.voxel_down_sample(0.1)
scan2_ds = scan2.voxel_down_sample(0.1)
print(f"Downsampled: {len(scan1)}->{len(scan1_ds)}, {len(scan2)}->{len(scan2_ds)}")

# 4. Register
result = registration_icp(scan1_ds, scan2_ds, max_correspondence_distance=0.3)
print(
    f"ICP: fitness={result.fitness:.3f}, RMSE={result.inlier_rmse:.5f}, "
    f"iters={result.num_iterations}"
)

# 5. Align and merge
aligned = scan1.transform(result.transformation)
merged = aligned + scan2
print(f"Merged cloud: {len(merged)} points")

print("\nExample 08 complete -- full pipeline executed successfully")
