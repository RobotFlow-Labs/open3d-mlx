"""TSDF Reconstruction -- Integrate synthetic depth frames into a 3D volume."""
import numpy as np, mlx.core as mx
from open3d_mlx.camera import PinholeCameraIntrinsic
from open3d_mlx.pipelines.integration import UniformTSDFVolume

# Camera setup
intrinsic = PinholeCameraIntrinsic.prime_sense_default()
print(f"Camera: {intrinsic.width}x{intrinsic.height}, fx={intrinsic.fx}")

# Create TSDF volume
volume = UniformTSDFVolume(length=2.0, resolution=64, sdf_trunc=0.04)
print(f"Volume: {volume.resolution}^3 voxels, {volume.voxel_size:.4f}m per voxel")

# Generate synthetic depth frame (flat wall at z=1.0m)
depth = np.full((intrinsic.height, intrinsic.width), 1000, dtype=np.uint16)  # 1.0m in mm

# Integrate from 3 viewpoints
for i, tx in enumerate([0.0, 0.1, -0.1]):
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[0, 3] = tx  # slight lateral shift
    volume.integrate(mx.array(depth), intrinsic, mx.array(extrinsic))
    print(f"Frame {i + 1}: integrated (offset x={tx})")

# Extract point cloud
pcd = volume.extract_point_cloud()
print(f"\nExtracted: {len(pcd)} surface points")
print(f"Point range: {np.array(pcd.get_min_bound())} to {np.array(pcd.get_max_bound())}")

print("\nExample 04 complete")
