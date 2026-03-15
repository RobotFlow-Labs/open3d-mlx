"""Raycasting -- Render synthetic depth and normal maps from a TSDF volume."""
import numpy as np, mlx.core as mx
from open3d_mlx.camera import PinholeCameraIntrinsic
from open3d_mlx.pipelines.integration import UniformTSDFVolume
from open3d_mlx.pipelines.raycasting import RaycastingScene

# Build a TSDF volume with a flat wall
intrinsic = PinholeCameraIntrinsic(160, 120, 130.0, 130.0, 80.0, 60.0)
volume = UniformTSDFVolume(length=2.0, resolution=64, sdf_trunc=0.04)

depth = np.full((120, 160), 800, dtype=np.uint16)  # 0.8m wall
extrinsic = np.eye(4, dtype=np.float32)
volume.integrate(mx.array(depth), intrinsic, mx.array(extrinsic))

# Raycast
scene = RaycastingScene()
scene.set_volume(volume)

depth_map = scene.render_depth(intrinsic, mx.array(extrinsic))
normal_map = scene.render_normal(intrinsic, mx.array(extrinsic))

depth_np = np.array(depth_map)
normal_np = np.array(normal_map)

valid = ~np.isinf(depth_np)
print(f"Depth map: {depth_np.shape}")
print(f"Valid pixels: {valid.sum()} / {valid.size}")
if valid.any():
    print(f"Depth range: {depth_np[valid].min():.3f} -- {depth_np[valid].max():.3f}m")
print(f"Normal map: {normal_np.shape}")

print("\nExample 05 complete")
