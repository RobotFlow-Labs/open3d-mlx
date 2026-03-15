"""File I/O -- Read and write PLY/PCD files."""
import tempfile, os
import numpy as np, mlx.core as mx
from open3d_mlx.geometry import PointCloud
from open3d_mlx.io import read_point_cloud, write_point_cloud

# Create a point cloud with colors
rng = np.random.default_rng(42)
pcd = PointCloud(mx.array(rng.random((1000, 3)).astype(np.float32)))
pcd.colors = mx.array(rng.random((1000, 3)).astype(np.float32))

with tempfile.TemporaryDirectory() as tmpdir:
    # PLY roundtrip
    ply_path = os.path.join(tmpdir, "test.ply")
    write_point_cloud(ply_path, pcd)
    loaded_ply = read_point_cloud(ply_path)
    print(f"PLY: wrote {len(pcd)} pts -> read {len(loaded_ply)} pts")

    # PCD roundtrip
    pcd_path = os.path.join(tmpdir, "test.pcd")
    write_point_cloud(pcd_path, pcd)
    loaded_pcd = read_point_cloud(pcd_path)
    print(f"PCD: wrote {len(pcd)} pts -> read {len(loaded_pcd)} pts")

print("\nExample 02 complete")
