"""Scalable hash-based TSDF volume for large scenes.

Matches Open3D: o3d.pipelines.integration.ScalableTSDFVolume

Unlike UniformTSDFVolume, this only allocates memory for observed regions
using a block-hashing approach. Suitable for room-scale and building-scale
reconstruction.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from open3d_mlx.camera.intrinsics import PinholeCameraIntrinsic
from open3d_mlx.geometry.pointcloud import PointCloud


class ScalableTSDFVolume:
    """Hash-based sparse TSDF volume for large scenes.

    Only allocates memory for observed regions. Each block is a small
    uniform grid of ``block_resolution^3`` voxels.

    Parameters
    ----------
    voxel_size : float
        Individual voxel size in meters.
    sdf_trunc : float
        Truncation distance for SDF values.
    color : bool
        If True, store per-voxel RGB color.
    block_resolution : int
        Number of voxels per block side (e.g. 8 means 8^3 voxels per block).
    block_count : int
        Initial block allocation capacity hint (unused, for API compat).
    """

    def __init__(
        self,
        voxel_size: float = 0.006,
        sdf_trunc: float = 0.04,
        color: bool = False,
        block_resolution: int = 8,
        block_count: int = 50000,
    ):
        self.voxel_size = float(voxel_size)
        self.sdf_trunc = float(sdf_trunc)
        self.block_resolution = int(block_resolution)
        self.block_size = self.voxel_size * self.block_resolution
        self._store_color = color

        # Hash table: (bx, by, bz) -> {"tsdf": np.array, "weight": np.array, "color": np.array|None}
        self._blocks: dict[tuple[int, int, int], dict] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_block_count(self) -> int:
        """Number of allocated blocks."""
        return len(self._blocks)

    # ------------------------------------------------------------------
    # Block management
    # ------------------------------------------------------------------

    def _block_key(self, world_pos: np.ndarray) -> tuple[int, int, int]:
        """Compute block key from a world-space position."""
        bk = np.floor(world_pos / self.block_size).astype(np.int64)
        return (int(bk[0]), int(bk[1]), int(bk[2]))

    def _get_or_create_block(self, key: tuple[int, int, int]) -> dict:
        """Get an existing block or allocate a new one."""
        if key not in self._blocks:
            R = self.block_resolution
            block = {
                "tsdf": np.ones((R, R, R), dtype=np.float32),
                "weight": np.zeros((R, R, R), dtype=np.float32),
            }
            if self._store_color:
                block["color"] = np.zeros((R, R, R, 3), dtype=np.float32)
            self._blocks[key] = block
        return self._blocks[key]

    def _block_origin(self, key: tuple[int, int, int]) -> np.ndarray:
        """World-space origin (min corner) of the block at *key*."""
        return np.array(key, dtype=np.float64) * self.block_size

    def _block_voxel_centers(self, key: tuple[int, int, int]) -> np.ndarray:
        """Return (R^3, 3) voxel centre positions for the given block."""
        R = self.block_resolution
        origin = self._block_origin(key)
        coords = np.arange(R, dtype=np.float32)
        gi, gj, gk = np.meshgrid(coords, coords, coords, indexing="ij")
        centers = np.stack([gi, gj, gk], axis=-1).reshape(-1, 3)
        return (centers + 0.5) * self.voxel_size + origin.astype(np.float32)

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        depth,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        color=None,
    ) -> None:
        """Integrate a single depth frame into the sparse volume.

        Parameters
        ----------
        depth : array-like
            ``(H, W)`` depth image.
        intrinsic : PinholeCameraIntrinsic
            Camera intrinsic parameters.
        extrinsic : array-like
            ``(4, 4)`` world-to-camera rigid transform.
        depth_scale : float
            Depth values are divided by this to obtain metres.
        depth_max : float
            Maximum depth in metres to integrate.
        color : array-like or None
            ``(H, W, 3)`` optional colour image.
        """
        depth_np = np.asarray(depth, dtype=np.float32) / depth_scale
        ext = np.asarray(extrinsic, dtype=np.float64)
        Rot = ext[:3, :3]
        tvec = ext[:3, 3]

        H, W = intrinsic.height, intrinsic.width

        # Inverse extrinsic for unprojecting depth pixels to world space
        R_inv = Rot.T
        t_inv = -R_inv @ tvec

        # 1. Determine which blocks to activate by sampling depth pixels
        stride = 4
        active_keys: set[tuple[int, int, int]] = set()
        for v_px in range(0, H, stride):
            for u_px in range(0, W, stride):
                d = depth_np[v_px, u_px]
                if d <= 0 or d >= depth_max:
                    continue
                # Unproject pixel to camera frame
                x_cam = (u_px - intrinsic.cx) * d / intrinsic.fx
                y_cam = (v_px - intrinsic.cy) * d / intrinsic.fy
                z_cam = d
                p_cam = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
                # Transform to world frame
                p_world = R_inv @ p_cam + t_inv
                key = self._block_key(p_world)
                active_keys.add(key)
                # Also add neighboring blocks within truncation distance
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            active_keys.add((key[0] + di, key[1] + dj, key[2] + dk))

        if not active_keys:
            return

        # 2. Integrate each active block
        for key in active_keys:
            block = self._get_or_create_block(key)
            voxels = self._block_voxel_centers(key)  # (R^3, 3)

            # Transform voxel centres to camera frame
            cam = (voxels @ Rot.T + tvec).astype(np.float32)
            z = cam[:, 2]

            # Project to pixel coordinates
            u = cam[:, 0] / z * intrinsic.fx + intrinsic.cx
            v = cam[:, 1] / z * intrinsic.fy + intrinsic.cy

            # Validity mask
            valid = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

            ui = np.clip(u.astype(np.int32), 0, W - 1)
            vi = np.clip(v.astype(np.int32), 0, H - 1)

            sampled = depth_np[vi, ui]

            # SDF
            sdf = sampled - z
            valid &= (sampled > 0) & (sampled < depth_max) & (np.abs(sdf) < self.sdf_trunc)

            if not np.any(valid):
                continue

            tsdf_val = np.clip(sdf / self.sdf_trunc, -1.0, 1.0)

            R = self.block_resolution
            tsdf_flat = block["tsdf"].reshape(-1)
            w_flat = block["weight"].reshape(-1)

            w_new = np.where(valid, w_flat + 1.0, w_flat)
            t_new = np.where(
                valid,
                (tsdf_flat * w_flat + tsdf_val) / np.maximum(w_new, 1.0),
                tsdf_flat,
            )

            block["tsdf"] = t_new.reshape(R, R, R)
            block["weight"] = np.minimum(w_new, 255.0).reshape(R, R, R)

            # Optional colour
            if color is not None and self._store_color and "color" in block:
                color_np = np.asarray(color, dtype=np.float32)
                sampled_color = color_np[vi, ui]
                color_flat = block["color"].reshape(-1, 3)
                c_new = np.where(
                    valid[:, None],
                    (color_flat * w_flat[:, None] + sampled_color)
                    / np.maximum(w_new[:, None], 1.0),
                    color_flat,
                )
                block["color"] = c_new.reshape(R, R, R, 3)

    # ------------------------------------------------------------------
    # Surface extraction
    # ------------------------------------------------------------------

    def extract_point_cloud(self) -> PointCloud:
        """Extract surface points at TSDF zero-crossings across all blocks.

        Returns
        -------
        PointCloud
        """
        all_points: list[np.ndarray] = []

        for key, block in self._blocks.items():
            tsdf = block["tsdf"]
            weight = block["weight"]
            R = self.block_resolution

            origin = self._block_origin(key).astype(np.float32)
            coords = np.arange(R, dtype=np.float32)
            gi, gj, gk = np.meshgrid(coords, coords, coords, indexing="ij")
            centers = np.stack([gi, gj, gk], axis=-1)
            centers = (centers + 0.5) * self.voxel_size + origin

            for axis in range(3):
                slc_cur = [slice(None)] * 3
                slc_nxt = [slice(None)] * 3
                slc_cur[axis] = slice(0, R - 1)
                slc_nxt[axis] = slice(1, R)

                t0 = tsdf[tuple(slc_cur)]
                t1 = tsdf[tuple(slc_nxt)]
                w0 = weight[tuple(slc_cur)]
                w1 = weight[tuple(slc_nxt)]

                crossing = (t0 * t1 < 0) & (w0 > 0) & (w1 > 0)
                if not np.any(crossing):
                    continue

                denom = t0 - t1
                denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
                t = t0 / denom
                t = np.clip(t, 0.0, 1.0)

                c0 = centers[tuple(slc_cur)]
                c1 = centers[tuple(slc_nxt)]
                interp = c0 + t[..., None] * (c1 - c0)
                pts = interp[crossing]
                all_points.append(pts)

        if not all_points:
            return PointCloud()

        return PointCloud(mx.array(np.concatenate(all_points, axis=0)))

    def extract_triangle_mesh(self) -> dict:
        """Extract triangle mesh via marching cubes on all active blocks.

        Returns
        -------
        dict
            ``{"vertices": mx.array, "triangles": mx.array}``
        """
        from open3d_mlx.pipelines.integration.marching_cubes import marching_cubes

        all_verts: list[np.ndarray] = []
        all_tris: list[np.ndarray] = []
        vert_offset = 0

        for key, block in self._blocks.items():
            origin = self._block_origin(key).astype(np.float32)
            spacing = (self.voxel_size,) * 3
            verts, tris = marching_cubes(
                block["tsdf"],
                level=0.0,
                spacing=spacing,
                origin=origin,
                weight=block["weight"],
                weight_threshold=0.0,
            )
            v_np = np.asarray(verts, dtype=np.float32)
            t_np = np.asarray(tris, dtype=np.int32)
            if len(v_np) > 0:
                all_verts.append(v_np)
                all_tris.append(t_np + vert_offset)
                vert_offset += len(v_np)

        if not all_verts:
            return {
                "vertices": mx.zeros((0, 3), dtype=mx.float32),
                "triangles": mx.zeros((0, 3), dtype=mx.int32),
            }

        return {
            "vertices": mx.array(np.concatenate(all_verts, axis=0)),
            "triangles": mx.array(np.concatenate(all_tris, axis=0)),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all blocks."""
        self._blocks.clear()
