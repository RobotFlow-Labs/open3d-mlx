"""Uniform TSDF volume for depth-frame integration.

Matches Open3D: o3d.pipelines.integration.UniformTSDFVolume

The implementation uses NumPy for the integration inner loop to maintain
float64 precision in the geometric transforms, and converts to MLX arrays
at the public property boundary.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from open3d_mlx.camera.intrinsics import PinholeCameraIntrinsic
from open3d_mlx.geometry.pointcloud import PointCloud


class UniformTSDFVolume:
    """Uniform voxel-grid TSDF volume.

    Parameters
    ----------
    length : float
        Side length of the cubic volume in meters.
    resolution : int
        Number of voxels per side (total voxels = resolution**3).
    sdf_trunc : float
        Truncation distance for SDF values.
    color : bool
        If True, store per-voxel RGB color.
    origin : array-like or None
        (3,) volume origin in world coordinates. Default ``[0, 0, 0]``.
    """

    def __init__(
        self,
        length: float = 4.0,
        resolution: int = 128,
        sdf_trunc: float = 0.04,
        color: bool = False,
        origin=None,
    ):
        self.length = float(length)
        self.resolution = int(resolution)
        self.voxel_size = self.length / self.resolution
        self.sdf_trunc = float(sdf_trunc)

        if origin is not None:
            self.origin = np.asarray(origin, dtype=np.float64).ravel()[:3]
        else:
            self.origin = np.zeros(3, dtype=np.float64)

        R = self.resolution
        # TSDF initialised to 1.0 (empty / far from surface)
        self._tsdf = np.ones((R, R, R), dtype=np.float32)
        self._weight = np.zeros((R, R, R), dtype=np.float32)
        self._color: np.ndarray | None = (
            np.zeros((R, R, R, 3), dtype=np.float32) if color else None
        )
        self._voxel_centers: np.ndarray | None = None  # lazy cache

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the volume to its initial state."""
        R = self.resolution
        self._tsdf = np.ones((R, R, R), dtype=np.float32)
        self._weight = np.zeros((R, R, R), dtype=np.float32)
        if self._color is not None:
            self._color = np.zeros((R, R, R, 3), dtype=np.float32)
        # Keep voxel_centers cache

    # ------------------------------------------------------------------
    # Vectorised integration
    # ------------------------------------------------------------------

    def _ensure_voxel_centers(self) -> None:
        """Lazily compute and cache voxel centre positions."""
        if self._voxel_centers is not None:
            return
        R = self.resolution
        coords = np.arange(R, dtype=np.float32)
        gi, gj, gk = np.meshgrid(coords, coords, coords, indexing="ij")
        centers = np.stack([gi, gj, gk], axis=-1).reshape(-1, 3)
        self._voxel_centers = (
            (centers + 0.5) * self.voxel_size + self.origin.astype(np.float32)
        )

    def integrate(
        self,
        depth,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic,
        depth_scale: float = 1000.0,
        depth_max: float = 3.0,
        color=None,
    ) -> None:
        """Integrate a single depth frame into the volume.

        Parameters
        ----------
        depth : array-like
            ``(H, W)`` depth image (uint16 or float).
        intrinsic : PinholeCameraIntrinsic
            Camera intrinsic parameters.
        extrinsic : array-like
            ``(4, 4)`` world-to-camera rigid transform.
        depth_scale : float
            Depth values are divided by this to obtain metres.
        depth_max : float
            Maximum depth (metres) to integrate.
        color : array-like or None
            ``(H, W, 3)`` optional colour image.
        """
        R = self.resolution
        self._ensure_voxel_centers()

        # --- all computation in numpy for precision ---
        voxels = self._voxel_centers  # (R^3, 3) float32

        ext = np.asarray(extrinsic, dtype=np.float64)
        Rot = ext[:3, :3]
        tvec = ext[:3, 3]

        # Transform voxel centres to camera frame
        cam = (voxels @ Rot.T + tvec).astype(np.float32)  # (R^3, 3)
        z = cam[:, 2]

        # Project to pixel coordinates
        u = cam[:, 0] / z * intrinsic.fx + intrinsic.cx
        v = cam[:, 1] / z * intrinsic.fy + intrinsic.cy

        H, W = intrinsic.height, intrinsic.width

        # Validity mask
        valid = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

        # Sample depth at projected pixel (nearest-neighbour)
        ui = np.clip(u.astype(np.int32), 0, W - 1)
        vi = np.clip(v.astype(np.int32), 0, H - 1)

        depth_np = np.asarray(depth, dtype=np.float32) / depth_scale
        sampled = depth_np[vi, ui]  # (R^3,)

        # SDF
        sdf = sampled - z
        valid &= (sampled > 0) & (sampled < depth_max) & (np.abs(sdf) < self.sdf_trunc)
        tsdf_val = np.clip(sdf / self.sdf_trunc, -1.0, 1.0)

        # Weighted average update
        tsdf_flat = self._tsdf.reshape(-1)
        w_flat = self._weight.reshape(-1)

        w_new = np.where(valid, w_flat + 1.0, w_flat)
        t_new = np.where(
            valid,
            (tsdf_flat * w_flat + tsdf_val) / np.maximum(w_new, 1.0),
            tsdf_flat,
        )

        self._tsdf = t_new.reshape(R, R, R)
        self._weight = np.minimum(w_new, 255.0).reshape(R, R, R)

        # Optional colour integration
        if color is not None and self._color is not None:
            color_np = np.asarray(color, dtype=np.float32)
            sampled_color = color_np[vi, ui]  # (R^3, 3)
            color_flat = self._color.reshape(-1, 3)
            c_new = np.where(
                valid[:, None],
                (color_flat * w_flat[:, None] + sampled_color) / np.maximum(w_new[:, None], 1.0),
                color_flat,
            )
            self._color = c_new.reshape(R, R, R, 3)

    # ------------------------------------------------------------------
    # Surface extraction
    # ------------------------------------------------------------------

    def extract_point_cloud(self) -> PointCloud:
        """Extract surface points at TSDF zero-crossings.

        For each axis direction, finds adjacent voxel pairs where the TSDF
        changes sign (both must have positive weight), then linearly
        interpolates the zero-crossing position.
        """
        R = self.resolution
        tsdf = self._tsdf
        weight = self._weight

        # Build voxel-centre grid (R, R, R, 3)
        coords = np.arange(R, dtype=np.float32)
        gi, gj, gk = np.meshgrid(coords, coords, coords, indexing="ij")
        centers = np.stack([gi, gj, gk], axis=-1)
        centers = (centers + 0.5) * self.voxel_size + self.origin.astype(np.float32)

        points_list: list[np.ndarray] = []

        for axis in range(3):
            # Slices: current and next along *axis*
            slc_cur = [slice(None)] * 3
            slc_nxt = [slice(None)] * 3
            slc_cur[axis] = slice(0, R - 1)
            slc_nxt[axis] = slice(1, R)

            t0 = tsdf[tuple(slc_cur)]
            t1 = tsdf[tuple(slc_nxt)]
            w0 = weight[tuple(slc_cur)]
            w1 = weight[tuple(slc_nxt)]

            # Zero-crossing: sign change with both weights > 0
            crossing = (t0 * t1 < 0) & (w0 > 0) & (w1 > 0)
            if not np.any(crossing):
                continue

            # Interpolation factor along the axis
            denom = t0 - t1
            denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            t = t0 / denom
            t = np.clip(t, 0.0, 1.0)

            c0 = centers[tuple(slc_cur)]
            c1 = centers[tuple(slc_nxt)]

            interp = c0 + t[..., None] * (c1 - c0)
            pts = interp[crossing]  # (K, 3)
            points_list.append(pts)

        if not points_list:
            return PointCloud()

        all_pts = np.concatenate(points_list, axis=0)
        return PointCloud(mx.array(all_pts))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tsdf(self) -> mx.array:
        """Raw TSDF values ``(R, R, R)``."""
        return mx.array(self._tsdf)

    @property
    def weight(self) -> mx.array:
        """Integration weights ``(R, R, R)``."""
        return mx.array(self._weight)
