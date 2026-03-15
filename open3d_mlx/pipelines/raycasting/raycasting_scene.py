"""Ray-TSDF volume intersection for rendering.

Implements sphere-tracing / adaptive ray marching through a TSDF volume
to render synthetic depth maps and normal maps.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from open3d_mlx.camera.intrinsics import PinholeCameraIntrinsic
from open3d_mlx.pipelines.raycasting.ray_utils import generate_rays


class RaycastingScene:
    """Ray-TSDF volume intersection for rendering.

    Renders synthetic depth and normal maps from a TSDF volume by marching
    rays through the volume and finding zero-crossings of the distance field.
    """

    def __init__(self):
        self._volume = None

    def set_volume(self, volume) -> None:
        """Set the TSDF volume to raycast against.

        Parameters
        ----------
        volume : UniformTSDFVolume
            A TSDF volume that has been integrated with depth frames.
        """
        self._volume = volume

    def cast_rays(
        self,
        rays: mx.array,
        max_steps: int = 200,
        min_step_size: float = 0.002,
    ) -> dict[str, mx.array]:
        """Cast rays against the TSDF volume.

        Parameters
        ----------
        rays : mx.array
            (N, 6) ray origins and directions.
        max_steps : int
            Maximum number of marching steps per ray.
        min_step_size : float
            Minimum step size in meters.

        Returns
        -------
        dict
            ``"t_hit"`` : (N,) float32 -- distance along ray to surface (inf = miss).
            ``"normals"`` : (N, 3) float32 -- surface normals at hit points.
            ``"positions"`` : (N, 3) float32 -- world-space hit positions.
        """
        if self._volume is None:
            raise RuntimeError("No volume set. Call set_volume() first.")

        vol = self._volume
        tsdf = vol._tsdf  # (R, R, R) numpy
        weight = vol._weight  # (R, R, R) numpy
        origin = vol.origin.astype(np.float64)
        voxel_size = vol.voxel_size
        resolution = vol.resolution

        rays_np = np.asarray(rays, dtype=np.float32)
        if rays_np.ndim == 1:
            rays_np = rays_np.reshape(1, 6)

        N = rays_np.shape[0]
        origins = rays_np[:, :3].astype(np.float64)
        directions = rays_np[:, 3:].astype(np.float64)

        # Normalize directions
        dir_norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        dir_norms = np.maximum(dir_norms, 1e-10)
        directions = directions / dir_norms

        # Ray-AABB intersection
        volume_min = origin
        volume_max = origin + resolution * voxel_size
        t_entry, t_exit = _ray_aabb_intersect(origins, directions, volume_min, volume_max)

        # Initialize outputs
        t_hit = np.full(N, np.inf, dtype=np.float64)
        hit_normals = np.zeros((N, 3), dtype=np.float64)

        # Active mask: rays that intersect the volume
        active = t_entry < t_exit
        if not np.any(active):
            return {
                "t_hit": mx.array(t_hit.astype(np.float32)),
                "normals": mx.array(hit_normals.astype(np.float32)),
                "positions": mx.array(np.zeros((N, 3), dtype=np.float32)),
            }

        # Start just inside the volume
        t_current = np.where(active, np.maximum(t_entry, 0.0) + min_step_size, 0.0)
        prev_tsdf = np.ones(N, dtype=np.float64)
        prev_t = t_current.copy()

        for _step in range(max_steps):
            if not np.any(active):
                break

            # Current positions
            pos = origins + t_current[:, None] * directions  # (N, 3)

            # Sample TSDF
            tsdf_val = _sample_tsdf_trilinear(
                pos, tsdf, weight, origin, voxel_size, resolution, active
            )

            # Check for zero-crossing: prev positive, current negative
            crossing = active & (prev_tsdf > 0) & (tsdf_val < 0)

            if np.any(crossing):
                # Bisect to find accurate surface location
                t_surface = _bisect_zero_crossing(
                    origins, directions, prev_t, t_current,
                    tsdf, weight, origin, voxel_size, resolution,
                    crossing, num_iters=6,
                )
                t_hit = np.where(crossing, t_surface, t_hit)

                # Compute normals from TSDF gradient
                hit_pos = origins + t_surface[:, None] * directions
                normals = _compute_tsdf_gradient(
                    hit_pos, tsdf, weight, origin, voxel_size, resolution, crossing
                )
                hit_normals = np.where(crossing[:, None], normals, hit_normals)

                # Deactivate rays that hit
                active = active & ~crossing

            # Adaptive step size
            step_size = np.maximum(np.abs(tsdf_val) * voxel_size, min_step_size)
            prev_t = t_current.copy()
            prev_tsdf = tsdf_val.copy()
            t_current = t_current + step_size

            # Deactivate rays that exit volume
            active = active & (t_current < t_exit)

        # Compute hit positions
        positions = origins + t_hit[:, None] * directions
        positions = np.where(np.isinf(t_hit[:, None]), 0.0, positions)

        return {
            "t_hit": mx.array(t_hit.astype(np.float32)),
            "normals": mx.array(hit_normals.astype(np.float32)),
            "positions": mx.array(positions.astype(np.float32)),
        }

    def render_depth(
        self,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic,
        max_steps: int = 200,
    ) -> mx.array:
        """Render a depth image from the TSDF volume.

        Parameters
        ----------
        intrinsic : PinholeCameraIntrinsic
            Camera intrinsics.
        extrinsic : array-like
            (4, 4) camera-to-world transformation.
        max_steps : int
            Maximum marching steps per ray.

        Returns
        -------
        depth : mx.array
            (H, W) float32 depth image in meters.
        """
        rays = generate_rays(intrinsic, extrinsic)
        H, W = intrinsic.height, intrinsic.width
        flat_rays = rays.reshape(H * W, 6)
        result = self.cast_rays(flat_rays, max_steps=max_steps)
        return result["t_hit"].reshape(H, W)

    def render_normal(
        self,
        intrinsic: PinholeCameraIntrinsic,
        extrinsic,
        max_steps: int = 200,
    ) -> mx.array:
        """Render a normal map from the TSDF volume.

        Parameters
        ----------
        intrinsic : PinholeCameraIntrinsic
            Camera intrinsics.
        extrinsic : array-like
            (4, 4) camera-to-world transformation.
        max_steps : int
            Maximum marching steps per ray.

        Returns
        -------
        normals : mx.array
            (H, W, 3) float32 normal map.
        """
        rays = generate_rays(intrinsic, extrinsic)
        H, W = intrinsic.height, intrinsic.width
        flat_rays = rays.reshape(H * W, 6)
        result = self.cast_rays(flat_rays, max_steps=max_steps)
        return result["normals"].reshape(H, W, 3)


# ==================================================================
# Internal helpers
# ==================================================================


def _ray_aabb_intersect(
    origins: np.ndarray,
    directions: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-AABB slab intersection for N rays.

    Parameters
    ----------
    origins : (N, 3)
    directions : (N, 3)
    box_min, box_max : (3,)

    Returns
    -------
    t_entry, t_exit : (N,) entry and exit distances (t_entry > t_exit means miss).
    """
    # Handle near-zero direction components to avoid division issues
    inv_dir = np.where(
        np.abs(directions) < 1e-12,
        np.sign(directions + 1e-30) * 1e12,
        1.0 / directions,
    )

    t_min = (box_min - origins) * inv_dir  # (N, 3)
    t_max = (box_max - origins) * inv_dir  # (N, 3)

    t_lo = np.minimum(t_min, t_max)
    t_hi = np.maximum(t_min, t_max)

    t_entry = np.max(t_lo, axis=-1)  # (N,)
    t_exit = np.min(t_hi, axis=-1)  # (N,)

    return t_entry, t_exit


def _sample_tsdf_trilinear(
    positions: np.ndarray,
    tsdf: np.ndarray,
    weight: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    resolution: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Trilinear interpolation of TSDF at arbitrary world positions.

    Parameters
    ----------
    positions : (N, 3) world-space positions.
    tsdf : (R, R, R) TSDF grid.
    weight : (R, R, R) weight grid.
    origin : (3,) volume origin.
    voxel_size : float
    resolution : int
    mask : (N,) bool -- only sample where True (others get 1.0).

    Returns
    -------
    values : (N,) interpolated TSDF values.
    """
    N = positions.shape[0]
    values = np.ones(N, dtype=np.float64)

    if mask is not None and not np.any(mask):
        return values

    # Convert world to voxel coordinates (center of voxel 0 is at 0.5)
    voxel_coords = (positions - origin) / voxel_size - 0.5  # (N, 3)

    # Floor indices
    i0 = np.floor(voxel_coords).astype(np.int64)
    frac = voxel_coords - i0  # fractional part [0, 1)

    # Clamp to valid range
    i0 = np.clip(i0, 0, resolution - 2)
    i1 = i0 + 1

    # Apply mask
    if mask is not None:
        idx = np.where(mask)[0]
    else:
        idx = np.arange(N)

    if len(idx) == 0:
        return values

    ix0 = i0[idx, 0]
    iy0 = i0[idx, 1]
    iz0 = i0[idx, 2]
    ix1 = i1[idx, 0]
    iy1 = i1[idx, 1]
    iz1 = i1[idx, 2]
    fx = frac[idx, 0]
    fy = frac[idx, 1]
    fz = frac[idx, 2]

    # 8 corner lookups
    c000 = tsdf[ix0, iy0, iz0]
    c001 = tsdf[ix0, iy0, iz1]
    c010 = tsdf[ix0, iy1, iz0]
    c011 = tsdf[ix0, iy1, iz1]
    c100 = tsdf[ix1, iy0, iz0]
    c101 = tsdf[ix1, iy0, iz1]
    c110 = tsdf[ix1, iy1, iz0]
    c111 = tsdf[ix1, iy1, iz1]

    # Trilinear interpolation
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    result = c0 * (1 - fz) + c1 * fz
    values[idx] = result

    return values


def _bisect_zero_crossing(
    origins: np.ndarray,
    directions: np.ndarray,
    t_lo: np.ndarray,
    t_hi: np.ndarray,
    tsdf: np.ndarray,
    weight: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    resolution: int,
    mask: np.ndarray,
    num_iters: int = 6,
) -> np.ndarray:
    """Bisection refinement to find zero-crossing between t_lo and t_hi.

    Parameters
    ----------
    origins, directions : (N, 3)
    t_lo, t_hi : (N,) -- bracket around zero-crossing.
    tsdf, weight : volume grids.
    origin, voxel_size, resolution : volume params.
    mask : (N,) bool -- only refine where True.
    num_iters : int -- number of bisection steps.

    Returns
    -------
    t_surface : (N,) refined t values at zero-crossing.
    """
    lo = t_lo.copy()
    hi = t_hi.copy()

    for _ in range(num_iters):
        mid = (lo + hi) * 0.5
        pos = origins + mid[:, None] * directions
        tsdf_mid = _sample_tsdf_trilinear(
            pos, tsdf, weight, origin, voxel_size, resolution, mask
        )
        # If tsdf_mid > 0, zero-crossing is between mid and hi
        # If tsdf_mid < 0, zero-crossing is between lo and mid
        move_lo = mask & (tsdf_mid > 0)
        lo = np.where(move_lo, mid, lo)
        hi = np.where(mask & ~move_lo, mid, hi)

    return (lo + hi) * 0.5


def _compute_tsdf_gradient(
    positions: np.ndarray,
    tsdf: np.ndarray,
    weight: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    resolution: int,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute TSDF gradient via central differences (for surface normals).

    Parameters
    ----------
    positions : (N, 3) world-space positions.
    tsdf, weight : volume grids.
    origin, voxel_size, resolution : volume params.
    mask : (N,) bool.

    Returns
    -------
    normals : (N, 3) normalized gradient vectors.
    """
    eps = voxel_size * 0.5
    N = positions.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)

    offsets = np.eye(3, dtype=np.float64) * eps

    for axis in range(3):
        pos_plus = positions + offsets[axis]
        pos_minus = positions - offsets[axis]
        val_plus = _sample_tsdf_trilinear(
            pos_plus, tsdf, weight, origin, voxel_size, resolution, mask
        )
        val_minus = _sample_tsdf_trilinear(
            pos_minus, tsdf, weight, origin, voxel_size, resolution, mask
        )
        normals[:, axis] = val_plus - val_minus

    # Normalize
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normals = normals / norms

    return normals
