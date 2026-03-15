"""MLX-backed PointCloud geometry.

This module implements the PointCloud class, the central data structure for
all 3D point-based operations in Open3D-MLX. It stores points, normals, and
colors as MLX arrays and provides transforms, downsampling, filtering, and
interop with NumPy.

Follows the tensor-based (t::geometry::PointCloud) API patterns from upstream
Open3D where practical.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np


def _ensure_float32(arr: mx.array) -> mx.array:
    """Cast to float32 if not already."""
    if arr.dtype != mx.float32:
        return arr.astype(mx.float32)
    return arr


def _validate_points(arr: mx.array, name: str = "points") -> None:
    """Validate that *arr* has shape (N, 3)."""
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3), got {tuple(arr.shape)}"
        )


def _to_mx(arr) -> mx.array:
    """Convert np.ndarray / list / mx.array to mx.array float32."""
    if isinstance(arr, np.ndarray):
        return mx.array(arr.astype(np.float32))
    if isinstance(arr, mx.array):
        return _ensure_float32(arr)
    # Attempt generic conversion (list, tuple, etc.)
    return mx.array(np.asarray(arr, dtype=np.float32))


class PointCloud:
    """MLX-backed point cloud with optional normals and colors.

    All heavy data is stored as ``mx.array`` of dtype ``float32``.
    Transformation and filtering methods return **new** PointCloud instances
    (immutable style) unless documented otherwise.

    Parameters
    ----------
    points : mx.array | np.ndarray | None
        ``(N, 3)`` array of 3-D coordinates, or ``None`` for an empty cloud.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        points: mx.array | np.ndarray | None = None,
    ) -> None:
        self._points: Optional[mx.array] = None
        self._normals: Optional[mx.array] = None
        self._colors: Optional[mx.array] = None

        if points is not None:
            pts = _to_mx(points)
            _validate_points(pts, "points")
            self._points = pts

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> mx.array:
        """``(N, 3)`` point positions. Returns empty ``(0, 3)`` if unset."""
        if self._points is None:
            return mx.zeros((0, 3))
        return self._points

    @points.setter
    def points(self, value: mx.array | np.ndarray) -> None:
        v = _to_mx(value)
        _validate_points(v, "points")
        self._points = v

    @property
    def normals(self) -> Optional[mx.array]:
        """``(N, 3)`` point normals, or ``None``."""
        return self._normals

    @normals.setter
    def normals(self, value: mx.array | np.ndarray | None) -> None:
        if value is None:
            self._normals = None
            return
        v = _to_mx(value)
        _validate_points(v, "normals")
        if self._points is not None and v.shape[0] != self._points.shape[0]:
            raise ValueError(
                f"normals length ({v.shape[0]}) must match points "
                f"length ({self._points.shape[0]})"
            )
        self._normals = v

    @property
    def colors(self) -> Optional[mx.array]:
        """``(N, 3)`` point colors in ``[0, 1]`` float32, or ``None``."""
        return self._colors

    @colors.setter
    def colors(self, value: mx.array | np.ndarray | None) -> None:
        if value is None:
            self._colors = None
            return
        v = _to_mx(value)
        _validate_points(v, "colors")
        if self._points is not None and v.shape[0] != self._points.shape[0]:
            raise ValueError(
                f"colors length ({v.shape[0]}) must match points "
                f"length ({self._points.shape[0]})"
            )
        self._colors = v

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of points."""
        if self._points is None:
            return 0
        return self._points.shape[0]

    def is_empty(self) -> bool:
        """Return ``True`` if the cloud contains zero points."""
        return len(self) == 0

    def has_normals(self) -> bool:
        """Return ``True`` if normals are set."""
        return self._normals is not None and self._normals.shape[0] > 0

    def has_colors(self) -> bool:
        """Return ``True`` if colors are set."""
        return self._colors is not None and self._colors.shape[0] > 0

    def __repr__(self) -> str:
        parts = [f"PointCloud(n={len(self)}"]
        if self.has_normals():
            parts.append("normals")
        if self.has_colors():
            parts.append("colors")
        return ", ".join(parts) + ")"

    # ------------------------------------------------------------------
    # Geometric queries
    # ------------------------------------------------------------------

    def get_min_bound(self) -> mx.array:
        """Return ``(3,)`` element-wise minimum of all points."""
        if self.is_empty():
            return mx.zeros((3,))
        return mx.min(self._points, axis=0)

    def get_max_bound(self) -> mx.array:
        """Return ``(3,)`` element-wise maximum of all points."""
        if self.is_empty():
            return mx.zeros((3,))
        return mx.max(self._points, axis=0)

    def get_center(self) -> mx.array:
        """Return ``(3,)`` centroid (mean of all points)."""
        if self.is_empty():
            return mx.zeros((3,))
        return mx.mean(self._points, axis=0)

    def get_axis_aligned_bounding_box(self) -> tuple[mx.array, mx.array]:
        """Return ``(min_bound, max_bound)``."""
        return self.get_min_bound(), self.get_max_bound()

    # ------------------------------------------------------------------
    # Transforms (return new PointCloud)
    # ------------------------------------------------------------------

    def transform(self, transformation: mx.array) -> "PointCloud":
        """Apply a 4x4 rigid (SE3) transformation.

        Points:  p' = R @ p + t
        Normals: n' = R @ n  (rotation only)
        Colors:  unchanged

        Parameters
        ----------
        transformation : mx.array
            ``(4, 4)`` transformation matrix.

        Returns
        -------
        PointCloud
            Transformed point cloud (new instance).
        """
        T = _ensure_float32(transformation)
        if T.shape != (4, 4):
            raise ValueError(
                f"transformation must be (4, 4), got {tuple(T.shape)}"
            )

        R = T[:3, :3]
        t = T[:3, 3]

        new_pcd = self.clone()
        if not self.is_empty():
            new_pcd._points = self._points @ R.T + t[None, :]

        if self.has_normals():
            new_pcd._normals = self._normals @ R.T

        return new_pcd

    def translate(
        self, translation: mx.array | np.ndarray, relative: bool = True
    ) -> "PointCloud":
        """Translate the point cloud.

        Parameters
        ----------
        translation : array-like
            ``(3,)`` translation vector.
        relative : bool
            If ``True`` (default), add *translation* to current positions.
            If ``False``, move the centroid to *translation*.

        Returns
        -------
        PointCloud
        """
        t = _to_mx(translation).reshape((3,))
        new_pcd = self.clone()
        if self.is_empty():
            return new_pcd

        if relative:
            new_pcd._points = self._points + t[None, :]
        else:
            center = self.get_center()
            new_pcd._points = self._points + (t - center)[None, :]

        return new_pcd

    def rotate(
        self,
        rotation: mx.array | np.ndarray,
        center: mx.array | np.ndarray | None = None,
    ) -> "PointCloud":
        """Rotate the point cloud.

        Parameters
        ----------
        rotation : array-like
            ``(3, 3)`` rotation matrix.
        center : array-like or None
            Rotation centre. Defaults to centroid.

        Returns
        -------
        PointCloud
        """
        R = _to_mx(rotation)
        if R.shape != (3, 3):
            raise ValueError(
                f"rotation must be (3, 3), got {tuple(R.shape)}"
            )

        new_pcd = self.clone()
        if self.is_empty():
            return new_pcd

        if center is None:
            c = self.get_center()
        else:
            c = _to_mx(center).reshape((3,))

        pts = self._points - c[None, :]
        new_pcd._points = pts @ R.T + c[None, :]

        if self.has_normals():
            new_pcd._normals = self._normals @ R.T

        return new_pcd

    def scale(
        self,
        factor: float,
        center: mx.array | np.ndarray | None = None,
    ) -> "PointCloud":
        """Uniformly scale the point cloud.

        Parameters
        ----------
        factor : float
            Scale factor.
        center : array-like or None
            Scale centre. Defaults to centroid.

        Returns
        -------
        PointCloud
        """
        new_pcd = self.clone()
        if self.is_empty():
            return new_pcd

        if center is None:
            c = self.get_center()
        else:
            c = _to_mx(center).reshape((3,))

        new_pcd._points = (self._points - c[None, :]) * factor + c[None, :]
        # Normals direction is unchanged by uniform scaling.
        return new_pcd

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def select_by_index(
        self, indices: mx.array | np.ndarray, invert: bool = False
    ) -> "PointCloud":
        """Select points by integer indices.

        Parameters
        ----------
        indices : array-like
            1-D array of integer indices.
        invert : bool
            If ``True``, select all points *except* those at *indices*.

        Returns
        -------
        PointCloud
        """
        if self.is_empty():
            return PointCloud()

        n = len(self)
        if isinstance(indices, mx.array):
            idx_np = np.asarray(indices, dtype=np.intp)
        else:
            idx_np = np.asarray(indices, dtype=np.intp)

        if invert:
            mask = np.ones(n, dtype=bool)
            mask[idx_np] = False
            idx_np = np.nonzero(mask)[0]

        new_pcd = PointCloud(self._points[mx.array(idx_np)])
        if self.has_normals():
            new_pcd._normals = self._normals[mx.array(idx_np)]
        if self.has_colors():
            new_pcd._colors = self._colors[mx.array(idx_np)]
        return new_pcd

    def select_by_mask(
        self, mask: mx.array | np.ndarray, invert: bool = False
    ) -> "PointCloud":
        """Select points by boolean mask.

        Parameters
        ----------
        mask : array-like
            Boolean array of length ``N``.
        invert : bool
            If ``True``, invert the mask before selecting.

        Returns
        -------
        PointCloud
        """
        if isinstance(mask, mx.array):
            m_np = np.asarray(mask, dtype=bool)
        else:
            m_np = np.asarray(mask, dtype=bool)

        if invert:
            m_np = ~m_np

        idx_np = np.nonzero(m_np)[0]
        idx = mx.array(idx_np)

        new_pcd = PointCloud()
        if self.is_empty() or len(idx_np) == 0:
            return new_pcd

        new_pcd._points = self._points[idx]
        if self.has_normals():
            new_pcd._normals = self._normals[idx]
        if self.has_colors():
            new_pcd._colors = self._colors[idx]
        return new_pcd

    def remove_non_finite_points(self) -> "PointCloud":
        """Remove points that contain NaN or Inf in any coordinate.

        Returns
        -------
        PointCloud
            Cleaned point cloud.
        """
        if self.is_empty():
            return PointCloud()

        pts_np = np.asarray(self._points, dtype=np.float32)
        finite_mask = np.all(np.isfinite(pts_np), axis=1)
        return self.select_by_mask(finite_mask)

    def remove_duplicated_points(self) -> "PointCloud":
        """Remove exact duplicate points, keeping the first occurrence.

        Returns
        -------
        PointCloud
        """
        if self.is_empty():
            return PointCloud()

        pts_np = np.asarray(self._points, dtype=np.float32)
        _, unique_idx = np.unique(pts_np, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)  # preserve original order
        return self.select_by_index(unique_idx)

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    def voxel_down_sample(self, voxel_size: float) -> "PointCloud":
        """Voxel grid downsampling via spatial hashing.

        Algorithm:
            1. Compute voxel indices: ``floor(points / voxel_size)``
            2. Hash indices with large primes to scalar keys
            3. Group by unique key, average positions (and attributes)

        Parameters
        ----------
        voxel_size : float
            Side length of each voxel. Must be positive.

        Returns
        -------
        PointCloud
            Down-sampled point cloud.
        """
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
        if self.is_empty():
            return PointCloud()

        pts_np = np.asarray(self._points, dtype=np.float32)
        n = pts_np.shape[0]

        # Voxel indices
        voxel_idx = np.floor(pts_np / voxel_size).astype(np.int64)

        # Collision-free uniqueness using np.unique on axis=0
        _, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
        n_voxels = int(inverse.max() + 1)

        # Average positions within each voxel
        new_pts = np.zeros((n_voxels, 3), dtype=np.float64)
        counts = np.zeros(n_voxels, dtype=np.float64)
        np.add.at(new_pts, inverse, pts_np.astype(np.float64))
        np.add.at(counts, inverse, 1.0)
        new_pts /= counts[:, None]

        result = PointCloud(mx.array(new_pts.astype(np.float32)))

        # Average normals (then re-normalize)
        if self.has_normals():
            nrm_np = np.asarray(self._normals, dtype=np.float64)
            new_nrm = np.zeros((n_voxels, 3), dtype=np.float64)
            np.add.at(new_nrm, inverse, nrm_np)
            lengths = np.linalg.norm(new_nrm, axis=1, keepdims=True)
            lengths = np.maximum(lengths, 1e-12)
            new_nrm /= lengths
            result._normals = mx.array(new_nrm.astype(np.float32))

        # Average colors (clamp to [0, 1])
        if self.has_colors():
            col_np = np.asarray(self._colors, dtype=np.float64)
            new_col = np.zeros((n_voxels, 3), dtype=np.float64)
            np.add.at(new_col, inverse, col_np)
            new_col /= counts[:, None]
            new_col = np.clip(new_col, 0.0, 1.0)
            result._colors = mx.array(new_col.astype(np.float32))

        return result

    def uniform_down_sample(self, every_k_points: int) -> "PointCloud":
        """Keep every k-th point.

        Parameters
        ----------
        every_k_points : int
            Sampling stride. Must be >= 1.

        Returns
        -------
        PointCloud
        """
        if every_k_points < 1:
            raise ValueError(
                f"every_k_points must be >= 1, got {every_k_points}"
            )
        if self.is_empty():
            return PointCloud()

        idx = np.arange(0, len(self), every_k_points)
        return self.select_by_index(idx)

    def random_down_sample(self, sampling_ratio: float) -> "PointCloud":
        """Randomly sample a fraction of points.

        Parameters
        ----------
        sampling_ratio : float
            Fraction in ``(0, 1]``.

        Returns
        -------
        PointCloud
        """
        if not (0.0 < sampling_ratio <= 1.0):
            raise ValueError(
                f"sampling_ratio must be in (0, 1], got {sampling_ratio}"
            )
        if self.is_empty():
            return PointCloud()

        n = len(self)
        k = max(1, int(round(n * sampling_ratio)))
        rng = np.random.default_rng()
        idx = rng.choice(n, size=k, replace=False)
        idx = np.sort(idx)
        return self.select_by_index(idx)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paint_uniform_color(
        self, color: mx.array | np.ndarray | list
    ) -> "PointCloud":
        """Set all point colors to a single RGB value.

        Parameters
        ----------
        color : array-like
            ``(3,)`` RGB color in ``[0, 1]``.

        Returns
        -------
        PointCloud
        """
        c = _to_mx(color).reshape((3,))
        new_pcd = self.clone()
        if self.is_empty():
            return new_pcd

        new_pcd._colors = mx.broadcast_to(c[None, :], (len(self), 3))
        # Ensure contiguous storage
        new_pcd._colors = mx.array(np.asarray(new_pcd._colors, dtype=np.float32))
        return new_pcd

    # ------------------------------------------------------------------
    # Normals stubs (deferred to PRD-04)
    # ------------------------------------------------------------------

    def crop(self, bounding_box) -> "PointCloud":
        """Return new PointCloud with only points inside the bounding box.

        Parameters
        ----------
        bounding_box : AxisAlignedBoundingBox
            The bounding box to crop to.

        Returns
        -------
        PointCloud
        """
        if self.is_empty():
            return PointCloud()

        pts_np = np.asarray(self.points, dtype=np.float32)
        min_b = np.asarray(bounding_box.min_bound, dtype=np.float32)
        max_b = np.asarray(bounding_box.max_bound, dtype=np.float32)
        mask = np.all((pts_np >= min_b) & (pts_np <= max_b), axis=1)
        return self.select_by_mask(mask)

    def farthest_point_down_sample(self, num_samples: int) -> "PointCloud":
        """Farthest point sampling -- greedily pick points maximizing min distance.

        Parameters
        ----------
        num_samples : int
            Number of points to select.

        Returns
        -------
        PointCloud
        """
        if self.is_empty():
            return PointCloud()

        pts_np = np.asarray(self.points, dtype=np.float64)
        N = len(pts_np)
        num_samples = min(num_samples, N)
        if num_samples <= 0:
            return PointCloud()

        selected = [0]
        min_dists = np.full(N, np.inf)
        for _ in range(num_samples - 1):
            last = pts_np[selected[-1]]
            dists = np.sum((pts_np - last) ** 2, axis=1)
            min_dists = np.minimum(min_dists, dists)
            selected.append(int(np.argmax(min_dists)))

        return self.select_by_index(selected)

    # ------------------------------------------------------------------
    # Normals
    # ------------------------------------------------------------------

    def estimate_normals(
        self,
        max_nn: int = 30,
        radius: float | None = None,
        search_param=None,
    ) -> None:
        """Estimate normals using PCA on local neighbourhoods.

        Uses KNN (or hybrid radius+KNN) search to find local neighborhoods,
        then computes normals via PCA (smallest eigenvector of covariance).

        Parameters
        ----------
        max_nn : int
            Maximum number of neighbors for PCA. Default 30.
        radius : float or None
            If given, use hybrid search (radius + max_nn constraint).
            If None, use pure KNN search with k=max_nn.
        search_param : KDTreeSearchParamKNN | KDTreeSearchParamRadius | KDTreeSearchParamHybrid | None
            If given, overrides max_nn and radius for Open3D API compatibility.
        """
        if search_param is not None:
            from open3d_mlx.geometry.kdtree import (
                KDTreeSearchParamHybrid,
                KDTreeSearchParamKNN,
                KDTreeSearchParamRadius,
            )

            if isinstance(search_param, KDTreeSearchParamHybrid):
                max_nn = search_param.max_nn
                radius = search_param.radius
            elif isinstance(search_param, KDTreeSearchParamKNN):
                max_nn = search_param.knn
                radius = None
            elif isinstance(search_param, KDTreeSearchParamRadius):
                max_nn = 30
                radius = search_param.radius
        if self.is_empty():
            return

        from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch
        from open3d_mlx.ops.normals import estimate_normals_pca

        nns = NearestNeighborSearch(self._points)
        if radius is not None:
            indices, _, _ = nns.hybrid_search(
                self._points, radius=radius, max_nn=max_nn
            )
        else:
            indices, _ = nns.knn_search(self._points, k=max_nn)
        self._normals = estimate_normals_pca(self._points, indices)

    def normalize_normals(self) -> None:
        """Normalize all normals to unit length. Modifies in-place."""
        if not self.has_normals():
            return
        nrm_np = np.asarray(self._normals, dtype=np.float64)
        lengths = np.linalg.norm(nrm_np, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        nrm_np /= lengths
        self._normals = mx.array(nrm_np.astype(np.float32))

    def orient_normals_towards_camera(
        self, camera_location: mx.array | None = None
    ) -> None:
        """Orient normals to face camera. Default camera at origin.

        Flips normals so that they point towards the given camera location.

        Parameters
        ----------
        camera_location : mx.array or None
            ``(3,)`` camera position. Defaults to origin ``[0, 0, 0]``.
        """
        if not self.has_normals():
            return

        from open3d_mlx.ops.normals import orient_normals_towards_viewpoint

        self._normals = orient_normals_towards_viewpoint(
            self._points, self._normals, viewpoint=camera_location
        )

    # ------------------------------------------------------------------
    # Outlier removal stubs (deferred to PRD-04)
    # ------------------------------------------------------------------

    def remove_statistical_outliers(
        self, nb_neighbors: int = 20, std_ratio: float = 2.0
    ) -> tuple["PointCloud", mx.array]:
        """Remove statistical outliers based on mean distance to neighbors.

        For each point, computes the mean distance to its K nearest neighbors.
        Points whose mean distance is beyond ``mean + std_ratio * std`` of the
        global distribution are considered outliers.

        Parameters
        ----------
        nb_neighbors : int
            Number of neighbors for mean distance computation.
        std_ratio : float
            Standard deviation multiplier threshold.

        Returns
        -------
        tuple[PointCloud, mx.array]
            (filtered_cloud, inlier_indices)
        """
        if self.is_empty():
            return PointCloud(), mx.array(np.array([], dtype=np.int32))

        from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch

        nns = NearestNeighborSearch(self._points)
        # k+1 because the point itself is its own nearest neighbor at distance 0
        k = min(nb_neighbors + 1, len(self))
        indices, sq_dists = nns.knn_search(self._points, k=k)

        # Skip the first neighbor (self) — take columns 1:
        if k > 1:
            sq_dists_neighbors = np.asarray(sq_dists[:, 1:], dtype=np.float64)
        else:
            sq_dists_neighbors = np.asarray(sq_dists, dtype=np.float64)

        mean_dists = np.sqrt(sq_dists_neighbors).mean(axis=1)
        global_mean = mean_dists.mean()
        global_std = mean_dists.std()

        threshold = global_mean + std_ratio * global_std
        inlier_mask = mean_dists <= threshold
        inlier_indices = np.nonzero(inlier_mask)[0]

        return self.select_by_index(inlier_indices), mx.array(inlier_indices.astype(np.int32))

    def remove_radius_outliers(
        self, nb_points: int = 2, search_radius: float = 1.0
    ) -> tuple["PointCloud", mx.array]:
        """Remove points that have fewer than ``nb_points`` neighbors within ``search_radius``.

        Parameters
        ----------
        nb_points : int
            Minimum number of neighbors required to keep a point.
        search_radius : float
            Radius for neighbor search.

        Returns
        -------
        tuple[PointCloud, mx.array]
            (filtered_cloud, inlier_indices)
        """
        if self.is_empty():
            return PointCloud(), mx.array(np.array([], dtype=np.int32))

        from open3d_mlx.ops.nearest_neighbor import NearestNeighborSearch

        nns = NearestNeighborSearch(self._points)
        idx_lists, _ = nns.radius_search(self._points, radius=search_radius)

        # Count neighbors (excluding self — the point itself is always found)
        inlier_indices = []
        for i, idx_arr in enumerate(idx_lists):
            # Subtract 1 for self-match
            n_neighbors = len(np.asarray(idx_arr)) - 1
            if n_neighbors >= nb_points:
                inlier_indices.append(i)

        inlier_indices = np.array(inlier_indices, dtype=np.int32)
        return self.select_by_index(inlier_indices), mx.array(inlier_indices)

    # ------------------------------------------------------------------
    # Interop
    # ------------------------------------------------------------------

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Export all attributes as a dict of NumPy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Always contains ``'points'``. May contain ``'normals'`` and
            ``'colors'``.
        """
        result: dict[str, np.ndarray] = {}
        if self.is_empty():
            result["points"] = np.zeros((0, 3), dtype=np.float32)
        else:
            result["points"] = np.asarray(self.points, dtype=np.float32)
        if self.has_normals():
            result["normals"] = np.asarray(self._normals, dtype=np.float32)
        if self.has_colors():
            result["colors"] = np.asarray(self._colors, dtype=np.float32)
        return result

    @classmethod
    def from_numpy(
        cls,
        points: np.ndarray,
        normals: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        **kwargs,
    ) -> "PointCloud":
        """Create a PointCloud from NumPy arrays.

        Parameters
        ----------
        points : np.ndarray
            ``(N, 3)`` positions.
        normals : np.ndarray or None
            ``(N, 3)`` normals.
        colors : np.ndarray or None
            ``(N, 3)`` colors.

        Returns
        -------
        PointCloud
        """
        pcd = cls(points)
        if normals is not None:
            pcd.normals = _to_mx(normals)
        if colors is not None:
            pcd.colors = _to_mx(colors)
        return pcd

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def clone(self) -> "PointCloud":
        """Create an independent deep copy.

        Returns
        -------
        PointCloud
        """
        new_pcd = PointCloud()
        if self._points is not None:
            # Use addition of zero to force a copy in MLX
            new_pcd._points = self._points + mx.zeros_like(self._points)
        if self._normals is not None:
            new_pcd._normals = self._normals + mx.zeros_like(self._normals)
        if self._colors is not None:
            new_pcd._colors = self._colors + mx.zeros_like(self._colors)
        return new_pcd

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def __add__(self, other: "PointCloud") -> "PointCloud":
        """Concatenate two point clouds.

        Parameters
        ----------
        other : PointCloud

        Returns
        -------
        PointCloud
        """
        if not isinstance(other, PointCloud):
            return NotImplemented

        if self.is_empty() and other.is_empty():
            return PointCloud()
        if self.is_empty():
            return other.clone()
        if other.is_empty():
            return self.clone()

        new_pts = mx.concatenate([self._points, other._points], axis=0)
        new_pcd = PointCloud(new_pts)

        # Normals: concatenate if both have them
        if self.has_normals() and other.has_normals():
            new_pcd._normals = mx.concatenate(
                [self._normals, other._normals], axis=0
            )

        # Colors: concatenate if both have them
        if self.has_colors() and other.has_colors():
            new_pcd._colors = mx.concatenate(
                [self._colors, other._colors], axis=0
            )

        return new_pcd
