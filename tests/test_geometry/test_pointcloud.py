"""Comprehensive tests for PointCloud geometry.

Covers construction, properties, geometric queries, transforms, filtering,
downsampling, painting, cloning, concatenation, interop, and edge cases.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import numpy.testing as npt
import pytest

from open3d_mlx.geometry.pointcloud import PointCloud


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_points(n: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 3)).astype(np.float32)


def _identity_4x4() -> mx.array:
    return mx.eye(4)


def _translation_matrix(tx: float, ty: float, tz: float) -> mx.array:
    T = np.eye(4, dtype=np.float32)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return mx.array(T)


def _rotation_z(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# ===================================================================
# Construction
# ===================================================================

class TestConstruction:
    def test_create_empty_pointcloud(self):
        pcd = PointCloud()
        assert len(pcd) == 0
        assert pcd.is_empty()
        assert pcd.points.shape == (0, 3)

    def test_create_from_mlx_array(self):
        pts = mx.array(_random_points(50))
        pcd = PointCloud(pts)
        assert len(pcd) == 50
        assert not pcd.is_empty()

    def test_create_from_numpy(self):
        pts_np = _random_points(30)
        pcd = PointCloud(pts_np)
        assert len(pcd) == 30

    def test_create_from_float64_numpy(self):
        pts = np.random.randn(10, 3).astype(np.float64)
        pcd = PointCloud(pts)
        assert pcd.points.dtype == mx.float32
        assert len(pcd) == 10

    def test_points_shape_validation_1d(self):
        with pytest.raises(ValueError, match="shape"):
            PointCloud(mx.array([1.0, 2.0, 3.0]))

    def test_points_shape_validation_wrong_cols(self):
        with pytest.raises(ValueError, match="shape"):
            PointCloud(mx.zeros((5, 4)))

    def test_from_numpy_classmethod(self):
        pts = _random_points(20)
        normals = _random_points(20, seed=7)
        colors = np.abs(_random_points(20, seed=8))
        pcd = PointCloud.from_numpy(pts, normals=normals, colors=colors)
        assert len(pcd) == 20
        assert pcd.has_normals()
        assert pcd.has_colors()


# ===================================================================
# Properties
# ===================================================================

class TestProperties:
    def test_len_and_is_empty(self):
        assert len(PointCloud()) == 0
        assert PointCloud().is_empty()
        pcd = PointCloud(_random_points(5))
        assert len(pcd) == 5
        assert not pcd.is_empty()

    def test_has_normals_colors_default(self):
        pcd = PointCloud(_random_points(5))
        assert not pcd.has_normals()
        assert not pcd.has_colors()

    def test_set_normals(self):
        pcd = PointCloud(_random_points(10))
        pcd.normals = mx.ones((10, 3))
        assert pcd.has_normals()
        assert pcd.normals.shape == (10, 3)

    def test_set_colors(self):
        pcd = PointCloud(_random_points(10))
        pcd.colors = mx.zeros((10, 3))
        assert pcd.has_colors()

    def test_normals_length_mismatch(self):
        pcd = PointCloud(_random_points(10))
        with pytest.raises(ValueError, match="match"):
            pcd.normals = mx.ones((5, 3))

    def test_colors_length_mismatch(self):
        pcd = PointCloud(_random_points(10))
        with pytest.raises(ValueError, match="match"):
            pcd.colors = mx.ones((7, 3))

    def test_set_normals_none(self):
        pcd = PointCloud(_random_points(5))
        pcd.normals = mx.ones((5, 3))
        assert pcd.has_normals()
        pcd.normals = None
        assert not pcd.has_normals()

    def test_set_colors_none(self):
        pcd = PointCloud(_random_points(5))
        pcd.colors = mx.ones((5, 3))
        pcd.colors = None
        assert not pcd.has_colors()

    def test_points_setter(self):
        pcd = PointCloud(_random_points(5))
        new_pts = _random_points(8, seed=99)
        pcd.points = new_pts
        assert len(pcd) == 8

    def test_repr(self):
        pcd = PointCloud(_random_points(3))
        r = repr(pcd)
        assert "n=3" in r


# ===================================================================
# Geometric queries
# ===================================================================

class TestGeometricQueries:
    def test_get_min_max_bound(self):
        pts = np.array([[1, 2, 3], [4, 5, 6], [0, -1, 2]], dtype=np.float32)
        pcd = PointCloud(pts)
        mn = np.array(pcd.get_min_bound().tolist())
        mx_ = np.array(pcd.get_max_bound().tolist())
        npt.assert_allclose(mn, [0, -1, 2])
        npt.assert_allclose(mx_, [4, 5, 6])

    def test_get_center(self):
        pts = np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32)
        pcd = PointCloud(pts)
        c = np.array(pcd.get_center().tolist())
        npt.assert_allclose(c, [1, 1, 1])

    def test_get_aabb(self):
        pcd = PointCloud(_random_points(50))
        mn, mx_ = pcd.get_axis_aligned_bounding_box()
        assert mn.shape == (3,)
        assert mx_.shape == (3,)

    def test_bounds_empty(self):
        pcd = PointCloud()
        npt.assert_allclose(np.array(pcd.get_min_bound().tolist()), [0, 0, 0])
        npt.assert_allclose(np.array(pcd.get_center().tolist()), [0, 0, 0])


# ===================================================================
# Transforms
# ===================================================================

class TestTransforms:
    def test_transform_identity(self):
        pts_np = _random_points(20)
        pcd = PointCloud(pts_np)
        pcd2 = pcd.transform(_identity_4x4())
        # Note: MLX matmul may use reduced precision on Apple Silicon,
        # so we allow atol=2e-3 for identity transform roundtrip.
        npt.assert_allclose(
            np.array(pcd2.points.tolist()), pts_np, atol=2e-3
        )

    def test_transform_translation(self):
        pts_np = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        pcd = PointCloud(pts_np)
        T = _translation_matrix(10, 20, 30)
        pcd2 = pcd.transform(T)
        result = np.array(pcd2.points.tolist())
        expected = pts_np + np.array([10, 20, 30])
        npt.assert_allclose(result, expected, atol=1e-5)

    def test_transform_rotation(self):
        pts_np = np.array([[1, 0, 0]], dtype=np.float32)
        pcd = PointCloud(pts_np)
        R = _rotation_z(90)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        pcd2 = pcd.transform(mx.array(T))
        result = np.array(pcd2.points.tolist())
        npt.assert_allclose(result, [[0, 1, 0]], atol=1e-5)

    def test_transform_preserves_normals_direction(self):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        normals = np.array([[1, 0, 0]], dtype=np.float32)
        pcd = PointCloud(pts)
        pcd.normals = mx.array(normals)

        T = _translation_matrix(100, 200, 300)
        pcd2 = pcd.transform(T)
        # Normals should NOT be translated, only rotated (identity R here)
        nrm = np.array(pcd2.normals.tolist())
        npt.assert_allclose(nrm, [[1, 0, 0]], atol=1e-5)

    def test_transform_rotates_normals(self):
        pts = np.zeros((1, 3), dtype=np.float32)
        normals = np.array([[1, 0, 0]], dtype=np.float32)
        pcd = PointCloud(pts)
        pcd.normals = mx.array(normals)

        R = _rotation_z(90)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        pcd2 = pcd.transform(mx.array(T))
        nrm = np.array(pcd2.normals.tolist())
        npt.assert_allclose(nrm, [[0, 1, 0]], atol=1e-5)

    def test_transform_bad_shape(self):
        pcd = PointCloud(_random_points(5))
        with pytest.raises(ValueError, match="4, 4"):
            pcd.transform(mx.eye(3))

    def test_translate_relative(self):
        pts = np.array([[1, 2, 3]], dtype=np.float32)
        pcd = PointCloud(pts)
        pcd2 = pcd.translate(mx.array([10.0, 20.0, 30.0]))
        result = np.array(pcd2.points.tolist())
        npt.assert_allclose(result, [[11, 22, 33]], atol=1e-5)

    def test_translate_absolute(self):
        pts = np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32)
        pcd = PointCloud(pts)
        target = np.array([0, 0, 0], dtype=np.float32)
        pcd2 = pcd.translate(mx.array(target), relative=False)
        center = np.array(pcd2.get_center().tolist())
        npt.assert_allclose(center, [0, 0, 0], atol=1e-5)

    def test_rotate_around_center(self):
        pts = np.array([[1, 0, 0], [-1, 0, 0]], dtype=np.float32)
        pcd = PointCloud(pts)
        R = _rotation_z(180)
        pcd2 = pcd.rotate(mx.array(R), center=mx.array([0.0, 0.0, 0.0]))
        result = np.array(pcd2.points.tolist())
        npt.assert_allclose(result, [[-1, 0, 0], [1, 0, 0]], atol=1e-4)

    def test_rotate_default_center(self):
        pts = np.array([[2, 0, 0], [0, 0, 0]], dtype=np.float32)
        pcd = PointCloud(pts)
        R = _rotation_z(90)
        pcd2 = pcd.rotate(mx.array(R))
        # Centre = (1, 0, 0). After rotation around centre, result should be
        # symmetric about the centre.
        center = np.array(pcd2.get_center().tolist())
        npt.assert_allclose(center, [1, 0, 0], atol=1e-4)

    def test_rotate_bad_shape(self):
        pcd = PointCloud(_random_points(3))
        with pytest.raises(ValueError, match="3, 3"):
            pcd.rotate(mx.eye(4))

    def test_scale_uniform(self):
        pts = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        pcd = PointCloud(pts)
        pcd2 = pcd.scale(2.0, center=mx.array([0.0, 0.0, 0.0]))
        result = np.array(pcd2.points.tolist())
        npt.assert_allclose(result, [[2, 0, 0], [0, 2, 0]], atol=1e-5)

    def test_scale_with_default_center(self):
        pts = np.array([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
        pcd = PointCloud(pts)
        pcd2 = pcd.scale(3.0)
        # Centre = (1,0,0). Scaled around centre.
        center = np.array(pcd2.get_center().tolist())
        npt.assert_allclose(center, [1, 0, 0], atol=1e-5)

    def test_transform_empty(self):
        pcd = PointCloud()
        pcd2 = pcd.transform(_identity_4x4())
        assert pcd2.is_empty()


# ===================================================================
# Filtering
# ===================================================================

class TestFiltering:
    def test_select_by_index(self):
        pcd = PointCloud(_random_points(10))
        pcd2 = pcd.select_by_index(np.array([0, 2, 4]))
        assert len(pcd2) == 3

    def test_select_by_index_invert(self):
        pcd = PointCloud(_random_points(10))
        pcd2 = pcd.select_by_index(np.array([0, 1, 2]), invert=True)
        assert len(pcd2) == 7

    def test_select_by_index_preserves_attributes(self):
        pcd = PointCloud(_random_points(10))
        pcd.normals = mx.ones((10, 3))
        pcd.colors = mx.ones((10, 3)) * 0.5
        pcd2 = pcd.select_by_index(np.array([1, 3, 5]))
        assert pcd2.has_normals()
        assert pcd2.has_colors()
        assert pcd2.normals.shape == (3, 3)

    def test_select_by_mask(self):
        pcd = PointCloud(_random_points(10))
        mask = np.array([True, False] * 5)
        pcd2 = pcd.select_by_mask(mask)
        assert len(pcd2) == 5

    def test_select_by_mask_invert(self):
        pcd = PointCloud(_random_points(10))
        mask = np.array([True, False] * 5)
        pcd2 = pcd.select_by_mask(mask, invert=True)
        assert len(pcd2) == 5

    def test_select_by_mask_mlx(self):
        pcd = PointCloud(_random_points(6))
        mask = mx.array([True, True, False, False, True, False])
        pcd2 = pcd.select_by_mask(mask)
        assert len(pcd2) == 3

    def test_remove_non_finite_points(self):
        pts = np.array(
            [[1, 2, 3], [np.nan, 0, 0], [0, np.inf, 0], [4, 5, 6]],
            dtype=np.float32,
        )
        pcd = PointCloud(pts)
        pcd2 = pcd.remove_non_finite_points()
        assert len(pcd2) == 2
        result = np.array(pcd2.points.tolist())
        npt.assert_allclose(result, [[1, 2, 3], [4, 5, 6]])

    def test_remove_non_finite_preserves_colors(self):
        pts = np.array([[1, 2, 3], [np.nan, 0, 0], [4, 5, 6]], dtype=np.float32)
        pcd = PointCloud(pts)
        pcd.colors = mx.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pcd2 = pcd.remove_non_finite_points()
        assert len(pcd2) == 2
        assert pcd2.has_colors()

    def test_remove_duplicated_points(self):
        pts = np.array(
            [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9]],
            dtype=np.float32,
        )
        pcd = PointCloud(pts)
        pcd2 = pcd.remove_duplicated_points()
        assert len(pcd2) == 3

    def test_filter_empty(self):
        pcd = PointCloud()
        assert pcd.remove_non_finite_points().is_empty()
        assert pcd.remove_duplicated_points().is_empty()
        assert pcd.select_by_index(np.array([], dtype=np.intp)).is_empty()


# ===================================================================
# Downsampling
# ===================================================================

class TestDownsampling:
    def test_voxel_down_sample_reduces_count(self):
        # Dense grid: 1000 points in [0, 1]^3 with voxel 0.2 -> ~125 voxels
        pts = np.random.default_rng(42).uniform(0, 1, (1000, 3)).astype(np.float32)
        pcd = PointCloud(pts)
        pcd2 = pcd.voxel_down_sample(0.2)
        assert len(pcd2) < len(pcd)
        assert len(pcd2) > 0
        # Should reduce by >50%
        assert len(pcd2) < len(pcd) * 0.5

    def test_voxel_down_sample_preserves_attributes(self):
        pts = np.random.default_rng(0).uniform(0, 1, (200, 3)).astype(np.float32)
        pcd = PointCloud(pts)
        pcd.normals = mx.array(np.random.default_rng(1).standard_normal((200, 3)).astype(np.float32))
        pcd.colors = mx.array(np.random.default_rng(2).uniform(0, 1, (200, 3)).astype(np.float32))

        pcd2 = pcd.voxel_down_sample(0.3)
        assert pcd2.has_normals()
        assert pcd2.has_colors()
        assert pcd2.normals.shape[0] == len(pcd2)
        assert pcd2.colors.shape[0] == len(pcd2)

    def test_voxel_down_sample_normals_unit_length(self):
        pts = np.random.default_rng(0).uniform(0, 1, (100, 3)).astype(np.float32)
        nrm = np.random.default_rng(1).standard_normal((100, 3)).astype(np.float32)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        pcd = PointCloud(pts)
        pcd.normals = mx.array(nrm)
        pcd2 = pcd.voxel_down_sample(0.3)
        nrm2 = np.array(pcd2.normals.tolist())
        lengths = np.linalg.norm(nrm2, axis=1)
        npt.assert_allclose(lengths, 1.0, atol=1e-4)

    def test_voxel_down_sample_bad_size(self):
        pcd = PointCloud(_random_points(10))
        with pytest.raises(ValueError, match="> 0"):
            pcd.voxel_down_sample(0.0)
        with pytest.raises(ValueError, match="> 0"):
            pcd.voxel_down_sample(-1.0)

    def test_voxel_down_sample_empty(self):
        pcd = PointCloud()
        assert pcd.voxel_down_sample(1.0).is_empty()

    def test_uniform_down_sample(self):
        pcd = PointCloud(_random_points(100))
        pcd2 = pcd.uniform_down_sample(5)
        assert len(pcd2) == 20

    def test_uniform_down_sample_every_1(self):
        pcd = PointCloud(_random_points(10))
        pcd2 = pcd.uniform_down_sample(1)
        assert len(pcd2) == 10

    def test_uniform_down_sample_bad_k(self):
        pcd = PointCloud(_random_points(10))
        with pytest.raises(ValueError):
            pcd.uniform_down_sample(0)

    def test_random_down_sample_ratio(self):
        n = 1000
        pcd = PointCloud(_random_points(n, seed=99))
        pcd2 = pcd.random_down_sample(0.5)
        # Should be approximately 500 +/- some tolerance
        assert 400 <= len(pcd2) <= 600

    def test_random_down_sample_full(self):
        pcd = PointCloud(_random_points(10))
        pcd2 = pcd.random_down_sample(1.0)
        assert len(pcd2) == 10

    def test_random_down_sample_bad_ratio(self):
        pcd = PointCloud(_random_points(10))
        with pytest.raises(ValueError):
            pcd.random_down_sample(0.0)
        with pytest.raises(ValueError):
            pcd.random_down_sample(1.5)

    def test_random_down_sample_empty(self):
        pcd = PointCloud()
        assert pcd.random_down_sample(0.5).is_empty()


# ===================================================================
# Painting
# ===================================================================

class TestPainting:
    def test_paint_uniform_color(self):
        pcd = PointCloud(_random_points(10))
        pcd2 = pcd.paint_uniform_color([1.0, 0.0, 0.0])
        assert pcd2.has_colors()
        assert pcd2.colors.shape == (10, 3)
        col = np.array(pcd2.colors.tolist())
        npt.assert_allclose(col, np.tile([1, 0, 0], (10, 1)), atol=1e-6)

    def test_paint_empty(self):
        pcd = PointCloud()
        pcd2 = pcd.paint_uniform_color([1, 0, 0])
        assert pcd2.is_empty()


# ===================================================================
# Clone
# ===================================================================

class TestClone:
    def test_clone_independence(self):
        pcd = PointCloud(_random_points(10))
        pcd.normals = mx.ones((10, 3))
        clone = pcd.clone()

        # Modify original
        pcd.points = _random_points(5, seed=99)

        # Clone should be unchanged
        assert len(clone) == 10
        assert clone.has_normals()

    def test_clone_empty(self):
        pcd = PointCloud()
        clone = pcd.clone()
        assert clone.is_empty()


# ===================================================================
# Concatenation
# ===================================================================

class TestConcatenation:
    def test_add_concatenation(self):
        pcd1 = PointCloud(_random_points(10, seed=1))
        pcd2 = PointCloud(_random_points(20, seed=2))
        pcd3 = pcd1 + pcd2
        assert len(pcd3) == 30

    def test_add_with_attributes(self):
        pcd1 = PointCloud(_random_points(5, seed=1))
        pcd1.normals = mx.ones((5, 3))
        pcd1.colors = mx.zeros((5, 3))

        pcd2 = PointCloud(_random_points(3, seed=2))
        pcd2.normals = mx.ones((3, 3)) * 0.5
        pcd2.colors = mx.ones((3, 3))

        pcd3 = pcd1 + pcd2
        assert len(pcd3) == 8
        assert pcd3.has_normals()
        assert pcd3.has_colors()
        assert pcd3.normals.shape == (8, 3)

    def test_add_empty_left(self):
        pcd1 = PointCloud()
        pcd2 = PointCloud(_random_points(5))
        pcd3 = pcd1 + pcd2
        assert len(pcd3) == 5

    def test_add_empty_right(self):
        pcd1 = PointCloud(_random_points(5))
        pcd2 = PointCloud()
        pcd3 = pcd1 + pcd2
        assert len(pcd3) == 5

    def test_add_both_empty(self):
        pcd = PointCloud() + PointCloud()
        assert pcd.is_empty()

    def test_add_mismatched_attributes(self):
        # One has normals, other doesn't -> no normals in result
        pcd1 = PointCloud(_random_points(5))
        pcd1.normals = mx.ones((5, 3))
        pcd2 = PointCloud(_random_points(5))
        pcd3 = pcd1 + pcd2
        assert len(pcd3) == 10
        assert not pcd3.has_normals()


# ===================================================================
# NumPy interop
# ===================================================================

class TestNumpyInterop:
    def test_numpy_roundtrip(self):
        pts_np = _random_points(15)
        pcd = PointCloud(pts_np)
        d = pcd.to_numpy()
        assert "points" in d
        npt.assert_allclose(d["points"], pts_np, atol=1e-5)

    def test_numpy_roundtrip_with_attributes(self):
        pts = _random_points(10)
        normals = _random_points(10, seed=3)
        colors = np.abs(_random_points(10, seed=4))
        pcd = PointCloud.from_numpy(pts, normals=normals, colors=colors)
        d = pcd.to_numpy()
        assert "normals" in d
        assert "colors" in d
        npt.assert_allclose(d["normals"], normals, atol=1e-5)

    def test_to_numpy_empty(self):
        d = PointCloud().to_numpy()
        assert d["points"].shape == (0, 3)
        assert "normals" not in d

    def test_from_numpy_no_extras(self):
        pcd = PointCloud.from_numpy(_random_points(5))
        assert not pcd.has_normals()
        assert not pcd.has_colors()


# ===================================================================
# Stubs / NotImplementedError
# ===================================================================

class TestStubs:
    def test_estimate_normals_raises(self):
        pcd = PointCloud(_random_points(5))
        with pytest.raises(NotImplementedError):
            pcd.estimate_normals()

    def test_statistical_outliers_raises(self):
        pcd = PointCloud(_random_points(5))
        with pytest.raises(NotImplementedError):
            pcd.remove_statistical_outliers()

    def test_radius_outliers_raises(self):
        pcd = PointCloud(_random_points(5))
        with pytest.raises(NotImplementedError):
            pcd.remove_radius_outliers()

    def test_orient_normals_raises(self):
        pcd = PointCloud(_random_points(5))
        with pytest.raises(NotImplementedError):
            pcd.orient_normals_towards_camera()


# ===================================================================
# Normalize normals
# ===================================================================

class TestNormalizeNormals:
    def test_normalize_normals(self):
        pcd = PointCloud(_random_points(5))
        nrm = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5],
                         [1, 1, 0], [1, 1, 1]], dtype=np.float32)
        pcd.normals = mx.array(nrm)
        pcd.normalize_normals()
        result = np.array(pcd.normals.tolist())
        lengths = np.linalg.norm(result, axis=1)
        npt.assert_allclose(lengths, 1.0, atol=1e-5)

    def test_normalize_normals_no_normals(self):
        pcd = PointCloud(_random_points(5))
        pcd.normalize_normals()  # should be a no-op
        assert not pcd.has_normals()


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_single_point(self):
        pcd = PointCloud(np.array([[1, 2, 3]], dtype=np.float32))
        assert len(pcd) == 1
        npt.assert_allclose(
            np.array(pcd.get_center().tolist()), [1, 2, 3]
        )
        pcd2 = pcd.voxel_down_sample(1.0)
        assert len(pcd2) == 1

    def test_transform_returns_new_instance(self):
        pcd = PointCloud(_random_points(5))
        pcd2 = pcd.transform(_identity_4x4())
        assert pcd is not pcd2

    def test_translate_returns_new_instance(self):
        pcd = PointCloud(_random_points(5))
        pcd2 = pcd.translate(mx.array([0.0, 0.0, 0.0]))
        assert pcd is not pcd2

    def test_large_cloud(self):
        """Smoke test with 10k points."""
        pts = np.random.default_rng(0).standard_normal((10000, 3)).astype(np.float32)
        pcd = PointCloud(pts)
        pcd2 = pcd.voxel_down_sample(0.5)
        assert len(pcd2) < 10000
        assert len(pcd2) > 0
