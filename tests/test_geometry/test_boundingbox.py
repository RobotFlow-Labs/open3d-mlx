"""Tests for AxisAlignedBoundingBox."""

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import AxisAlignedBoundingBox


class TestCreation:
    def test_from_min_max(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([1.0, 2.0, 3.0]),
        )
        np.testing.assert_allclose(np.asarray(bbox.min_bound), [0, 0, 0])
        np.testing.assert_allclose(np.asarray(bbox.max_bound), [1, 2, 3])

    def test_from_points(self):
        pts = mx.array([
            [1.0, 2.0, 3.0],
            [-1.0, 0.0, 5.0],
            [0.5, 1.5, 0.0],
        ])
        bbox = AxisAlignedBoundingBox.create_from_points(pts)
        np.testing.assert_allclose(np.asarray(bbox.min_bound), [-1, 0, 0])
        np.testing.assert_allclose(np.asarray(bbox.max_bound), [1, 2, 5])

    def test_default_is_zero(self):
        bbox = AxisAlignedBoundingBox()
        np.testing.assert_allclose(np.asarray(bbox.min_bound), [0, 0, 0])
        np.testing.assert_allclose(np.asarray(bbox.max_bound), [0, 0, 0])


class TestProperties:
    def test_center(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([2.0, 4.0, 6.0]),
        )
        center = np.asarray(bbox.get_center())
        np.testing.assert_allclose(center, [1, 2, 3])

    def test_extent(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([1.0, 2.0, 3.0]),
            max_bound=mx.array([4.0, 6.0, 9.0]),
        )
        ext = np.asarray(bbox.get_extent())
        np.testing.assert_allclose(ext, [3, 4, 6])

    def test_half_extent(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([2.0, 4.0, 6.0]),
        )
        he = np.asarray(bbox.get_half_extent())
        np.testing.assert_allclose(he, [1, 2, 3])

    def test_volume(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([2.0, 3.0, 4.0]),
        )
        assert abs(bbox.volume() - 24.0) < 1e-6


class TestContains:
    def test_points_inside(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([1.0, 1.0, 1.0]),
        )
        pts = mx.array([
            [0.5, 0.5, 0.5],  # inside
            [2.0, 0.5, 0.5],  # outside (x)
            [0.5, -0.1, 0.5], # outside (y)
        ])
        result = np.asarray(bbox.contains(pts))
        assert result[0] == True
        assert result[1] == False
        assert result[2] == False

    def test_boundary_points(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([1.0, 1.0, 1.0]),
        )
        pts = mx.array([
            [0.0, 0.0, 0.0],  # on min boundary
            [1.0, 1.0, 1.0],  # on max boundary
        ])
        result = np.asarray(bbox.contains(pts))
        assert result[0] == True
        assert result[1] == True

    def test_repr(self):
        bbox = AxisAlignedBoundingBox(
            min_bound=mx.array([0.0, 0.0, 0.0]),
            max_bound=mx.array([1.0, 1.0, 1.0]),
        )
        r = repr(bbox)
        assert "AxisAlignedBoundingBox" in r
