"""Tests for FPFH feature descriptor computation."""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from open3d_mlx.geometry import PointCloud
from open3d_mlx.geometry.kdtree import KDTreeSearchParamHybrid, KDTreeSearchParamKNN
from open3d_mlx.pipelines.registration.feature import compute_fpfh_feature


def _make_sphere_pcd(n: int = 50, radius: float = 1.0) -> PointCloud:
    """Create a point cloud on a sphere with outward normals."""
    # Fibonacci sphere for roughly uniform distribution
    indices = np.arange(0, n, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(phi) * radius

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)

    pcd = PointCloud(mx.array(points))
    pcd.normals = mx.array(normals.astype(np.float32))
    return pcd


def _make_plane_pcd(n: int = 25) -> PointCloud:
    """Create a flat grid point cloud with z-up normals."""
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(0, 1, side)
    ys = np.linspace(0, 1, side)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel(), np.zeros(side * side)], axis=1)
    points = points[:n].astype(np.float32)
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1)).astype(np.float32)

    pcd = PointCloud(mx.array(points))
    pcd.normals = mx.array(normals)
    return pcd


class TestFPFHShape:
    """Test FPFH output shape and dtype."""

    def test_output_shape_sphere(self):
        pcd = _make_sphere_pcd(30)
        param = KDTreeSearchParamHybrid(radius=1.0, max_nn=20)
        features = compute_fpfh_feature(pcd, param)
        assert features.shape == (30, 33)

    def test_output_shape_plane(self):
        pcd = _make_plane_pcd(16)
        param = KDTreeSearchParamHybrid(radius=0.5, max_nn=20)
        features = compute_fpfh_feature(pcd, param)
        assert features.shape == (16, 33)

    def test_output_dtype(self):
        pcd = _make_sphere_pcd(20)
        param = KDTreeSearchParamHybrid(radius=1.0, max_nn=15)
        features = compute_fpfh_feature(pcd, param)
        assert features.dtype == mx.float32


class TestFPFHRequiresNormals:
    """FPFH must raise if normals are missing."""

    def test_no_normals_raises(self):
        pts = np.random.rand(10, 3).astype(np.float32)
        pcd = PointCloud(mx.array(pts))
        param = KDTreeSearchParamHybrid(radius=1.0, max_nn=10)
        with pytest.raises(ValueError, match="normals"):
            compute_fpfh_feature(pcd, param)

    def test_no_radius_raises(self):
        pcd = _make_sphere_pcd(10)
        param = KDTreeSearchParamKNN(knn=5)
        with pytest.raises(ValueError, match="radius"):
            compute_fpfh_feature(pcd, param)


class TestFPFHNonZero:
    """FPFH descriptors should be non-zero for non-degenerate geometry."""

    def test_sphere_nonzero(self):
        pcd = _make_sphere_pcd(40)
        param = KDTreeSearchParamHybrid(radius=1.5, max_nn=30)
        features = compute_fpfh_feature(pcd, param)
        feat_np = np.array(features)
        # At least most descriptors should be non-zero
        nonzero_count = np.sum(np.any(feat_np != 0, axis=1))
        assert nonzero_count > len(feat_np) * 0.5

    def test_plane_nonzero(self):
        pcd = _make_plane_pcd(25)
        param = KDTreeSearchParamHybrid(radius=0.5, max_nn=20)
        features = compute_fpfh_feature(pcd, param)
        feat_np = np.array(features)
        nonzero_count = np.sum(np.any(feat_np != 0, axis=1))
        assert nonzero_count > len(feat_np) * 0.5


class TestFPFHDifferentGeometry:
    """Different geometry should produce different descriptors."""

    def test_sphere_vs_plane(self):
        sphere = _make_sphere_pcd(30)
        plane = _make_plane_pcd(30)
        param = KDTreeSearchParamHybrid(radius=1.0, max_nn=20)

        feat_sphere = np.array(compute_fpfh_feature(sphere, param))
        feat_plane = np.array(compute_fpfh_feature(plane, param))

        # Mean descriptors should differ meaningfully
        mean_sphere = feat_sphere.mean(axis=0)
        mean_plane = feat_plane.mean(axis=0)
        diff = np.linalg.norm(mean_sphere - mean_plane)
        assert diff > 0.01, "Sphere and plane FPFH should differ"
