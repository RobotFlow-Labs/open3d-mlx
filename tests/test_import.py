"""Tests for import completeness and top-level API accessibility.

These tests verify that all public modules and key classes/functions are
accessible from top-level imports, ensuring the package is usable without
digging into internal paths.
"""

from __future__ import annotations

import re


class TestTopLevelImport:
    """Verify ``import open3d_mlx`` and submodule access."""

    def test_import_open3d_mlx(self):
        import open3d_mlx  # noqa: F401

    def test_version_is_string(self):
        import open3d_mlx
        assert isinstance(open3d_mlx.__version__, str)

    def test_version_is_valid_semver(self):
        import open3d_mlx
        assert re.match(r"^\d+\.\d+\.\d+", open3d_mlx.__version__)


class TestCoreAccess:
    """open3d_mlx.core accessible."""

    def test_core_module(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "core")

    def test_core_device(self):
        from open3d_mlx.core import Device
        assert Device is not None


class TestGeometryAccess:
    """open3d_mlx.geometry.PointCloud accessible."""

    def test_geometry_module(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "geometry")

    def test_pointcloud(self):
        from open3d_mlx.geometry import PointCloud
        assert PointCloud is not None

    def test_pointcloud_via_top_level(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.geometry, "PointCloud")


class TestIOAccess:
    """open3d_mlx.io.read_point_cloud accessible."""

    def test_io_module(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "io")

    def test_read_point_cloud(self):
        from open3d_mlx.io import read_point_cloud
        assert callable(read_point_cloud)

    def test_read_point_cloud_via_top_level(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.io, "read_point_cloud")


class TestPipelinesAccess:
    """open3d_mlx.pipelines submodules accessible."""

    def test_pipelines_module(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "pipelines")

    def test_registration_submodule(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.pipelines, "registration")

    def test_registration_icp(self):
        from open3d_mlx.pipelines.registration import registration_icp
        assert callable(registration_icp)

    def test_registration_icp_via_top_level(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.pipelines.registration, "registration_icp")

    def test_integration_submodule(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.pipelines, "integration")

    def test_uniform_tsdf_volume(self):
        from open3d_mlx.pipelines.integration import UniformTSDFVolume
        assert UniformTSDFVolume is not None

    def test_uniform_tsdf_via_top_level(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.pipelines.integration, "UniformTSDFVolume")

    def test_raycasting_submodule(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.pipelines, "raycasting")


class TestCameraAccess:
    """open3d_mlx.camera.PinholeCameraIntrinsic accessible."""

    def test_camera_module(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "camera")

    def test_pinhole_camera_intrinsic(self):
        from open3d_mlx.camera import PinholeCameraIntrinsic
        assert PinholeCameraIntrinsic is not None

    def test_pinhole_via_top_level(self):
        import open3d_mlx
        assert hasattr(open3d_mlx.camera, "PinholeCameraIntrinsic")


class TestInteropAccess:
    """open3d_mlx.interop accessible."""

    def test_interop_module(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "interop")

    def test_interop_functions_exist(self):
        from open3d_mlx.interop import to_open3d, from_open3d, to_open3d_tensor
        assert callable(to_open3d)
        assert callable(from_open3d)
        assert callable(to_open3d_tensor)


class TestDunderAll:
    """All public modules define __all__."""

    def test_top_level_all(self):
        import open3d_mlx
        assert hasattr(open3d_mlx, "__all__")
        assert "core" in open3d_mlx.__all__
        assert "geometry" in open3d_mlx.__all__
        assert "io" in open3d_mlx.__all__
        assert "pipelines" in open3d_mlx.__all__
        assert "camera" in open3d_mlx.__all__

    def test_pipelines_all(self):
        from open3d_mlx import pipelines
        assert hasattr(pipelines, "__all__")
        assert "registration" in pipelines.__all__
        assert "integration" in pipelines.__all__
        assert "raycasting" in pipelines.__all__

    def test_no_private_in_top_level_all(self):
        import open3d_mlx
        for name in open3d_mlx.__all__:
            assert not name.startswith("_") or name == "__version__"
