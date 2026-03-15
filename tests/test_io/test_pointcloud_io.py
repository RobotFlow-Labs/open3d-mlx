"""Tests for the top-level point cloud I/O dispatcher."""

from __future__ import annotations

import numpy as np
import pytest

from open3d_mlx.io.pointcloud_io import read_point_cloud, write_point_cloud
from tests.test_io.conftest import CUBE_COLORS_UINT8, CUBE_NORMALS, CUBE_POINTS


class TestAutoDetect:
    """Test format auto-detection."""

    def test_detect_ply(self, cube_ascii_ply):
        result = read_point_cloud(cube_ascii_ply)
        # Result is either a PointCloud or a dict
        if isinstance(result, dict):
            pts = result["points"]
        else:
            pts = np.array(result.points)
        assert pts.shape == (8, 3)

    def test_detect_pcd(self, cube_ascii_pcd):
        result = read_point_cloud(cube_ascii_pcd)
        if isinstance(result, dict):
            pts = result["points"]
        else:
            pts = np.array(result.points)
        assert pts.shape == (8, 3)

    def test_unsupported_extension(self, data_dir):
        path = data_dir / "test.xyz"
        path.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported"):
            read_point_cloud(str(path))

    def test_explicit_format_ply(self, cube_ascii_ply):
        result = read_point_cloud(str(cube_ascii_ply), format="ply")
        if isinstance(result, dict):
            assert result["points"].shape == (8, 3)
        else:
            assert np.array(result.points).shape == (8, 3)


class TestRemoveNonFinite:
    """Test NaN / Inf removal."""

    def test_remove_nan(self, data_dir):
        pts = CUBE_POINTS.copy()
        pts[0] = [np.nan, 0.0, 0.0]
        pts[3] = [0.0, np.nan, 0.0]

        from open3d_mlx.io.ply import write_ply
        path = data_dir / "nan.ply"
        write_ply(path, {"points": pts}, ascii=True)

        result = read_point_cloud(str(path), remove_nan_points=True)
        if isinstance(result, dict):
            assert result["points"].shape == (6, 3)
        else:
            assert np.array(result.points).shape == (6, 3)

    def test_remove_inf(self, data_dir):
        pts = CUBE_POINTS.copy()
        pts[2] = [np.inf, 0.0, 0.0]

        from open3d_mlx.io.ply import write_ply
        path = data_dir / "inf.ply"
        write_ply(path, {"points": pts}, ascii=True)

        result = read_point_cloud(str(path), remove_infinite_points=True)
        if isinstance(result, dict):
            assert result["points"].shape == (7, 3)
        else:
            assert np.array(result.points).shape == (7, 3)

    def test_remove_nan_preserves_clean(self, cube_ascii_ply):
        result = read_point_cloud(str(cube_ascii_ply), remove_nan_points=True)
        if isinstance(result, dict):
            assert result["points"].shape == (8, 3)
        else:
            assert np.array(result.points).shape == (8, 3)


class TestRoundtrip:
    """Test full write → read roundtrip via dispatcher."""

    def test_roundtrip_ply(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "normals": CUBE_NORMALS.copy(),
            "colors": CUBE_COLORS_UINT8.astype(np.float32) / 255.0,
        }
        path = data_dir / "roundtrip.ply"
        write_point_cloud(str(path), data, write_ascii=True)

        result = read_point_cloud(str(path))
        if isinstance(result, dict):
            np.testing.assert_allclose(result["points"], CUBE_POINTS, atol=1e-5)
        else:
            np.testing.assert_allclose(
                np.array(result.points), CUBE_POINTS, atol=1e-5
            )

    def test_roundtrip_pcd(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "roundtrip.pcd"
        write_point_cloud(str(path), data, write_ascii=False)

        result = read_point_cloud(str(path))
        if isinstance(result, dict):
            np.testing.assert_allclose(result["points"], CUBE_POINTS, atol=1e-5)
        else:
            np.testing.assert_allclose(
                np.array(result.points), CUBE_POINTS, atol=1e-5
            )

    def test_roundtrip_preserves_all_ply(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "normals": CUBE_NORMALS.copy(),
            "colors": CUBE_COLORS_UINT8.copy(),
        }
        path = data_dir / "all_attrs.ply"
        write_point_cloud(str(path), data, write_ascii=False)

        result = read_point_cloud(str(path))
        if isinstance(result, dict):
            np.testing.assert_allclose(result["points"], CUBE_POINTS, atol=1e-5)
            np.testing.assert_allclose(result["normals"], CUBE_NORMALS, atol=1e-5)
            # Colors come back as uint8 from PLY reader, then normalized to float
            colors = result.get("colors")
            if colors is not None:
                if colors.dtype == np.float32:
                    expected = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
                    np.testing.assert_allclose(colors, expected, atol=1e-2)
                else:
                    np.testing.assert_array_equal(colors, CUBE_COLORS_UINT8)

    def test_roundtrip_preserves_all_pcd(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "colors": CUBE_COLORS_UINT8.copy(),
        }
        path = data_dir / "all_attrs.pcd"
        write_point_cloud(str(path), data, write_ascii=False)

        result = read_point_cloud(str(path))
        if isinstance(result, dict):
            np.testing.assert_allclose(result["points"], CUBE_POINTS, atol=1e-5)
            colors = result.get("colors")
            if colors is not None:
                if colors.dtype == np.float32:
                    expected = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
                    np.testing.assert_allclose(colors, expected, atol=1e-2)
                else:
                    np.testing.assert_array_equal(colors, CUBE_COLORS_UINT8)

    def test_empty_roundtrip_ply(self, data_dir):
        data = {"points": np.empty((0, 3), dtype=np.float32)}
        path = data_dir / "empty.ply"
        write_point_cloud(str(path), data)
        result = read_point_cloud(str(path))
        if isinstance(result, dict):
            assert result["points"].shape == (0, 3)
        else:
            assert np.array(result.points).shape == (0, 3)


class TestWritePointCloud:
    """Test write_point_cloud return value and format detection."""

    def test_returns_true(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "ok.ply"
        assert write_point_cloud(str(path), data) is True

    def test_unsupported_format(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "bad.xyz"
        with pytest.raises(ValueError, match="Unsupported"):
            write_point_cloud(str(path), data)
