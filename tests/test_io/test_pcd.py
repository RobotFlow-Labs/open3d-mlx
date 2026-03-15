"""Tests for PCD reader and writer."""

from __future__ import annotations

import numpy as np
import pytest

from open3d_mlx.io.pcd import (
    _pack_pcd_rgb,
    _unpack_pcd_rgb,
    read_pcd,
    write_pcd,
)
from tests.test_io.conftest import CUBE_COLORS_UINT8, CUBE_POINTS


class TestPcdRgbPacking:
    """Test RGB float packing / unpacking."""

    def test_roundtrip(self):
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        packed = _pack_pcd_rgb(colors)
        assert packed.dtype == np.float32
        assert packed.shape == (3,)

        unpacked = _unpack_pcd_rgb(packed)
        np.testing.assert_array_equal(unpacked, colors)

    def test_all_colors(self):
        colors = CUBE_COLORS_UINT8.copy()
        packed = _pack_pcd_rgb(colors)
        unpacked = _unpack_pcd_rgb(packed)
        np.testing.assert_array_equal(unpacked, colors)

    def test_single_channel(self):
        colors = np.array([[128, 0, 0]], dtype=np.uint8)
        packed = _pack_pcd_rgb(colors)
        unpacked = _unpack_pcd_rgb(packed)
        np.testing.assert_array_equal(unpacked, colors)


class TestReadPcdAscii:
    """Test reading ASCII PCD files."""

    def test_positions_only(self, cube_ascii_pcd):
        data = read_pcd(cube_ascii_pcd)
        assert "points" in data
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_no_normals_or_colors(self, cube_ascii_pcd):
        data = read_pcd(cube_ascii_pcd)
        assert "normals" not in data
        assert "colors" not in data


class TestReadPcdBinary:
    """Test reading binary PCD files."""

    def test_positions_only(self, cube_binary_pcd):
        data = read_pcd(cube_binary_pcd)
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_with_rgb(self, cube_rgb_pcd):
        data = read_pcd(cube_rgb_pcd)
        assert "colors" in data
        assert data["colors"].dtype == np.uint8
        assert data["colors"].shape == (8, 3)
        np.testing.assert_array_equal(data["colors"], CUBE_COLORS_UINT8)


class TestWritePcd:
    """Test writing PCD files and roundtrip."""

    def test_write_ascii_roundtrip(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "out.pcd"
        write_pcd(path, data, ascii=True)

        loaded = read_pcd(path)
        np.testing.assert_allclose(loaded["points"], CUBE_POINTS, atol=1e-5)

    def test_write_binary_roundtrip(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "out_bin.pcd"
        write_pcd(path, data, ascii=False)

        loaded = read_pcd(path)
        np.testing.assert_allclose(loaded["points"], CUBE_POINTS, atol=1e-5)

    def test_write_with_colors_roundtrip(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "colors": CUBE_COLORS_UINT8.copy(),
        }
        path = data_dir / "out_rgb.pcd"
        write_pcd(path, data, ascii=False)

        loaded = read_pcd(path)
        np.testing.assert_array_equal(loaded["colors"], CUBE_COLORS_UINT8)

    def test_write_ascii_with_colors_roundtrip(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "colors": CUBE_COLORS_UINT8.copy(),
        }
        path = data_dir / "out_rgb_ascii.pcd"
        write_pcd(path, data, ascii=True)

        loaded = read_pcd(path)
        np.testing.assert_allclose(loaded["points"], CUBE_POINTS, atol=1e-5)
        np.testing.assert_array_equal(loaded["colors"], CUBE_COLORS_UINT8)

    def test_preserves_float32_precision(self, data_dir):
        rng = np.random.default_rng(456)
        pts = rng.standard_normal((50, 3)).astype(np.float32)
        data = {"points": pts}
        path = data_dir / "precision.pcd"
        write_pcd(path, data, ascii=False)

        loaded = read_pcd(path)
        np.testing.assert_array_equal(loaded["points"], pts)

    def test_empty_pointcloud(self, data_dir):
        data = {"points": np.empty((0, 3), dtype=np.float32)}
        path = data_dir / "empty.pcd"
        write_pcd(path, data, ascii=True)

        loaded = read_pcd(path)
        assert loaded["points"].shape == (0, 3)


class TestPcdErrors:
    """Test error handling."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_pcd("/nonexistent/path.pcd")

    def test_invalid_shape(self, data_dir):
        with pytest.raises(ValueError):
            write_pcd(data_dir / "bad.pcd", {"points": np.array([1, 2, 3])})
