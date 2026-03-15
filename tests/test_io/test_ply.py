"""Tests for PLY reader and writer."""

from __future__ import annotations

import numpy as np
import pytest

from open3d_mlx.io.ply import read_ply, write_ply
from tests.test_io.conftest import CUBE_COLORS_UINT8, CUBE_NORMALS, CUBE_POINTS


class TestReadPlyAscii:
    """Test reading ASCII PLY files."""

    def test_positions_only(self, cube_ascii_ply):
        data = read_ply(cube_ascii_ply)
        assert "points" in data
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_no_normals_or_colors(self, cube_ascii_ply):
        data = read_ply(cube_ascii_ply)
        assert "normals" not in data
        assert "colors" not in data

    def test_with_normals(self, cube_normals_ply):
        data = read_ply(cube_normals_ply)
        assert "normals" in data
        assert data["normals"].shape == (8, 3)
        np.testing.assert_allclose(data["normals"], CUBE_NORMALS, atol=1e-5)

    def test_with_colors_uint8(self, cube_colors_ply):
        data = read_ply(cube_colors_ply)
        assert "colors" in data
        assert data["colors"].shape == (8, 3)
        assert data["colors"].dtype == np.uint8
        np.testing.assert_array_equal(data["colors"], CUBE_COLORS_UINT8)

    def test_with_colors_float(self, cube_float_colors_ply):
        data = read_ply(cube_float_colors_ply)
        assert "colors" in data
        assert data["colors"].dtype == np.float32
        expected = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
        np.testing.assert_allclose(data["colors"], expected, atol=1e-5)


class TestReadPlyBinary:
    """Test reading binary PLY files."""

    def test_binary_little_endian(self, cube_binary_ply):
        data = read_ply(cube_binary_ply)
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_binary_big_endian(self, cube_big_endian_ply):
        data = read_ply(cube_big_endian_ply)
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)


class TestWritePly:
    """Test writing PLY files and roundtrip."""

    def test_write_ascii_roundtrip(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "out_ascii.ply"
        write_ply(path, data, ascii=True)

        loaded = read_ply(path)
        np.testing.assert_allclose(loaded["points"], CUBE_POINTS, atol=1e-5)

    def test_write_binary_roundtrip(self, data_dir):
        data = {"points": CUBE_POINTS.copy()}
        path = data_dir / "out_binary.ply"
        write_ply(path, data, ascii=False)

        loaded = read_ply(path)
        np.testing.assert_allclose(loaded["points"], CUBE_POINTS, atol=1e-5)

    def test_write_with_normals_roundtrip(self, data_dir):
        data = {"points": CUBE_POINTS.copy(), "normals": CUBE_NORMALS.copy()}
        path = data_dir / "out_normals.ply"
        write_ply(path, data, ascii=True)

        loaded = read_ply(path)
        np.testing.assert_allclose(loaded["normals"], CUBE_NORMALS, atol=1e-5)

    def test_write_with_colors_roundtrip(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "colors": CUBE_COLORS_UINT8.copy(),
        }
        path = data_dir / "out_colors.ply"
        write_ply(path, data, ascii=True)

        loaded = read_ply(path)
        np.testing.assert_array_equal(loaded["colors"], CUBE_COLORS_UINT8)

    def test_write_float_colors_become_uint8(self, data_dir):
        """Float colors in [0,1] should be saved as uint8 and round-trip."""
        colors_float = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
        data = {"points": CUBE_POINTS.copy(), "colors": colors_float}
        path = data_dir / "out_fcolors.ply"
        write_ply(path, data, ascii=True)

        loaded = read_ply(path)
        assert loaded["colors"].dtype == np.uint8
        np.testing.assert_array_equal(loaded["colors"], CUBE_COLORS_UINT8)

    def test_write_binary_with_all_attributes(self, data_dir):
        data = {
            "points": CUBE_POINTS.copy(),
            "normals": CUBE_NORMALS.copy(),
            "colors": CUBE_COLORS_UINT8.copy(),
        }
        path = data_dir / "out_all.ply"
        write_ply(path, data, ascii=False)

        loaded = read_ply(path)
        np.testing.assert_allclose(loaded["points"], CUBE_POINTS, atol=1e-5)
        np.testing.assert_allclose(loaded["normals"], CUBE_NORMALS, atol=1e-5)
        np.testing.assert_array_equal(loaded["colors"], CUBE_COLORS_UINT8)

    def test_preserves_float32_precision(self, data_dir):
        rng = np.random.default_rng(123)
        pts = rng.standard_normal((100, 3)).astype(np.float32)
        data = {"points": pts}
        path = data_dir / "precision.ply"
        write_ply(path, data, ascii=False)

        loaded = read_ply(path)
        np.testing.assert_array_equal(loaded["points"], pts)

    def test_empty_pointcloud(self, data_dir):
        data = {"points": np.empty((0, 3), dtype=np.float32)}
        path = data_dir / "empty.ply"
        write_ply(path, data, ascii=True)

        loaded = read_ply(path)
        assert loaded["points"].shape == (0, 3)


class TestPlyErrors:
    """Test error handling."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_ply("/nonexistent/path.ply")

    def test_not_a_ply(self, data_dir):
        path = data_dir / "bad.ply"
        path.write_text("not a ply file\n")
        with pytest.raises(ValueError, match="Not a PLY"):
            read_ply(path)

    def test_invalid_shape(self, data_dir):
        with pytest.raises(ValueError):
            write_ply(data_dir / "bad.ply", {"points": np.array([1, 2, 3])})
