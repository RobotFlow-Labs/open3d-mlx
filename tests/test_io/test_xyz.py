"""Tests for XYZ point cloud I/O."""

from __future__ import annotations

import numpy as np
import pytest

from open3d_mlx.io.xyz import read_xyz, write_xyz
from tests.test_io.conftest import CUBE_COLORS_UINT8, CUBE_NORMALS, CUBE_POINTS


class TestReadXYZ:
    """Test reading XYZ files."""

    def test_read_points_only(self, data_dir):
        path = data_dir / "points.xyz"
        np.savetxt(path, CUBE_POINTS, fmt="%.6f")

        data = read_xyz(str(path))
        assert "points" in data
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_read_with_normals(self, data_dir):
        path = data_dir / "normals.xyz"
        arr = np.hstack([CUBE_POINTS, CUBE_NORMALS])
        np.savetxt(path, arr, fmt="%.6f")

        data = read_xyz(str(path))
        assert "points" in data
        assert "normals" in data
        assert data["normals"].shape == (8, 3)
        np.testing.assert_allclose(data["normals"], CUBE_NORMALS, atol=1e-5)

    def test_read_with_colors(self, data_dir):
        path = data_dir / "colors.xyz"
        colors_float = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
        arr = np.hstack([CUBE_POINTS, CUBE_NORMALS, CUBE_COLORS_UINT8.astype(np.float32)])
        np.savetxt(path, arr, fmt="%.6f")

        data = read_xyz(str(path))
        assert "colors" in data
        assert data["colors"].shape == (8, 3)
        # Colors > 1.0 should be normalized
        assert data["colors"].max() <= 1.0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_xyz("/nonexistent/path.xyz")


class TestWriteXYZ:
    """Test writing XYZ files."""

    def test_write_points_only(self, data_dir):
        path = data_dir / "out.xyz"
        write_xyz(str(path), {"points": CUBE_POINTS})
        assert path.exists()

        data = read_xyz(str(path))
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)
        assert "normals" not in data

    def test_write_with_normals(self, data_dir):
        path = data_dir / "out_n.xyz"
        write_xyz(str(path), {"points": CUBE_POINTS, "normals": CUBE_NORMALS})

        data = read_xyz(str(path))
        assert "normals" in data
        np.testing.assert_allclose(data["normals"], CUBE_NORMALS, atol=1e-5)

    def test_write_exclude_normals(self, data_dir):
        path = data_dir / "no_n.xyz"
        write_xyz(
            str(path),
            {"points": CUBE_POINTS, "normals": CUBE_NORMALS},
            include_normals=False,
        )
        data = read_xyz(str(path))
        assert "normals" not in data


class TestRoundtripXYZ:
    """Test write -> read roundtrip."""

    def test_roundtrip_points(self, data_dir):
        path = data_dir / "rt.xyz"
        original = {"points": CUBE_POINTS.copy()}
        write_xyz(str(path), original)
        loaded = read_xyz(str(path))
        np.testing.assert_allclose(loaded["points"], original["points"], atol=1e-5)

    def test_roundtrip_full(self, data_dir):
        path = data_dir / "rt_full.xyz"
        colors_float = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
        original = {
            "points": CUBE_POINTS.copy(),
            "normals": CUBE_NORMALS.copy(),
            "colors": colors_float,
        }
        write_xyz(str(path), original)
        loaded = read_xyz(str(path))
        np.testing.assert_allclose(loaded["points"], original["points"], atol=1e-5)
        np.testing.assert_allclose(loaded["normals"], original["normals"], atol=1e-5)
        np.testing.assert_allclose(loaded["colors"], original["colors"], atol=1e-2)
