"""Tests for PTS point cloud I/O."""

from __future__ import annotations

import numpy as np
import pytest

from open3d_mlx.io.pts import read_pts, write_pts
from tests.test_io.conftest import CUBE_COLORS_UINT8, CUBE_POINTS


class TestReadPTS:
    """Test reading PTS files."""

    def test_read_points_only(self, data_dir):
        path = data_dir / "points.pts"
        with open(path, "w") as f:
            f.write(f"{len(CUBE_POINTS)}\n")
            for p in CUBE_POINTS:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        data = read_pts(str(path))
        assert "points" in data
        assert data["points"].shape == (8, 3)
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_read_with_colors(self, data_dir):
        path = data_dir / "colors.pts"
        with open(path, "w") as f:
            f.write(f"{len(CUBE_POINTS)}\n")
            for p, c in zip(CUBE_POINTS, CUBE_COLORS_UINT8):
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0 {c[0]} {c[1]} {c[2]}\n"
                )

        data = read_pts(str(path))
        assert "colors" in data
        assert data["colors"].shape == (8, 3)
        assert data["colors"].max() <= 1.0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_pts("/nonexistent/path.pts")


class TestWritePTS:
    """Test writing PTS files."""

    def test_write_points_only(self, data_dir):
        path = data_dir / "out.pts"
        write_pts(str(path), {"points": CUBE_POINTS})
        assert path.exists()

        data = read_pts(str(path))
        np.testing.assert_allclose(data["points"], CUBE_POINTS, atol=1e-5)

    def test_write_with_colors(self, data_dir):
        path = data_dir / "out_c.pts"
        colors_float = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
        write_pts(str(path), {"points": CUBE_POINTS, "colors": colors_float})

        data = read_pts(str(path))
        assert "colors" in data
        np.testing.assert_allclose(
            data["colors"],
            colors_float,
            atol=0.01,  # uint8 roundtrip tolerance
        )


class TestRoundtripPTS:
    """Test write -> read roundtrip."""

    def test_roundtrip_points(self, data_dir):
        path = data_dir / "rt.pts"
        original = {"points": CUBE_POINTS.copy()}
        write_pts(str(path), original)
        loaded = read_pts(str(path))
        np.testing.assert_allclose(loaded["points"], original["points"], atol=1e-5)

    def test_roundtrip_with_colors(self, data_dir):
        path = data_dir / "rt_c.pts"
        colors_float = CUBE_COLORS_UINT8.astype(np.float32) / 255.0
        original = {"points": CUBE_POINTS.copy(), "colors": colors_float}
        write_pts(str(path), original)
        loaded = read_pts(str(path))
        np.testing.assert_allclose(loaded["points"], original["points"], atol=1e-5)
        np.testing.assert_allclose(loaded["colors"], original["colors"], atol=0.01)
