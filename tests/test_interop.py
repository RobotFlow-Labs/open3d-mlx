"""Tests for Open3D interop conversions.

Tests are skipped when the ``open3d`` package is not installed.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.geometry import PointCloud

# Check once whether open3d is available
_has_open3d = True
try:
    import open3d  # noqa: F401
except ImportError:
    _has_open3d = False

skip_no_open3d = pytest.mark.skipif(
    not _has_open3d, reason="open3d not installed"
)


# ------------------------------------------------------------------
# Tests that require open3d
# ------------------------------------------------------------------

@skip_no_open3d
class TestToOpen3D:
    """to_open3d preserves geometry attributes."""

    def test_preserves_points(self):
        from open3d_mlx.interop import to_open3d

        pts = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pcd = PointCloud(pts)
        o3d_pcd = to_open3d(pcd)
        result = np.asarray(o3d_pcd.points)
        np.testing.assert_allclose(result, np.array(pts), atol=1e-5)

    def test_preserves_normals(self):
        from open3d_mlx.interop import to_open3d

        pts = mx.array([[1.0, 0.0, 0.0]])
        norms = mx.array([[0.0, 0.0, 1.0]])
        pcd = PointCloud(pts)
        pcd.normals = norms
        o3d_pcd = to_open3d(pcd)
        result = np.asarray(o3d_pcd.normals)
        np.testing.assert_allclose(result, np.array(norms), atol=1e-5)

    def test_preserves_colors(self):
        from open3d_mlx.interop import to_open3d

        pts = mx.array([[1.0, 0.0, 0.0]])
        colors = mx.array([[0.5, 0.5, 0.5]])
        pcd = PointCloud(pts)
        pcd.colors = colors
        o3d_pcd = to_open3d(pcd)
        result = np.asarray(o3d_pcd.colors)
        np.testing.assert_allclose(result, np.array(colors), atol=1e-5)


@skip_no_open3d
class TestFromOpen3D:
    """from_open3d preserves geometry attributes."""

    def test_preserves_points(self):
        import open3d as o3d
        from open3d_mlx.interop import from_open3d

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )
        pcd = from_open3d(o3d_pcd)
        np.testing.assert_allclose(
            np.array(pcd.points), np.array([[1, 2, 3], [4, 5, 6]]),
            atol=1e-5,
        )

    def test_preserves_normals(self):
        import open3d as o3d
        from open3d_mlx.interop import from_open3d

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 0.0]]))
        o3d_pcd.normals = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 0.0]]))
        pcd = from_open3d(o3d_pcd)
        assert pcd.has_normals()
        np.testing.assert_allclose(
            np.array(pcd.normals), np.array([[0, 1, 0]]), atol=1e-5
        )


@skip_no_open3d
class TestRoundtrip:
    """Roundtrip conversion preserves data."""

    def test_roundtrip_points_and_normals(self):
        from open3d_mlx.interop import from_open3d, to_open3d

        pts = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        norms = mx.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        pcd = PointCloud(pts)
        pcd.normals = norms

        pcd2 = from_open3d(to_open3d(pcd))
        np.testing.assert_allclose(np.array(pcd2.points), np.array(pts), atol=1e-5)
        np.testing.assert_allclose(np.array(pcd2.normals), np.array(norms), atol=1e-5)


@skip_no_open3d
class TestToOpen3DTensor:
    """to_open3d_tensor produces a tensor-based point cloud."""

    def test_basic(self):
        from open3d_mlx.interop import to_open3d_tensor

        pts = mx.array([[1.0, 2.0, 3.0]])
        pcd = PointCloud(pts)
        o3d_pcd = to_open3d_tensor(pcd)
        result = o3d_pcd.point.positions.numpy()
        np.testing.assert_allclose(result, np.array(pts), atol=1e-5)


# ------------------------------------------------------------------
# Tests that do NOT require open3d
# ------------------------------------------------------------------

class TestImportError:
    """ImportError raised when open3d is missing."""

    def test_to_open3d_raises(self):
        """to_open3d raises ImportError when open3d is absent."""
        with mock.patch.dict(sys.modules, {"open3d": None}):
            # Force reimport of interop to pick up the mocked module
            import open3d_mlx.interop as interop_mod
            # Manually simulate the import failure by calling with patched import
            pts = mx.array([[1.0, 2.0, 3.0]])
            pcd = PointCloud(pts)

            # The function does `import open3d as o3d` inside -- patch builtins
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "open3d":
                    raise ImportError("No module named 'open3d'")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="open3d"):
                    interop_mod.to_open3d(pcd)

    def test_to_open3d_tensor_raises(self):
        """to_open3d_tensor raises ImportError when open3d is absent."""
        with mock.patch.dict(sys.modules, {"open3d": None}):
            import open3d_mlx.interop as interop_mod
            pts = mx.array([[1.0, 2.0, 3.0]])
            pcd = PointCloud(pts)

            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "open3d":
                    raise ImportError("No module named 'open3d'")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="open3d"):
                    interop_mod.to_open3d_tensor(pcd)
