"""Tests for correspondence checkers."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.pipelines.registration import (
    CorrespondenceCheckerBasedOnDistance,
    CorrespondenceCheckerBasedOnEdgeLength,
    CorrespondenceCheckerBasedOnNormal,
)


# ---------------------------------------------------------------------------
# CorrespondenceCheckerBasedOnDistance
# ---------------------------------------------------------------------------


class TestDistanceChecker:
    """Tests for CorrespondenceCheckerBasedOnDistance."""

    def test_invalid_threshold_raises(self):
        """Negative threshold should raise ValueError."""
        with pytest.raises(ValueError, match="distance_threshold"):
            CorrespondenceCheckerBasedOnDistance(-1.0)

    def test_zero_threshold_raises(self):
        """Zero threshold should raise ValueError."""
        with pytest.raises(ValueError, match="distance_threshold"):
            CorrespondenceCheckerBasedOnDistance(0.0)

    def test_all_within_threshold(self):
        """All correspondences within threshold should be valid."""
        checker = CorrespondenceCheckerBasedOnDistance(1.0)

        source = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=mx.float32)
        target = mx.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=mx.float32)
        corr = mx.array([0, 1], dtype=mx.int32)
        # Squared distances: 0.01, 0.01
        sq_dists = mx.array([0.01, 0.01], dtype=mx.float32)

        valid = checker.check(source, target, corr, sq_dists)
        assert valid.all()

    def test_filters_far_correspondences(self):
        """Correspondences beyond threshold should be filtered out."""
        checker = CorrespondenceCheckerBasedOnDistance(0.5)

        source = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=mx.float32,
        )
        target = mx.array(
            [[0.1, 0.0, 0.0], [5.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
            dtype=mx.float32,
        )
        corr = mx.array([0, 1, 2], dtype=mx.int32)
        # Squared distances: 0.01, 16.0, 0.01
        sq_dists = mx.array([0.01, 16.0, 0.01], dtype=mx.float32)

        valid = checker.check(source, target, corr, sq_dists)
        assert valid[0] == True
        assert valid[1] == False  # 4.0 > 0.5
        assert valid[2] == True

    def test_no_match_entries(self):
        """Entries with correspondences = -1 should be invalid."""
        checker = CorrespondenceCheckerBasedOnDistance(1.0)

        source = mx.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=mx.float32)
        target = mx.array([[0.1, 0.0, 0.0]], dtype=mx.float32)
        corr = mx.array([0, -1], dtype=mx.int32)
        sq_dists = mx.array([0.01, float("inf")], dtype=mx.float32)

        valid = checker.check(source, target, corr, sq_dists)
        assert valid[0] == True
        assert valid[1] == False


# ---------------------------------------------------------------------------
# CorrespondenceCheckerBasedOnEdgeLength
# ---------------------------------------------------------------------------


class TestEdgeLengthChecker:
    """Tests for CorrespondenceCheckerBasedOnEdgeLength."""

    def test_invalid_threshold_raises(self):
        """Threshold outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="similarity_threshold"):
            CorrespondenceCheckerBasedOnEdgeLength(0.0)
        with pytest.raises(ValueError, match="similarity_threshold"):
            CorrespondenceCheckerBasedOnEdgeLength(1.5)

    def test_consistent_edges_pass(self):
        """Consistent edge lengths should all pass."""
        checker = CorrespondenceCheckerBasedOnEdgeLength(0.9)

        # Source and target have same point structure
        source = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=mx.float32,
        )
        target = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=mx.float32,
        )
        corr = mx.array([0, 1, 2, 3], dtype=mx.int32)

        valid = checker.check(source, target, corr)
        # All should pass since edge lengths are identical
        assert valid.all()

    def test_inconsistent_edges_filtered(self):
        """Inconsistent edge lengths should be filtered."""
        checker = CorrespondenceCheckerBasedOnEdgeLength(0.9)

        # Source points are evenly spaced
        source = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=mx.float32,
        )
        # Target: first two close, third very far -> edge length mismatch
        target = mx.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=mx.float32,
        )
        corr = mx.array([0, 1, 2], dtype=mx.int32)

        valid = checker.check(source, target, corr)
        # At least one should be marked invalid due to edge mismatch
        assert not valid.all()

    def test_no_match_handled(self):
        """Entries with -1 correspondences should remain invalid."""
        checker = CorrespondenceCheckerBasedOnEdgeLength(0.9)

        source = mx.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=mx.float32
        )
        target = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)
        corr = mx.array([0, -1], dtype=mx.int32)

        valid = checker.check(source, target, corr)
        assert valid[1] == False


# ---------------------------------------------------------------------------
# CorrespondenceCheckerBasedOnNormal
# ---------------------------------------------------------------------------


class TestNormalChecker:
    """Tests for CorrespondenceCheckerBasedOnNormal."""

    def test_invalid_threshold_raises(self):
        """Negative angle threshold should raise ValueError."""
        with pytest.raises(ValueError, match="normal_angle_threshold"):
            CorrespondenceCheckerBasedOnNormal(-0.1)

    def test_aligned_normals_pass(self):
        """Perfectly aligned normals should pass."""
        checker = CorrespondenceCheckerBasedOnNormal(math.pi / 4)  # 45 degrees

        src_normals = mx.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32
        )
        tgt_normals = mx.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32
        )
        corr = mx.array([0, 1], dtype=mx.int32)

        valid = checker.check(src_normals, tgt_normals, corr)
        assert valid.all()

    def test_perpendicular_normals_filtered(self):
        """Perpendicular normals (90 degrees) should be filtered with 45-degree threshold."""
        checker = CorrespondenceCheckerBasedOnNormal(math.pi / 4)  # 45 degrees

        src_normals = mx.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=mx.float32
        )
        tgt_normals = mx.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=mx.float32
        )
        corr = mx.array([0, 1], dtype=mx.int32)

        valid = checker.check(src_normals, tgt_normals, corr)
        assert valid[0] == True  # Aligned
        assert valid[1] == False  # Perpendicular

    def test_opposite_normals_pass(self):
        """Opposite normals should pass (direction-agnostic check)."""
        checker = CorrespondenceCheckerBasedOnNormal(math.pi / 4)  # 45 degrees

        src_normals = mx.array([[0.0, 0.0, 1.0]], dtype=mx.float32)
        tgt_normals = mx.array([[0.0, 0.0, -1.0]], dtype=mx.float32)
        corr = mx.array([0], dtype=mx.int32)

        valid = checker.check(src_normals, tgt_normals, corr)
        # Should pass because we use abs(dot) to be direction-agnostic
        assert valid[0] == True

    def test_no_match_entries(self):
        """Entries with -1 correspondences should be invalid."""
        checker = CorrespondenceCheckerBasedOnNormal(math.pi / 2)

        src_normals = mx.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=mx.float32
        )
        tgt_normals = mx.array([[0.0, 0.0, 1.0]], dtype=mx.float32)
        corr = mx.array([0, -1], dtype=mx.int32)

        valid = checker.check(src_normals, tgt_normals, corr)
        assert valid[0] == True
        assert valid[1] == False

    def test_wide_threshold_passes_all(self):
        """Very wide threshold (pi) should pass all valid correspondences."""
        checker = CorrespondenceCheckerBasedOnNormal(math.pi)

        rng = np.random.default_rng(42)
        N = 20
        src_n = rng.standard_normal((N, 3)).astype(np.float32)
        src_n /= np.linalg.norm(src_n, axis=1, keepdims=True)
        tgt_n = rng.standard_normal((N, 3)).astype(np.float32)
        tgt_n /= np.linalg.norm(tgt_n, axis=1, keepdims=True)

        corr = mx.arange(N, dtype=mx.int32)
        valid = checker.check(mx.array(src_n), mx.array(tgt_n), corr)
        assert valid.all()
