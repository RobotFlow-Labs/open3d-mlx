"""Comprehensive tests for robust loss kernels.

Tests cover:
- Correctness of weight values at known inputs
- Monotonicity / symmetry properties
- Edge cases: zero, large, negative, empty, single-element inputs
- Shape preservation
- Value bounds
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from open3d_mlx.pipelines.registration.robust_kernel import (
    CauchyLoss,
    GMLoss,
    HuberLoss,
    L2Loss,
    RobustKernel,
    TukeyLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_np(a: mx.array) -> np.ndarray:
    """Convert MLX array to numpy, forcing evaluation."""
    mx.eval(a)
    return np.array(a)


ALL_KERNELS = [L2Loss, HuberLoss, TukeyLoss, CauchyLoss, GMLoss]


# ===========================================================================
# L2Loss
# ===========================================================================

class TestL2Loss:
    def test_weight_always_one(self):
        k = L2Loss()
        r = mx.array([0.0, 0.5, 1.0, 10.0, -3.0, 100.0])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w, 1.0)

    def test_weight_zero_input(self):
        w = _to_np(L2Loss().weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0)

    def test_weight_large_input(self):
        w = _to_np(L2Loss().weight(mx.array([1e6])))
        np.testing.assert_allclose(w, 1.0)

    def test_weight_negative_input(self):
        w = _to_np(L2Loss().weight(mx.array([-5.0, -0.1])))
        np.testing.assert_allclose(w, 1.0)

    def test_shape_preserved(self):
        r = mx.zeros((3, 4))
        w = L2Loss().weight(r)
        assert w.shape == (3, 4)

    def test_empty_array(self):
        r = mx.array([]).reshape(0)
        w = L2Loss().weight(r)
        assert w.shape == (0,)


# ===========================================================================
# HuberLoss
# ===========================================================================

class TestHuberLoss:
    def test_weight_one_inside_threshold(self):
        k = HuberLoss(k=2.0)
        r = mx.array([0.0, 0.5, 1.0, 1.99, -1.5])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w, 1.0, atol=1e-6)

    def test_weight_at_threshold(self):
        k = HuberLoss(k=2.0)
        w = _to_np(k.weight(mx.array([2.0, -2.0])))
        np.testing.assert_allclose(w, 1.0, atol=1e-6)

    def test_weight_outside_threshold(self):
        k = HuberLoss(k=2.0)
        r = mx.array([4.0, -4.0])
        w = _to_np(k.weight(r))
        expected = 2.0 / 4.0  # k / |r|
        np.testing.assert_allclose(w, expected, atol=1e-6)

    def test_monotonically_decreasing_outside(self):
        k = HuberLoss(k=1.0)
        residuals = mx.array([1.0, 2.0, 3.0, 5.0, 10.0])
        w = _to_np(k.weight(residuals))
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1], f"w[{i}]={w[i]} < w[{i+1}]={w[i+1]}"

    def test_symmetric(self):
        k = HuberLoss(k=1.5)
        r = mx.array([3.0, -3.0, 0.5, -0.5])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w[0], w[1], atol=1e-6)
        np.testing.assert_allclose(w[2], w[3], atol=1e-6)

    def test_weight_zero_input(self):
        w = _to_np(HuberLoss().weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0)

    def test_weight_large_input(self):
        w = _to_np(HuberLoss(k=1.0).weight(mx.array([1e6])))
        assert w[0] > 0
        assert w[0] < 1e-3

    def test_weight_in_unit_range(self):
        k = HuberLoss(k=1.0)
        r = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        w = _to_np(k.weight(r))
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_shape_preserved(self):
        r = mx.ones((5, 3))
        w = HuberLoss().weight(r)
        assert w.shape == (5, 3)

    def test_empty_array(self):
        w = HuberLoss().weight(mx.array([]).reshape(0))
        assert w.shape == (0,)

    def test_single_element(self):
        w = _to_np(HuberLoss(k=1.0).weight(mx.array([0.5])))
        np.testing.assert_allclose(w, 1.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            HuberLoss(k=0.0)
        with pytest.raises(ValueError):
            HuberLoss(k=-1.0)


# ===========================================================================
# TukeyLoss
# ===========================================================================

class TestTukeyLoss:
    def test_weight_positive_inside(self):
        k = TukeyLoss(k=5.0)
        r = mx.array([0.0, 1.0, 2.0, 3.0, 4.0, 4.99])
        w = _to_np(k.weight(r))
        assert np.all(w > 0), f"Expected all positive, got {w}"

    def test_weight_zero_outside(self):
        k = TukeyLoss(k=5.0)
        r = mx.array([5.01, 10.0, 100.0, -6.0])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w, 0.0, atol=1e-6)

    def test_maximum_at_zero(self):
        k = TukeyLoss(k=4.0)
        w = _to_np(k.weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0, atol=1e-6)

    def test_weight_at_threshold(self):
        k = TukeyLoss(k=4.0)
        w = _to_np(k.weight(mx.array([4.0, -4.0])))
        np.testing.assert_allclose(w, 0.0, atol=1e-6)

    def test_symmetric(self):
        k = TukeyLoss(k=3.0)
        r = mx.array([1.5, -1.5, 2.9, -2.9])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w[0], w[1], atol=1e-6)
        np.testing.assert_allclose(w[2], w[3], atol=1e-6)

    def test_weight_decreasing_inside(self):
        k = TukeyLoss(k=5.0)
        r = mx.array([0.0, 1.0, 2.0, 3.0, 4.0])
        w = _to_np(k.weight(r))
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1], f"w[{i}]={w[i]} < w[{i+1}]={w[i+1]}"

    def test_weight_in_unit_range(self):
        k = TukeyLoss(k=4.0)
        r = mx.array([-5.0, -2.0, 0.0, 2.0, 5.0])
        w = _to_np(k.weight(r))
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_shape_preserved(self):
        r = mx.ones((2, 3, 4))
        w = TukeyLoss().weight(r)
        assert w.shape == (2, 3, 4)

    def test_empty_array(self):
        w = TukeyLoss().weight(mx.array([]).reshape(0))
        assert w.shape == (0,)

    def test_single_element(self):
        w = _to_np(TukeyLoss(k=5.0).weight(mx.array([2.0])))
        expected = (1.0 - (2.0 / 5.0) ** 2) ** 2
        np.testing.assert_allclose(w, expected, atol=1e-6)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            TukeyLoss(k=0.0)


# ===========================================================================
# CauchyLoss
# ===========================================================================

class TestCauchyLoss:
    def test_weight_one_at_zero(self):
        w = _to_np(CauchyLoss(k=1.0).weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0)

    def test_weight_decreasing(self):
        k = CauchyLoss(k=1.0)
        r = mx.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        w = _to_np(k.weight(r))
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1], f"w[{i}]={w[i]} < w[{i+1}]={w[i+1]}"

    def test_weight_never_zero(self):
        k = CauchyLoss(k=1.0)
        r = mx.array([100.0, 1000.0, 1e6])
        w = _to_np(k.weight(r))
        assert np.all(w > 0)

    def test_symmetric(self):
        k = CauchyLoss(k=2.0)
        r = mx.array([3.0, -3.0])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w[0], w[1], atol=1e-6)

    def test_known_value(self):
        k = CauchyLoss(k=2.0)
        r = mx.array([2.0])  # 1 / (1 + (2/2)^2) = 1/2
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w, 0.5, atol=1e-6)

    def test_weight_in_unit_range(self):
        k = CauchyLoss(k=1.0)
        r = mx.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        w = _to_np(k.weight(r))
        assert np.all(w > 0.0)
        assert np.all(w <= 1.0)

    def test_shape_preserved(self):
        r = mx.ones((4, 2))
        w = CauchyLoss().weight(r)
        assert w.shape == (4, 2)

    def test_empty_array(self):
        w = CauchyLoss().weight(mx.array([]).reshape(0))
        assert w.shape == (0,)

    def test_single_element(self):
        w = _to_np(CauchyLoss(k=1.0).weight(mx.array([1.0])))
        np.testing.assert_allclose(w, 0.5, atol=1e-6)

    def test_large_input(self):
        w = _to_np(CauchyLoss(k=1.0).weight(mx.array([1e4])))
        assert w[0] > 0
        assert w[0] < 1e-6

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            CauchyLoss(k=-0.5)


# ===========================================================================
# GMLoss
# ===========================================================================

class TestGMLoss:
    def test_weight_one_at_zero(self):
        w = _to_np(GMLoss(k=1.0).weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0)

    def test_weight_decreasing(self):
        k = GMLoss(k=1.0)
        r = mx.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
        w = _to_np(k.weight(r))
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1], f"w[{i}]={w[i]} < w[{i+1}]={w[i+1]}"

    def test_weight_never_zero(self):
        k = GMLoss(k=1.0)
        r = mx.array([100.0, 1000.0])
        w = _to_np(k.weight(r))
        assert np.all(w > 0)

    def test_symmetric(self):
        k = GMLoss(k=2.0)
        r = mx.array([3.0, -3.0])
        w = _to_np(k.weight(r))
        np.testing.assert_allclose(w[0], w[1], atol=1e-6)

    def test_known_value(self):
        # k=1, r=1 => (1/(1+1))^2 = 0.25
        k = GMLoss(k=1.0)
        w = _to_np(k.weight(mx.array([1.0])))
        np.testing.assert_allclose(w, 0.25, atol=1e-6)

    def test_weight_in_unit_range(self):
        k = GMLoss(k=1.0)
        r = mx.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        w = _to_np(k.weight(r))
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0)

    def test_decreases_faster_than_cauchy(self):
        """GM suppresses outliers more aggressively than Cauchy."""
        r = mx.array([3.0])
        gm_w = _to_np(GMLoss(k=1.0).weight(r))[0]
        cauchy_w = _to_np(CauchyLoss(k=1.0).weight(r))[0]
        assert gm_w < cauchy_w

    def test_shape_preserved(self):
        r = mx.ones((3,))
        w = GMLoss().weight(r)
        assert w.shape == (3,)

    def test_empty_array(self):
        w = GMLoss().weight(mx.array([]).reshape(0))
        assert w.shape == (0,)

    def test_single_element(self):
        w = _to_np(GMLoss(k=2.0).weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0)

    def test_large_input(self):
        w = _to_np(GMLoss(k=1.0).weight(mx.array([1e4])))
        assert w[0] > 0
        assert w[0] < 1e-10

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            GMLoss(k=0.0)


# ===========================================================================
# Cross-kernel tests
# ===========================================================================

class TestAllKernels:
    """Properties that should hold for every kernel."""

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_zero_input_weight_is_one(self, KernelClass):
        k = KernelClass()
        w = _to_np(k.weight(mx.array([0.0])))
        np.testing.assert_allclose(w, 1.0, atol=1e-6)

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_output_shape_matches_input(self, KernelClass):
        k = KernelClass()
        for shape in [(5,), (3, 4), (2, 3, 2)]:
            r = mx.ones(shape)
            w = k.weight(r)
            assert w.shape == shape, f"{KernelClass.__name__}: {w.shape} != {shape}"

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_handles_negative_input(self, KernelClass):
        k = KernelClass()
        r = mx.array([-1.0, -0.5, -0.01])
        w = _to_np(k.weight(r))
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_handles_large_input(self, KernelClass):
        k = KernelClass()
        r = mx.array([1e4, -1e4])
        w = _to_np(k.weight(r))
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_empty_array(self, KernelClass):
        k = KernelClass()
        r = mx.array([]).reshape(0)
        w = k.weight(r)
        assert w.shape == (0,)

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_single_element(self, KernelClass):
        k = KernelClass()
        w = _to_np(k.weight(mx.array([0.5])))
        assert np.all(np.isfinite(w))
        assert w.shape == (1,)

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_unit_scale_values_in_0_1(self, KernelClass):
        """For residuals in [-1, 1], weights should be in [0, 1]."""
        k = KernelClass()
        r = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        w = _to_np(k.weight(r))
        assert np.all(w >= 0.0), f"{KernelClass.__name__}: min={w.min()}"
        assert np.all(w <= 1.0 + 1e-7), f"{KernelClass.__name__}: max={w.max()}"

    @pytest.mark.parametrize("KernelClass", ALL_KERNELS)
    def test_batch_of_values(self, KernelClass):
        """Process a larger batch without errors."""
        k = KernelClass()
        r = mx.array(np.linspace(-10, 10, 1000).astype(np.float32))
        w = _to_np(k.weight(r))
        assert w.shape == (1000,)
        assert np.all(np.isfinite(w))
        assert np.all(w >= 0.0)


class TestBaseClass:
    def test_base_raises_not_implemented(self):
        k = RobustKernel()
        with pytest.raises(NotImplementedError):
            k.weight(mx.array([1.0]))
