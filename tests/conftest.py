"""Shared test fixtures for open3d-mlx."""

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic numpy RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_points(rng):
    """100 random 3D points as MLX array."""
    return mx.array(rng.standard_normal((100, 3)).astype(np.float32))


@pytest.fixture
def identity_4x4():
    """4x4 identity matrix as MLX array."""
    return mx.eye(4)
