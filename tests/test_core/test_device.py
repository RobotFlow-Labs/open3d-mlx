"""Tests for open3d_mlx.core.device module."""

import pytest

from open3d_mlx.core.device import Device


class TestDeviceInit:
    """Tests for Device construction."""

    def test_default_device_str(self):
        """Default device string is 'MLX:0'."""
        d = Device()
        assert d.device_str == "MLX:0"

    def test_custom_device_str(self):
        """Custom device strings are uppercased."""
        d = Device("gpu:0")
        assert d.device_str == "GPU:0"

    def test_cpu_device(self):
        """CPU device string is accepted."""
        d = Device("CPU:0")
        assert d.device_str == "CPU:0"


class TestDeviceProperties:
    """Tests for Device properties."""

    def test_type_mlx(self):
        """type property extracts the prefix."""
        assert Device("MLX:0").type == "MLX"

    def test_type_gpu(self):
        """type property works for GPU prefix."""
        assert Device("GPU:0").type == "GPU"

    def test_index_zero(self):
        """index property returns the numeric index."""
        assert Device("MLX:0").index == 0

    def test_index_nonzero(self):
        """index property handles non-zero indices."""
        assert Device("GPU:1").index == 1

    def test_index_default_when_no_colon(self):
        """index defaults to 0 when no colon in device string."""
        d = Device("MLX")
        assert d.index == 0


class TestDeviceRepr:
    """Tests for Device __repr__ and __str__."""

    def test_repr(self):
        """repr returns Device('MLX:0') format."""
        assert repr(Device("MLX:0")) == "Device('MLX:0')"

    def test_str(self):
        """str returns the raw device string."""
        assert str(Device("MLX:0")) == "MLX:0"


class TestDeviceEquality:
    """Tests for Device __eq__ and __hash__."""

    def test_equality_with_device(self):
        """Two Device instances with same string are equal."""
        assert Device("MLX:0") == Device("MLX:0")

    def test_inequality_with_device(self):
        """Two Device instances with different strings are not equal."""
        assert Device("MLX:0") != Device("GPU:0")

    def test_equality_with_string(self):
        """Device compares equal to a matching string."""
        assert Device("MLX:0") == "MLX:0"

    def test_equality_with_string_case_insensitive(self):
        """Device compares equal to string regardless of case."""
        assert Device("MLX:0") == "mlx:0"
        assert Device("gpu:0") == "GPU:0"

    def test_inequality_with_string(self):
        """Device is not equal to a non-matching string."""
        assert Device("MLX:0") != "GPU:0"

    def test_equality_with_other_type(self):
        """Device does not equal non-string, non-Device types."""
        assert Device("MLX:0") != 42
        assert Device("MLX:0") != None  # noqa: E711

    def test_hash_equal_devices(self):
        """Equal devices have the same hash."""
        assert hash(Device("MLX:0")) == hash(Device("MLX:0"))

    def test_hash_usable_in_set(self):
        """Device instances can be used in sets and as dict keys."""
        s = {Device("MLX:0"), Device("MLX:0"), Device("GPU:0")}
        assert len(s) == 2


class TestDeviceStaticMethods:
    """Tests for Device static methods."""

    def test_get_default(self):
        """get_default() returns MLX:0 device."""
        d = Device.get_default()
        assert d == "MLX:0"
        assert isinstance(d, Device)

    def test_is_available(self):
        """MLX backend is available on this machine."""
        assert Device.is_available() is True
