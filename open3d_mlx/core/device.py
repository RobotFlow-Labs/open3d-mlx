"""Device abstraction for MLX.

MLX uses unified memory on Apple Silicon -- there is no CPU/GPU distinction
for data placement.  This module provides a ``Device`` class for API
compatibility with Open3D patterns that pass ``device=`` arguments.  All
computation is dispatched to the MLX default device (Apple GPU when available,
CPU otherwise).
"""

from __future__ import annotations

import mlx.core as mx


class Device:
    """Device descriptor, compatible with Open3D's device strings.

    On MLX every array lives in unified memory and computation is dispatched
    to the default MLX device (typically the Apple GPU).  This class exists
    so that code which passes ``device="GPU:0"`` or ``device="MLX:0"`` does
    not break.

    Examples:
        >>> d = Device()
        >>> d
        Device('MLX:0')
        >>> d == "mlx:0"
        True
        >>> Device.is_available()
        True
    """

    __slots__ = ("_device_str",)

    def __init__(self, device_str: str = "MLX:0") -> None:
        """Create a Device from a device string.

        Args:
            device_str: Device identifier.  Recognised prefixes are
                ``"MLX"``, ``"GPU"``, ``"CPU"``.  The string is normalised
                to uppercase.  Defaults to ``"MLX:0"``.
        """
        self._device_str: str = device_str.upper()

    # -- Properties ----------------------------------------------------------

    @property
    def type(self) -> str:
        """Return the device type prefix (e.g. ``"MLX"``, ``"GPU"``)."""
        return self._device_str.split(":")[0]

    @property
    def index(self) -> int:
        """Return the device index (e.g. ``0``)."""
        parts = self._device_str.split(":")
        if len(parts) >= 2:
            return int(parts[1])
        return 0

    @property
    def device_str(self) -> str:
        """Return the full device string."""
        return self._device_str

    # -- Dunder methods ------------------------------------------------------

    def __repr__(self) -> str:
        return f"Device('{self._device_str}')"

    def __str__(self) -> str:
        return self._device_str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._device_str == other.upper()
        if isinstance(other, Device):
            return self._device_str == other._device_str
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._device_str)

    # -- Static helpers ------------------------------------------------------

    @staticmethod
    def get_default() -> Device:
        """Return the default device (``MLX:0``).

        Returns:
            A ``Device`` instance representing the default MLX device.
        """
        return Device("MLX:0")

    @staticmethod
    def is_available() -> bool:
        """Check whether the MLX backend can execute computation.

        Performs a minimal eval to verify that the Metal backend (or CPU
        fallback) is functional.

        Returns:
            ``True`` if MLX can run operations, ``False`` otherwise.
        """
        try:
            mx.eval(mx.zeros(1))
            return True
        except Exception:
            return False
