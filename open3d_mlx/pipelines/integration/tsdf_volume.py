"""TSDF volume color type enumeration."""

from enum import Enum


class TSDFVolumeColorType(Enum):
    """Color integration mode for TSDF volumes.

    Matches Open3D: o3d.pipelines.integration.TSDFVolumeColorType
    """

    NoColor = 0
    RGB8 = 1
    Gray32 = 2
