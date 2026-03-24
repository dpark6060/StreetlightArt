"""Road segment data model — geometry, classification, and fixture profile."""

import typing as t

import numpy as np
from dataclasses import dataclass, field

from .road_types import RoadType
from .streetlight import FixtureProfile


# Road type → default road width in meters
_DEFAULT_WIDTHS: t.Dict[RoadType, float] = {
    RoadType.HIGHWAY:     12.0,
    RoadType.MAJOR_ROAD:   8.0,
    RoadType.COLLECTOR:    6.0,
    RoadType.RESIDENTIAL:  4.0,
    RoadType.SERVICE:      3.0,
    RoadType.MINOR:        4.0,
}


@dataclass
class RoadSegment:
    """A single road segment — geometry and classification only.

    No light instances are stored here; those are added in Stage 5.

    Attributes:
        geometry: (N, 3) array of (x, y, z) points in local ENU meters.
        road_type: Classification of this road.
        width: Road width in meters.
        fixture_profile: Lighting character for this segment. None means no streetlights.
    """

    geometry: np.ndarray
    road_type: RoadType
    width: float
    fixture_profile: t.Optional[FixtureProfile] = field(default=None, compare=False)


def get_default_width(road_type: RoadType) -> float:
    """Return the default road width in meters for a given road type.

    Args:
        road_type (RoadType): The road classification.

    Returns:
        float: Default width in meters. Falls back to 4.0m for unknown types.
    """
    return _DEFAULT_WIDTHS.get(road_type, 4.0)
