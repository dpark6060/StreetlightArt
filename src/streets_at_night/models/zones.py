"""Zone data containers — residential, commercial, and dark zones."""

import typing as t
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from .house import House


@dataclass
class ResidentialZone:
    """A residential zone boundary with its collection of houses.

    Data container only.  Houses are lit via extruded building geometry
    (building_pass), not by a dedicated light-placement engine.
    """

    boundary: Polygon
    houses: t.List[House] = field(default_factory=list)


@dataclass
class CommercialZone:
    """A commercial zone boundary with building footprints and parking areas.

    Data container only.  Lights (SignLight, ParkingLotLight, StorefrontLight)
    are created by commercial_placer in Stage 5 and stored in the Scene's
    flat light collection.
    """

    boundary: Polygon
    building_footprints: t.List[Polygon] = field(default_factory=list)
    parking_areas: t.List[Polygon] = field(default_factory=list)


@dataclass
class DarkZone:
    """A dark zone (park, forest, water) with a uniform base color.

    Data container only.  No light placement occurs within dark zones
    (parks may gain sparse path lights in a future stage, but that is
    not implemented here).
    """

    boundary: Polygon
    zone_type: str
    base_color: t.Tuple[int, int, int]
