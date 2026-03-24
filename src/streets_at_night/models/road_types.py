"""Road type enumeration used for classification and rendering."""

from enum import Enum


class RoadType(Enum):
    """Categories of roads derived from OSM highway tags."""

    HIGHWAY = "highway"
    MAJOR_ROAD = "major_road"
    COLLECTOR = "collector"
    RESIDENTIAL = "residential"
    SERVICE = "service"
    MINOR = "minor"
