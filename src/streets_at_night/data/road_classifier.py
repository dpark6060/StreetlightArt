"""Classify OSM road edges into RoadType categories."""

import geopandas as gpd

from ..models.road_types import RoadType


_HIGHWAY_TO_ROAD_TYPE: dict = {
    "motorway": RoadType.HIGHWAY,
    "motorway_link": RoadType.HIGHWAY,
    "trunk": RoadType.HIGHWAY,
    "trunk_link": RoadType.HIGHWAY,
    "primary": RoadType.MAJOR_ROAD,
    "primary_link": RoadType.MAJOR_ROAD,
    "secondary": RoadType.MAJOR_ROAD,
    "secondary_link": RoadType.MAJOR_ROAD,
    "tertiary": RoadType.COLLECTOR,
    "tertiary_link": RoadType.COLLECTOR,
    "residential": RoadType.RESIDENTIAL,
    "living_street": RoadType.RESIDENTIAL,
    "service": RoadType.SERVICE,
    "unclassified": RoadType.MINOR,
}

_IGNORED_HIGHWAY_TYPES: set = {
    "footway",
    "cycleway",
    "path",
    "pedestrian",
    "track",
    "steps",
    "bridleway",
}


def classify_roads(roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a road_type column and drop irrelevant highway types.

    Rows with highway tags in the ignored set (footways, cycleways, etc.)
    are dropped. Rows with unrecognized tags that are not in the mapping
    are also dropped.

    Args:
        roads_gdf (GeoDataFrame): Road edges with an OSM 'highway' column.

    Returns:
        GeoDataFrame: Filtered roads with an added 'road_type' column.
    """
    gdf = roads_gdf.copy()
    gdf = _normalize_highway_column(gdf)
    gdf = _drop_ignored_types(gdf)
    gdf = _apply_road_type_mapping(gdf)
    gdf = _drop_unrecognized_types(gdf)
    return gdf


def _normalize_highway_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Coerce the highway column to a simple string, handling list values.

    OSM edges can have multiple highway tags stored as a list. This takes
    the first value in that case.

    Args:
        gdf (GeoDataFrame): Roads GeoDataFrame.

    Returns:
        GeoDataFrame: GeoDataFrame with highway column as plain strings.
    """
    gdf = gdf.copy()
    gdf["highway"] = gdf["highway"].apply(
        lambda v: v[0] if isinstance(v, list) else v
    )
    return gdf


def _drop_ignored_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove rows whose highway tag is in the ignore list.

    Args:
        gdf (GeoDataFrame): Roads with a string 'highway' column.

    Returns:
        GeoDataFrame: Roads with ignored types removed.
    """
    mask = ~gdf["highway"].isin(_IGNORED_HIGHWAY_TYPES)
    return gdf[mask].copy()


def _apply_road_type_mapping(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map OSM highway tag strings to RoadType enum values.

    Args:
        gdf (GeoDataFrame): Roads with a string 'highway' column.

    Returns:
        GeoDataFrame: Roads with a new 'road_type' column (may contain None).
    """
    gdf = gdf.copy()
    gdf["road_type"] = gdf["highway"].map(_HIGHWAY_TO_ROAD_TYPE)
    return gdf


def _drop_unrecognized_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop rows that could not be mapped to a known RoadType.

    Args:
        gdf (GeoDataFrame): Roads with 'road_type' column (may contain NaN).

    Returns:
        GeoDataFrame: Roads with all unrecognized types removed.
    """
    return gdf[gdf["road_type"].notna()].copy()
