# Radical
"""Tests for streets_at_night.data.road_classifier.classify_roads."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from streets_at_night.data.road_classifier import classify_roads
from streets_at_night.models.road_types import RoadType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_roads_gdf(highway_values: list) -> gpd.GeoDataFrame:
    """Build a minimal GeoDataFrame with a highway column for testing."""
    return gpd.GeoDataFrame(
        {
            "highway": highway_values,
            "geometry": [LineString([(0, 0), (1, 1)])] * len(highway_values),
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Classification — mapped types
# ---------------------------------------------------------------------------

def test_classify_roads_motorway_returns_highway_type():
    # Arrange
    gdf = make_roads_gdf(["motorway"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.HIGHWAY


def test_classify_roads_trunk_link_returns_highway_type():
    # Arrange
    gdf = make_roads_gdf(["trunk_link"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.HIGHWAY


def test_classify_roads_primary_returns_major_road_type():
    # Arrange
    gdf = make_roads_gdf(["primary"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.MAJOR_ROAD


def test_classify_roads_secondary_link_returns_major_road_type():
    # Arrange
    gdf = make_roads_gdf(["secondary_link"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.MAJOR_ROAD


def test_classify_roads_tertiary_returns_collector_type():
    # Arrange
    gdf = make_roads_gdf(["tertiary"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.COLLECTOR


def test_classify_roads_residential_returns_residential_type():
    # Arrange
    gdf = make_roads_gdf(["residential"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.RESIDENTIAL


def test_classify_roads_living_street_returns_residential_type():
    # Arrange
    gdf = make_roads_gdf(["living_street"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.RESIDENTIAL


def test_classify_roads_service_returns_service_type():
    # Arrange
    gdf = make_roads_gdf(["service"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.SERVICE


def test_classify_roads_unclassified_returns_minor_type():
    # Arrange
    gdf = make_roads_gdf(["unclassified"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.MINOR


# ---------------------------------------------------------------------------
# Dropped types — ignored set
# ---------------------------------------------------------------------------

def test_classify_roads_footway_is_dropped():
    # Arrange
    gdf = make_roads_gdf(["footway"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 0


def test_classify_roads_cycleway_is_dropped():
    # Arrange
    gdf = make_roads_gdf(["cycleway"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 0


def test_classify_roads_path_is_dropped():
    # Arrange
    gdf = make_roads_gdf(["path"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 0


def test_classify_roads_steps_is_dropped():
    # Arrange
    gdf = make_roads_gdf(["steps"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Dropped types — unrecognized tags
# ---------------------------------------------------------------------------

def test_classify_roads_unknown_tag_is_dropped():
    # Arrange
    gdf = make_roads_gdf(["unknown_future_tag"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 0


# ---------------------------------------------------------------------------
# List-valued highway tags
# ---------------------------------------------------------------------------

def test_classify_roads_list_highway_tag_classified_by_first_match():
    # Arrange — list value with motorway first; should resolve to HIGHWAY
    gdf = make_roads_gdf([["motorway", "residential"]])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 1
    assert result.iloc[0]["road_type"] == RoadType.HIGHWAY


# ---------------------------------------------------------------------------
# Mixed input
# ---------------------------------------------------------------------------

def test_classify_roads_mixed_input_returns_only_valid_types():
    # Arrange — motorway (keep HIGHWAY), footway (drop), unknown_tag (drop), residential (keep RESIDENTIAL)
    gdf = make_roads_gdf(["motorway", "footway", "unknown_tag", "residential"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert len(result) == 2
    road_types = list(result["road_type"])
    assert RoadType.HIGHWAY in road_types
    assert RoadType.RESIDENTIAL in road_types


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_classify_roads_returns_road_type_column():
    # Arrange
    gdf = make_roads_gdf(["motorway", "residential"])

    # Act
    result = classify_roads(gdf)

    # Assert
    assert "road_type" in result.columns
    assert all(isinstance(v, RoadType) for v in result["road_type"])
