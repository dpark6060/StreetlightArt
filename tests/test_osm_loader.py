# Radical
"""Tests for streets_at_night.data.osm_loader cache correctness.

Covers three bugs in the OSM cache round-trip:
  A. list-valued OSM tags getting stringified by the GPKG round-trip
  B. schema divergence between first-run and cached-run GeoDataFrames
  C. cache key not being sensitive to per-loader download params
"""

from pathlib import Path
from unittest import mock

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from streets_at_night.data import osm_loader
from streets_at_night.data.osm_loader import _cache_path, _save_to_gpkg, load_roads


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_BBOX = (34.20, 34.10, -117.20, -117.30)


def _make_roads_gdf(highway_values: list) -> gpd.GeoDataFrame:
    """Build a minimal roads GeoDataFrame with a highway column for testing."""
    return gpd.GeoDataFrame(
        {
            "highway": highway_values,
            "geometry": [LineString([(0, 0), (1, 1)])] * len(highway_values),
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Bug A — list-valued cells are scalarized before the GPKG round-trip
# ---------------------------------------------------------------------------


def test_save_to_gpkg_scalarizes_list_valued_cells_on_round_trip(tmp_path):
    # Arrange — a list-valued highway cell, as osmnx produces for merged ways
    gdf = _make_roads_gdf([["residential", "unclassified"]])
    path = tmp_path / "roads.gpkg"

    # Act
    _save_to_gpkg(gdf, path, keep_cols=["geometry", "highway"])
    result = gpd.read_file(path)

    # Assert — the round-tripped value is the scalar first element, not a
    # stringified list
    assert result["highway"].iloc[0] == "residential"


def test_save_to_gpkg_empty_list_cell_does_not_raise(tmp_path):
    # Arrange — an empty-list highway cell should not raise IndexError
    gdf = _make_roads_gdf([[]])
    path = tmp_path / "roads.gpkg"

    # Act
    _save_to_gpkg(gdf, path, keep_cols=["geometry", "highway"])
    result = gpd.read_file(path)

    # Assert — falls through to a missing value, not a crash
    assert result["highway"].iloc[0] is None


def test_save_to_gpkg_colon_column_survives_round_trip(tmp_path):
    # Arrange — a "building:levels" column, as OSM building footprints carry
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [LineString([(0, 0), (1, 1)])],
            "building:levels": ["3"],
        },
        crs="EPSG:4326",
    )
    path = tmp_path / "buildings.gpkg"

    # Act
    _save_to_gpkg(gdf, path, keep_cols=["geometry", "building:levels"])
    result = gpd.read_file(path)

    # Assert — the colon-bearing column name and its value both survive.
    # If GDAL sanitizes the colon out of the column name, this fails and
    # signals a real bug: building:levels silently breaks downstream.
    assert "building:levels" in result.columns
    assert result["building:levels"].iloc[0] == "3"


# ---------------------------------------------------------------------------
# Bug B — cold and warm loads return identical schemas
# ---------------------------------------------------------------------------


def test_load_roads_cold_and_warm_return_identical_schema(tmp_path):
    # Arrange — a raw-shaped GDF like osmnx's graph_to_gdfs output, with
    # extra columns and a list-valued highway cell
    raw_gdf = gpd.GeoDataFrame(
        {
            "highway": [["residential", "unclassified"]],
            "name": ["Main St"],
            "osmid": [123],
            "oneway": [False],
            "length": [42.0],
            "maxspeed": ["25 mph"],
            "some_other_osm_column": ["irrelevant"],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        crs="EPSG:4326",
    )

    with mock.patch.object(osm_loader, "_download_roads", return_value=raw_gdf):
        cold_result = load_roads(_BBOX, tmp_path)
        warm_result = load_roads(_BBOX, tmp_path)

    # Assert — identical columns and dtypes regardless of cache state
    assert list(cold_result.columns) == list(warm_result.columns)
    assert list(cold_result.dtypes) == list(warm_result.dtypes)
    assert cold_result["highway"].iloc[0] == "residential"
    assert warm_result["highway"].iloc[0] == "residential"


# ---------------------------------------------------------------------------
# Bug C — cache key incorporates params and schema version, not just bbox
# ---------------------------------------------------------------------------


def test_cache_path_different_params_produce_different_paths(tmp_path):
    # Arrange
    params_a = {"network_type": "drive"}
    params_b = {"network_type": "walk"}

    # Act
    path_a = _cache_path(tmp_path, "roads", _BBOX, params_a)
    path_b = _cache_path(tmp_path, "roads", _BBOX, params_b)

    # Assert
    assert path_a != path_b


def test_cache_path_same_bbox_as_list_or_tuple_produces_same_path(tmp_path):
    # Arrange
    params = {"network_type": "drive"}
    bbox_tuple = _BBOX
    bbox_list = list(_BBOX)

    # Act
    path_tuple = _cache_path(tmp_path, "roads", bbox_tuple, params)
    path_list = _cache_path(tmp_path, "roads", bbox_list, params)

    # Assert
    assert path_tuple == path_list
