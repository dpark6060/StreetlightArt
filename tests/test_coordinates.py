# Radical
"""Unit tests for streets_at_night.camera.coordinates."""

import numpy as np
import pytest
import geopandas as gpd
from shapely.geometry import LineString

from streets_at_night.camera.coordinates import (
    CoordinateTransformer,
    create_transformer,
    wgs84_to_enu,
    wgs84_to_enu_batch,
    convert_road_geometry,
)

# ---------------------------------------------------------------------------
# San Bernardino area constants
# ---------------------------------------------------------------------------
_ORIGIN_LAT = 34.17
_ORIGIN_LON = -117.30

_EAST_LAT = 34.17
_EAST_LON = -117.20   # ~8.8 km east of origin

_NORTH_LAT = 34.27
_NORTH_LON = -117.30  # ~11.1 km north of origin

_TOLERANCE_M = 100.0  # 100-metre absolute tolerance for geographic assertions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def transformer() -> CoordinateTransformer:
    """Return a CoordinateTransformer centred on the San Bernardino origin."""
    return create_transformer(_ORIGIN_LAT, _ORIGIN_LON)


@pytest.fixture
def single_road_gdf() -> gpd.GeoDataFrame:
    """Return a minimal GeoDataFrame with one east-running road."""
    return gpd.GeoDataFrame(
        {"geometry": [LineString([(_ORIGIN_LON, _ORIGIN_LAT), (_EAST_LON, _EAST_LAT)])]},
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# create_transformer
# ---------------------------------------------------------------------------

def test_create_transformer_returns_coordinate_transformer():
    # Arrange / Act
    result = create_transformer(_ORIGIN_LAT, _ORIGIN_LON)

    # Assert
    assert isinstance(result, CoordinateTransformer)


# ---------------------------------------------------------------------------
# wgs84_to_enu — single-point
# ---------------------------------------------------------------------------

def test_wgs84_to_enu_origin_returns_zero_xy(transformer):
    # Act
    east, north, up = wgs84_to_enu(transformer, _ORIGIN_LAT, _ORIGIN_LON)

    # Assert
    assert east == pytest.approx(0.0, abs=_TOLERANCE_M)
    assert north == pytest.approx(0.0, abs=_TOLERANCE_M)


def test_wgs84_to_enu_altitude_in_meters(transformer):
    # Act
    _, _, up = wgs84_to_enu(transformer, _ORIGIN_LAT, _ORIGIN_LON, alt_m=1000.0)

    # Assert
    assert up == pytest.approx(1000.0, abs=0.01)


def test_wgs84_to_enu_east_displacement(transformer):
    # Act
    east, north, _ = wgs84_to_enu(transformer, _EAST_LAT, _EAST_LON)

    # Assert — point is east of origin: x positive, y near zero
    assert east > 0.0
    assert north == pytest.approx(0.0, abs=_TOLERANCE_M)


def test_wgs84_to_enu_north_displacement(transformer):
    # Act
    east, north, _ = wgs84_to_enu(transformer, _NORTH_LAT, _NORTH_LON)

    # Assert — point is north of origin: y positive, x near zero
    assert north > 0.0
    assert east == pytest.approx(0.0, abs=_TOLERANCE_M)


# ---------------------------------------------------------------------------
# wgs84_to_enu_batch
# ---------------------------------------------------------------------------

def test_wgs84_to_enu_batch_shape(transformer):
    # Arrange
    lats = np.array([_ORIGIN_LAT, _EAST_LAT, _NORTH_LAT])
    lons = np.array([_ORIGIN_LON, _EAST_LON, _NORTH_LON])

    # Act
    result = wgs84_to_enu_batch(transformer, lats, lons)

    # Assert
    assert result.shape == (3, 3)


def test_wgs84_to_enu_batch_matches_single(transformer):
    # Arrange
    lats = np.array([_EAST_LAT])
    lons = np.array([_EAST_LON])

    # Act
    batch_result = wgs84_to_enu_batch(transformer, lats, lons)
    single_east, single_north, single_up = wgs84_to_enu(transformer, _EAST_LAT, _EAST_LON)

    # Assert
    assert batch_result[0, 0] == pytest.approx(single_east, abs=0.01)
    assert batch_result[0, 1] == pytest.approx(single_north, abs=0.01)
    assert batch_result[0, 2] == pytest.approx(single_up, abs=0.01)


# ---------------------------------------------------------------------------
# convert_road_geometry
# ---------------------------------------------------------------------------

def test_convert_road_geometry_returns_list(transformer, single_road_gdf):
    # Act
    result = convert_road_geometry(transformer, single_road_gdf)

    # Assert
    assert isinstance(result, list)


def test_convert_road_geometry_z_is_zero(transformer, single_road_gdf):
    # Act
    segments = convert_road_geometry(transformer, single_road_gdf)

    # Assert — all z values should be exactly 0 (ground level)
    for segment in segments:
        assert np.all(segment[:, 2] == 0.0)


def test_convert_road_geometry_batched_transform_matches_per_segment_conversion(transformer):
    # Arrange — multiple segments of different vertex counts, to verify the
    # batched-then-split transform lines up correctly per segment
    multi_segment_gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(_ORIGIN_LON, _ORIGIN_LAT), (_EAST_LON, _EAST_LAT)]),
                LineString(
                    [
                        (_ORIGIN_LON, _ORIGIN_LAT),
                        (_NORTH_LON, _NORTH_LAT),
                        (_EAST_LON, _EAST_LAT),
                    ]
                ),
            ]
        },
        crs="EPSG:4326",
    )

    # Act
    batched_segments = convert_road_geometry(transformer, multi_segment_gdf)

    # Assert — each batched segment matches converting its own coordinates alone
    for segment, geom in zip(batched_segments, multi_segment_gdf.geometry):
        coords = np.array(geom.coords)
        expected = wgs84_to_enu_batch(transformer, coords[:, 1], coords[:, 0])
        assert np.allclose(segment, expected, atol=1e-6)
