# Radical
"""Integration-style tests for streets_at_night.data.zone_builder.build_zones."""

import random

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from streets_at_night.camera.coordinates import create_transformer
from streets_at_night.config.schema import (
    ColorPoolConfig,
    HouseConfig,
    ZoneColorsConfig,
    ZoneConfig,
)
from streets_at_night.data.zone_builder import build_zones
from streets_at_night.models.house import House
from streets_at_night.models.zones import CommercialZone, DarkZone, ResidentialZone


# ---------------------------------------------------------------------------
# Constants — San Bernardino area (lon, lat order for shapely box())
# ---------------------------------------------------------------------------

_ORIGIN_LAT = 34.10
_ORIGIN_LON = -117.30

# A small residential polygon ~1km across
_RES_POLY = box(-117.31, 34.10, -117.30, 34.11)

# A small commercial polygon just east of residential
_COM_POLY = box(-117.30, 34.10, -117.29, 34.11)

# A small park polygon north of residential
_PARK_POLY = box(-117.31, 34.11, -117.30, 34.12)

# A small water polygon
_WATER_POLY = box(-117.29, 34.10, -117.28, 34.11)

# A tiny building footprint well inside the residential polygon
_BUILDING_POLY = box(-117.308, 34.102, -117.306, 34.104)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transformer():
    """Real CoordinateTransformer centred on the San Bernardino test area."""
    return create_transformer(_ORIGIN_LAT, _ORIGIN_LON)


@pytest.fixture
def zone_config():
    """Minimal ZoneConfig built directly from schema dataclasses."""
    house_config = HouseConfig(
        exterior_colors=ColorPoolConfig(
            colors=[
                ("#0D0A08", 1.0),
                ("#0C0907", 1.0),
            ]
        ),
        yard_base_color="#060A04",
        yard_variation=2,
        min_lights=0,
        max_lights=2,
        light_weights=[1.0, 1.0, 1.0],
        footprint_coverage=0.65,
        lot_width=15.0,
        lot_depth=25.0,
        setback=10.0,
    )
    colors = ZoneColorsConfig(
        commercial="#0A0A0C",
        industrial="#080808",
        park_forest="#040804",
        water="#040408",
        road_surface="#0C0C0C",
        default_ground="#080706",
    )
    return ZoneConfig(
        house=house_config,
        colors=colors,
        tile_size_meters=256.0,
        tile_resolution_px=128,
    )


@pytest.fixture
def rng():
    """Seeded RNG for reproducible test runs."""
    return random.Random(42)


@pytest.fixture
def empty_land_use_gdf():
    """Empty land use GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"landuse": [], "geometry": []},
        crs="EPSG:4326",
    )


@pytest.fixture
def empty_buildings_gdf():
    """Empty buildings GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"building": [], "geometry": []},
        crs="EPSG:4326",
    )


@pytest.fixture
def residential_land_use_gdf():
    """Land use GDF containing one residential polygon."""
    return gpd.GeoDataFrame(
        {"landuse": ["residential"], "geometry": [_RES_POLY]},
        crs="EPSG:4326",
    )


@pytest.fixture
def list_valued_residential_land_use_gdf():
    """Land use GDF with a list-valued landuse tag (OSM merged-feature convention)."""
    return gpd.GeoDataFrame(
        {"landuse": [["residential"]], "geometry": [_RES_POLY]},
        crs="EPSG:4326",
    )


@pytest.fixture
def commercial_land_use_gdf():
    """Land use GDF containing one commercial polygon."""
    return gpd.GeoDataFrame(
        {"landuse": ["commercial"], "geometry": [_COM_POLY]},
        crs="EPSG:4326",
    )


@pytest.fixture
def park_land_use_gdf():
    """Land use GDF for a leisure=park polygon."""
    return gpd.GeoDataFrame(
        {"landuse": [None], "leisure": ["park"], "geometry": [_PARK_POLY]},
        crs="EPSG:4326",
    )


@pytest.fixture
def water_natural_gdf():
    """Land use GDF with natural=water polygon."""
    return gpd.GeoDataFrame(
        {"landuse": [None], "natural": ["water"], "geometry": [_WATER_POLY]},
        crs="EPSG:4326",
    )


@pytest.fixture
def building_inside_residential_gdf():
    """Buildings GDF with one footprint that lies inside the residential polygon."""
    return gpd.GeoDataFrame(
        {"building": ["yes"], "geometry": [_BUILDING_POLY]},
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_zones_empty_gdfs_returns_three_empty_lists(
    empty_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — all fixtures are empty GDFs

    # Act
    residential, commercial, dark = build_zones(
        empty_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    assert residential == []
    assert commercial == []
    assert dark == []


def test_build_zones_residential_land_use_produces_residential_zone(
    residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — one residential polygon, no buildings

    # Act
    residential, commercial, dark = build_zones(
        residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    assert len(residential) >= 1
    assert all(isinstance(z, ResidentialZone) for z in residential)
    assert commercial == []


def test_build_zones_list_valued_landuse_tag_produces_residential_zone(
    list_valued_residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — landuse tag is a list (OSM merged-feature convention), not a plain string

    # Act
    residential, commercial, dark = build_zones(
        list_valued_residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    assert len(residential) >= 1
    assert all(isinstance(z, ResidentialZone) for z in residential)
    assert commercial == []


def test_build_zones_commercial_land_use_produces_commercial_zone(
    commercial_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — one commercial polygon, no buildings

    # Act
    residential, commercial, dark = build_zones(
        commercial_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    assert len(commercial) >= 1
    assert all(isinstance(z, CommercialZone) for z in commercial)
    assert residential == []


def test_build_zones_park_land_use_produces_dark_zone_with_park_type(
    park_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — one park polygon, classified via leisure tag

    # Act
    residential, commercial, dark = build_zones(
        park_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    assert len(dark) >= 1
    assert all(isinstance(z, DarkZone) for z in dark)
    park_zones = [z for z in dark if z.zone_type == "park"]
    assert len(park_zones) >= 1


def test_build_zones_water_natural_produces_dark_zone_with_water_type(
    water_natural_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — one natural=water polygon

    # Act
    residential, commercial, dark = build_zones(
        water_natural_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    assert len(dark) >= 1
    water_zones = [z for z in dark if z.zone_type == "water"]
    assert len(water_zones) >= 1


def test_build_zones_residential_zone_has_enu_polygon_boundary(
    residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — residential land use with no buildings (forces synthetic lots)

    # Act
    residential, _, _ = build_zones(
        residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert — boundary should be a valid Polygon in ENU metre scale (not WGS84 degrees)
    assert len(residential) >= 1
    zone = residential[0]
    assert isinstance(zone.boundary, Polygon)
    assert not zone.boundary.is_empty
    # ENU coordinates for this area should be in the order of hundreds to thousands of metres
    # not ~0.01 degrees (which would indicate WGS84 was not converted)
    bounds = zone.boundary.bounds
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    assert extent > 10.0, "Boundary extent looks like degrees, not metres"


def test_build_zones_residential_with_buildings_creates_houses_from_footprints(
    residential_land_use_gdf,
    building_inside_residential_gdf,
    transformer,
    zone_config,
    rng,
):
    # Arrange — one building footprint inside the residential zone

    # Act
    residential, _, _ = build_zones(
        residential_land_use_gdf,
        building_inside_residential_gdf,
        transformer,
        zone_config,
        rng,
    )

    # Assert
    assert len(residential) >= 1
    zone = residential[0]
    assert len(zone.houses) >= 1
    assert all(isinstance(h, House) for h in zone.houses)


def test_build_zones_residential_synthetic_lots_generates_houses_without_buildings(
    residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — residential zone, no building footprints; should fall back to grid lots

    # Act
    residential, _, _ = build_zones(
        residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert — synthetic grid should produce at least one house
    assert len(residential) >= 1
    zone = residential[0]
    assert len(zone.houses) >= 1
    assert all(isinstance(h, House) for h in zone.houses)


def test_build_zones_reproducible_with_same_seed(
    residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config
):
    # Arrange — two calls with identical seeds

    rng_a = random.Random(99)
    rng_b = random.Random(99)

    # Act
    residential_a, _, _ = build_zones(
        residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng_a
    )
    residential_b, _, _ = build_zones(
        residential_land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng_b
    )

    # Assert — same seed produces same number of houses with same colors
    assert len(residential_a) == len(residential_b)
    for zone_a, zone_b in zip(residential_a, residential_b):
        assert len(zone_a.houses) == len(zone_b.houses)
        for house_a, house_b in zip(zone_a.houses, zone_b.houses):
            assert house_a.base_color == house_b.base_color
            assert house_a.yard_color == house_b.yard_color
            assert house_a.num_lights == house_b.num_lights
