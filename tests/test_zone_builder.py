# Radical
"""Integration-style tests for streets_at_night.data.zone_builder.build_zones."""

import random

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from streets_at_night.camera.coordinates import create_transformer
from streets_at_night.config.schema import (
    ColorPoolConfig,
    HouseConfig,
    ZoneColorsConfig,
    ZoneConfig,
)
from streets_at_night.data.zone_builder import (
    _build_parking_areas,
    _convert_polygons_individually,
    _filter_lots_within_boundary,
    _get_buildings_within,
    _polygons_wgs84_to_enu_batch,
    build_zones,
)
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


# ---------------------------------------------------------------------------
# A — Polygon holes preserved
# ---------------------------------------------------------------------------


def test_build_zones_park_with_hole_preserves_interior_ring(
    empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — a park polygon with a lake-shaped hole carved out of the middle
    hole = Polygon(
        [(-117.309, 34.111), (-117.302, 34.111), (-117.302, 34.118), (-117.309, 34.118)]
    )
    park_with_hole = Polygon(_PARK_POLY.exterior.coords, holes=[hole.exterior.coords])
    land_use_gdf = gpd.GeoDataFrame(
        {"landuse": [None], "leisure": ["park"], "geometry": [park_with_hole]},
        crs="EPSG:4326",
    )

    # Act
    _, _, dark = build_zones(
        land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert — the hole survives WGS84 -> ENU conversion
    park_zones = [z for z in dark if z.zone_type == "park"]
    assert len(park_zones) == 1
    zone = park_zones[0]
    assert len(zone.boundary.interiors) == 1

    point_inside_hole = Polygon(zone.boundary.interiors[0]).representative_point()
    assert not zone.boundary.contains(point_inside_hole)


# ---------------------------------------------------------------------------
# B — Boundary-straddling buildings assigned to at most one zone
# ---------------------------------------------------------------------------


def test_get_buildings_within_straddling_building_assigned_to_only_one_zone():
    # Arrange — two adjacent zones sharing an edge at lon=-117.31, and a
    # building whose footprint straddles that shared edge
    zone_a = box(-117.32, 34.10, -117.31, 34.11)
    zone_b = box(-117.31, 34.10, -117.30, 34.11)
    straddling_building = box(-117.312, 34.104, -117.305, 34.106)
    buildings_gdf = gpd.GeoDataFrame(
        {"building": ["yes"], "geometry": [straddling_building]},
        crs="EPSG:4326",
    )

    # Act
    buildings_in_a = _get_buildings_within(zone_a, buildings_gdf)
    buildings_in_b = _get_buildings_within(zone_b, buildings_gdf)

    # Assert — assigned to exactly one zone, not both
    assert len(buildings_in_a) + len(buildings_in_b) == 1


def test_build_zones_straddling_building_produces_exactly_one_house_across_two_zones(
    transformer, zone_config, rng
):
    # Arrange — two adjacent residential polygons sharing an edge at
    # lon=-117.31, and a single, large building footprint straddling that
    # edge. The footprint is far larger than any synthetic lot footprint
    # (~158 sq m at this zone_config's lot dimensions), so real-building
    # houses can be distinguished from density-fallback synthetic houses by
    # footprint area alone.
    zone_a = box(-117.32, 34.10, -117.31, 34.11)
    zone_b = box(-117.31, 34.10, -117.30, 34.11)
    straddling_building = box(-117.312, 34.104, -117.305, 34.106)
    land_use_gdf = gpd.GeoDataFrame(
        {"landuse": ["residential", "residential"], "geometry": [zone_a, zone_b]},
        crs="EPSG:4326",
    )
    buildings_gdf = gpd.GeoDataFrame(
        {"building": ["yes"], "geometry": [straddling_building]},
        crs="EPSG:4326",
    )

    # Act
    residential, _, _ = build_zones(
        land_use_gdf, buildings_gdf, transformer, zone_config, rng
    )

    # Assert — the straddling footprint is converted into exactly one House
    # total across both zones, not duplicated into both
    assert len(residential) == 2
    real_footprint_houses = [
        house
        for zone in residential
        for house in zone.houses
        if house.footprint.area > 1000.0
    ]
    assert len(real_footprint_houses) == 1


def test_build_zones_single_corner_building_below_density_threshold_still_gets_synthetic_lots(
    residential_land_use_gdf, transformer, zone_config, rng
):
    # Arrange — one tiny building near a corner of the zone; far too sparse
    # to reach _MIN_HOUSES_PER_KM2 on its own
    corner_building = box(-117.302, 34.108, -117.301, 34.109)
    buildings_gdf = gpd.GeoDataFrame(
        {"building": ["yes"], "geometry": [corner_building]},
        crs="EPSG:4326",
    )

    # Act
    residential, _, _ = build_zones(
        residential_land_use_gdf, buildings_gdf, transformer, zone_config, rng
    )

    # Assert — synthetic lots were generated in addition to the real building
    assert len(residential) == 1
    assert len(residential[0].houses) > 1


# ---------------------------------------------------------------------------
# C — Zone classification (grass -> park)
# ---------------------------------------------------------------------------


def test_build_zones_grass_landuse_produces_dark_zone_with_park_type(
    empty_buildings_gdf, transformer, zone_config, rng
):
    # Arrange — landuse=grass should classify as a park DarkZone
    land_use_gdf = gpd.GeoDataFrame(
        {"landuse": ["grass"], "geometry": [_PARK_POLY]}, crs="EPSG:4326"
    )

    # Act
    _, _, dark = build_zones(
        land_use_gdf, empty_buildings_gdf, transformer, zone_config, rng
    )

    # Assert
    park_zones = [z for z in dark if z.zone_type == "park"]
    assert len(park_zones) == 1


# ---------------------------------------------------------------------------
# D — Parking-area computation (union-difference, not looped difference)
# ---------------------------------------------------------------------------


def test_build_parking_areas_excludes_all_footprints():
    # Arrange
    boundary = box(0.0, 0.0, 100.0, 100.0)
    footprints = [
        box(10.0, 10.0, 20.0, 20.0),
        box(40.0, 40.0, 50.0, 50.0),
        box(70.0, 70.0, 80.0, 80.0),
    ]

    # Act
    parking = _build_parking_areas(boundary, footprints)

    # Assert — no parking polygon overlaps the interior of any footprint
    parking_union = unary_union(parking)
    for footprint in footprints:
        assert not parking_union.intersects(footprint.buffer(-0.5))


def test_build_parking_areas_matches_iterative_difference_equivalence():
    # Arrange — same small case computed two ways: single union-difference
    # (current implementation) vs. the old per-footprint looped difference
    boundary = box(0.0, 0.0, 100.0, 100.0)
    footprints = [
        box(10.0, 10.0, 20.0, 20.0),
        box(40.0, 40.0, 50.0, 50.0),
        box(70.0, 70.0, 80.0, 80.0),
    ]

    # Act
    result = _build_parking_areas(boundary, footprints)
    result_union = unary_union(result)

    expected = boundary
    for footprint in footprints:
        expected = expected.difference(footprint)

    # Assert
    assert result_union.equals(expected)


# ---------------------------------------------------------------------------
# E — Vectorized geometry equivalence (behavior-preserving perf changes)
# ---------------------------------------------------------------------------


def test_filter_lots_within_boundary_matches_naive_per_lot_contains():
    # Arrange — an L-shaped boundary and a grid of candidate lots, some of
    # which fall in the boundary's missing corner
    zone_boundary = Polygon(
        [(0, 0), (100, 0), (100, 80), (60, 80), (60, 40), (0, 40)]
    )
    lots = [
        box(x, y, x + 10, y + 10)
        for x in range(0, 100, 15)
        for y in range(0, 80, 15)
    ]

    # Act
    vectorized_result = _filter_lots_within_boundary(lots, zone_boundary)
    naive_result = [lot for lot in lots if zone_boundary.contains(lot.centroid)]

    # Assert — vectorized shapely.contains_xy matches the naive per-lot loop
    assert vectorized_result == naive_result


def test_polygons_wgs84_to_enu_batch_matches_individual_conversion(transformer):
    # Arrange — several polygons, including one with a hole, converted both
    # via the single batched transform and one-at-a-time
    hole = Polygon(
        [(-117.309, 34.111), (-117.302, 34.111), (-117.302, 34.118), (-117.309, 34.118)]
    )
    poly_with_hole = Polygon(_PARK_POLY.exterior.coords, holes=[hole.exterior.coords])
    polygons = [_RES_POLY, _COM_POLY, poly_with_hole]

    # Act
    batch_result = _polygons_wgs84_to_enu_batch(polygons, transformer)
    individual_result = _convert_polygons_individually(polygons, transformer)

    # Assert — same polygons, coordinates equal to 1e-6
    assert len(batch_result) == len(individual_result)
    for batch_poly, individual_poly in zip(batch_result, individual_result):
        batch_coords = np.array(batch_poly.exterior.coords)
        individual_coords = np.array(individual_poly.exterior.coords)
        assert np.allclose(batch_coords, individual_coords, atol=1e-6)

        # Assert — holes are reconstructed onto the correct polygon, not
        # misassigned by the flatten/split/rebuild index bookkeeping
        assert len(batch_poly.interiors) == len(individual_poly.interiors)
        for batch_hole, individual_hole in zip(batch_poly.interiors, individual_poly.interiors):
            batch_hole_coords = np.array(batch_hole.coords)
            individual_hole_coords = np.array(individual_hole.coords)
            assert np.allclose(batch_hole_coords, individual_hole_coords, atol=1e-6)
