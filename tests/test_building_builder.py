# Radical
"""Unit tests for streets_at_night.placement.building_builder."""

import dataclasses
import random

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from streets_at_night.camera.coordinates import create_transformer
from streets_at_night.config.schema import BuildingConfig, ColorPoolConfig, HouseConfig
from streets_at_night.models.building import Building
from streets_at_night.models.house import House
from streets_at_night.models.zones import ResidentialZone
from streets_at_night.placement.building_builder import (
    _apply_height_jitter,
    _build_house_buildings,
    _filter_houses_without_osm_match,
    _parse_height_tag,
    _parse_levels_tag,
    _resolve_building_height,
    build_buildings,
)


_ORIGIN_LAT = 34.10
_ORIGIN_LON = -117.30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transformer():
    """Real CoordinateTransformer centred on the San Bernardino test area."""
    return create_transformer(_ORIGIN_LAT, _ORIGIN_LON)


@pytest.fixture
def building_config() -> BuildingConfig:
    """BuildingConfig with height_jitter disabled for deterministic fallback tests."""
    return BuildingConfig(
        enabled=True,
        floor_height_m=3.2,
        height_jitter=0.0,
        building_max_distance_m=12000.0,
        window_max_distance_m=6000.0,
        wall_color="#0B0A09",
        roof_color="#080807",
        face_shade_strength=0.2,
        window_lit_fraction=0.12,
        window_width_m=1.4,
        window_height_m=1.1,
        window_pitch_m=2.4,
        window_side_margin_m=0.8,
        window_sill_margin_m=0.9,
        window_roof_margin_m=0.6,
        window_color_temp_min_k=1900.0,
        window_color_temp_max_k=5500.0,
        window_saturation=0.85,
        window_brightness_min=0.15,
        window_brightness_max=0.5,
        window_glow_px=0.0,
        height_defaults={"house": 4.5, "retail": 6.5, "default": 5.0},
    )


@pytest.fixture
def building_config_jittered(building_config: BuildingConfig) -> BuildingConfig:
    """`building_config` with height_jitter enabled, for jitter-scope tests."""
    return dataclasses.replace(building_config, height_jitter=0.2)


@pytest.fixture
def rng() -> random.Random:
    """Seeded RNG for reproducible test runs."""
    return random.Random(42)


@pytest.fixture
def house_config() -> HouseConfig:
    """Minimal HouseConfig for constructing real House objects."""
    return HouseConfig(
        exterior_colors=ColorPoolConfig(colors=[("#0D0A08", 1.0)]),
        yard_base_color="#060A04",
        yard_variation=2,
        min_lights=0,
        max_lights=1,
        light_weights=[1.0, 1.0],
        footprint_coverage=0.65,
        lot_width=15.0,
        lot_depth=25.0,
        setback=10.0,
    )


def _make_buildings_gdf(polygon, building_tag="retail", height=None, levels=None):
    """Build a minimal OSM buildings GeoDataFrame for one footprint."""
    return gpd.GeoDataFrame(
        {
            "geometry": [polygon],
            "building": [building_tag],
            "height": [height],
            "building:levels": [levels],
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Pure parser — height tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("12", 12.0),
        ("12 m", 12.0),
        ("39'", 39 * 0.3048),
        ("abc", None),
        (None, None),
        (float("nan"), None),
        ("", None),
    ],
)
def test_parse_height_tag_table(value, expected):
    result = _parse_height_tag(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Pure parser — levels tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("3", 3.0),
        ("2.5", 2.5),
        ("abc", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_levels_tag_table(value, expected):
    result = _parse_levels_tag(value)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Height resolution — fallback order
# ---------------------------------------------------------------------------


def test_resolve_building_height_uses_height_tag_when_parseable(building_config, rng):
    height = _resolve_building_height("12 m", "5", "retail", building_config, rng)
    assert height == pytest.approx(12.0)


def test_resolve_building_height_falls_back_to_levels_when_height_unparseable(
    building_config, rng
):
    height = _resolve_building_height(None, "3", "retail", building_config, rng)
    assert height == pytest.approx(3 * building_config.floor_height_m)


def test_resolve_building_height_falls_back_to_type_default(building_config, rng):
    height = _resolve_building_height(None, None, "retail", building_config, rng)
    assert height == pytest.approx(building_config.height_defaults["retail"])


def test_resolve_building_height_unknown_type_uses_generic_default(building_config, rng):
    height = _resolve_building_height(None, None, "totally_unknown_tag", building_config, rng)
    assert height == pytest.approx(building_config.height_defaults["default"])


def test_resolve_building_height_parsed_height_tag_not_jittered(
    building_config_jittered, rng
):
    height = _resolve_building_height("12 m", None, "retail", building_config_jittered, rng)
    assert height == pytest.approx(12.0)


def test_resolve_building_height_levels_derived_height_not_jittered(
    building_config_jittered, rng
):
    height = _resolve_building_height(
        None, "3", "retail", building_config_jittered, rng
    )
    assert height == pytest.approx(3 * building_config_jittered.floor_height_m)


def test_resolve_building_height_type_default_is_jittered(building_config_jittered, rng):
    raw_default = building_config_jittered.height_defaults["retail"]
    jitter = building_config_jittered.height_jitter

    height = _resolve_building_height(None, None, "retail", building_config_jittered, rng)

    assert height != pytest.approx(raw_default)
    assert raw_default * (1.0 - jitter) <= height <= raw_default * (1.0 + jitter)


def test_apply_height_jitter_disabled_returns_unchanged():
    rng_local = random.Random(1)
    assert _apply_height_jitter(10.0, 0.0, rng_local) == pytest.approx(10.0)


def test_apply_height_jitter_stays_within_configured_bounds():
    rng_local = random.Random(1)
    result = _apply_height_jitter(10.0, 0.12, rng_local)
    assert 8.8 <= result <= 11.2


# ---------------------------------------------------------------------------
# ENU conversion sanity
# ---------------------------------------------------------------------------


def test_build_buildings_converts_osm_footprint_to_enu(transformer, building_config, rng):
    polygon = box(-117.301, 34.101, -117.300, 34.102)
    gdf = _make_buildings_gdf(polygon)

    buildings = build_buildings(gdf, [], transformer, building_config, rng)

    assert len(buildings) == 1
    footprint = buildings[0].footprint
    assert footprint.shape == (4, 2)
    # Footprint should be in ENU meters (hundreds of meters from the origin),
    # not still in WGS84 degree units.
    assert np.abs(footprint).max() > 10.0


def test_build_buildings_footprint_has_no_closing_duplicate(transformer, building_config, rng):
    polygon = box(-117.301, 34.101, -117.300, 34.102)
    gdf = _make_buildings_gdf(polygon)

    buildings = build_buildings(gdf, [], transformer, building_config, rng)

    footprint = buildings[0].footprint
    assert not np.allclose(footprint[0], footprint[-1])


def test_build_buildings_footprint_is_ccw(transformer, building_config, rng):
    polygon = box(-117.301, 34.101, -117.300, 34.102)
    gdf = _make_buildings_gdf(polygon)

    buildings = build_buildings(gdf, [], transformer, building_config, rng)

    footprint = buildings[0].footprint
    x = footprint[:, 0]
    y = footprint[:, 1]
    signed_area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert signed_area > 0


def test_build_buildings_uses_resolved_height_from_tag(transformer, building_config, rng):
    polygon = box(-117.301, 34.101, -117.300, 34.102)
    gdf = _make_buildings_gdf(polygon, height="20 m")

    buildings = build_buildings(gdf, [], transformer, building_config, rng)

    assert buildings[0].height == pytest.approx(20.0)


def test_build_buildings_disabled_returns_empty_list(transformer, building_config, rng):
    polygon = box(-117.301, 34.101, -117.300, 34.102)
    gdf = _make_buildings_gdf(polygon)
    building_config.enabled = False

    buildings = build_buildings(gdf, [], transformer, building_config, rng)

    assert buildings == []


def test_build_buildings_none_gdf_with_no_zones_returns_empty_list(
    transformer, building_config, rng
):
    buildings = build_buildings(None, [], transformer, building_config, rng)
    assert buildings == []


# ---------------------------------------------------------------------------
# House-footprint dedupe
# ---------------------------------------------------------------------------


def test_filter_houses_without_osm_match_drops_overlapping_house():
    osm_polygons = [box(0, 0, 20, 20)]
    house_polygons = [box(5, 5, 10, 10)]

    result = _filter_houses_without_osm_match(house_polygons, osm_polygons)

    assert result == []


def test_filter_houses_without_osm_match_keeps_non_overlapping_house():
    osm_polygons = [box(0, 0, 20, 20)]
    house_polygons = [box(100, 100, 110, 110)]

    result = _filter_houses_without_osm_match(house_polygons, osm_polygons)

    assert result == house_polygons


def test_filter_houses_without_osm_match_no_osm_footprints_keeps_all_houses():
    house_polygons = [box(100, 100, 110, 110), box(200, 200, 210, 210)]

    result = _filter_houses_without_osm_match(house_polygons, [])

    assert result == house_polygons


def test_build_house_buildings_skips_house_overlapping_osm_building(
    building_config, rng, house_config
):
    osm_building = Building(
        footprint=np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
        height=6.5,
        building_kind="retail",
        window_seed=1,
    )
    overlapping_house = House(footprint=box(5, 5, 10, 10), config=house_config, rng=rng)
    zone = ResidentialZone(boundary=box(-100, -100, 300, 300), houses=[overlapping_house])

    result = _build_house_buildings([zone], [osm_building], building_config, rng)

    assert result == []


def test_build_house_buildings_includes_house_without_osm_overlap(
    building_config, rng, house_config
):
    osm_building = Building(
        footprint=np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]]),
        height=6.5,
        building_kind="retail",
        window_seed=1,
    )
    distant_house = House(footprint=box(100, 100, 110, 110), config=house_config, rng=rng)
    zone = ResidentialZone(boundary=box(-100, -100, 300, 300), houses=[distant_house])

    result = _build_house_buildings([zone], [osm_building], building_config, rng)

    assert len(result) == 1
    assert result[0].building_kind == "house"
    assert result[0].height == pytest.approx(building_config.height_defaults["house"])


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------


def test_build_buildings_reproducible_with_same_seed(transformer, building_config):
    polygon = box(-117.301, 34.101, -117.300, 34.102)
    gdf = _make_buildings_gdf(polygon, height=None, levels=None)

    buildings_a = build_buildings(gdf, [], transformer, building_config, random.Random(7))
    buildings_b = build_buildings(gdf, [], transformer, building_config, random.Random(7))

    assert buildings_a[0].height == pytest.approx(buildings_b[0].height)
    assert buildings_a[0].window_seed == buildings_b[0].window_seed
