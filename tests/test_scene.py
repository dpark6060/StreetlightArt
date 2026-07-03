# Radical
"""Tests for Scene, SpatialIndex, RoadSegment, and pygame smoke tests."""

from unittest import mock
import numpy as np
import pytest
from shapely.geometry import Polygon

from streets_at_night.models.road_segment import RoadSegment, get_default_width
from streets_at_night.models.road_types import RoadType
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.zones import CommercialZone, DarkZone, ResidentialZone
from streets_at_night.scene.spatial_index import SpatialIndex
from streets_at_night.scene.scene import (
    Scene,
    _compute_combined_bounds,
    _compute_world_bounds,
    _segment_bounds,
)
from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import (
    AtmosphereConfig,
    BoundsConfig,
    CameraConfig,
    ColorPoolConfig,
    HouseConfig,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    LightConfig,
    QualityPreset,
    SceneConfig,
    ZoneColorsConfig,
    ZoneConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_geometry() -> np.ndarray:
    """Return a simple 3-point (N, 3) geometry array."""
    return np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 50.0, 0.0]])


@pytest.fixture
def road_segment(simple_geometry: np.ndarray) -> RoadSegment:
    """Return a basic highway RoadSegment."""
    return RoadSegment(
        geometry=simple_geometry,
        road_type=RoadType.HIGHWAY,
        width=12.0,
    )


@pytest.fixture
def camera() -> object:
    """Return a basic camera for testing."""
    return create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 5000.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=100.0,
        far=60000.0,
    )


@pytest.fixture
def zone_config() -> ZoneConfig:
    """Return a minimal ZoneConfig for testing."""
    return ZoneConfig(
        house=HouseConfig(
            exterior_colors=ColorPoolConfig(colors=[("#0D0A08", 1.0)]),
            yard_base_color="#060A04",
            yard_variation=2,
            min_lights=0,
            max_lights=5,
            light_weights=[5.0, 3.0, 1.5, 0.5, 0.3, 0.1],
            footprint_coverage=0.65,
            lot_width=15.0,
            lot_depth=25.0,
            setback=10.0,
        ),
        colors=ZoneColorsConfig(
            commercial="#0A0A0C",
            industrial="#080808",
            park_forest="#040804",
            water="#040408",
            road_surface="#0C0C0C",
            default_ground="#080706",
        ),
        tile_size_meters=256.0,
        tile_resolution_px=128,
    )


@pytest.fixture
def scene_config(
    zone_config: ZoneConfig,
    light_config: LightConfig,
    atmosphere_config: AtmosphereConfig,
) -> SceneConfig:
    """Return a minimal SceneConfig for testing."""
    _style = RoadStyleConfig(
        highway=RoadStyleEntry(color="#141210", width=4),
        major_road=RoadStyleEntry(color="#121010", width=3),
        collector=RoadStyleEntry(color="#100E0E", width=2),
        residential=RoadStyleEntry(color="#0E0C0C", width=1),
        service=RoadStyleEntry(color="#0C0A0A", width=1),
        minor=RoadStyleEntry(color="#0A0908", width=1),
    )
    return SceneConfig(
        camera=CameraConfig(
            lat=34.2349,
            lon=-117.3116,
            altitude_ft=5186.0,
            target_lat=34.1,
            target_lon=-117.3,
            target_altitude_ft=0.0,
            fov=60.0,
        ),
        bounds=BoundsConfig(north=34.3, south=34.0, east=-117.1, west=-117.5),
        quality_presets={"low": QualityPreset(width=640, height=360)},
        seed=42,
        zones=zone_config,
        lights=light_config,
        roads=RoadsConfig(export=_style, preview=_style),
        atmosphere=atmosphere_config,
    )


@pytest.fixture
def three_segments() -> list:
    """Return three RoadSegments spread across different spatial areas."""
    return [
        RoadSegment(
            geometry=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]),
            road_type=RoadType.HIGHWAY,
            width=12.0,
        ),
        RoadSegment(
            geometry=np.array([[1000.0, 1000.0, 0.0], [1100.0, 1000.0, 0.0]]),
            road_type=RoadType.RESIDENTIAL,
            width=4.0,
        ),
        RoadSegment(
            geometry=np.array([[5000.0, 5000.0, 0.0], [5100.0, 5000.0, 0.0]]),
            road_type=RoadType.COLLECTOR,
            width=6.0,
        ),
    ]


@pytest.fixture
def sample_light() -> LightSource:
    """Return a LightSource at a known position."""
    return LightSource(
        position=np.array([200.0, 200.0, 5.0]),
        color=(255, 200, 100),
        brightness=1.0,
        size=0.5,
        glow_radius=10.0,
        illumination_radius=20.0,
        illumination_intensity=0.8,
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
    )


@pytest.fixture
def residential_zone_near() -> ResidentialZone:
    """Return a ResidentialZone whose boundary is near the origin."""
    return ResidentialZone(
        boundary=Polygon([(50.0, 50.0), (250.0, 50.0), (250.0, 250.0), (50.0, 250.0)])
    )


@pytest.fixture
def commercial_zone_far() -> CommercialZone:
    """Return a CommercialZone whose boundary is far from the origin."""
    return CommercialZone(
        boundary=Polygon(
            [(8000.0, 8000.0), (9000.0, 8000.0), (9000.0, 9000.0), (8000.0, 9000.0)]
        )
    )


@pytest.fixture
def dark_zone_near() -> DarkZone:
    """Return a DarkZone whose boundary is near the origin."""
    return DarkZone(
        boundary=Polygon([(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]),
        zone_type="park",
        base_color=(10, 20, 10),
    )


# ---------------------------------------------------------------------------
# RoadSegment tests
# ---------------------------------------------------------------------------


def test_road_segment_stores_geometry_and_type(simple_geometry: np.ndarray) -> None:
    segment = RoadSegment(
        geometry=simple_geometry,
        road_type=RoadType.MAJOR_ROAD,
        width=8.0,
    )

    assert segment.road_type == RoadType.MAJOR_ROAD
    assert segment.width == 8.0
    np.testing.assert_array_equal(segment.geometry, simple_geometry)


def test_road_segment_fixture_profile_defaults_to_none(simple_geometry: np.ndarray) -> None:
    segment = RoadSegment(
        geometry=simple_geometry,
        road_type=RoadType.RESIDENTIAL,
        width=4.0,
    )

    assert segment.fixture_profile is None


def test_get_default_width_highway() -> None:
    result = get_default_width(RoadType.HIGHWAY)

    assert result == 12.0


def test_get_default_width_residential() -> None:
    result = get_default_width(RoadType.RESIDENTIAL)

    assert result == 4.0


def test_get_default_width_service() -> None:
    result = get_default_width(RoadType.SERVICE)

    assert result == 3.0


def test_get_default_width_all_types_return_positive() -> None:
    for road_type in RoadType:
        width = get_default_width(road_type)
        assert width > 0.0, f"{road_type} returned non-positive width"


# ---------------------------------------------------------------------------
# SpatialIndex tests
# ---------------------------------------------------------------------------


def test_spatial_index_insert_and_query_rect_returns_item() -> None:
    index = SpatialIndex(min_x=0.0, min_y=0.0, max_x=10000.0, max_y=10000.0)
    item = "road_a"

    index.insert(item, 0.0, 0.0, 100.0, 100.0)
    result = index.query_rect(0.0, 0.0, 200.0, 200.0)

    assert item in result


def test_spatial_index_query_rect_excludes_far_item() -> None:
    index = SpatialIndex(min_x=0.0, min_y=0.0, max_x=10000.0, max_y=10000.0)
    near_item = "near"
    far_item = "far"

    index.insert(near_item, 0.0, 0.0, 100.0, 100.0)
    index.insert(far_item, 9000.0, 9000.0, 9100.0, 9100.0)
    result = index.query_rect(0.0, 0.0, 200.0, 200.0)

    assert near_item in result
    assert far_item not in result


def test_spatial_index_query_radius_finds_nearby_item() -> None:
    index = SpatialIndex(min_x=0.0, min_y=0.0, max_x=10000.0, max_y=10000.0)
    item = "near"

    index.insert(item, 490.0, 490.0, 510.0, 510.0)
    result = index.query_radius(500.0, 500.0, 600.0)

    assert item in result


def test_spatial_index_get_item_count_counts_duplicates() -> None:
    index = SpatialIndex(min_x=0.0, min_y=0.0, max_x=5000.0, max_y=5000.0, cell_size=500.0)
    # An item spanning two cells should appear in both
    index.insert("span", 400.0, 0.0, 600.0, 100.0)

    assert index.get_item_count() >= 2


def test_spatial_index_empty_query_returns_empty_list() -> None:
    index = SpatialIndex(min_x=0.0, min_y=0.0, max_x=10000.0, max_y=10000.0)

    result = index.query_rect(0.0, 0.0, 100.0, 100.0)

    assert result == []


# ---------------------------------------------------------------------------
# Scene tests
# ---------------------------------------------------------------------------


def test_scene_stores_road_segments_and_camera(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene(three_segments, camera, scene_config)

    assert scene.road_segments is three_segments
    assert scene.camera is camera
    assert scene.config is scene_config


def test_scene_build_spatial_index_enables_rect_query(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene(three_segments, camera, scene_config)
    scene.build_spatial_index()

    # Query near first segment only
    result = scene.get_roads_in_rect(0.0, -50.0, 200.0, 50.0)
    result_ids = {id(s) for s in result}

    assert id(three_segments[0]) in result_ids


def test_scene_get_roads_in_rect_excludes_far_segments(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene(three_segments, camera, scene_config)
    scene.build_spatial_index()

    result = scene.get_roads_in_rect(0.0, -50.0, 200.0, 50.0)
    result_ids = {id(s) for s in result}

    assert id(three_segments[2]) not in result_ids


def test_scene_get_roads_in_rect_deduplicates_results(
    camera: object,
    scene_config: SceneConfig,
) -> None:
    # A long segment spanning many cells
    long_segment = RoadSegment(
        geometry=np.array([[0.0, 0.0, 0.0], [5000.0, 0.0, 0.0]]),
        road_type=RoadType.HIGHWAY,
        width=12.0,
    )
    scene = Scene([long_segment], camera, scene_config)
    scene.build_spatial_index()

    result = scene.get_roads_in_rect(-100.0, -100.0, 6000.0, 100.0)
    long_segment_id = id(long_segment)
    count = sum(1 for s in result if id(s) == long_segment_id)

    assert count == 1


def test_scene_get_roads_in_rect_without_index_returns_all(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene(three_segments, camera, scene_config)
    # Intentionally do not build the spatial index

    result = scene.get_roads_in_rect(0.0, 0.0, 100.0, 100.0)

    assert result is three_segments


def test_scene_build_spatial_index_noop_for_empty_scene(
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene([], camera, scene_config)

    # Should not raise
    scene.build_spatial_index()

    assert scene._spatial_index is None


# ---------------------------------------------------------------------------
# _segment_bounds and _compute_world_bounds helpers
# ---------------------------------------------------------------------------


def test_segment_bounds_returns_correct_extents(road_segment: RoadSegment) -> None:
    min_x, min_y, max_x, max_y = _segment_bounds(road_segment)

    assert min_x == 0.0
    assert min_y == 0.0
    assert max_x == 200.0
    assert max_y == 50.0


def test_compute_world_bounds_spans_all_segments(three_segments: list) -> None:
    min_x, min_y, max_x, max_y = _compute_world_bounds(three_segments)

    assert min_x == 0.0
    assert min_y == 0.0
    assert max_x == 5100.0
    assert max_y == 5000.0


def test_compute_combined_bounds_expands_for_zone_beyond_roads(
    three_segments: list,
    commercial_zone_far: CommercialZone,
) -> None:
    min_x, min_y, max_x, max_y = _compute_combined_bounds(
        three_segments, [], [commercial_zone_far], []
    )

    assert max_x == 9000.0
    assert max_y == 9000.0


def test_compute_combined_bounds_without_zones_matches_road_bounds(
    three_segments: list,
) -> None:
    min_x, min_y, max_x, max_y = _compute_combined_bounds(
        three_segments, [], [], []
    )

    assert min_x == 0.0
    assert min_y == 0.0
    assert max_x == 5100.0
    assert max_y == 5000.0


# ---------------------------------------------------------------------------
# Scene light and zone storage tests
# ---------------------------------------------------------------------------


def test_scene_initialises_empty_lights_and_car_pairs(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene(three_segments, camera, scene_config)

    assert scene.lights == []
    assert scene.car_pairs == []


def test_scene_add_lights_appends_to_lights(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    sample_light: LightSource,
) -> None:
    scene = Scene(three_segments, camera, scene_config)

    scene.add_lights([sample_light])

    assert sample_light in scene.lights


def test_scene_add_car_pairs_appends_to_car_pairs(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    scene = Scene(three_segments, camera, scene_config)
    fake_pair = object()

    scene.add_car_pairs([fake_pair])

    assert fake_pair in scene.car_pairs


def test_scene_get_lights_near_without_index_returns_all(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    sample_light: LightSource,
) -> None:
    scene = Scene(three_segments, camera, scene_config)
    scene.add_lights([sample_light])

    result = scene.get_lights_near(0.0, 0.0, 10.0)

    assert result is scene.lights


def test_scene_get_lights_near_with_index_finds_nearby_light(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    sample_light: LightSource,
) -> None:
    scene = Scene(three_segments, camera, scene_config)
    scene.build_spatial_index()
    scene.add_lights([sample_light])

    result = scene.get_lights_near(200.0, 200.0, 100.0)

    assert sample_light in result


def test_scene_get_lights_near_with_index_excludes_far_light(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
) -> None:
    far_light = LightSource(
        position=np.array([4000.0, 4000.0, 5.0]),
        color=(255, 255, 255),
        brightness=1.0,
        size=0.5,
        glow_radius=10.0,
        illumination_radius=20.0,
        illumination_intensity=0.8,
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
    )
    scene = Scene(three_segments, camera, scene_config)
    scene.build_spatial_index()
    scene.add_lights([far_light])

    result = scene.get_lights_near(0.0, 0.0, 100.0)

    assert far_light not in result


# ---------------------------------------------------------------------------
# Zone spatial index tests
# ---------------------------------------------------------------------------


def test_scene_build_spatial_index_indexes_zones(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    residential_zone_near: ResidentialZone,
) -> None:
    scene = Scene(
        three_segments,
        camera,
        scene_config,
        residential_zones=[residential_zone_near],
    )
    scene.build_spatial_index()

    result_res, _, _ = scene.get_zones_in_rect(0.0, 0.0, 300.0, 300.0)

    assert residential_zone_near in result_res


def test_scene_get_zones_in_rect_without_index_returns_all_zones(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    residential_zone_near: ResidentialZone,
    commercial_zone_far: CommercialZone,
    dark_zone_near: DarkZone,
) -> None:
    scene = Scene(
        three_segments,
        camera,
        scene_config,
        residential_zones=[residential_zone_near],
        commercial_zones=[commercial_zone_far],
        dark_zones=[dark_zone_near],
    )

    res, com, dark = scene.get_zones_in_rect(0.0, 0.0, 100.0, 100.0)

    assert res is scene.residential_zones
    assert com is scene.commercial_zones
    assert dark is scene.dark_zones


def test_scene_get_zones_in_rect_excludes_far_zone(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    residential_zone_near: ResidentialZone,
    commercial_zone_far: CommercialZone,
) -> None:
    scene = Scene(
        three_segments,
        camera,
        scene_config,
        residential_zones=[residential_zone_near],
        commercial_zones=[commercial_zone_far],
    )
    scene.build_spatial_index()

    res, com, _ = scene.get_zones_in_rect(0.0, 0.0, 500.0, 500.0)

    assert residential_zone_near in res
    assert commercial_zone_far not in com


def test_scene_get_zones_in_rect_deduplicates_zones(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    residential_zone_near: ResidentialZone,
) -> None:
    # A zone spanning many cells should appear only once
    large_zone = ResidentialZone(
        boundary=Polygon([
            (0.0, 0.0), (5000.0, 0.0), (5000.0, 5000.0), (0.0, 5000.0)
        ])
    )
    scene = Scene(
        three_segments,
        camera,
        scene_config,
        residential_zones=[large_zone],
    )
    scene.build_spatial_index()

    res, _, _ = scene.get_zones_in_rect(0.0, 0.0, 6000.0, 6000.0)
    count = sum(1 for z in res if z is large_zone)

    assert count == 1


def test_scene_build_spatial_index_covers_zone_beyond_roads(
    three_segments: list,
    camera: object,
    scene_config: SceneConfig,
    commercial_zone_far: CommercialZone,
) -> None:
    # Zone extends beyond road bounds — index must still cover it
    scene = Scene(
        three_segments,
        camera,
        scene_config,
        commercial_zones=[commercial_zone_far],
    )
    scene.build_spatial_index()

    _, com, _ = scene.get_zones_in_rect(7000.0, 7000.0, 9500.0, 9500.0)

    assert commercial_zone_far in com


# ---------------------------------------------------------------------------
# Pygame smoke test — import and headless init
# ---------------------------------------------------------------------------


def test_pygame_can_be_imported() -> None:
    import pygame

    assert pygame is not None
