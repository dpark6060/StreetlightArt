# Radical
"""Tests for devmode synthetic scene construction and config round-trip."""

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.config.schema import (
    AtmosphereConfig,
    BoundsConfig,
    CameraConfig,
    LightConfig,
    QualityPreset,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    SceneConfig,
    ZoneColorsConfig,
    ZoneConfig,
    HouseConfig,
    ColorPoolConfig,
)
from streets_at_night.devmode.config_writer import config_to_dict
from streets_at_night.devmode.synthetic_scene import (
    ROW_LABELS,
    build_synthetic_scene,
)
from streets_at_night.models.road_types import RoadType
from streets_at_night.scene.scene import Scene


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def road_style() -> RoadStyleConfig:
    """Return a minimal road style config for testing."""
    entry = RoadStyleEntry(color="#101010", width=1)
    return RoadStyleConfig(
        highway=entry,
        major_road=entry,
        collector=entry,
        residential=entry,
        service=entry,
        minor=entry,
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
            max_lights=3,
            light_weights=[1.0, 1.0, 1.0, 0.5, 0.3, 0.1],
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
    road_style: RoadStyleConfig,
) -> SceneConfig:
    """Return a minimal SceneConfig for devmode tests."""
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
        lights=dataclass_from_dict(LightConfig, DEFAULTS["lights"]),
        roads=RoadsConfig(export=road_style, preview=road_style),
        atmosphere=dataclass_from_dict(AtmosphereConfig, DEFAULTS["atmosphere"]),
        devmode=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def config_with_buildings() -> SceneConfig:
    """Return a full default SceneConfig including a [buildings] section."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


def test_build_synthetic_scene_returns_scene(scene_config: SceneConfig) -> None:
    """build_synthetic_scene returns a Scene with lights populated."""

    result = build_synthetic_scene(scene_config)

    assert isinstance(result, Scene)
    assert len(result.lights) > 0


def test_synthetic_scene_has_buildings_when_configured(
    config_with_buildings: SceneConfig,
) -> None:
    """The Houses row adds buildings when the config has a [buildings] section."""

    result = build_synthetic_scene(config_with_buildings)

    assert len(result.buildings) > 0


def test_synthetic_scene_has_no_buildings_when_unconfigured(
    scene_config: SceneConfig,
) -> None:
    """A config without a [buildings] section produces a scene with no buildings."""

    result = build_synthetic_scene(scene_config)

    assert result.buildings == []


def test_synthetic_scene_buildings_have_mixed_heights(
    config_with_buildings: SceneConfig,
) -> None:
    """The Houses row mixes short houses and taller apartment blocks."""

    result = build_synthetic_scene(config_with_buildings)

    heights = [b.height for b in result.buildings]
    assert any(h < 5.0 for h in heights), "expected at least one short house"
    assert any(h > 10.0 for h in heights), "expected at least one tall apartment block"


def test_synthetic_building_footprints_are_ccw(
    config_with_buildings: SceneConfig,
) -> None:
    """Synthetic footprints wind counter-clockwise (positive shoelace area)."""

    result = build_synthetic_scene(config_with_buildings)

    footprint = result.buildings[0].footprint
    x = footprint[:, 0]
    y = footprint[:, 1]
    signed_area = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert signed_area > 0.0


def test_synthetic_scene_buildings_row_adds_no_road_segment(
    config_with_buildings: SceneConfig,
) -> None:
    """The Houses row does not add an eleventh road segment (still 10)."""

    result = build_synthetic_scene(config_with_buildings)

    assert len(result.road_segments) == 10


def test_synthetic_scene_has_correct_road_segments(scene_config: SceneConfig) -> None:
    """build_synthetic_scene creates exactly 10 road segments (rows 0-9)."""

    result = build_synthetic_scene(scene_config)

    assert len(result.road_segments) == 10


def test_road_segments_cover_all_five_road_types(scene_config: SceneConfig) -> None:
    """The first 5 road segments cover all five non-MINOR road types."""

    result = build_synthetic_scene(scene_config)

    road_types = [seg.road_type for seg in result.road_segments[:5]]
    assert RoadType.HIGHWAY in road_types
    assert RoadType.MAJOR_ROAD in road_types
    assert RoadType.COLLECTOR in road_types
    assert RoadType.RESIDENTIAL in road_types
    assert RoadType.SERVICE in road_types


def test_car_row_has_no_fixture_profile(scene_config: SceneConfig) -> None:
    """The car-row segment (index 5) has fixture_profile=None."""

    result = build_synthetic_scene(scene_config)

    car_row_segment = result.road_segments[5]
    assert car_row_segment.fixture_profile is None


def test_config_writer_round_trips(scene_config: SceneConfig) -> None:
    """config_to_dict produces a dict that _build_scene_config re-loads correctly."""

    data = config_to_dict(scene_config)
    reloaded = dataclass_from_dict(SceneConfig, data)

    assert reloaded.seed == scene_config.seed
    assert reloaded.bounds.north == scene_config.bounds.north
    assert reloaded.atmosphere.falloff_distance == scene_config.atmosphere.falloff_distance
    assert reloaded.atmosphere.haze_color == scene_config.atmosphere.haze_color
    assert reloaded.devmode is None


def test_config_writer_round_trips_devmode(scene_config: SceneConfig) -> None:
    """config_to_dict preserves devmode section when present."""
    from streets_at_night.config.schema import DevmodeCameraConfig, DevmodeConfig

    scene_config.devmode = DevmodeConfig(
        camera=DevmodeCameraConfig(
            pos_x=-500.0,
            pos_y=0.0,
            pos_z=800.0,
            target_x=5000.0,
            target_y=0.0,
            target_z=0.0,
            fov=60.0,
            near=100.0,
            far=60000.0,
        ),
        render_width=896,
        render_height=720,
    )

    data = config_to_dict(scene_config)
    reloaded = dataclass_from_dict(SceneConfig, data)

    assert reloaded.devmode is not None
    assert reloaded.devmode.render_width == 896
    assert reloaded.devmode.camera.pos_x == -500.0

# ---------------------------------------------------------------------------
# _pick_first_real_profile
# ---------------------------------------------------------------------------


def test_pick_first_real_profile_returns_profile_for_highway(scene_config: SceneConfig) -> None:
    """_pick_first_real_profile returns a FixtureProfile for HIGHWAY road type."""
    import random
    from streets_at_night.devmode.synthetic_scene import _pick_first_real_profile
    from streets_at_night.models.road_types import RoadType
    from streets_at_night.models.streetlight import FixtureProfile

    rng = random.Random(1)

    result = _pick_first_real_profile(RoadType.HIGHWAY, scene_config, rng)

    assert isinstance(result, FixtureProfile)


def test_pick_first_real_profile_returns_profile_for_residential(scene_config: SceneConfig) -> None:
    """_pick_first_real_profile returns a FixtureProfile even when the first pool entry is no-lights."""
    import random
    from streets_at_night.devmode.synthetic_scene import _pick_first_real_profile
    from streets_at_night.models.road_types import RoadType
    from streets_at_night.models.streetlight import FixtureProfile

    # Residential pool has a no-lights first entry — function must skip it
    rng = random.Random(1)

    result = _pick_first_real_profile(RoadType.RESIDENTIAL, scene_config, rng)

    assert isinstance(result, FixtureProfile)


def test_pick_first_real_profile_returns_none_when_all_entries_are_no_lights(
    scene_config: SceneConfig,
) -> None:
    """_pick_first_real_profile returns None when every pool entry has no profile."""
    import random
    from streets_at_night.config.schema import FixturePoolEntry
    from streets_at_night.devmode.synthetic_scene import _pick_first_real_profile
    from streets_at_night.models.road_types import RoadType

    rng = random.Random(1)
    # Replace all highway entries with no-lights entries
    scene_config.lights.streetlights.highway = [
        FixturePoolEntry(weight=1.0, profile=None),
    ]

    result = _pick_first_real_profile(RoadType.HIGHWAY, scene_config, rng)

    assert result is None


def test_pick_first_real_profile_profile_has_correct_glow_radius(scene_config: SceneConfig) -> None:
    """_pick_first_real_profile returns a profile whose glow_radius matches config."""
    import random
    from streets_at_night.devmode.synthetic_scene import _pick_first_real_profile
    from streets_at_night.models.road_types import RoadType

    rng = random.Random(1)
    # Set a known glow_radius on the first highway profile
    scene_config.lights.streetlights.highway[0].profile.glow_radius = 55.0

    result = _pick_first_real_profile(RoadType.HIGHWAY, scene_config, rng)

    assert result.glow_radius == 55.0


# ---------------------------------------------------------------------------
# _force_lights_on_dark_segments
# ---------------------------------------------------------------------------


def test_force_lights_on_dark_segments_returns_empty_when_all_segments_already_lit(
    scene_config: SceneConfig,
) -> None:
    """_force_lights_on_dark_segments returns no lights when all segments have profiles."""
    import random

    import numpy as np

    from streets_at_night.devmode.synthetic_scene import _force_lights_on_dark_segments
    from streets_at_night.models.road_segment import RoadSegment
    from streets_at_night.models.road_types import RoadType
    from streets_at_night.models.streetlight import FixtureProfile

    rng = random.Random(1)
    profile = FixtureProfile(
        fixture_color=(255, 200, 100),
        height=8.0,
        brightness=0.7,
        glow_radius=12.0,
        beam_angle_deg=140.0,
        illumination_intensity=0.4,
        spacing=40.0,
        side="both",
    )
    geom = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    segment = RoadSegment(geometry=geom, road_type=RoadType.HIGHWAY, width=12.0, fixture_profile=profile)

    result = _force_lights_on_dark_segments([segment], scene_config, rng)

    assert result == []


def test_force_lights_on_dark_segments_places_lights_on_dark_segment(
    scene_config: SceneConfig,
) -> None:
    """_force_lights_on_dark_segments returns lights for a segment with fixture_profile=None."""
    import random

    import numpy as np

    from streets_at_night.devmode.synthetic_scene import _force_lights_on_dark_segments
    from streets_at_night.models.road_segment import RoadSegment
    from streets_at_night.models.road_types import RoadType

    rng = random.Random(1)
    geom = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
    segment = RoadSegment(geometry=geom, road_type=RoadType.HIGHWAY, width=12.0, fixture_profile=None)

    result = _force_lights_on_dark_segments([segment], scene_config, rng)

    assert len(result) > 0


def test_force_lights_on_dark_segments_assigns_fixture_profile_to_segment(
    scene_config: SceneConfig,
) -> None:
    """_force_lights_on_dark_segments sets fixture_profile on previously-dark segment."""
    import random

    import numpy as np

    from streets_at_night.devmode.synthetic_scene import _force_lights_on_dark_segments
    from streets_at_night.models.road_segment import RoadSegment
    from streets_at_night.models.road_types import RoadType
    from streets_at_night.models.streetlight import FixtureProfile

    rng = random.Random(1)
    geom = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
    segment = RoadSegment(geometry=geom, road_type=RoadType.RESIDENTIAL, width=4.0, fixture_profile=None)

    _force_lights_on_dark_segments([segment], scene_config, rng)

    assert isinstance(segment.fixture_profile, FixtureProfile)


def test_force_lights_on_dark_segments_skips_segment_with_all_no_lights_pool(
    scene_config: SceneConfig,
) -> None:
    """_force_lights_on_dark_segments returns no lights for a road type whose entire pool is no-lights."""
    import random

    import numpy as np

    from streets_at_night.config.schema import FixturePoolEntry
    from streets_at_night.devmode.synthetic_scene import _force_lights_on_dark_segments
    from streets_at_night.models.road_segment import RoadSegment
    from streets_at_night.models.road_types import RoadType

    rng = random.Random(1)
    # Empty the highway pool entirely
    scene_config.lights.streetlights.highway = [FixturePoolEntry(weight=1.0, profile=None)]
    geom = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
    segment = RoadSegment(geometry=geom, road_type=RoadType.HIGHWAY, width=12.0, fixture_profile=None)

    result = _force_lights_on_dark_segments([segment], scene_config, rng)

    assert result == []


def test_all_road_rows_have_lights_after_build_synthetic_scene(scene_config: SceneConfig) -> None:
    """After build_synthetic_scene, every road-type segment (indices 0-4) has a fixture_profile."""
    from streets_at_night.devmode.synthetic_scene import build_synthetic_scene
    from streets_at_night.models.streetlight import FixtureProfile

    scene = build_synthetic_scene(scene_config)

    for i, segment in enumerate(scene.road_segments[:5]):
        assert isinstance(segment.fixture_profile, FixtureProfile), (
            f"Road segment {i} ({segment.road_type}) has no fixture_profile after build"
        )


def test_direct_light_row_segments_have_no_fixture_profile(scene_config: SceneConfig) -> None:
    """Segments 6-9 (direct light rows) have fixture_profile=None."""

    result = build_synthetic_scene(scene_config)

    for i in range(6, 10):
        assert result.road_segments[i].fixture_profile is None, (
            f"Direct light row segment {i} should have fixture_profile=None"
        )


def test_direct_light_row_segments_are_residential_type(scene_config: SceneConfig) -> None:
    """Segments 6-9 use RoadType.RESIDENTIAL."""

    result = build_synthetic_scene(scene_config)

    for i in range(6, 10):
        assert result.road_segments[i].road_type == RoadType.RESIDENTIAL, (
            f"Direct light row segment {i} should be RESIDENTIAL"
        )


def test_direct_light_row_segments_have_wide_width(scene_config: SceneConfig) -> None:
    """Segments 6-9 have a wider-than-default width to showcase illumination pools."""
    from streets_at_night.models.road_segment import get_default_width

    result = build_synthetic_scene(scene_config)

    default_residential_width = get_default_width(RoadType.RESIDENTIAL)
    for i in range(6, 10):
        assert result.road_segments[i].width > default_residential_width, (
            f"Direct light row segment {i} width should exceed default residential width"
        )


def test_direct_light_row_segments_match_row_y_offsets(scene_config: SceneConfig) -> None:
    """Segments 6-9 Y positions match the ROW_LABELS y offsets for rows 6-9."""

    result = build_synthetic_scene(scene_config)

    for i in range(6, 10):
        expected_y = ROW_LABELS[i][1]
        actual_y = result.road_segments[i].geometry[0, 1]
        assert actual_y == expected_y, (
            f"Segment {i} Y={actual_y} does not match ROW_LABELS[{i}] Y={expected_y}"
        )
