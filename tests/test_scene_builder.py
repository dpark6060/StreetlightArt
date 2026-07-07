# Radical
"""Unit tests for streets_at_night.placement.scene_builder.populate_lights."""

import random
from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_light_config
from streets_at_night.config.schema import (
    AtmosphereConfig,
    BoundsConfig,
    CameraConfig,
    ColorPoolConfig,
    HouseConfig,
    LightConfig,
    QualityPreset,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    SceneConfig,
    ZoneColorsConfig,
    ZoneConfig,
)
from streets_at_night.models.illumination import ConeIllumination
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.road_segment import RoadSegment
from streets_at_night.models.road_types import RoadType
from streets_at_night.placement.scene_builder import populate_lights


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def light_config() -> LightConfig:
    """Return a default LightConfig from project defaults."""
    return _build_light_config(DEFAULTS["lights"])


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
        tile_resolution_px=64,
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


def _make_mock_scene(segments: list, scene_config: SceneConfig) -> mock.MagicMock:
    """Create a mock Scene with the given segments and config."""
    scene = mock.MagicMock()
    scene.road_segments = segments
    scene.config = scene_config
    scene.commercial_zones = []
    scene.residential_zones = []
    return scene


def _make_straight_segment(road_type: RoadType, length: float = 500.0) -> RoadSegment:
    """Return a straight road segment of given length along the X axis."""
    return RoadSegment(
        geometry=np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]]),
        road_type=road_type,
        width=8.0,
    )


# ---------------------------------------------------------------------------
# populate_lights — calls all placers
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_calls_streetlight_placer(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    mock_streetlights.return_value = []
    mock_cars.return_value = ([], [])
    mock_commercial.return_value = []
    mock_house.return_value = []
    segments = [_make_straight_segment(RoadType.HIGHWAY)]
    scene = _make_mock_scene(segments, scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert
    mock_streetlights.assert_called_once()


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_calls_car_placer(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    mock_streetlights.return_value = []
    mock_cars.return_value = ([], [])
    mock_commercial.return_value = []
    mock_house.return_value = []
    segments = [_make_straight_segment(RoadType.HIGHWAY)]
    scene = _make_mock_scene(segments, scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert
    mock_cars.assert_called_once()


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_calls_commercial_placer(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    mock_streetlights.return_value = []
    mock_cars.return_value = ([], [])
    mock_commercial.return_value = []
    mock_house.return_value = []
    scene = _make_mock_scene([], scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert
    mock_commercial.assert_called_once()


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_calls_house_light_placer(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    mock_streetlights.return_value = []
    mock_cars.return_value = ([], [])
    mock_commercial.return_value = []
    mock_house.return_value = []
    scene = _make_mock_scene([], scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert
    mock_house.assert_called_once()


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_adds_all_lights_to_scene(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    street_light = LightSource(
        position=np.array([0.0, 0.0, 9.0]),
        color=(255, 160, 64),
        brightness=0.9,
        size=0.3,
        glow_radius=25.0,
        illumination_intensity=0.85,
        illumination=ConeIllumination(40.0, 2.0, 0.0),
    )
    car_light = LightSource(
        position=np.array([10.0, 0.0, 0.5]),
        color=(255, 255, 255),
        brightness=0.8,
        size=0.15,
        glow_radius=8.0,
        illumination_intensity=0.6,
        illumination=ConeIllumination(15.0, 2.0, 0.0),
    )
    mock_streetlights.return_value = [street_light]
    mock_cars.return_value = ([car_light], [])
    mock_commercial.return_value = []
    mock_house.return_value = []
    scene = _make_mock_scene([], scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert — add_lights called for each placer's result
    assert scene.add_lights.call_count == 4


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_adds_car_pairs_to_scene(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    mock_pair = mock.MagicMock()
    mock_streetlights.return_value = []
    mock_cars.return_value = ([], [mock_pair])
    mock_commercial.return_value = []
    mock_house.return_value = []
    scene = _make_mock_scene([], scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert — car pairs are registered with the scene
    scene.add_car_pairs.assert_called_once_with([mock_pair])


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_passes_road_segments_to_streetlight_placer(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange
    mock_streetlights.return_value = []
    mock_cars.return_value = ([], [])
    mock_commercial.return_value = []
    mock_house.return_value = []
    segments = [_make_straight_segment(RoadType.MAJOR_ROAD)]
    scene = _make_mock_scene(segments, scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert — road segments forwarded to streetlight placer
    call_args = mock_streetlights.call_args[0]
    assert call_args[0] is segments


@mock.patch("streets_at_night.placement.scene_builder.place_streetlights")
@mock.patch("streets_at_night.placement.scene_builder.place_cars")
@mock.patch("streets_at_night.placement.scene_builder.place_commercial_lights")
@mock.patch("streets_at_night.placement.scene_builder.collect_house_lights")
def test_populate_lights_uses_independent_child_rngs(
    mock_house, mock_commercial, mock_cars, mock_streetlights, scene_config
):
    # Arrange — capture the RNG passed to each placer
    captured_rngs = []
    mock_streetlights.side_effect = lambda segs, cfg, rng: captured_rngs.append(rng) or []
    mock_cars.side_effect = lambda segs, cfg, rng: captured_rngs.append(rng) or ([], [])
    mock_commercial.side_effect = lambda zones, cfg, rng: captured_rngs.append(rng) or []
    mock_house.side_effect = lambda zones, cfg, rng: captured_rngs.append(rng) or []
    scene = _make_mock_scene([], scene_config)
    rng = random.Random(42)

    # Act
    populate_lights(scene, scene_config, rng)

    # Assert — each placer gets a distinct RNG object (child RNG, not the master)
    assert len(captured_rngs) == 4
    assert len(set(id(r) for r in captured_rngs)) == 4
