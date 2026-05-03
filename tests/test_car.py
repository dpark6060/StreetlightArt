# Radical
"""Tests for streets_at_night.models.car.CarLightPair."""

import math
import random

import numpy as np
import pytest

from streets_at_night.config.schema import CarConfig, ColorPoolConfig
from streets_at_night.models.car import CarLightPair
from streets_at_night.models.light_source import LightSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def road_position() -> np.ndarray:
    """Return a center road position in ENU meters."""
    return np.array([500.0, 300.0, 0.0])


@pytest.fixture
def car_config() -> CarConfig:
    """Return a CarConfig built from DEFAULTS values."""
    return CarConfig(
        headlight_colors=ColorPoolConfig(colors=[
            ("#FFFFFF", 0.5),
            ("#FFF4E0", 0.3),
            ("#FFEE80", 0.15),
            ("#CCE0FF", 0.05),
        ]),
        taillight_colors=ColorPoolConfig(colors=[
            ("#FF2020", 0.6),
            ("#CC1010", 0.25),
            ("#FF3010", 0.15),
        ]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_radius=15.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        headlight_illumination_inner_fraction=0.3,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
    )


@pytest.fixture
def car_pair(road_position, rng, car_config) -> CarLightPair:
    """Return a CarLightPair traveling East (bearing=0)."""
    return CarLightPair(
        position=road_position,
        travel_bearing=0.0,
        config=car_config,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# __init__ — always exactly two lights
# ---------------------------------------------------------------------------


def test_get_lights_returns_exactly_two(car_pair):
    # Act
    lights = car_pair.get_lights()

    # Assert
    assert len(lights) == 2


def test_get_lights_are_light_source_instances(car_pair):
    # Act
    lights = car_pair.get_lights()

    # Assert
    assert all(isinstance(light, LightSource) for light in lights)


def test_travel_bearing_stored(road_position, rng, car_config):
    # Arrange
    bearing = math.pi / 4

    # Act
    pair = CarLightPair(
        position=road_position, travel_bearing=bearing, config=car_config, rng=rng
    )

    # Assert
    assert pair.travel_bearing == bearing


# ---------------------------------------------------------------------------
# __init__ — light positions are offset perpendicular to bearing
# ---------------------------------------------------------------------------


def test_light_positions_differ_from_center(car_pair, road_position):
    # Act
    lights = car_pair.get_lights()

    # Assert — at least one light must be offset from the road center
    positions = [light.position for light in lights]
    same_as_center = [
        np.allclose(p[:2], road_position[:2]) for p in positions
    ]
    assert not all(same_as_center)


def test_light_positions_symmetric_about_center(car_pair, road_position):
    # Act
    lights = car_pair.get_lights()

    # Assert — midpoint of the two lights equals the road center position
    midpoint = (lights[0].position + lights[1].position) / 2.0
    np.testing.assert_allclose(midpoint[:2], road_position[:2], atol=1e-9)


def test_light_separation_within_range(car_config):
    # Arrange — run many samples to verify separation is within config range
    for seed in range(30):
        rng = random.Random(seed)
        pos = np.array([0.0, 0.0, 0.0])
        pair = CarLightPair(position=pos, travel_bearing=0.0, config=car_config, rng=rng)
        lights = pair.get_lights()
        separation = np.linalg.norm(lights[0].position - lights[1].position)

        # Assert
        assert (
            car_config.separation_min <= separation <= car_config.separation_max
        ), f"seed={seed} separation={separation}"


# ---------------------------------------------------------------------------
# resolve_facing — headlights when traveling toward camera
# ---------------------------------------------------------------------------


def test_resolve_facing_toward_camera_sets_bright_glow(car_pair):
    # Arrange — car travels East, camera also looks East → dot > 0 → headlights
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — headlights have large glow radius
    assert all(light.glow_radius == 8.0 for light in lights)


def test_resolve_facing_toward_camera_sets_brightness_range(car_pair, car_config):
    # Arrange
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — headlight brightness within config range
    assert all(
        car_config.headlight_brightness_min <= light.brightness <= car_config.headlight_brightness_max
        for light in lights
    )


# ---------------------------------------------------------------------------
# resolve_facing — taillights when traveling away from camera
# ---------------------------------------------------------------------------


def test_resolve_facing_away_from_camera_sets_dim_glow(car_pair):
    # Arrange — car travels East, camera looks West → dot < 0 → taillights
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — taillights have small glow radius
    assert all(light.glow_radius == 3.0 for light in lights)


def test_resolve_facing_away_from_camera_sets_brightness_range(car_pair, car_config):
    # Arrange
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — taillight brightness within config range
    assert all(
        car_config.taillight_brightness_min <= light.brightness <= car_config.taillight_brightness_max
        for light in lights
    )


# ---------------------------------------------------------------------------
# resolve_facing — both lights get the same color
# ---------------------------------------------------------------------------


def test_resolve_facing_headlights_both_same_color(car_pair):
    # Arrange
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert
    assert lights[0].color == lights[1].color


def test_resolve_facing_taillights_both_same_color(car_pair):
    # Arrange
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert
    assert lights[0].color == lights[1].color


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_light_positions(car_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])
    pair_a = CarLightPair(
        position=pos, travel_bearing=0.5, config=car_config, rng=random.Random(99)
    )
    pair_b = CarLightPair(
        position=pos, travel_bearing=0.5, config=car_config, rng=random.Random(99)
    )

    # Act
    lights_a = pair_a.get_lights()
    lights_b = pair_b.get_lights()

    # Assert
    for la, lb in zip(lights_a, lights_b):
        np.testing.assert_array_equal(la.position, lb.position)


# ---------------------------------------------------------------------------
# Config passthrough
# ---------------------------------------------------------------------------


def test_headlight_color_comes_from_config():
    # Arrange — single-entry pool so we know exactly which color will be picked
    single_color_config = CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#AABBCC", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#FF0000", 1.0)]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_radius=15.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
        headlight_illumination_inner_fraction=0.3,
    )
    rng = random.Random(0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=single_color_config,
        rng=rng,
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — headlight color matches the only color in the pool
    expected = single_color_config.headlight_colors.colors[0]
    expected_hex = expected[0]
    r = int(expected_hex[1:3], 16)
    g = int(expected_hex[3:5], 16)
    b = int(expected_hex[5:7], 16)
    assert lights[0].color == (r, g, b)


def test_taillight_color_comes_from_config():
    # Arrange — single-entry pool so we know exactly which color will be picked
    single_color_config = CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#FFFFFF", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#DD1122", 1.0)]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_radius=15.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
        headlight_illumination_inner_fraction=0.3,
    )
    rng = random.Random(0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=single_color_config,
        rng=rng,
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — taillight color matches the only color in the pool
    expected = single_color_config.taillight_colors.colors[0]
    expected_hex = expected[0]
    r = int(expected_hex[1:3], 16)
    g = int(expected_hex[3:5], 16)
    b = int(expected_hex[5:7], 16)
    assert lights[0].color == (r, g, b)


def test_separation_respects_config_range():
    # Arrange — tight separation range so any sample must fall within it
    tight_config = CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#FFFFFF", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#FF0000", 1.0)]),
        separation_min=3.0,
        separation_max=3.1,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_radius=15.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
        headlight_illumination_inner_fraction=0.3,
    )
    for seed in range(20):
        rng = random.Random(seed)
        pair = CarLightPair(
            position=np.array([0.0, 0.0, 0.0]),
            travel_bearing=0.0,
            config=tight_config,
            rng=rng,
        )
        lights = pair.get_lights()
        separation = np.linalg.norm(lights[0].position - lights[1].position)

        # Assert
        assert (
            tight_config.separation_min <= separation <= tight_config.separation_max
        ), f"seed={seed} separation={separation}"


# ---------------------------------------------------------------------------
# Config-driven glow/illumination properties
# ---------------------------------------------------------------------------


def _make_config_with_light_overrides(
    headlight_glow_radius: float = 8.0,
    headlight_illumination_radius: float = 15.0,
    headlight_illumination_intensity: float = 0.6,
    headlight_illumination_falloff: float = 1.5,
    headlight_illumination_inner_fraction: float = 0.3,
    taillight_glow_radius: float = 3.0,
    taillight_illumination_radius: float = 2.0,
    taillight_illumination_intensity: float = 0.1,
    taillight_illumination_falloff: float = 2.0,
    taillight_illumination_inner_fraction: float = 0.0,
    headlight_brightness_min: float = 0.7,
    headlight_brightness_max: float = 0.9,
    taillight_brightness_min: float = 0.3,
    taillight_brightness_max: float = 0.5,
) -> CarConfig:
    """Build a CarConfig with controllable glow/illumination and brightness fields."""
    return CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#FFFFFF", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#FF0000", 1.0)]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=headlight_brightness_min,
        headlight_brightness_max=headlight_brightness_max,
        taillight_brightness_min=taillight_brightness_min,
        taillight_brightness_max=taillight_brightness_max,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=headlight_glow_radius,
        headlight_illumination_radius=headlight_illumination_radius,
        headlight_illumination_intensity=headlight_illumination_intensity,
        headlight_illumination_falloff=headlight_illumination_falloff,
        headlight_illumination_inner_fraction=headlight_illumination_inner_fraction,
        taillight_glow_radius=taillight_glow_radius,
        taillight_illumination_radius=taillight_illumination_radius,
        taillight_illumination_intensity=taillight_illumination_intensity,
        taillight_illumination_falloff=taillight_illumination_falloff,
        taillight_illumination_inner_fraction=taillight_illumination_inner_fraction,
    )


def test_headlight_glow_radius_comes_from_config():
    # Arrange — use non-default value to confirm it is read from config
    config = _make_config_with_light_overrides(headlight_glow_radius=20.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert
    assert all(light.glow_radius == 20.0 for light in lights)


def test_headlight_illumination_radius_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(headlight_illumination_radius=40.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert
    assert all(light.illumination_radius == 40.0 for light in lights)


def test_headlight_illumination_intensity_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(headlight_illumination_intensity=0.9)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert
    assert all(light.illumination_intensity == 0.9 for light in lights)


def test_taillight_glow_radius_comes_from_config():
    # Arrange — use non-default value to confirm it is read from config
    config = _make_config_with_light_overrides(taillight_glow_radius=10.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert
    assert all(light.glow_radius == 10.0 for light in lights)


def test_taillight_illumination_radius_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(taillight_illumination_radius=7.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert
    assert all(light.illumination_radius == 7.0 for light in lights)


def test_taillight_illumination_intensity_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(taillight_illumination_intensity=0.05)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert
    assert all(light.illumination_intensity == 0.05 for light in lights)


# ---------------------------------------------------------------------------
# Brightness config threading
# ---------------------------------------------------------------------------


def test_headlight_brightness_comes_from_config():
    # Arrange — tight range so any sampled value must fall within it
    config = _make_config_with_light_overrides(
        headlight_brightness_min=0.85,
        headlight_brightness_max=0.86,
    )
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — brightness falls within the tight configured range
    assert all(0.85 <= light.brightness <= 0.86 for light in lights)


def test_taillight_brightness_comes_from_config():
    # Arrange — tight range so any sampled value must fall within it
    config = _make_config_with_light_overrides(
        taillight_brightness_min=0.35,
        taillight_brightness_max=0.36,
    )
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — brightness falls within the tight configured range
    assert all(0.35 <= light.brightness <= 0.36 for light in lights)
