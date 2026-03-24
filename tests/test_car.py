# Radical
"""Tests for streets_at_night.models.car.CarLightPair."""

import math
import random

import numpy as np
import pytest

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
def car_pair(road_position, rng) -> CarLightPair:
    """Return a CarLightPair traveling East (bearing=0)."""
    return CarLightPair(
        position=road_position,
        travel_bearing=0.0,
        config={},
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


def test_travel_bearing_stored(road_position, rng):
    # Arrange
    bearing = math.pi / 4

    # Act
    pair = CarLightPair(
        position=road_position, travel_bearing=bearing, config={}, rng=rng
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


def test_light_separation_within_range():
    # Arrange — run many samples to verify separation is in [1.6, 2.4]
    for seed in range(30):
        rng = random.Random(seed)
        pos = np.array([0.0, 0.0, 0.0])
        pair = CarLightPair(position=pos, travel_bearing=0.0, config={}, rng=rng)
        lights = pair.get_lights()
        separation = np.linalg.norm(lights[0].position - lights[1].position)

        # Assert
        assert 1.6 <= separation <= 2.4, f"seed={seed} separation={separation}"


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


def test_resolve_facing_toward_camera_sets_brightness_range(car_pair):
    # Arrange
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — headlight brightness in [0.7, 0.9]
    assert all(0.7 <= light.brightness <= 0.9 for light in lights)


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


def test_resolve_facing_away_from_camera_sets_brightness_range(car_pair):
    # Arrange
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — taillight brightness in [0.3, 0.5]
    assert all(0.3 <= light.brightness <= 0.5 for light in lights)


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


def test_same_seed_produces_same_light_positions():
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])
    pair_a = CarLightPair(position=pos, travel_bearing=0.5, config={}, rng=random.Random(99))
    pair_b = CarLightPair(position=pos, travel_bearing=0.5, config={}, rng=random.Random(99))

    # Act
    lights_a = pair_a.get_lights()
    lights_b = pair_b.get_lights()

    # Assert
    for la, lb in zip(lights_a, lights_b):
        np.testing.assert_array_equal(la.position, lb.position)
