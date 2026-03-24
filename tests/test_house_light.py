# Radical
"""Tests for streets_at_night.models.house_light.create_house_light."""

import random

import numpy as np
import pytest

from streets_at_night.models.house_light import create_house_light
from streets_at_night.models.light_source import LightSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def position() -> np.ndarray:
    """Return a standard ENU position for tests."""
    return np.array([50.0, 75.0, 0.0])


# ---------------------------------------------------------------------------
# create_house_light — return type
# ---------------------------------------------------------------------------


def test_create_house_light_returns_light_source(position, rng):
    # Act
    light = create_house_light(position, {}, rng)

    # Assert
    assert isinstance(light, LightSource)


# ---------------------------------------------------------------------------
# create_house_light — position
# ---------------------------------------------------------------------------


def test_create_house_light_stores_position(position, rng):
    # Act
    light = create_house_light(position, {}, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


# ---------------------------------------------------------------------------
# create_house_light — brightness
# ---------------------------------------------------------------------------


def test_create_house_light_brightness_in_range(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_house_light(pos, {}, rng)

    # Assert — house lights are dim
    assert 0.3 <= light.brightness <= 0.6


# ---------------------------------------------------------------------------
# create_house_light — color
# ---------------------------------------------------------------------------


def test_create_house_light_color_is_valid_rgb(position, rng):
    # Act
    light = create_house_light(position, {}, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# create_house_light — illumination parameters
# ---------------------------------------------------------------------------


def test_create_house_light_has_correct_glow_radius(position, rng):
    # Act
    light = create_house_light(position, {}, rng)

    # Assert
    assert light.glow_radius == 5.0


def test_create_house_light_has_correct_illumination_radius(position, rng):
    # Act
    light = create_house_light(position, {}, rng)

    # Assert
    assert light.illumination_radius == 8.0


# ---------------------------------------------------------------------------
# create_house_light — reproducibility
# ---------------------------------------------------------------------------


def test_create_house_light_reproducible_with_same_seed():
    # Arrange
    pos = np.array([10.0, 20.0, 0.0])
    rng_a = random.Random(7)
    rng_b = random.Random(7)

    # Act
    light_a = create_house_light(pos, {}, rng_a)
    light_b = create_house_light(pos, {}, rng_b)

    # Assert
    assert light_a.color == light_b.color
    assert light_a.brightness == light_b.brightness
