# Radical
"""Tests for streets_at_night.models.house_light.create_house_light."""

import random

import numpy as np
import pytest

from streets_at_night.config.schema import ColorPoolConfig, HouseLightConfig
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


@pytest.fixture
def house_light_config() -> HouseLightConfig:
    """Return a HouseLightConfig built from DEFAULTS values."""
    return HouseLightConfig(
        colors=ColorPoolConfig(colors=[
            ("#FFE4A0", 0.45),
            ("#FFF0D0", 0.25),
            ("#FFCC80", 0.15),
            ("#D0E0FF", 0.15),
        ]),
        brightness_min=0.3,
        brightness_max=0.6,
        glow_radius=6.0,
        illumination_radius=8.0,
        illumination_intensity=0.35,
    )


# ---------------------------------------------------------------------------
# create_house_light — return type
# ---------------------------------------------------------------------------


def test_create_house_light_returns_light_source(position, rng, house_light_config):
    # Act
    light = create_house_light(position, house_light_config, rng)

    # Assert
    assert isinstance(light, LightSource)


# ---------------------------------------------------------------------------
# create_house_light — position
# ---------------------------------------------------------------------------


def test_create_house_light_stores_position(position, rng, house_light_config):
    # Act
    light = create_house_light(position, house_light_config, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


# ---------------------------------------------------------------------------
# create_house_light — brightness
# ---------------------------------------------------------------------------


def test_create_house_light_brightness_in_range(rng, house_light_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_house_light(pos, house_light_config, rng)

    # Assert — house lights are dim
    assert 0.3 <= light.brightness <= 0.6


# ---------------------------------------------------------------------------
# create_house_light — color
# ---------------------------------------------------------------------------


def test_create_house_light_color_is_valid_rgb(position, rng, house_light_config):
    # Act
    light = create_house_light(position, house_light_config, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# create_house_light — illumination parameters
# ---------------------------------------------------------------------------


def test_create_house_light_has_correct_glow_radius(position, rng, house_light_config):
    # Act
    light = create_house_light(position, house_light_config, rng)

    # Assert
    assert light.glow_radius == house_light_config.glow_radius


def test_create_house_light_has_correct_illumination_radius(position, rng, house_light_config):
    # Act
    light = create_house_light(position, house_light_config, rng)

    # Assert
    assert light.illumination_radius == house_light_config.illumination_radius


# ---------------------------------------------------------------------------
# create_house_light — reproducibility
# ---------------------------------------------------------------------------


def test_create_house_light_reproducible_with_same_seed(house_light_config):
    # Arrange
    pos = np.array([10.0, 20.0, 0.0])
    rng_a = random.Random(7)
    rng_b = random.Random(7)

    # Act
    light_a = create_house_light(pos, house_light_config, rng_a)
    light_b = create_house_light(pos, house_light_config, rng_b)

    # Assert
    assert light_a.color == light_b.color
    assert light_a.brightness == light_b.brightness


# ---------------------------------------------------------------------------
# create_house_light — config passthrough
# ---------------------------------------------------------------------------


def test_create_house_light_glow_radius_from_config(rng, house_light_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_house_light(pos, house_light_config, rng)

    # Assert
    assert light.glow_radius == house_light_config.glow_radius


def test_create_house_light_illumination_radius_from_config(rng, house_light_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_house_light(pos, house_light_config, rng)

    # Assert
    assert light.illumination_radius == house_light_config.illumination_radius
