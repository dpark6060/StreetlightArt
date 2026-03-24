# Radical
"""Tests for streets_at_night.models.commercial factory functions."""

import random

import numpy as np
import pytest

from streets_at_night.models.commercial import (
    create_parking_lot_light,
    create_sign_light,
    create_storefront_light,
)
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
    return np.array([200.0, 150.0, 3.0])


# ---------------------------------------------------------------------------
# create_sign_light
# ---------------------------------------------------------------------------


def test_create_sign_light_returns_light_source(position, rng):
    # Act
    light = create_sign_light(position, {}, rng)

    # Assert
    assert isinstance(light, LightSource)


def test_create_sign_light_stores_position(position, rng):
    # Act
    light = create_sign_light(position, {}, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


def test_create_sign_light_brightness_in_range(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_sign_light(pos, {}, rng)

    # Assert
    assert 0.6 <= light.brightness <= 0.9


def test_create_sign_light_illumination_radius_in_range(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_sign_light(pos, {}, rng)

    # Assert
    assert 15.0 <= light.illumination_radius <= 25.0


def test_create_sign_light_color_is_valid_rgb(position, rng):
    # Act
    light = create_sign_light(position, {}, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# create_parking_lot_light
# ---------------------------------------------------------------------------


def test_create_parking_lot_light_returns_light_source(position, rng):
    # Act
    light = create_parking_lot_light(position, {}, rng)

    # Assert
    assert isinstance(light, LightSource)


def test_create_parking_lot_light_stores_position(position, rng):
    # Act
    light = create_parking_lot_light(position, {}, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


def test_create_parking_lot_light_brightness_in_range(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_parking_lot_light(pos, {}, rng)

    # Assert
    assert 0.75 <= light.brightness <= 0.9


def test_create_parking_lot_light_has_large_illumination(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_parking_lot_light(pos, {}, rng)

    # Assert — parking lots have wide coverage
    assert light.illumination_radius == 30.0


def test_create_parking_lot_light_color_is_valid_rgb(position, rng):
    # Act
    light = create_parking_lot_light(position, {}, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


# ---------------------------------------------------------------------------
# create_storefront_light
# ---------------------------------------------------------------------------


def test_create_storefront_light_returns_light_source(position, rng):
    # Act
    light = create_storefront_light(position, {}, rng)

    # Assert
    assert isinstance(light, LightSource)


def test_create_storefront_light_stores_position(position, rng):
    # Act
    light = create_storefront_light(position, {}, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


def test_create_storefront_light_brightness_in_range(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, {}, rng)

    # Assert
    assert 0.5 <= light.brightness <= 0.7


def test_create_storefront_light_has_correct_glow(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, {}, rng)

    # Assert
    assert light.glow_radius == 8.0


def test_create_storefront_light_has_correct_illumination(rng):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, {}, rng)

    # Assert
    assert light.illumination_radius == 12.0


def test_create_storefront_light_color_is_valid_rgb(position, rng):
    # Act
    light = create_storefront_light(position, {}, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255
