# Radical
"""Tests for streets_at_night.models.commercial factory functions."""

import random

import numpy as np
import pytest

from streets_at_night.config.schema import ColorPoolConfig, CommercialLightConfig
from streets_at_night.models.commercial import (
    create_parking_lot_light,
    create_sign_light,
    create_storefront_light,
)
from streets_at_night.models.illumination import BeamIllumination
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


@pytest.fixture
def commercial_config() -> CommercialLightConfig:
    """Return a CommercialLightConfig built from DEFAULTS values."""
    return CommercialLightConfig(
        sign_colors=ColorPoolConfig(colors=[
            ("#FFF8E0", 0.35),
            ("#FF2020", 0.12),
            ("#2040FF", 0.12),
            ("#20CC40", 0.10),
            ("#FFE020", 0.10),
            ("#8020CC", 0.08),
            ("#FF8020", 0.08),
            ("#FF40A0", 0.05),
        ]),
        sign_brightness_min=0.6,
        sign_brightness_max=0.9,
        sign_glow_radius=12.0,
        sign_illumination_radius=20.0,
        sign_illumination_intensity=0.7,
        sign_illumination_falloff=2.0,
        sign_illumination_inner_fraction=0.0,
        parking_colors=ColorPoolConfig(colors=[
            ("#E0E8FF", 0.7),
            ("#FFA040", 0.3),
        ]),
        parking_brightness_min=0.75,
        parking_brightness_max=0.9,
        parking_glow_radius=20.0,
        parking_height_min=8.0,
        parking_height_max=12.0,
        parking_beam_angle_deg=125.0,
        parking_bulb_fraction=0.6,
        parking_reflector_falloff=2.0,
        parking_illumination_intensity=0.8,
        parking_grid_spacing_min=20.0,
        parking_grid_spacing_max=30.0,
        storefront_colors=ColorPoolConfig(colors=[
            ("#FFE8CC", 0.6),
            ("#FFE080", 0.25),
            ("#F0F0FF", 0.15),
        ]),
        storefront_brightness_min=0.5,
        storefront_brightness_max=0.7,
        storefront_glow_radius=8.0,
        storefront_illumination_radius=12.0,
        storefront_illumination_intensity=0.5,
        storefront_illumination_falloff=2.0,
        storefront_illumination_inner_fraction=0.0,
        storefront_spacing_min=5.0,
        storefront_spacing_max=8.0,
    )


# ---------------------------------------------------------------------------
# create_sign_light
# ---------------------------------------------------------------------------


def test_create_sign_light_returns_light_source(position, rng, commercial_config):
    # Act
    light = create_sign_light(position, commercial_config, rng)

    # Assert
    assert isinstance(light, LightSource)


def test_create_sign_light_stores_position(position, rng, commercial_config):
    # Act
    light = create_sign_light(position, commercial_config, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


def test_create_sign_light_brightness_in_range(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_sign_light(pos, commercial_config, rng)

    # Assert
    assert 0.6 <= light.brightness <= 0.9


def test_create_sign_light_illumination_radius_in_range(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_sign_light(pos, commercial_config, rng)

    # Assert
    assert light.illumination.radius == commercial_config.sign_illumination_radius


def test_create_sign_light_color_is_valid_rgb(position, rng, commercial_config):
    # Act
    light = create_sign_light(position, commercial_config, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


def test_create_sign_light_glow_radius_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_sign_light(pos, commercial_config, rng)

    # Assert
    assert light.glow_radius == commercial_config.sign_glow_radius


# ---------------------------------------------------------------------------
# create_parking_lot_light
# ---------------------------------------------------------------------------


def test_create_parking_lot_light_returns_light_source(position, rng, commercial_config):
    # Act
    light = create_parking_lot_light(position, commercial_config, rng)

    # Assert
    assert isinstance(light, LightSource)


def test_create_parking_lot_light_stores_position(position, rng, commercial_config):
    # Act
    light = create_parking_lot_light(position, commercial_config, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


def test_create_parking_lot_light_brightness_in_range(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_parking_lot_light(pos, commercial_config, rng)

    # Assert
    assert 0.75 <= light.brightness <= 0.9


def test_create_parking_lot_light_uses_physical_beam_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 10.0])

    # Act
    light = create_parking_lot_light(pos, commercial_config, rng)

    # Assert — pole-mounted, so the pool is physical like a streetlight's
    assert light.illumination == BeamIllumination(
        commercial_config.parking_beam_angle_deg,
        commercial_config.parking_bulb_fraction,
        commercial_config.parking_reflector_falloff,
    )


def test_create_parking_lot_light_color_is_valid_rgb(position, rng, commercial_config):
    # Act
    light = create_parking_lot_light(position, commercial_config, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


def test_create_parking_lot_light_glow_radius_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_parking_lot_light(pos, commercial_config, rng)

    # Assert
    assert light.glow_radius == commercial_config.parking_glow_radius


# ---------------------------------------------------------------------------
# create_storefront_light
# ---------------------------------------------------------------------------


def test_create_storefront_light_returns_light_source(position, rng, commercial_config):
    # Act
    light = create_storefront_light(position, commercial_config, rng)

    # Assert
    assert isinstance(light, LightSource)


def test_create_storefront_light_stores_position(position, rng, commercial_config):
    # Act
    light = create_storefront_light(position, commercial_config, rng)

    # Assert
    np.testing.assert_array_equal(light.position, position)


def test_create_storefront_light_brightness_in_range(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, commercial_config, rng)

    # Assert
    assert 0.5 <= light.brightness <= 0.7


def test_create_storefront_light_has_correct_glow(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, commercial_config, rng)

    # Assert
    assert light.glow_radius == commercial_config.storefront_glow_radius


def test_create_storefront_light_has_correct_illumination(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, commercial_config, rng)

    # Assert
    assert light.illumination.radius == commercial_config.storefront_illumination_radius


def test_create_storefront_light_color_is_valid_rgb(position, rng, commercial_config):
    # Act
    light = create_storefront_light(position, commercial_config, rng)

    # Assert
    r, g, b = light.color
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255


def test_create_storefront_light_glow_radius_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, commercial_config, rng)

    # Assert
    assert light.glow_radius == commercial_config.storefront_glow_radius


# ---------------------------------------------------------------------------
# illumination_intensity comes from config
# ---------------------------------------------------------------------------


def test_create_sign_light_illumination_intensity_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_sign_light(pos, commercial_config, rng)

    # Assert
    assert light.illumination_intensity == commercial_config.sign_illumination_intensity


def test_create_parking_lot_light_illumination_intensity_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_parking_lot_light(pos, commercial_config, rng)

    # Assert
    assert light.illumination_intensity == commercial_config.parking_illumination_intensity


def test_create_storefront_light_illumination_intensity_from_config(rng, commercial_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])

    # Act
    light = create_storefront_light(pos, commercial_config, rng)

    # Assert
    assert light.illumination_intensity == commercial_config.storefront_illumination_intensity
