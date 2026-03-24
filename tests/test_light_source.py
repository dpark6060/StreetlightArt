# Radical
"""Tests for streets_at_night.models.light_source.LightSource."""

import dataclasses

import numpy as np
import pytest

from streets_at_night.models.light_source import LightSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_position():
    """Return a standard ENU position array for tests."""
    return np.array([100.0, 200.0, 5.0])


@pytest.fixture
def sample_light(sample_position):
    """Return a fully populated LightSource instance."""
    return LightSource(
        position=sample_position,
        color=(255, 180, 80),
        brightness=1.5,
        size=0.3,
        glow_radius=20.0,
        illumination_radius=15.0,
        illumination_intensity=0.6,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_succeeds_with_all_fields(sample_position):
    # Arrange — all required fields

    # Act
    light = LightSource(
        position=sample_position,
        color=(200, 150, 50),
        brightness=1.0,
        size=0.5,
        glow_radius=10.0,
        illumination_radius=8.0,
        illumination_intensity=0.4,
    )

    # Assert
    assert light is not None


# ---------------------------------------------------------------------------
# Attribute storage
# ---------------------------------------------------------------------------


def test_attributes_position_stored_correctly(sample_light, sample_position):
    # Arrange — light and position from fixtures

    # Act & Assert
    np.testing.assert_array_equal(sample_light.position, sample_position)


def test_attributes_color_stored_correctly(sample_light):
    # Arrange — fixture light has color (255, 180, 80)

    # Act & Assert
    assert sample_light.color == (255, 180, 80)


def test_attributes_brightness_stored_correctly(sample_light):
    # Arrange — fixture light has brightness 1.5

    # Act & Assert
    assert sample_light.brightness == 1.5


def test_attributes_size_stored_correctly(sample_light):
    # Arrange — fixture light has size 0.3

    # Act & Assert
    assert sample_light.size == 0.3


def test_attributes_glow_radius_stored_correctly(sample_light):
    # Arrange — fixture light has glow_radius 20.0

    # Act & Assert
    assert sample_light.glow_radius == 20.0


def test_attributes_illumination_radius_stored_correctly(sample_light):
    # Arrange — fixture light has illumination_radius 15.0

    # Act & Assert
    assert sample_light.illumination_radius == 15.0


def test_attributes_illumination_intensity_stored_correctly(sample_light):
    # Arrange — fixture light has illumination_intensity 0.6

    # Act & Assert
    assert sample_light.illumination_intensity == 0.6


# ---------------------------------------------------------------------------
# Dataclass identity
# ---------------------------------------------------------------------------


def test_is_dataclass_instance(sample_light):
    # Arrange — use the standard library dataclasses helper

    # Act
    result = dataclasses.is_dataclass(sample_light)

    # Assert
    assert result is True
