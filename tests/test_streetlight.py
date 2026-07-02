# Radical
"""Tests for streets_at_night.models.streetlight module."""

import math
import random
import typing as t

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_streetlight_config
from streets_at_night.config.schema import FixtureProfileConfig
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.road_types import RoadType
from streets_at_night.models.streetlight import (
    FixturePool,
    FixtureProfile,
    StreetLight,
    _config_to_fixture_profile,
    create_fixture_pool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def streetlight_config():
    """Return a StreetLightConfig built from project defaults."""
    return _build_streetlight_config(DEFAULTS["lights"]["streetlights"])


@pytest.fixture
def sample_profile() -> FixtureProfile:
    """Return a fully populated FixtureProfile for tests."""
    return FixtureProfile(
        fixture_color=(255, 160, 64),
        height=10.0,
        brightness=0.8,
        glow_radius=18.0,
        beam_angle_deg=147.0,
        illumination_intensity=0.75,
        spacing=45.0,
        side="both",
    )


@pytest.fixture
def sample_position() -> np.ndarray:
    """Return a standard ENU position for tests."""
    return np.array([100.0, 200.0, 10.0])


# ---------------------------------------------------------------------------
# FixtureProfile — construction
# ---------------------------------------------------------------------------


def test_fixture_profile_stores_all_fields(sample_profile):
    # Assert
    assert sample_profile.fixture_color == (255, 160, 64)
    assert sample_profile.height == 10.0
    assert sample_profile.brightness == 0.8
    assert sample_profile.glow_radius == 18.0
    assert sample_profile.beam_angle_deg == 147.0
    assert sample_profile.illumination_intensity == 0.75
    assert sample_profile.spacing == 45.0
    assert sample_profile.side == "both"


# ---------------------------------------------------------------------------
# FixturePool — pick
# ---------------------------------------------------------------------------


def test_fixture_pool_pick_returns_profile(sample_profile, rng):
    # Arrange
    pool = FixturePool([(sample_profile, 1.0)], rng)

    # Act
    result = pool.pick()

    # Assert
    assert result is sample_profile


def test_fixture_pool_pick_can_return_none(rng):
    # Arrange — 100% chance of None
    pool = FixturePool([(None, 1.0)], rng)

    # Act
    result = pool.pick()

    # Assert
    assert result is None


def test_fixture_pool_pick_respects_weights(sample_profile):
    # Arrange — profile weight=0, None weight=1 → always None
    rng = random.Random(1)
    pool = FixturePool([(sample_profile, 0.0), (None, 1.0)], rng)

    # Act
    results = [pool.pick() for _ in range(20)]

    # Assert — all None since weight is 0 for profile
    assert all(r is None for r in results)


def test_fixture_pool_pick_is_reproducible(sample_profile):
    # Arrange
    rng_a = random.Random(7)
    rng_b = random.Random(7)
    pool_a = FixturePool([(sample_profile, 0.7), (None, 0.3)], rng_a)
    pool_b = FixturePool([(sample_profile, 0.7), (None, 0.3)], rng_b)

    # Act
    results_a = [pool_a.pick() for _ in range(10)]
    results_b = [pool_b.pick() for _ in range(10)]

    # Assert
    assert results_a == results_b


# ---------------------------------------------------------------------------
# StreetLight.create_from_profile — factory
# ---------------------------------------------------------------------------


def test_create_from_profile_returns_light_source(sample_profile, sample_position):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert
    assert isinstance(light, LightSource)


def test_create_from_profile_uses_fixture_color(sample_profile, sample_position):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert
    assert light.color == sample_profile.fixture_color


def test_create_from_profile_uses_fixture_brightness(sample_profile, sample_position):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert
    assert light.brightness == sample_profile.brightness


def test_create_from_profile_uses_fixture_glow_radius(sample_profile, sample_position):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert
    assert light.glow_radius == sample_profile.glow_radius


def test_create_from_profile_stores_position(sample_profile, sample_position):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert
    np.testing.assert_array_equal(light.position, sample_position)


def test_create_from_profile_passes_beam_angle(sample_profile, sample_position):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert — beam angle flows through so the illumination pass uses the physical model
    assert light.beam_angle_deg == sample_profile.beam_angle_deg


def test_create_from_profile_derives_illumination_radius_from_height_and_beam(
    sample_profile, sample_position
):
    # Act
    light = StreetLight.create_from_profile(
        sample_profile, sample_position, RoadType.MAJOR_ROAD
    )

    # Assert — radius = height * tan(beam / 2)
    expected = sample_profile.height * math.tan(math.radians(sample_profile.beam_angle_deg / 2.0))
    assert light.illumination_radius == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _config_to_fixture_profile — color temperature + master shift
# ---------------------------------------------------------------------------


def _make_profile_config(color_temp_k: float) -> FixtureProfileConfig:
    """Return a minimal FixtureProfileConfig at the given color temperature."""
    return FixtureProfileConfig(
        color_temp_k=color_temp_k,
        saturation=1.0,
        height_min=8.0,
        height_max=8.0,
        brightness_min=0.8,
        brightness_max=0.8,
        glow_radius=18.0,
        beam_angle_deg=147.0,
        illumination_intensity=0.75,
        spacing_min=45.0,
        spacing_max=45.0,
        side="both",
    )


def test_config_to_fixture_profile_positive_shift_makes_color_cooler(rng):
    # Arrange — same 2700K base, one with no shift and one shifted +2000K
    cfg = _make_profile_config(2700.0)

    # Act
    base = _config_to_fixture_profile(cfg, rng, 0.0)
    shifted = _config_to_fixture_profile(cfg, rng, 2000.0)

    # Assert — a higher color temperature carries more blue
    assert shifted.fixture_color[2] > base.fixture_color[2]


# ---------------------------------------------------------------------------
# create_fixture_pool — per road type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("road_type", [
    RoadType.HIGHWAY,
    RoadType.MAJOR_ROAD,
    RoadType.COLLECTOR,
    RoadType.RESIDENTIAL,
    RoadType.SERVICE,
    RoadType.MINOR,
])
def test_create_fixture_pool_returns_pool_for_each_road_type(road_type, streetlight_config):
    # Arrange
    rng = random.Random(0)

    # Act
    pool = create_fixture_pool(road_type, streetlight_config, rng)

    # Assert
    assert isinstance(pool, FixturePool)


def test_create_fixture_pool_highway_never_returns_none(rng, streetlight_config):
    # Arrange — highway should always have lights
    pool = create_fixture_pool(RoadType.HIGHWAY, streetlight_config, rng)

    # Act
    results = [pool.pick() for _ in range(50)]

    # Assert
    assert all(r is not None for r in results)


def test_create_fixture_pool_residential_can_return_none(streetlight_config):
    # Arrange — residential has 25% no-lights weight
    rng = random.Random(0)
    pool = create_fixture_pool(RoadType.RESIDENTIAL, streetlight_config, rng)

    # Act — run many picks to get statistical coverage
    results = [pool.pick() for _ in range(200)]

    # Assert — some None results expected from the 25% weight
    assert any(r is None for r in results)


def test_create_fixture_pool_service_can_return_none(streetlight_config):
    # Arrange — service has 70% no-lights weight
    rng = random.Random(0)
    pool = create_fixture_pool(RoadType.SERVICE, streetlight_config, rng)

    # Act
    results = [pool.pick() for _ in range(100)]

    # Assert — majority should be None
    none_count = sum(1 for r in results if r is None)
    assert none_count > 40


def test_create_fixture_pool_highway_profile_has_both_sides(rng, streetlight_config):
    # Arrange
    pool = create_fixture_pool(RoadType.HIGHWAY, streetlight_config, rng)

    # Act
    profile = pool.pick()

    # Assert
    assert profile is not None
    assert profile.side == "both"
