# Radical
"""Unit tests for streets_at_night.placement.house_light_placer."""

import random
from unittest import mock

import pytest
from shapely.geometry import box

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_light_config
from streets_at_night.config.schema import ColorPoolConfig, HouseConfig, LightConfig
from streets_at_night.models.house import House
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.zones import ResidentialZone
from streets_at_night.placement.house_light_placer import (
    _collect_zone_lights,
    collect_house_lights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(7)


@pytest.fixture
def light_config() -> LightConfig:
    """Return a default LightConfig built from project defaults."""
    return _build_light_config(DEFAULTS["lights"])


@pytest.fixture
def house_config() -> HouseConfig:
    """Return a minimal HouseConfig with lights enabled."""
    return HouseConfig(
        exterior_colors=ColorPoolConfig(colors=[("#0D0A08", 1.0)]),
        yard_base_color="#060A04",
        yard_variation=2,
        min_lights=2,
        max_lights=2,
        light_weights=[1.0],
        footprint_coverage=0.65,
        lot_width=15.0,
        lot_depth=25.0,
        setback=10.0,
    )


@pytest.fixture
def house_with_lights(house_config) -> House:
    """Return a House guaranteed to have 2 lights."""
    footprint = box(0, 0, 15, 25)
    rng = random.Random(42)
    return House(footprint=footprint, config=house_config, rng=rng)


@pytest.fixture
def house_config_no_lights() -> HouseConfig:
    """Return a HouseConfig that produces houses with zero lights."""
    return HouseConfig(
        exterior_colors=ColorPoolConfig(colors=[("#0D0A08", 1.0)]),
        yard_base_color="#060A04",
        yard_variation=2,
        min_lights=0,
        max_lights=0,
        light_weights=[1.0],
        footprint_coverage=0.65,
        lot_width=15.0,
        lot_depth=25.0,
        setback=10.0,
    )


@pytest.fixture
def house_no_lights(house_config_no_lights) -> House:
    """Return a House with zero lights."""
    footprint = box(0, 0, 15, 25)
    rng = random.Random(42)
    return House(footprint=footprint, config=house_config_no_lights, rng=rng)


@pytest.fixture
def residential_zone_with_lights(house_with_lights) -> ResidentialZone:
    """Return a ResidentialZone with one house that has lights."""
    return ResidentialZone(
        boundary=box(0, 0, 100, 100),
        houses=[house_with_lights],
    )


@pytest.fixture
def empty_residential_zone() -> ResidentialZone:
    """Return a ResidentialZone with no houses."""
    return ResidentialZone(
        boundary=box(0, 0, 100, 100),
        houses=[],
    )


# ---------------------------------------------------------------------------
# _collect_zone_lights
# ---------------------------------------------------------------------------


def test_collect_zone_lights_returns_list(residential_zone_with_lights, rng, light_config):
    # Act
    lights = _collect_zone_lights(residential_zone_with_lights, rng, light_config.house_lights)

    # Assert
    assert isinstance(lights, list)


def test_collect_zone_lights_returns_light_sources(residential_zone_with_lights, rng, light_config):
    # Act
    lights = _collect_zone_lights(residential_zone_with_lights, rng, light_config.house_lights)

    # Assert
    assert all(isinstance(l, LightSource) for l in lights)


def test_collect_zone_lights_count_matches_house_lights(
    residential_zone_with_lights, house_with_lights, rng, light_config
):
    # Arrange — house has exactly 2 lights
    expected_count = house_with_lights.num_lights

    # Act
    lights = _collect_zone_lights(residential_zone_with_lights, rng, light_config.house_lights)

    # Assert
    assert len(lights) == expected_count


def test_collect_zone_lights_empty_zone_returns_empty(empty_residential_zone, rng, light_config):
    # Act
    lights = _collect_zone_lights(empty_residential_zone, rng, light_config.house_lights)

    # Assert
    assert lights == []


def test_collect_zone_lights_house_with_no_lights_contributes_nothing(
    house_no_lights, rng, light_config
):
    # Arrange
    zone = ResidentialZone(
        boundary=box(0, 0, 100, 100),
        houses=[house_no_lights],
    )

    # Act
    lights = _collect_zone_lights(zone, rng, light_config.house_lights)

    # Assert
    assert lights == []


def test_collect_zone_lights_calls_get_lights_on_each_house(rng, light_config):
    # Arrange
    mock_house = mock.MagicMock()
    mock_house.get_lights.return_value = []
    zone = ResidentialZone(
        boundary=box(0, 0, 100, 100),
        houses=[mock_house],
    )

    # Act
    _collect_zone_lights(zone, rng, light_config.house_lights)

    # Assert
    mock_house.get_lights.assert_called_once_with(rng, light_config.house_lights)


def test_collect_zone_lights_accumulates_from_multiple_houses(
    house_with_lights, rng, light_config
):
    # Arrange — two houses, each with 2 lights
    zone = ResidentialZone(
        boundary=box(0, 0, 200, 100),
        houses=[house_with_lights, house_with_lights],
    )
    expected_count = house_with_lights.num_lights * 2

    # Act
    lights = _collect_zone_lights(zone, rng, light_config.house_lights)

    # Assert
    assert len(lights) == expected_count


# ---------------------------------------------------------------------------
# collect_house_lights — orchestration
# ---------------------------------------------------------------------------


def test_collect_house_lights_returns_list(
    residential_zone_with_lights, light_config, rng
):
    # Act
    lights = collect_house_lights([residential_zone_with_lights], light_config, rng)

    # Assert
    assert isinstance(lights, list)


def test_collect_house_lights_returns_light_sources(
    residential_zone_with_lights, light_config, rng
):
    # Act
    lights = collect_house_lights([residential_zone_with_lights], light_config, rng)

    # Assert
    assert all(isinstance(l, LightSource) for l in lights)


def test_collect_house_lights_empty_zones(light_config, rng):
    # Act
    lights = collect_house_lights([], light_config, rng)

    # Assert
    assert lights == []


def test_collect_house_lights_accumulates_across_zones(
    house_with_lights, light_config, rng
):
    # Arrange — two separate zones, each with one house that has 2 lights
    zone_a = ResidentialZone(boundary=box(0, 0, 100, 100), houses=[house_with_lights])
    zone_b = ResidentialZone(boundary=box(200, 200, 300, 300), houses=[house_with_lights])
    expected = house_with_lights.num_lights * 2

    # Act
    lights = collect_house_lights([zone_a, zone_b], light_config, rng)

    # Assert
    assert len(lights) == expected


def test_collect_house_lights_reproducible_with_same_seed(
    residential_zone_with_lights, light_config
):
    # Arrange
    rng_a = random.Random(55)
    rng_b = random.Random(55)

    # Act
    lights_a = collect_house_lights([residential_zone_with_lights], light_config, rng_a)
    lights_b = collect_house_lights([residential_zone_with_lights], light_config, rng_b)

    # Assert
    assert len(lights_a) == len(lights_b)
    for la, lb in zip(lights_a, lights_b):
        assert la.color == lb.color
        assert la.brightness == lb.brightness
