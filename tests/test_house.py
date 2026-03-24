# Radical
"""Unit tests for streets_at_night.models.house.House."""

import random

import pytest
from shapely.geometry import box

from streets_at_night.config.schema import ColorPoolConfig, HouseConfig
from streets_at_night.models.house import House


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def house_config() -> HouseConfig:
    """Return a HouseConfig built from the project defaults."""
    return HouseConfig(
        exterior_colors=ColorPoolConfig(
            colors=[
                ("#0D0A08", 1.0),
                ("#0C0907", 1.0),
                ("#0B0808", 1.0),
                ("#0A0907", 1.0),
                ("#0C0A09", 1.0),
            ]
        ),
        yard_base_color="#060A04",
        yard_variation=2,
        min_lights=0,
        max_lights=5,
        light_weights=[5.0, 3.0, 1.5, 0.5, 0.3, 0.1],
        footprint_coverage=0.65,
        lot_width=15.0,
        lot_depth=25.0,
        setback=10.0,
    )


@pytest.fixture
def footprint():
    """Return a simple 10x15 metre rectangular footprint."""
    return box(0, 0, 10, 15)


@pytest.fixture
def house(house_config, footprint) -> House:
    """Return a House built with a fixed seed for reproducibility."""
    rng = random.Random(42)
    return House(footprint=footprint, config=house_config, rng=rng)


# ---------------------------------------------------------------------------
# __init__ — base_color
# ---------------------------------------------------------------------------

def test_init_sets_base_color(house):
    # Assert
    assert isinstance(house.base_color, tuple)
    assert len(house.base_color) == 3
    assert all(isinstance(c, int) for c in house.base_color)
    assert all(0 <= c <= 255 for c in house.base_color)


# ---------------------------------------------------------------------------
# __init__ — yard_color
# ---------------------------------------------------------------------------

def test_init_sets_yard_color(house):
    # Assert
    assert isinstance(house.yard_color, tuple)
    assert len(house.yard_color) == 3
    assert all(isinstance(c, int) for c in house.yard_color)
    assert all(0 <= c <= 255 for c in house.yard_color)


# ---------------------------------------------------------------------------
# __init__ — num_lights
# ---------------------------------------------------------------------------

def test_init_sets_num_lights(house, house_config):
    # Assert
    assert isinstance(house.num_lights, int)
    assert house_config.min_lights <= house.num_lights <= house_config.max_lights


# ---------------------------------------------------------------------------
# __init__ — light_positions
# ---------------------------------------------------------------------------

def test_init_computes_light_positions(house):
    # Assert
    assert isinstance(house.light_positions, list)
    assert len(house.light_positions) == house.num_lights
    for pos in house.light_positions:
        assert isinstance(pos, tuple)
        assert len(pos) == 2


# ---------------------------------------------------------------------------
# __init__ — footprint stored
# ---------------------------------------------------------------------------

def test_footprint_stored(house, footprint):
    # Assert
    assert house.footprint is footprint


# ---------------------------------------------------------------------------
# get_lights
# ---------------------------------------------------------------------------

def test_get_lights_returns_light_sources_for_each_position(house):
    # Arrange
    from streets_at_night.models.light_source import LightSource
    rng = random.Random(1)

    # Act
    result = house.get_lights(rng)

    # Assert
    assert len(result) == house.num_lights
    assert all(isinstance(light, LightSource) for light in result)


def test_get_lights_returns_empty_list_when_no_lights(house_config, footprint):
    # Arrange — force num_lights to 0 by using a config with max_lights=0
    config = HouseConfig(
        exterior_colors=house_config.exterior_colors,
        yard_base_color=house_config.yard_base_color,
        yard_variation=house_config.yard_variation,
        min_lights=0,
        max_lights=0,
        light_weights=[1.0],
        footprint_coverage=house_config.footprint_coverage,
        lot_width=house_config.lot_width,
        lot_depth=house_config.lot_depth,
        setback=house_config.setback,
    )
    rng = random.Random(5)
    no_light_house = House(footprint=footprint, config=config, rng=rng)

    # Act
    result = no_light_house.get_lights(rng)

    # Assert
    assert result == []


# ---------------------------------------------------------------------------
# Reproducibility — same seed + footprint → same properties
# ---------------------------------------------------------------------------

def test_seeded_rng_reproducible(house_config, footprint):
    # Arrange
    rng_a = random.Random(99)
    rng_b = random.Random(99)

    # Act
    house_a = House(footprint=footprint, config=house_config, rng=rng_a)
    house_b = House(footprint=footprint, config=house_config, rng=rng_b)

    # Assert
    assert house_a.base_color == house_b.base_color
    assert house_a.yard_color == house_b.yard_color
    assert house_a.num_lights == house_b.num_lights
    assert house_a.light_positions == house_b.light_positions


# ---------------------------------------------------------------------------
# yard_color — values are close to the configured yard_base_color
# ---------------------------------------------------------------------------

def test_yard_color_near_base(house_config, footprint):
    """Yard color channels must each be within yard_variation of the base color."""
    # Arrange — yard_base_color "#060A04" == (6, 10, 4); yard_variation == 2
    base_r, base_g, base_b = 0x06, 0x0A, 0x04
    variation = house_config.yard_variation
    rng = random.Random(7)

    # Act
    h = House(footprint=footprint, config=house_config, rng=rng)

    # Assert — each channel within variation of base (accounting for clamping at 0/255)
    assert abs(h.yard_color[0] - base_r) <= variation
    assert abs(h.yard_color[1] - base_g) <= variation
    assert abs(h.yard_color[2] - base_b) <= variation
