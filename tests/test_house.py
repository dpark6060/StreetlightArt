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
# __init__ — footprint stored
# ---------------------------------------------------------------------------

def test_footprint_stored(house, footprint):
    # Assert
    assert house.footprint is footprint


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
