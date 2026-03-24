# Radical
"""Unit tests for streets_at_night.models.zones."""

import random

import pytest
from shapely.geometry import box

from streets_at_night.config.schema import ColorPoolConfig, HouseConfig
from streets_at_night.models.house import House
from streets_at_night.models.zones import CommercialZone, DarkZone, ResidentialZone


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def boundary():
    """Return a simple 100x100 metre boundary polygon."""
    return box(0, 0, 100, 100)


@pytest.fixture
def house_config() -> HouseConfig:
    """Return a minimal HouseConfig for building test House objects."""
    return HouseConfig(
        exterior_colors=ColorPoolConfig(
            colors=[("#0D0A08", 1.0)]
        ),
        yard_base_color="#060A04",
        yard_variation=2,
        min_lights=0,
        max_lights=1,
        light_weights=[1.0, 1.0],
        footprint_coverage=0.65,
        lot_width=15.0,
        lot_depth=25.0,
        setback=10.0,
    )


@pytest.fixture
def house(house_config) -> House:
    """Return a single House for use in zone tests."""
    footprint = box(5, 5, 15, 20)
    rng = random.Random(42)
    return House(footprint=footprint, config=house_config, rng=rng)


# ---------------------------------------------------------------------------
# ResidentialZone — init
# ---------------------------------------------------------------------------

def test_residential_zone_init_stores_boundary(boundary, house):
    # Arrange / Act
    zone = ResidentialZone(boundary=boundary, houses=[house])

    # Assert
    assert zone.boundary is boundary


def test_residential_zone_init_stores_houses(boundary, house):
    # Arrange / Act
    zone = ResidentialZone(boundary=boundary, houses=[house])

    # Assert
    assert zone.houses == [house]


def test_residential_zone_empty_houses(boundary):
    # Arrange / Act
    zone = ResidentialZone(boundary=boundary, houses=[])

    # Assert
    assert zone.houses == []
    assert zone.boundary is boundary


# ---------------------------------------------------------------------------
# CommercialZone — init
# ---------------------------------------------------------------------------

def test_commercial_zone_init_stores_boundary(boundary):
    # Arrange
    footprints = [box(10, 10, 30, 40)]
    parking = [box(35, 10, 60, 40)]

    # Act
    zone = CommercialZone(
        boundary=boundary,
        building_footprints=footprints,
        parking_areas=parking,
    )

    # Assert
    assert zone.boundary is boundary


def test_commercial_zone_init_stores_building_footprints(boundary):
    # Arrange
    footprints = [box(10, 10, 30, 40), box(50, 10, 70, 40)]
    parking = []

    # Act
    zone = CommercialZone(
        boundary=boundary,
        building_footprints=footprints,
        parking_areas=parking,
    )

    # Assert
    assert zone.building_footprints == footprints


def test_commercial_zone_init_stores_parking_areas(boundary):
    # Arrange
    footprints = []
    parking = [box(35, 10, 60, 40)]

    # Act
    zone = CommercialZone(
        boundary=boundary,
        building_footprints=footprints,
        parking_areas=parking,
    )

    # Assert
    assert zone.parking_areas == parking


def test_commercial_zone_empty_lists(boundary):
    # Act
    zone = CommercialZone(
        boundary=boundary,
        building_footprints=[],
        parking_areas=[],
    )

    # Assert
    assert zone.building_footprints == []
    assert zone.parking_areas == []
    assert zone.boundary is boundary


# ---------------------------------------------------------------------------
# DarkZone — init
# ---------------------------------------------------------------------------

def test_dark_zone_init_stores_boundary(boundary):
    # Act
    zone = DarkZone(boundary=boundary, zone_type="park", base_color=(4, 8, 4))

    # Assert
    assert zone.boundary is boundary


def test_dark_zone_init_stores_zone_type(boundary):
    # Act
    zone = DarkZone(boundary=boundary, zone_type="park", base_color=(4, 8, 4))

    # Assert
    assert zone.zone_type == "park"


def test_dark_zone_init_stores_base_color(boundary):
    # Act
    zone = DarkZone(boundary=boundary, zone_type="park", base_color=(4, 8, 4))

    # Assert
    assert zone.base_color == (4, 8, 4)


def test_dark_zone_types_water(boundary):
    # Act
    zone = DarkZone(boundary=boundary, zone_type="water", base_color=(4, 4, 8))

    # Assert
    assert zone.zone_type == "water"


def test_dark_zone_types_forest(boundary):
    # Act
    zone = DarkZone(boundary=boundary, zone_type="forest", base_color=(2, 6, 2))

    # Assert
    assert zone.zone_type == "forest"
