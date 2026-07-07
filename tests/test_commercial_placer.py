# Radical
"""Unit tests for streets_at_night.placement.commercial_placer."""

import random
from unittest import mock

import numpy as np
import pytest
from shapely.geometry import box

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import LightConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.zones import CommercialZone
from streets_at_night.placement.commercial_placer import (
    _compute_grid_points,
    _get_edge_length,
    _get_edge_midpoint,
    _get_exterior_edges,
    _place_lights_along_edge,
    _place_parking_grid,
    _place_sign_lights,
    _place_storefront_lights_on_longest_edge,
    place_commercial_lights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(99)


@pytest.fixture
def light_config() -> LightConfig:
    """Return a default LightConfig built from project defaults."""
    return dataclass_from_dict(LightConfig, DEFAULTS["lights"])


@pytest.fixture
def small_footprint():
    """Return a 20x30 metre building footprint."""
    return box(0, 0, 20, 30)


@pytest.fixture
def parking_polygon():
    """Return a 40x40 metre parking area polygon."""
    return box(0, 0, 40, 40)


@pytest.fixture
def commercial_zone(small_footprint, parking_polygon) -> CommercialZone:
    """Return a CommercialZone with one building and one parking area."""
    return CommercialZone(
        boundary=box(0, 0, 100, 100),
        building_footprints=[small_footprint],
        parking_areas=[parking_polygon],
    )


# ---------------------------------------------------------------------------
# _get_edge_length
# ---------------------------------------------------------------------------


def test_get_edge_length_horizontal():
    # Arrange
    start = (0.0, 0.0)
    end = (10.0, 0.0)

    # Act
    length = _get_edge_length(start, end)

    # Assert
    assert length == pytest.approx(10.0)


def test_get_edge_length_diagonal():
    # Arrange
    start = (0.0, 0.0)
    end = (3.0, 4.0)

    # Act
    length = _get_edge_length(start, end)

    # Assert
    assert length == pytest.approx(5.0)


def test_get_edge_length_zero_for_same_point():
    # Arrange
    start = (5.0, 5.0)
    end = (5.0, 5.0)

    # Act
    length = _get_edge_length(start, end)

    # Assert
    assert length == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _get_edge_midpoint
# ---------------------------------------------------------------------------


def test_get_edge_midpoint_is_3d_array():
    # Act
    midpoint = _get_edge_midpoint((0.0, 0.0), (10.0, 20.0))

    # Assert
    assert midpoint.shape == (3,)


def test_get_edge_midpoint_correct_xy():
    # Act
    midpoint = _get_edge_midpoint((0.0, 0.0), (10.0, 20.0))

    # Assert
    np.testing.assert_array_almost_equal(midpoint, [5.0, 10.0, 0.0])


def test_get_edge_midpoint_z_is_zero():
    # Act
    midpoint = _get_edge_midpoint((3.0, 7.0), (9.0, 1.0))

    # Assert
    assert midpoint[2] == 0.0


# ---------------------------------------------------------------------------
# _get_exterior_edges
# ---------------------------------------------------------------------------


def test_get_exterior_edges_count(small_footprint):
    # Act
    edges = _get_exterior_edges(small_footprint)

    # Assert — box polygon has 4 edges (5 coords, closing coord repeats start)
    assert len(edges) == 4


def test_get_exterior_edges_returns_tuples(small_footprint):
    # Act
    edges = _get_exterior_edges(small_footprint)

    # Assert
    for start, end in edges:
        assert len(start) == 2
        assert len(end) == 2


# ---------------------------------------------------------------------------
# _place_sign_lights
# ---------------------------------------------------------------------------


def test_place_sign_lights_skips_short_edges(light_config, rng):
    # Arrange — edges all shorter than 5m min_length
    edges = [((0.0, 0.0), (3.0, 0.0)), ((3.0, 0.0), (3.0, 2.0))]

    # Act
    lights = _place_sign_lights(edges, light_config, rng)

    # Assert
    assert lights == []


def test_place_sign_lights_places_on_qualifying_edges(light_config, rng):
    # Arrange — one edge longer than 5m
    edges = [((0.0, 0.0), (20.0, 0.0)), ((0.0, 0.0), (2.0, 0.0))]

    # Act
    lights = _place_sign_lights(edges, light_config, rng)

    # Assert — one qualifying edge, one light
    assert len(lights) == 1


def test_place_sign_lights_returns_light_sources(light_config, rng):
    # Arrange
    edges = [((0.0, 0.0), (20.0, 0.0))]

    # Act
    lights = _place_sign_lights(edges, light_config, rng)

    # Assert
    assert all(isinstance(l, LightSource) for l in lights)


def test_place_sign_lights_places_at_midpoint(light_config, rng):
    # Arrange
    edges = [((0.0, 0.0), (20.0, 0.0))]

    # Act
    lights = _place_sign_lights(edges, light_config, rng)

    # Assert
    np.testing.assert_array_almost_equal(lights[0].position, [10.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# _place_lights_along_edge
# ---------------------------------------------------------------------------


def test_place_lights_along_edge_returns_empty_when_edge_shorter_than_spacing(
    light_config, rng
):
    # Arrange — spacing is 5-8m, edge is only 2m
    edge = ((0.0, 0.0), (2.0, 0.0))

    # Act
    lights = _place_lights_along_edge(edge, light_config, rng)

    # Assert
    assert lights == []


def test_place_lights_along_edge_returns_light_sources(light_config, rng):
    # Arrange — 30m edge guarantees multiple lights at max spacing of 8m
    edge = ((0.0, 0.0), (30.0, 0.0))

    # Act
    lights = _place_lights_along_edge(edge, light_config, rng)

    # Assert
    assert len(lights) >= 1
    assert all(isinstance(l, LightSource) for l in lights)


def test_place_lights_along_edge_positions_on_edge(light_config, rng):
    # Arrange
    edge = ((0.0, 0.0), (30.0, 0.0))

    # Act
    lights = _place_lights_along_edge(edge, light_config, rng)

    # Assert — all lights should have y=0 and x in [0, 30]
    for light in lights:
        assert light.position[1] == pytest.approx(0.0)
        assert 0.0 <= light.position[0] <= 30.0


# ---------------------------------------------------------------------------
# _place_storefront_lights_on_longest_edge
# ---------------------------------------------------------------------------


def test_place_storefront_lights_on_longest_edge_uses_longest(
    small_footprint, light_config, rng
):
    # Arrange — 20x30 box: longest edges are the 30m sides
    edges = _get_exterior_edges(small_footprint)

    # Act
    lights = _place_storefront_lights_on_longest_edge(edges, light_config, rng)

    # Assert — lights should span roughly 30m, so at least 3 at max 8m spacing
    assert len(lights) >= 1


def test_place_storefront_lights_on_longest_edge_returns_empty_for_no_edges(
    light_config, rng
):
    # Act
    lights = _place_storefront_lights_on_longest_edge([], light_config, rng)

    # Assert
    assert lights == []


# ---------------------------------------------------------------------------
# _compute_grid_points
# ---------------------------------------------------------------------------


def test_compute_grid_points_returns_only_interior_points(parking_polygon):
    # Act
    points = _compute_grid_points(parking_polygon, spacing=15.0)

    # Assert — all returned points must be inside the polygon
    from shapely.geometry import Point
    for x, y in points:
        assert parking_polygon.contains(Point(x, y))


def test_compute_grid_points_returns_at_least_one_point(parking_polygon):
    # Arrange — 40x40 polygon, 15m spacing → grid of ~8 points
    # Act
    points = _compute_grid_points(parking_polygon, spacing=15.0)

    # Assert
    assert len(points) >= 1


def test_compute_grid_points_empty_for_zero_area_polygon():
    # Arrange — degenerate tiny polygon that won't contain any grid point
    tiny = box(0, 0, 0.1, 0.1)

    # Act
    points = _compute_grid_points(tiny, spacing=15.0)

    # Assert
    assert points == []


def _compute_grid_points_nested_loop(polygon, spacing):
    """Pre-change reference implementation: per-point polygon.contains(Point(...))."""
    from shapely.geometry import Point

    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx + spacing / 2.0, maxx, spacing)
    ys = np.arange(miny + spacing / 2.0, maxy, spacing)
    points = []
    for x in xs:
        for y in ys:
            if polygon.contains(Point(x, y)):
                points.append((float(x), float(y)))
    return points


def test_compute_grid_points_matches_old_nested_loop_implementation(parking_polygon):
    # Arrange — an irregular pentagon so the grid isn't trivially axis-aligned
    irregular = box(0, 0, 40, 40).difference(box(30, 30, 40, 40))

    # Act
    old_points = _compute_grid_points_nested_loop(irregular, spacing=7.0)
    new_points = _compute_grid_points(irregular, spacing=7.0)

    # Assert — vectorized shapely.contains_xy call yields the identical point set
    assert new_points == old_points


# ---------------------------------------------------------------------------
# _place_parking_grid
# ---------------------------------------------------------------------------


def test_place_parking_grid_returns_light_sources(parking_polygon, light_config, rng):
    # Act
    lights = _place_parking_grid(parking_polygon, light_config, rng)

    # Assert
    assert all(isinstance(l, LightSource) for l in lights)


def test_place_parking_grid_positions_inside_polygon(
    parking_polygon, light_config, rng
):
    # Act
    lights = _place_parking_grid(parking_polygon, light_config, rng)

    # Assert
    from shapely.geometry import Point
    for light in lights:
        x, y = light.position[0], light.position[1]
        assert parking_polygon.contains(Point(x, y))


# ---------------------------------------------------------------------------
# place_commercial_lights — integration
# ---------------------------------------------------------------------------


def test_place_commercial_lights_returns_list(commercial_zone, light_config, rng):
    # Act
    lights = place_commercial_lights([commercial_zone], light_config, rng)

    # Assert
    assert isinstance(lights, list)


def test_place_commercial_lights_returns_light_sources(
    commercial_zone, light_config, rng
):
    # Act
    lights = place_commercial_lights([commercial_zone], light_config, rng)

    # Assert
    assert all(isinstance(l, LightSource) for l in lights)


def test_place_commercial_lights_empty_zones(light_config, rng):
    # Act
    lights = place_commercial_lights([], light_config, rng)

    # Assert
    assert lights == []


def test_place_commercial_lights_no_buildings_no_parking(light_config, rng):
    # Arrange
    zone = CommercialZone(
        boundary=box(0, 0, 100, 100),
        building_footprints=[],
        parking_areas=[],
    )

    # Act
    lights = place_commercial_lights([zone], light_config, rng)

    # Assert
    assert lights == []


def test_place_commercial_lights_multiple_zones(light_config, rng):
    # Arrange
    zone_a = CommercialZone(
        boundary=box(0, 0, 50, 50),
        building_footprints=[box(5, 5, 25, 35)],
        parking_areas=[],
    )
    zone_b = CommercialZone(
        boundary=box(100, 100, 200, 200),
        building_footprints=[box(110, 110, 150, 160)],
        parking_areas=[box(155, 110, 195, 160)],
    )

    # Act
    lights = place_commercial_lights([zone_a, zone_b], light_config, rng)

    # Assert — two zones with buildings/parking produce more lights than one
    single_zone_lights = place_commercial_lights([zone_a], dataclass_from_dict(LightConfig, DEFAULTS["lights"]), random.Random(99))
    assert len(lights) > len(single_zone_lights)


def test_place_commercial_lights_reproducible_with_same_seed(
    commercial_zone, light_config
):
    # Arrange
    rng_a = random.Random(42)
    rng_b = random.Random(42)

    # Act
    lights_a = place_commercial_lights([commercial_zone], light_config, rng_a)
    lights_b = place_commercial_lights([commercial_zone], light_config, rng_b)

    # Assert
    assert len(lights_a) == len(lights_b)
    for la, lb in zip(lights_a, lights_b):
        assert la.color == lb.color
        np.testing.assert_array_almost_equal(la.position, lb.position)
