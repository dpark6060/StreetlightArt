# Radical
"""Tests for streets_at_night.placement.car_placer module."""

import math
import random
from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import LightConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.models.car import CarLightPair
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.road_segment import RoadSegment
from streets_at_night.models.road_types import RoadType
from streets_at_night.placement.car_placer import (
    _bearing_from_tangent,
    _create_car_at_point,
    _get_density_range,
    _offset_to_lane,
    _perpendicular_2d,
    _walk_with_random_spacing,
    place_cars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def light_config():
    """Return a default LightConfig for tests."""
    return dataclass_from_dict(LightConfig, DEFAULTS["lights"])


@pytest.fixture
def straight_segment() -> RoadSegment:
    """Return a 500m straight road segment along the X axis."""
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0],
    ])
    return RoadSegment(geometry=geometry, road_type=RoadType.MAJOR_ROAD, width=8.0)


@pytest.fixture
def highway_segment() -> RoadSegment:
    """Return a 500m highway segment."""
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0],
    ])
    return RoadSegment(geometry=geometry, road_type=RoadType.HIGHWAY, width=12.0)


# ---------------------------------------------------------------------------
# place_cars — return types and structure
# ---------------------------------------------------------------------------


def test_place_cars_returns_tuple_of_lights_and_pairs(straight_segment, light_config, rng):
    # Act
    lights, pairs = place_cars([straight_segment], light_config, rng)

    # Assert — both collections returned
    assert isinstance(lights, list)
    assert isinstance(pairs, list)


def test_place_cars_lights_are_light_source_instances(straight_segment, light_config, rng):
    # Act
    lights, _ = place_cars([straight_segment], light_config, rng)

    # Assert
    assert all(isinstance(l, LightSource) for l in lights)


def test_place_cars_pairs_are_car_light_pair_instances(straight_segment, light_config, rng):
    # Act
    _, pairs = place_cars([straight_segment], light_config, rng)

    # Assert
    assert all(isinstance(p, CarLightPair) for p in pairs)


def test_place_cars_each_pair_contributes_three_lights(straight_segment, light_config, rng):
    # Act
    lights, pairs = place_cars([straight_segment], light_config, rng)

    # Assert — each CarLightPair produces two point lights plus the cone carrier
    assert len(lights) == len(pairs) * 3


def test_place_cars_produces_cars_on_long_segment(straight_segment, light_config, rng):
    # Act — major road has density 40-80m, 500m segment should yield multiple cars
    _, pairs = place_cars([straight_segment], light_config, rng)

    # Assert
    assert len(pairs) > 0


def test_place_cars_empty_segment_list_returns_empty(light_config, rng):
    # Act
    lights, pairs = place_cars([], light_config, rng)

    # Assert
    assert lights == []
    assert pairs == []


def test_place_cars_same_seed_reproducible(straight_segment, light_config):
    # Act
    lights_a, _ = place_cars([straight_segment], light_config, random.Random(5))
    lights_b, _ = place_cars([straight_segment], light_config, random.Random(5))

    # Assert
    assert len(lights_a) == len(lights_b)


def test_place_cars_shared_pools_preserve_seeded_colors(straight_segment, light_config):
    # Arrange — reference path: build a fresh headlight/taillight ColorPool
    # per car (the pre-change behavior), rather than the shared-per-run pools
    # place_cars now builds. ColorPool construction draws no RNG, so this
    # should produce identical colors to the shared-pool path for the same seed.
    camera_forward = np.array([-1.0, 0.0, 0.0])
    density_min, density_max = _get_density_range(straight_segment.road_type, light_config.cars)
    old_rng = random.Random(123)
    sample_points, sample_tangents = _walk_with_random_spacing(
        straight_segment.geometry, density_min, density_max, old_rng
    )
    old_pairs = [
        _create_car_at_point(point, tangent, straight_segment.road_type, light_config.cars, old_rng)
        for point, tangent in zip(sample_points, sample_tangents)
    ]
    for pair in old_pairs:
        pair.resolve_facing(camera_forward)
    old_colors = [light.color for pair in old_pairs for light in pair.get_lights()]

    # Act — current shared-pool path via place_cars
    _, new_pairs = place_cars([straight_segment], light_config, random.Random(123))
    for pair in new_pairs:
        pair.resolve_facing(camera_forward)
    new_colors = [light.color for pair in new_pairs for light in pair.get_lights()]

    # Assert
    assert old_colors == new_colors


# ---------------------------------------------------------------------------
# _get_density_range
# ---------------------------------------------------------------------------


def test_get_density_range_highway_returns_highway_values(light_config):
    # Act
    density_min, density_max = _get_density_range(RoadType.HIGHWAY, light_config.cars)

    # Assert
    assert density_min == light_config.cars.density_highway_min
    assert density_max == light_config.cars.density_highway_max


def test_get_density_range_major_returns_major_values(light_config):
    # Act
    density_min, density_max = _get_density_range(RoadType.MAJOR_ROAD, light_config.cars)

    # Assert
    assert density_min == light_config.cars.density_major_min
    assert density_max == light_config.cars.density_major_max


def test_get_density_range_residential_returns_residential_values(light_config):
    # Act
    density_min, density_max = _get_density_range(RoadType.RESIDENTIAL, light_config.cars)

    # Assert
    assert density_min == light_config.cars.density_residential_min
    assert density_max == light_config.cars.density_residential_max


def test_get_density_range_minor_falls_back_to_residential(light_config):
    # Arrange — MINOR is not in CarConfig, should use residential density
    minor_min, minor_max = _get_density_range(RoadType.MINOR, light_config.cars)
    res_min, res_max = _get_density_range(RoadType.RESIDENTIAL, light_config.cars)

    # Assert
    assert minor_min == res_min
    assert minor_max == res_max


def test_get_density_range_service_returns_service_values(light_config):
    # Act
    density_min, density_max = _get_density_range(RoadType.SERVICE, light_config.cars)

    # Assert
    assert density_min == light_config.cars.density_service_min
    assert density_max == light_config.cars.density_service_max


# ---------------------------------------------------------------------------
# _bearing_from_tangent
# ---------------------------------------------------------------------------


def test_bearing_from_tangent_north_pointing_tangent():
    # Arrange — tangent pointing north (y-positive in ENU)
    tangent = np.array([0.0, 1.0, 0.0])

    # Act
    bearing = _bearing_from_tangent(tangent)

    # Assert — math convention: atan2(1, 0) = pi/2 (north)
    assert bearing == pytest.approx(math.pi / 2.0)


def test_bearing_from_tangent_east_pointing_tangent():
    # Arrange — tangent pointing east (x-positive in ENU)
    tangent = np.array([1.0, 0.0, 0.0])

    # Act
    bearing = _bearing_from_tangent(tangent)

    # Assert — math convention: atan2(0, 1) = 0 (east)
    assert bearing == pytest.approx(0.0)


def test_bearing_from_tangent_opposite_direction_differs_by_pi():
    # Arrange
    tangent_fwd = np.array([1.0, 0.0, 0.0])
    tangent_rev = np.array([-1.0, 0.0, 0.0])

    # Act
    bearing_fwd = _bearing_from_tangent(tangent_fwd)
    bearing_rev = _bearing_from_tangent(tangent_rev)

    # Assert — opposite directions differ by pi
    assert abs(abs(bearing_fwd - bearing_rev) - math.pi) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _offset_to_lane
# ---------------------------------------------------------------------------


def test_offset_to_lane_moves_position_off_centerline():
    # Arrange
    center = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    # Act
    pos = _offset_to_lane(center, tangent, 1)

    # Assert — position is no longer at center
    assert not np.allclose(pos[:2], center[:2])


def test_offset_to_lane_opposite_signs_are_symmetric():
    # Arrange
    center = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    # Act
    pos_left = _offset_to_lane(center, tangent, 1)
    pos_right = _offset_to_lane(center, tangent, -1)

    # Assert — symmetric about centerline
    midpoint = (pos_left + pos_right) / 2.0
    np.testing.assert_allclose(midpoint[:2], center[:2], atol=1e-9)


# ---------------------------------------------------------------------------
# _perpendicular_2d
# ---------------------------------------------------------------------------


def test_car_placer_perpendicular_2d_returns_unit_vector():
    # Arrange
    tangent = np.array([3.0, 4.0, 0.0])

    # Act
    perp = _perpendicular_2d(tangent)

    # Assert
    assert np.linalg.norm(perp) == pytest.approx(1.0, abs=1e-9)


def test_car_placer_perpendicular_2d_is_orthogonal():
    # Arrange
    tangent = np.array([0.0, 1.0, 0.0])

    # Act
    perp = _perpendicular_2d(tangent)

    # Assert
    assert abs(np.dot(tangent, perp)) < 1e-9


# ---------------------------------------------------------------------------
# _walk_with_random_spacing
# ---------------------------------------------------------------------------


def test_walk_with_random_spacing_returns_positions_within_segment(rng):
    # Arrange
    geometry = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])

    # Act
    positions, tangents = _walk_with_random_spacing(geometry, 40.0, 80.0, rng)

    # Assert — at least one car on a 500m segment with 40-80m spacing
    assert len(positions) > 0
    assert len(positions) == len(tangents)


def test_walk_with_random_spacing_positions_along_segment(rng):
    # Arrange — segment along X axis from 0 to 500
    geometry = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])

    # Act
    positions, _ = _walk_with_random_spacing(geometry, 40.0, 80.0, rng)

    # Assert — all X coordinates are within the segment range
    for pos in positions:
        assert 0.0 <= pos[0] <= 500.0


def test_walk_with_random_spacing_empty_for_short_segment(rng):
    # Arrange — segment much shorter than minimum spacing
    geometry = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    # Act
    positions, tangents = _walk_with_random_spacing(geometry, 40.0, 80.0, rng)

    # Assert
    assert positions == []
    assert tangents == []


def test_walk_with_random_spacing_same_seed_reproducible():
    # Arrange
    geometry = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])

    # Act
    positions_a, _ = _walk_with_random_spacing(geometry, 40.0, 80.0, random.Random(3))
    positions_b, _ = _walk_with_random_spacing(geometry, 40.0, 80.0, random.Random(3))

    # Assert
    assert len(positions_a) == len(positions_b)
    for pa, pb in zip(positions_a, positions_b):
        np.testing.assert_array_equal(pa, pb)


# ---------------------------------------------------------------------------
# Highway vs bidirectional direction logic
# ---------------------------------------------------------------------------


def test_place_cars_highway_cars_have_valid_bearings(highway_segment, light_config):
    # Arrange — seed chosen to yield cars on the segment
    _, pairs = place_cars([highway_segment], light_config, random.Random(1))

    # Assert — all bearings are finite numbers
    assert len(pairs) > 0
    for pair in pairs:
        assert math.isfinite(pair.travel_bearing)


def test_place_cars_highway_denser_than_residential(light_config):
    # Arrange — same length for both segment types, same seed
    geometry = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
    highway_seg = RoadSegment(geometry=geometry, road_type=RoadType.HIGHWAY, width=12.0)
    residential_seg = RoadSegment(geometry=geometry, road_type=RoadType.RESIDENTIAL, width=4.0)

    # Act
    _, highway_pairs = place_cars([highway_seg], light_config, random.Random(7))
    _, residential_pairs = place_cars([residential_seg], light_config, random.Random(7))

    # Assert — highway density_min/max is lower spacing (more cars per km) than residential
    assert len(highway_pairs) > len(residential_pairs)


# ---------------------------------------------------------------------------
# Bearing convention regression — placer and CarLightPair must agree
# ---------------------------------------------------------------------------


def test_create_car_at_point_east_tangent_headlights_face_oncoming_camera(light_config):
    # Arrange — east-west tangent, force lane_sign=+1 so travel_bearing is
    # exactly the road bearing (east, i.e. +x). Under the old (buggy) placer
    # convention this car would be interpreted as traveling north-south.
    point = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    rng = mock.MagicMock(spec=random.Random)
    rng.choice.return_value = 1
    rng.uniform.return_value = 2.0

    # Act
    pair = _create_car_at_point(point, tangent, RoadType.HIGHWAY, light_config.cars, rng)
    pair.resolve_facing(np.array([-1.0, 0.0, 0.0]))
    lights = pair.get_lights()

    # Assert — camera looking west at an eastbound car sees headlights on the
    # two point lights (the cone carrier has no glow)
    assert all(light.glow_radius == light_config.cars.headlight_glow_radius for light in lights[:2])


def test_create_car_at_point_east_west_road_lights_separated_along_y(light_config):
    # Arrange — same setup as above; the visible symptom of the bearing
    # mismatch was that the two lights ended up separated along the travel
    # axis (x) instead of perpendicular to it (y).
    point = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])
    rng = mock.MagicMock(spec=random.Random)
    rng.choice.return_value = 1
    rng.uniform.return_value = 2.0

    # Act
    pair = _create_car_at_point(point, tangent, RoadType.HIGHWAY, light_config.cars, rng)
    light_a, light_b = pair.get_lights()[:2]

    # Assert — lights differ in y (perpendicular to travel), match in x
    assert abs(light_a.position[1] - light_b.position[1]) == pytest.approx(2.0)
    assert light_a.position[0] == pytest.approx(light_b.position[0])
