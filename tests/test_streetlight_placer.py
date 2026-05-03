# Radical
"""Tests for streets_at_night.placement.streetlight_placer module."""

import random
from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_streetlight_config
from streets_at_night.models.light_source import LightSource
from streets_at_night.models.road_segment import RoadSegment
from streets_at_night.models.road_types import RoadType
from streets_at_night.models.streetlight import FixtureProfile
from streets_at_night.placement.streetlight_placer import (
    _compute_arc_lengths,
    _compute_light_position,
    _get_sample_distances,
    _get_sides_for_step,
    _perpendicular_2d,
    place_streetlights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG."""
    return random.Random(42)


@pytest.fixture
def sample_profile() -> FixtureProfile:
    """Return a FixtureProfile with well-known values for testing."""
    return FixtureProfile(
        fixture_color=(255, 160, 64),
        height=9.0,
        brightness=0.8,
        glow_radius=18.0,
        illumination_radius=30.0,
        illumination_intensity=0.75,
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
        spacing=50.0,
        side="both",
    )


@pytest.fixture
def straight_segment(sample_profile) -> RoadSegment:
    """Return a 200m straight road segment along the X axis."""
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ])
    seg = RoadSegment(geometry=geometry, road_type=RoadType.MAJOR_ROAD, width=8.0)
    seg.fixture_profile = sample_profile
    return seg


@pytest.fixture
def short_segment() -> RoadSegment:
    """Return a 10m segment — too short to fit a 50m-spaced light."""
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    seg = RoadSegment(geometry=geometry, road_type=RoadType.MAJOR_ROAD, width=8.0)
    seg.fixture_profile = FixtureProfile(
        fixture_color=(255, 160, 64),
        height=9.0,
        brightness=0.8,
        glow_radius=18.0,
        illumination_radius=30.0,
        illumination_intensity=0.75,
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
        spacing=50.0,
        side="both",
    )
    return seg


# ---------------------------------------------------------------------------
# place_streetlights — integration contract
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_assigns_fixture_profile_to_each_segment(
    mock_pool_factory, sample_profile, rng
):
    # Arrange
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = sample_profile
    mock_pool_factory.return_value = mock_pool

    geometry = np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    segments = [
        RoadSegment(geometry=geometry, road_type=RoadType.MAJOR_ROAD, width=8.0),
        RoadSegment(geometry=geometry, road_type=RoadType.RESIDENTIAL, width=4.0),
    ]
    config = mock.MagicMock()

    # Act
    place_streetlights(segments, config, rng)

    # Assert — each segment got a fixture_profile assigned
    for seg in segments:
        assert seg.fixture_profile is sample_profile


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_skips_segments_with_none_profile(
    mock_pool_factory, rng
):
    # Arrange
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = None
    mock_pool_factory.return_value = mock_pool

    geometry = np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    segment = RoadSegment(geometry=geometry, road_type=RoadType.RESIDENTIAL, width=4.0)
    config = mock.MagicMock()

    # Act
    lights = place_streetlights([segment], config, rng)

    # Assert — no lights produced for dark segment
    assert lights == []
    assert segment.fixture_profile is None


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_returns_light_sources(
    mock_pool_factory, sample_profile, straight_segment, rng
):
    # Arrange
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = sample_profile
    mock_pool_factory.return_value = mock_pool
    config = mock.MagicMock()
    straight_segment.fixture_profile = sample_profile

    # Act
    lights = place_streetlights([straight_segment], config, rng)

    # Assert
    assert len(lights) > 0
    assert all(isinstance(l, LightSource) for l in lights)


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_both_sides_produces_even_count(
    mock_pool_factory, rng
):
    # Arrange — "both" side produces exactly two lights per spacing step
    profile = FixtureProfile(
        fixture_color=(255, 160, 64),
        height=9.0,
        brightness=0.8,
        glow_radius=18.0,
        illumination_radius=30.0,
        illumination_intensity=0.75,
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
        spacing=50.0,
        side="both",
    )
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = profile
    mock_pool_factory.return_value = mock_pool

    geometry = np.array([[0.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    segment = RoadSegment(geometry=geometry, road_type=RoadType.MAJOR_ROAD, width=8.0)
    config = mock.MagicMock()

    # Act
    lights = place_streetlights([segment], config, rng)

    # Assert — "both" side produces a paired (even) count of lights
    assert len(lights) % 2 == 0


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_same_seed_reproducible(mock_pool_factory, sample_profile):
    # Arrange
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = sample_profile
    mock_pool_factory.return_value = mock_pool

    geometry = np.array([[0.0, 0.0, 0.0], [300.0, 0.0, 0.0]])
    config = mock.MagicMock()

    def make_segment():
        return RoadSegment(geometry=geometry, road_type=RoadType.MAJOR_ROAD, width=8.0)

    # Act
    lights_a = place_streetlights([make_segment()], config, random.Random(7))
    lights_b = place_streetlights([make_segment()], config, random.Random(7))

    # Assert
    assert len(lights_a) == len(lights_b)
    for la, lb in zip(lights_a, lights_b):
        np.testing.assert_array_equal(la.position, lb.position)


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_short_segment_produces_no_lights(
    mock_pool_factory, short_segment, rng
):
    # Arrange
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = short_segment.fixture_profile
    mock_pool_factory.return_value = mock_pool
    config = mock.MagicMock()

    # Act
    lights = place_streetlights([short_segment], config, rng)

    # Assert
    assert lights == []


@mock.patch("streets_at_night.placement.streetlight_placer.create_fixture_pool")
def test_place_streetlights_empty_segments_returns_empty_list(mock_pool_factory, rng):
    # Arrange
    mock_pool = mock.MagicMock()
    mock_pool.pick.return_value = None
    mock_pool_factory.return_value = mock_pool
    config = mock.MagicMock()

    # Act
    lights = place_streetlights([], config, rng)

    # Assert — no lights produced for empty segment list
    assert lights == []


def test_place_streetlights_residential_pool_can_produce_none_profile():
    # Arrange — use the real residential pool, which has 25% None weight
    # Run enough iterations to statistically guarantee we see None
    from streets_at_night.models.streetlight import create_fixture_pool

    rng = random.Random(0)
    streetlight_config = _build_streetlight_config(DEFAULTS["lights"]["streetlights"])
    pool = create_fixture_pool(RoadType.RESIDENTIAL, streetlight_config, rng)

    # Act — draw many picks and collect results
    picks = [pool.pick() for _ in range(200)]

    # Assert — at least one None profile must appear (25% weight → nearly certain in 200 draws)
    assert any(p is None for p in picks)


def test_place_streetlights_highway_pool_never_produces_none_profile():
    # Arrange — highway pool has no None entry
    from streets_at_night.models.streetlight import create_fixture_pool

    rng = random.Random(99)
    streetlight_config = _build_streetlight_config(DEFAULTS["lights"]["streetlights"])
    pool = create_fixture_pool(RoadType.HIGHWAY, streetlight_config, rng)

    # Act — draw many picks
    picks = [pool.pick() for _ in range(100)]

    # Assert — highway segments must always have a fixture profile
    assert all(p is not None for p in picks)


# ---------------------------------------------------------------------------
# _get_sides_for_step
# ---------------------------------------------------------------------------


def test_get_sides_for_step_both_returns_two_sides():
    # Act
    sides = _get_sides_for_step("both", 0, 1)

    # Assert
    assert len(sides) == 2
    assert 1 in sides
    assert -1 in sides


def test_get_sides_for_step_one_returns_side_choice_positive():
    # Act
    sides = _get_sides_for_step("one", 0, 1)

    # Assert
    assert sides == [1]


def test_get_sides_for_step_one_returns_side_choice_negative():
    # Act
    sides = _get_sides_for_step("one", 0, -1)

    # Assert
    assert sides == [-1]


def test_get_sides_for_step_alternating_flips_between_steps():
    # Act
    side_even = _get_sides_for_step("alternating", 0, 1)
    side_odd = _get_sides_for_step("alternating", 1, 1)

    # Assert — must be different sides
    assert side_even != side_odd


# ---------------------------------------------------------------------------
# _compute_light_position
# ---------------------------------------------------------------------------


def test_compute_light_position_z_equals_pole_height():
    # Arrange
    point = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    # Act
    pos = _compute_light_position(point, tangent, 1, road_width=8.0, pole_height=9.0)

    # Assert
    assert pos[2] == pytest.approx(9.0)


def test_compute_light_position_offset_perpendicular_to_tangent():
    # Arrange — tangent along X axis; CCW perpendicular is +Y
    point = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    # Act — side_sign=+1 offsets in the perpendicular direction
    pos = _compute_light_position(point, tangent, 1, road_width=8.0, pole_height=0.0)

    # Assert — offset equals half road width
    expected_offset = 4.0
    assert abs(pos[1]) == pytest.approx(expected_offset)
    assert pos[0] == pytest.approx(0.0)


def test_compute_light_position_negative_side_offsets_opposite():
    # Arrange
    point = np.array([0.0, 0.0, 0.0])
    tangent = np.array([1.0, 0.0, 0.0])

    # Act
    pos_left = _compute_light_position(point, tangent, 1, road_width=8.0, pole_height=0.0)
    pos_right = _compute_light_position(point, tangent, -1, road_width=8.0, pole_height=0.0)

    # Assert — symmetric about centerline
    assert pos_left[1] == pytest.approx(-pos_right[1])


# ---------------------------------------------------------------------------
# _perpendicular_2d
# ---------------------------------------------------------------------------


def test_perpendicular_2d_returns_unit_vector():
    # Arrange
    tangent = np.array([3.0, 4.0, 0.0])

    # Act
    perp = _perpendicular_2d(tangent)

    # Assert
    assert np.linalg.norm(perp) == pytest.approx(1.0, abs=1e-9)


def test_perpendicular_2d_is_orthogonal_to_tangent():
    # Arrange
    tangent = np.array([1.0, 0.0, 0.0])

    # Act
    perp = _perpendicular_2d(tangent)

    # Assert
    assert abs(np.dot(tangent, perp)) < 1e-9


def test_perpendicular_2d_degenerate_tangent_returns_fallback():
    # Arrange — near-zero vector
    tangent = np.array([0.0, 0.0, 0.0])

    # Act
    perp = _perpendicular_2d(tangent)

    # Assert — fallback must be a valid unit vector
    assert np.linalg.norm(perp) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _compute_arc_lengths
# ---------------------------------------------------------------------------


def test_compute_arc_lengths_starts_at_zero():
    # Arrange
    geometry = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    # Act
    cum, _ = _compute_arc_lengths(geometry)

    # Assert
    assert cum[0] == pytest.approx(0.0)


def test_compute_arc_lengths_total_equals_segment_length():
    # Arrange
    geometry = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    # Act
    cum, seg_lens = _compute_arc_lengths(geometry)

    # Assert
    assert cum[-1] == pytest.approx(100.0)
    assert seg_lens[0] == pytest.approx(100.0)


def test_compute_arc_lengths_multi_segment_sums_correctly():
    # Arrange — two 50m segments
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [50.0, 0.0, 0.0],
        [50.0, 50.0, 0.0],
    ])

    # Act
    cum, seg_lens = _compute_arc_lengths(geometry)

    # Assert
    assert cum[-1] == pytest.approx(100.0)
    assert len(seg_lens) == 2


# ---------------------------------------------------------------------------
# _get_sample_distances
# ---------------------------------------------------------------------------


def test_get_sample_distances_empty_when_total_less_than_half_spacing():
    # _get_sample_distances starts at spacing/2; if total_length < spacing/2
    # there are no valid sample points.
    # Act
    distances = _get_sample_distances(total_length=10.0, spacing=50.0)

    # Assert
    assert distances == []


def test_get_sample_distances_starts_at_half_spacing():
    # Act
    distances = _get_sample_distances(total_length=200.0, spacing=50.0)

    # Assert — first sample is at spacing/2
    assert distances[0] == pytest.approx(25.0)


def test_get_sample_distances_spaced_evenly():
    # Act
    distances = _get_sample_distances(total_length=300.0, spacing=50.0)

    # Assert — consecutive distances differ by spacing
    for i in range(1, len(distances)):
        assert distances[i] - distances[i - 1] == pytest.approx(50.0)
