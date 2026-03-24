# Radical
"""Tests for streets_at_night.models.range_param.RangeParam."""

import random

import pytest

from streets_at_night.models.range_param import RangeParam


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_rng():
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


# ---------------------------------------------------------------------------
# pick — distribution range bounds
# ---------------------------------------------------------------------------


def test_pick_uniform_within_range(seeded_rng):
    # Arrange
    param = RangeParam(2.0, 10.0, "uniform", seeded_rng)

    # Act
    results = [param.pick() for _ in range(100)]

    # Assert
    assert all(2.0 <= v <= 10.0 for v in results)


def test_pick_normal_within_range(seeded_rng):
    # Arrange — normal is clamped so all values must be in range
    param = RangeParam(0.0, 100.0, "normal", seeded_rng)

    # Act
    results = [param.pick() for _ in range(200)]

    # Assert
    assert all(0.0 <= v <= 100.0 for v in results)


def test_pick_exponential_within_range(seeded_rng):
    # Arrange
    param = RangeParam(5.0, 15.0, "exponential", seeded_rng)

    # Act
    results = [param.pick() for _ in range(100)]

    # Assert
    assert all(5.0 <= v <= 15.0 for v in results)


# ---------------------------------------------------------------------------
# pick — reproducibility
# ---------------------------------------------------------------------------


def test_seeded_rng_reproducible():
    # Arrange — two params with identical seeds and config
    rng_a = random.Random(7)
    rng_b = random.Random(7)
    param_a = RangeParam(1.0, 5.0, "uniform", rng_a)
    param_b = RangeParam(1.0, 5.0, "uniform", rng_b)

    # Act
    picks_a = [param_a.pick() for _ in range(20)]
    picks_b = [param_b.pick() for _ in range(20)]

    # Assert
    assert picks_a == picks_b


# ---------------------------------------------------------------------------
# __init__ — validation errors
# ---------------------------------------------------------------------------


def test_invalid_distribution_raises(seeded_rng):
    # Arrange — unsupported distribution name

    # Act & Assert
    with pytest.raises(ValueError):
        RangeParam(0.0, 1.0, "gamma", seeded_rng)


def test_min_greater_than_max_raises(seeded_rng):
    # Arrange — min > max is invalid

    # Act & Assert
    with pytest.raises(ValueError):
        RangeParam(10.0, 5.0, "uniform", seeded_rng)


def test_min_equal_to_max_raises(seeded_rng):
    # Arrange — min == max is also invalid (not strictly less than)

    # Act & Assert
    with pytest.raises(ValueError):
        RangeParam(5.0, 5.0, "uniform", seeded_rng)
