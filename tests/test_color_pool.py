# Radical
"""Tests for streets_at_night.models.color_pool.ColorPool."""

import random

import pytest

from streets_at_night.models.color_pool import ColorPool
from streets_at_night.utils.colors import hex_to_rgb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_rng():
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def simple_pool(seeded_rng):
    """Return a ColorPool with two colors and equal weights."""
    colors = [("#FF0000", 1.0), ("#0000FF", 1.0)]
    return ColorPool(colors, seeded_rng)


# ---------------------------------------------------------------------------
# pick — return type and value range
# ---------------------------------------------------------------------------


def test_pick_returns_rgb_tuple(simple_pool):
    # Arrange — pool already constructed

    # Act
    result = simple_pool.pick()

    # Assert
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_pick_values_in_valid_range(simple_pool):
    # Arrange — sample many picks to exercise both colors

    # Act
    results = [simple_pool.pick() for _ in range(50)]

    # Assert
    for r, g, b in results:
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255


def test_seeded_rng_reproducible():
    # Arrange — two pools with identical seeds and palettes
    colors = [("#AABBCC", 1.0), ("#112233", 1.0), ("#FFFFFF", 0.5)]
    rng_a = random.Random(99)
    rng_b = random.Random(99)
    pool_a = ColorPool(colors, rng_a)
    pool_b = ColorPool(colors, rng_b)

    # Act
    picks_a = [pool_a.pick() for _ in range(20)]
    picks_b = [pool_b.pick() for _ in range(20)]

    # Assert
    assert picks_a == picks_b


def test_single_color_always_picked():
    # Arrange — pool with exactly one entry
    only_color = "#0D0A08"
    expected = (13, 10, 8)
    pool = ColorPool([(only_color, 1.0)], random.Random(7))

    # Act
    results = [pool.pick() for _ in range(30)]

    # Assert
    assert all(r == expected for r in results)


# ---------------------------------------------------------------------------
# hex_to_rgb — conversion
# ---------------------------------------------------------------------------


def test_hex_to_rgb_valid_red():
    # Arrange — pure red in uppercase hex

    # Act
    result = hex_to_rgb("#FF0000")

    # Assert
    assert result == (255, 0, 0)


def test_hex_to_rgb_valid_dark_color():
    # Arrange — a specific dark color used in the docstring example

    # Act
    result = hex_to_rgb("#0D0A08")

    # Assert
    assert result == (13, 10, 8)


def test_hex_to_rgb_lowercase():
    # Arrange — same red value in lowercase

    # Act
    result = hex_to_rgb("#ff0000")

    # Assert
    assert result == (255, 0, 0)


# ---------------------------------------------------------------------------
# weights — distribution behaviour
# ---------------------------------------------------------------------------


def test_weights_affect_distribution():
    # Arrange — color_a has overwhelmingly higher weight than color_b
    color_a = "#FFFFFF"
    color_b = "#000000"
    rng = random.Random(1)
    pool = ColorPool([(color_a, 100.0), (color_b, 0.001)], rng)

    # Act
    picks = [pool.pick() for _ in range(200)]

    # Assert
    count_a = sum(1 for p in picks if p == (255, 255, 255))
    count_b = sum(1 for p in picks if p == (0, 0, 0))
    assert count_a > count_b * 10
