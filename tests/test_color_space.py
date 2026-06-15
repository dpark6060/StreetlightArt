"""Tests for the color_space conversion and tone-mapping helpers."""

import numpy as np
import pytest

from streets_at_night.rendering.color_space import (
    hex_to_linear,
    linear_to_srgb,
    srgb255_to_linear,
    srgb_to_linear,
    tonemap_aces,
)


def test_srgb_to_linear_maps_endpoints_to_themselves():
    result = srgb_to_linear(np.array([0.0, 1.0], dtype=np.float32))

    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert result[1] == pytest.approx(1.0, abs=1e-6)


def test_srgb_to_linear_midpoint_known_value():
    # sRGB 0.5 -> linear ~0.214 is the canonical anchor for the transfer curve.
    result = srgb_to_linear(np.array([0.5], dtype=np.float32))

    assert result[0] == pytest.approx(0.2140, abs=1e-3)


def test_srgb_to_linear_uses_linear_segment_below_threshold():
    # Below 0.04045 the curve is a straight line s/12.92.
    result = srgb_to_linear(np.array([0.02], dtype=np.float32))

    assert result[0] == pytest.approx(0.02 / 12.92, abs=1e-6)


def test_linear_to_srgb_is_inverse_of_srgb_to_linear():
    srgb = np.linspace(0.0, 1.0, 50, dtype=np.float32)

    round_trip = linear_to_srgb(srgb_to_linear(srgb))

    np.testing.assert_allclose(round_trip, srgb, atol=1e-4)


def test_linear_to_srgb_clips_values_above_one():
    result = linear_to_srgb(np.array([5.0], dtype=np.float32))

    assert result[0] == pytest.approx(1.0, abs=1e-6)


def test_hex_to_linear_white_is_one():
    result = hex_to_linear("#FFFFFF")

    assert result == pytest.approx((1.0, 1.0, 1.0), abs=1e-6)


def test_hex_to_linear_matches_srgb_decode():
    # #BFBFBF is sRGB 0.75; linear ~0.522.
    result = hex_to_linear("#BFBFBF")

    assert result[0] == pytest.approx(0.522, abs=2e-3)


def test_srgb255_to_linear_matches_hex_path():
    from_tuple = srgb255_to_linear((255, 160, 64))
    from_hex = hex_to_linear("#FFA040")

    assert from_tuple == pytest.approx(from_hex, abs=1e-6)


def test_tonemap_aces_maps_zero_to_zero():
    result = tonemap_aces(np.array([0.0], dtype=np.float32))

    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_tonemap_aces_is_monotonic_and_bounded():
    x = np.linspace(0.0, 20.0, 200, dtype=np.float32)

    result = tonemap_aces(x)

    assert np.all(np.diff(result) >= -1e-7)
    assert result.max() <= 1.0
    assert result.min() >= 0.0


def test_tonemap_aces_rolls_off_bright_values_without_hard_clip():
    # Two half-bright lights summing to 1.2 linear should tone-map below 1.0,
    # i.e. roll off rather than the old hard clamp to white.
    result = tonemap_aces(np.array([1.2], dtype=np.float32))

    assert 0.7 < result[0] < 1.0
