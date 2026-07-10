# Radical
"""Tests for devmode field_meta: substring rule matching, precedence, fallback."""

import pytest

from streets_at_night.devmode.field_meta import FieldMeta, get_field_meta


@pytest.mark.parametrize(
    "field_name, sample_value, expected_min, expected_max, expected_int",
    [
        # sample_value sits inside each rule's base range so containment does
        # not widen it — this isolates rule matching.
        ("color_temp_k", 4000.0, 1000.0, 10000.0, False),
        ("window_lit_fraction", 0.5, 0.0, 1.0, False),
        ("saturation", 1.0, 0.0, 2.0, False),
        ("beam_angle_deg", 90.0, 10.0, 170.0, False),
        ("illumination_inner_fraction", 0.5, 0.0, 1.0, False),
        ("illumination_falloff", 2.0, 0.1, 8.0, False),
        ("illumination_intensity", 0.5, 0.0, 1.0, False),
        ("illumination_radius", 30.0, 0.0, 60.0, False),
        ("glow_radius", 50.0, 0.0, 120.0, False),
        ("brightness_max", 5.0, 0.0, 10.0, False),
        ("height_min", 10.0, 0.0, 30.0, False),
        ("spacing_min", 50.0, 1.0, 300.0, False),
        ("separation_max", 5.0, 0.5, 10.0, False),
        ("density_highway_min", 0.1, 0.0, 0.2, False),
    ],
)
def test_get_field_meta_matches_rules(field_name, sample_value, expected_min, expected_max, expected_int):
    # Act
    meta = get_field_meta(field_name, sample_value)

    # Assert
    assert meta.min_val == expected_min
    assert meta.max_val == expected_max
    assert meta.is_int == expected_int


def test_get_field_meta_knee_range():
    # Act
    meta = get_field_meta("knee_start", 1.5)

    # Assert
    assert (meta.min_val, meta.max_val) == (0.0, 10.0)


def test_get_field_meta_exposure_range():
    # Act
    meta = get_field_meta("exposure", 0.0)

    # Assert
    assert (meta.min_val, meta.max_val) == (-3.0, 3.0)


def test_get_field_meta_color_temp_shift_precedes_color_temp():
    # Act — "color_temp_shift_k" contains both substrings; shift must win.
    meta = get_field_meta("color_temp_shift_k", 0.0)

    # Assert
    assert meta.min_val == -3000.0
    assert meta.max_val == 3000.0


def test_get_field_meta_extinction_strength_range():
    # Act
    meta = get_field_meta("window_extinction_strength", 1.0)

    # Assert
    assert (meta.min_val, meta.max_val) == (0.0, 2.0)


def test_get_field_meta_value_above_rule_max_expands_max_with_headroom():
    # Act — brightness rule caps at 10; an authored 40 must not be clamped.
    meta = get_field_meta("brightness_max", 40.0)

    # Assert — max lifts to value*1.5 (headroom), min unchanged.
    assert meta.min_val == 0.0
    assert meta.max_val == 60.0


def test_get_field_meta_value_at_rule_max_gets_headroom():
    # Act — a value sitting exactly at the cap should not be pinned there.
    meta = get_field_meta("brightness_max", 10.0)

    # Assert
    assert meta.max_val == 15.0


def test_get_field_meta_value_below_rule_min_expands_min():
    # Act — spacing rule floors at 1.0; an authored 0.5 must fit.
    meta = get_field_meta("spacing_min", 0.5)

    # Assert — min drops to value*0.5, max unchanged.
    assert meta.min_val == 0.25
    assert meta.max_val == 300.0


def test_get_field_meta_negative_below_shift_min_expands_min():
    # Act — signed shift lever authored beyond -3000.
    meta = get_field_meta("color_temp_shift_k", -5000.0)

    # Assert — min goes further negative, max unchanged.
    assert meta.min_val == -7500.0
    assert meta.max_val == 3000.0


def test_get_field_meta_value_inside_rule_range_unchanged():
    # Act
    meta = get_field_meta("glow_radius", 50.0)

    # Assert — comfortably inside 0..120, no widening.
    assert meta.min_val == 0.0
    assert meta.max_val == 120.0


def test_get_field_meta_fallback_positive_value():
    # Act
    meta = get_field_meta("some_unknown_field", 5.0)

    # Assert — fallback spans 0..max(1, value*4).
    assert meta == FieldMeta(0.0, 20.0)


def test_get_field_meta_fallback_small_positive_floors_at_one():
    # Act
    meta = get_field_meta("some_unknown_field", 0.1)

    # Assert
    assert meta == FieldMeta(0.0, 1.0)


def test_get_field_meta_fallback_negative_value():
    # Act
    meta = get_field_meta("some_unknown_field", -50.0)

    # Assert — negative fallback spans (value*4, abs(value)*4).
    assert meta == FieldMeta(-200.0, 200.0)
