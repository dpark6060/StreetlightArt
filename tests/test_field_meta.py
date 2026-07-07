# Radical
"""Tests for devmode field_meta: substring rule matching, precedence, fallback."""

import pytest

from streets_at_night.devmode.field_meta import FieldMeta, get_field_meta


@pytest.mark.parametrize(
    "field_name, expected_min, expected_max, expected_int",
    [
        ("color_temp_k", 1000.0, 10000.0, False),
        ("saturation", 0.0, 2.0, False),
        ("beam_angle_deg", 10.0, 170.0, False),
        ("illumination_inner_fraction", 0.0, 1.0, False),
        ("illumination_falloff", 0.1, 8.0, False),
        ("illumination_intensity", 0.0, 1.0, False),
        ("illumination_radius", 0.0, 60.0, False),
        ("glow_radius", 0.0, 120.0, False),
        ("brightness_max", 0.0, 10.0, False),
        ("height_min", 0.0, 30.0, False),
        ("spacing_min", 1.0, 300.0, False),
        ("separation_max", 0.5, 10.0, False),
        ("density_highway_min", 0.0, 0.2, False),
    ],
)
def test_get_field_meta_matches_rules(field_name, expected_min, expected_max, expected_int):
    # Act
    meta = get_field_meta(field_name, 1.0)

    # Assert
    assert meta.min_val == expected_min
    assert meta.max_val == expected_max
    assert meta.is_int == expected_int


def test_get_field_meta_color_temp_shift_precedes_color_temp():
    # Act — "color_temp_shift_k" contains both substrings; shift must win.
    meta = get_field_meta("color_temp_shift_k", 0.0)

    # Assert
    assert meta.min_val == -3000.0
    assert meta.max_val == 3000.0


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
