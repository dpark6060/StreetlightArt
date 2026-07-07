"""Slider range metadata for reflected light-config fields.

The dynamic panel has no per-field hand-authored ranges, so it infers a
sensible slider min/max from the field name via ordered substring rules. Rules
are checked in order and the first match wins — ``color_temp_shift`` must
therefore precede ``color_temp``, since the shift lever is a signed offset with
a very different range than an absolute Kelvin temperature.
"""

import typing as t
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldMeta:
    """Slider range for one config field.

    Attributes:
        min_val: Slider minimum value.
        max_val: Slider maximum value.
        is_int: When True, the value is an integer slider.
    """

    min_val: float
    max_val: float
    is_int: bool = False


# Ordered (substring, FieldMeta) rules. First substring match wins.
_FIELD_META_RULES: t.List[t.Tuple[str, FieldMeta]] = [
    ("color_temp_shift", FieldMeta(-3000.0, 3000.0)),
    ("color_temp", FieldMeta(1000.0, 10000.0)),
    ("saturation", FieldMeta(0.0, 2.0)),
    ("beam_angle", FieldMeta(10.0, 170.0)),
    ("inner_fraction", FieldMeta(0.0, 1.0)),
    ("falloff", FieldMeta(0.1, 8.0)),
    ("illumination_intensity", FieldMeta(0.0, 1.0)),
    ("illumination_radius", FieldMeta(0.0, 60.0)),
    ("glow_radius", FieldMeta(0.0, 120.0)),
    ("brightness", FieldMeta(0.0, 10.0)),
    ("height", FieldMeta(0.0, 30.0)),
    ("spacing", FieldMeta(1.0, 300.0)),
    ("separation", FieldMeta(0.5, 10.0)),
    ("density", FieldMeta(0.0, 0.2)),
]


def get_field_meta(field_name: str, current_value: float) -> FieldMeta:
    """Return the slider range for a field, by name substring or a fallback.

    Checks the ordered rules and returns the first whose substring appears in
    field_name. When no rule matches, the fallback spans the current value:
    ``FieldMeta(0.0, max(1.0, current_value * 4))`` for non-negative values, or
    ``FieldMeta(current_value * 4, abs(current_value) * 4)`` for negative ones.

    Args:
        field_name (str): The dataclass field name being edited.
        current_value (float): The field's current value, used for the fallback.

    Returns:
        FieldMeta: Slider range for the field.
    """
    for substring, meta in _FIELD_META_RULES:
        if substring in field_name:
            return meta
    if current_value < 0:
        return FieldMeta(current_value * 4, abs(current_value) * 4)
    return FieldMeta(0.0, max(1.0, current_value * 4))
