"""Slider range metadata for reflected light-config fields.

The dynamic panel has no per-field hand-authored ranges, so it infers a
sensible slider min/max from the field name via ordered substring rules. Rules
are checked in order and the first match wins — ``color_temp_shift`` must
therefore precede ``color_temp``, since the shift lever is a signed offset with
a very different range than an absolute Kelvin temperature.

The rule ranges are *typical editing* ranges, not hard caps. The UI must never
clamp a value the config legitimately holds, so ``get_field_meta`` widens the
matched range whenever the current value sits at or past an edge — a value
above the rule max pushes the max up (with headroom), a value below the rule
min pushes the min down. Combined with the text entry expanding the range to
fit anything the user types (see ``widget_sync.apply_entry_to_slider``), the
slider imposes no artificial ceiling.
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
    ("lit_fraction", FieldMeta(0.0, 1.0)),
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
    """Return the slider range for a field, widened to contain its current value.

    Picks a base range by name substring (or a value-spanning fallback when no
    rule matches), then widens it so the current value is always representable
    without clamping: a value at or above the base max raises the max to
    ``current_value * 1.5`` (headroom to slide higher); a value below the base
    min lowers the min to fit it. A value comfortably inside the base range
    leaves it untouched.

    Args:
        field_name (str): The dataclass field name being edited.
        current_value (float): The field's current value.

    Returns:
        FieldMeta: Slider range guaranteed to contain current_value.
    """
    return _contain_value(_base_meta(field_name, current_value), current_value)


def _base_meta(field_name: str, current_value: float) -> FieldMeta:
    """Return the base (pre-widening) range for a field.

    Checks the ordered rules and returns the first whose substring appears in
    field_name. When no rule matches, the fallback spans the current value:
    ``FieldMeta(0.0, max(1.0, current_value * 4))`` for non-negative values, or
    ``FieldMeta(current_value * 4, abs(current_value) * 4)`` for negative ones.

    Args:
        field_name (str): The dataclass field name being edited.
        current_value (float): The field's current value, used for the fallback.

    Returns:
        FieldMeta: Base slider range before value-containment widening.
    """
    for substring, meta in _FIELD_META_RULES:
        if substring in field_name:
            return meta
    if current_value < 0:
        return FieldMeta(current_value * 4, abs(current_value) * 4)
    return FieldMeta(0.0, max(1.0, current_value * 4))


def _contain_value(base: FieldMeta, current_value: float) -> FieldMeta:
    """Widen a base range so it contains current_value, with headroom at the cap.

    A value at or above ``base.max_val`` raises the max to ``current_value *
    1.5`` so the slider is not pinned at the top and can be pushed higher. A
    value below ``base.min_val`` lowers the min to fit it (further negative for
    signed fields, toward zero otherwise). Values inside the base range return
    it unchanged.

    Args:
        base (FieldMeta): The base range from ``_base_meta``.
        current_value (float): The value that must be representable.

    Returns:
        FieldMeta: Range containing current_value.
    """
    min_val, max_val = base.min_val, base.max_val
    if current_value >= max_val:
        max_val = current_value * 1.5
    if current_value < min_val:
        min_val = current_value * 1.5 if current_value < 0 else current_value * 0.5
    return FieldMeta(min_val, max_val, base.is_int)
