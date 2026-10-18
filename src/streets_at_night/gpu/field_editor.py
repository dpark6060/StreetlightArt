"""Dear ImGui editor rows for one LightKind's dataclass fields.

The imgui counterpart of devmode/dynamic_panel: discovers fields with
``dataclasses.fields``, filters by the kind's prefix, and picks slider ranges
from devmode/field_meta. Edits write straight into the live config dataclass.
"""

import dataclasses
import typing as t

from imgui_bundle import imgui

from ..devmode.field_meta import FieldMeta, get_field_meta, get_field_unit
from ..devmode.light_catalog import LightKind


class FieldEditor:
    """Draws editor rows and remembers each slider's range between frames.

    Ranges are fixed the first time a field is shown and only widen when the
    value leaves them, so a slider's scale doesn't jump while it's dragged.
    """

    def __init__(self) -> None:
        """Start with no cached ranges."""
        self._ranges: t.Dict[t.Tuple[str, str], FieldMeta] = {}

    def draw(self, kind: LightKind) -> bool:
        """Draw one row per editable field of kind.target.

        Args:
            kind (LightKind): The light kind to edit.

        Returns:
            bool: True if any value changed this frame.
        """
        changed = False
        for f in dataclasses.fields(kind.target):
            if kind.field_prefix is not None and not f.name.startswith(kind.field_prefix):
                continue
            changed |= self._draw_field(kind, f.name)
        return changed

    def _draw_field(self, kind: LightKind, name: str) -> bool:
        """Draw the widget matching one field's current value type.

        Args:
            kind (LightKind): The kind owning the field.
            name (str): Field name.

        Returns:
            bool: True if the value changed.
        """
        value = getattr(kind.target, name)
        label = f"{_display_name(name, kind.field_prefix)}##{kind.label}.{name}"
        if isinstance(value, bool):
            changed, new = imgui.checkbox(label, value)
        elif isinstance(value, int):
            meta = self._range(kind, name, value)
            changed, new = imgui.slider_int(label, value, int(meta.min_val), int(meta.max_val))
        elif isinstance(value, float):
            meta = self._range(kind, name, value)
            changed, new = imgui.slider_float(label, value, meta.min_val, meta.max_val, "%.3f")
        elif isinstance(value, str):
            changed, new = imgui.input_text(label, value)
        else:
            imgui.text_disabled(f"{_display_name(name, kind.field_prefix)}: edit in TOML")
            return False
        if changed:
            setattr(kind.target, name, new)
        return changed

    def _range(self, kind: LightKind, name: str, value: float) -> FieldMeta:
        """Return the cached slider range, widening it only if value escaped.

        Args:
            kind (LightKind): The kind owning the field.
            name (str): Field name.
            value (float): Current value.

        Returns:
            FieldMeta: Range containing value.
        """
        key = (kind.label, name)
        meta = self._ranges.get(key)
        if meta is None or not (meta.min_val <= value <= meta.max_val):
            meta = get_field_meta(name, value)
            self._ranges[key] = meta
        return meta


def _display_name(name: str, prefix: t.Optional[str]) -> str:
    """Turn a field name into a label with its unit, e.g. 'Beam Angle Deg (deg)'.

    Args:
        name (str): Field name.
        prefix (str | None): Kind prefix to strip.

    Returns:
        str: Display label.
    """
    stripped = name[len(prefix):] if prefix and name.startswith(prefix) else name
    pretty = stripped.replace("_", " ").strip().title()
    unit = get_field_unit(name)
    return f"{pretty} ({unit})" if unit else pretty
