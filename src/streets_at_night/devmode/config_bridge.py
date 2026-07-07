"""Two-way sync between SceneConfig and slider widget values."""

import typing as t

from .slider_registry import SliderRegistry


def get_nested(obj: t.Any, path: tuple) -> t.Any:
    """Traverse a nested dataclass or list by a path tuple.

    String path items are treated as attribute names; integer items are treated
    as list indices. Raises AttributeError or IndexError on bad paths.

    Args:
        obj: The root object to traverse (e.g. a SceneConfig instance).
        path (tuple): Sequence of string attribute names or integer list indices.

    Returns:
        Any: The value found at the end of the path.
    """
    current = obj
    for key in path:
        if isinstance(key, int):
            current = current[key]
        else:
            current = getattr(current, key)
    return current


def set_nested(obj: t.Any, path: tuple, value: t.Any) -> None:
    """Set a value in a nested dataclass or list by a path tuple.

    Traverses to the parent object using all but the last path element, then
    uses setattr or __setitem__ on the final key.

    Args:
        obj: The root object to traverse (e.g. a SceneConfig instance).
        path (tuple): Sequence of string attribute names or integer list indices.
        value: The value to set at the target location.
    """
    parent = get_nested(obj, path[:-1])
    final_key = path[-1]
    if isinstance(final_key, int):
        parent[final_key] = value
    else:
        setattr(parent, final_key, value)


def read_config_to_widgets(config: t.Any, registry: SliderRegistry) -> None:
    """Populate all slider, button, and text widgets from current config values.

    Reads each registered widget's config path, extracts the current value,
    and pushes it to the widget. Also updates value entry text for sliders.

    Args:
        config (Any): The root object widget paths traverse from — a full
            SceneConfig for hand-written sections, or a light-config dataclass
            for reflected dynamic sections.
        registry (SliderRegistry): Widget registry for the active panel section.
    """
    _read_sliders(config, registry)
    _read_bool_buttons(config, registry)
    _read_text_entries(config, registry)


def apply_widgets_to_config(registry: SliderRegistry, config: t.Any) -> None:
    """Write all current widget values back into config.

    Reads each widget's current state and writes it to the appropriate config
    path. Integer sliders are cast before writing. Bool buttons are read from
    their tracked state in registry.bool_states. Text entries are written as
    strings.

    Args:
        registry (SliderRegistry): Widget registry holding all active widgets.
        config (Any): The root object widget paths traverse from — a full
            SceneConfig, or a light-config dataclass for dynamic sections.
    """
    _apply_sliders(registry, config)
    _apply_bool_buttons(registry, config)
    _apply_text_entries(registry, config)


def _read_sliders(config: t.Any, registry: SliderRegistry) -> None:
    """Push current config values to all registered sliders and their value entries.

    Args:
        config (Any): Root object providing the source values.
        registry (SliderRegistry): Registry with slider-to-descriptor mappings.
    """
    for slider, descriptor in registry.sliders.items():
        value = get_nested(config, descriptor.config_path)
        slider.set_current_value(float(value))
        value_entry = registry.value_entries.get(slider)
        if value_entry is not None:
            value_entry.set_text(_format_value(value, descriptor.is_int))


def _read_bool_buttons(config: t.Any, registry: SliderRegistry) -> None:
    """Push current config boolean values to all registered toggle buttons.

    Also syncs registry.bool_states so the in-memory state matches config.

    Args:
        config (Any): Root object providing the source values.
        registry (SliderRegistry): Registry with button-to-path mappings.
    """
    for button, config_path in registry.bool_buttons.items():
        value = bool(get_nested(config, config_path))
        registry.bool_states[button] = value
        button.set_text("ON" if value else "OFF")


def _read_text_entries(config: t.Any, registry: SliderRegistry) -> None:
    """Push current config string values to all registered text entries.

    Converts haze_color tuples to '#RRGGBB' hex strings when encountered.

    Args:
        config (Any): Root object providing the source values.
        registry (SliderRegistry): Registry with entry-to-path mappings.
    """
    for entry, config_path in registry.text_entries.items():
        value = get_nested(config, config_path)
        if isinstance(value, tuple):
            text = _tuple_to_hex(value)
        else:
            text = str(value)
        entry.set_text(text)


def _apply_sliders(registry: SliderRegistry, config: t.Any) -> None:
    """Read dirty slider values and write them into config.

    Writes the slider value to descriptor.config_path and to every path in
    descriptor.extra_config_paths. The extra paths allow a single slider to
    control the same field across multiple pool entries (e.g. all fixture
    profiles for a road type) so that whichever entry the seeded RNG selects
    at scene-build time reflects the slider change.

    Only sliders in registry.dirty (i.e. actually moved by the user since the
    section was built) are written. Without this gate, a slider's initial
    value — read from only the *first* profiled pool entry, see
    panel_sections.build_streetlight_section — would get broadcast to every
    entry in extra_config_paths on every apply, flattening an authored mix
    (e.g. sodium + LED streetlights) down to one fixture's values the moment
    Re-render is clicked, even with nothing touched.

    A better fix would give each pool entry its own subsection of sliders
    instead of one slider fanning out to extra_config_paths; that is more UI
    work and is deferred (see panel_sections.build_streetlight_section).

    Args:
        registry (SliderRegistry): Registry with slider-to-descriptor mappings.
        config (Any): Root object to update in-place.
    """
    for slider, descriptor in registry.sliders.items():
        if descriptor.label not in registry.dirty:
            continue
        raw = slider.get_current_value()
        value = int(round(raw)) if descriptor.is_int else float(raw)
        set_nested(config, descriptor.config_path, value)
        for extra_path in descriptor.extra_config_paths:
            set_nested(config, extra_path, value)
        value_entry = registry.value_entries.get(slider)
        if value_entry is not None:
            value_entry.set_text(_format_value(value, descriptor.is_int))


def _apply_bool_buttons(registry: SliderRegistry, config: t.Any) -> None:
    """Read all toggle button states from registry.bool_states and write into config.

    Args:
        registry (SliderRegistry): Registry with button-to-path mappings and bool_states.
        config (Any): Root object to update in-place.
    """
    for button, config_path in registry.bool_buttons.items():
        value = registry.bool_states.get(button, False)
        set_nested(config, config_path, value)


def _apply_text_entries(registry: SliderRegistry, config: t.Any) -> None:
    """Read all text entry values and write them into config.

    Converts '#RRGGBB' strings back to (R, G, B) tuples for color fields.

    Args:
        registry (SliderRegistry): Registry with entry-to-path mappings.
        config (Any): Root object to update in-place.
    """
    for entry, config_path in registry.text_entries.items():
        text = entry.get_text().strip()
        current = get_nested(config, config_path)
        if isinstance(current, tuple):
            value = _hex_to_tuple(text)
        else:
            value = text
        set_nested(config, config_path, value)


def _format_value(value: t.Any, is_int: bool) -> str:
    """Format a slider value for display in a value entry.

    Args:
        value: The numeric value to format.
        is_int (bool): When True, format as integer; otherwise as 3-decimal float.

    Returns:
        str: Formatted string representation.
    """
    if is_int:
        return str(int(round(float(value))))
    return f"{float(value):.3f}"


def _tuple_to_hex(color: tuple) -> str:
    """Convert an (R, G, B) integer tuple to a '#RRGGBB' hex string.

    Args:
        color (tuple): (R, G, B) with integer components 0-255.

    Returns:
        str: Hex color string like '#3A2010'.
    """
    r, g, b = color
    return f"#{int(r):02X}{int(g):02X}{int(b):02X}"


def _hex_to_tuple(hex_str: str) -> t.Tuple[int, int, int]:
    """Convert a '#RRGGBB' hex string to an (R, G, B) integer tuple.

    Falls back to (0, 0, 0) if the string cannot be parsed.

    Args:
        hex_str (str): Hex color string like '#3A2010'.

    Returns:
        tuple[int, int, int]: (R, G, B) integer components.
    """
    try:
        clean = hex_str.lstrip("#")
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
        return (r, g, b)
    except (ValueError, IndexError):
        return (0, 0, 0)
