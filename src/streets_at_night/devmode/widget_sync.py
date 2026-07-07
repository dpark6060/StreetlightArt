"""Shared slider/entry/toggle widget-sync handlers for devmode panels.

Public versions of the paired-widget update logic that ``cmd_devmode`` keeps as
private copies. The UI Devmode tab imports these; ``cmd_devmode`` keeps its own
copies until change 09 removes it. All three mark the touched slider dirty so
``config_bridge`` only writes widgets the user actually changed.
"""

import typing as t

from .slider_registry import SliderRegistry


def toggle_bool_button(button: t.Any, registry: SliderRegistry) -> None:
    """Flip the tracked state of a toggle button and update its display text.

    Args:
        button: The toggle UIButton to flip.
        registry (SliderRegistry): Registry holding bool_states for all buttons.
    """
    current = registry.bool_states.get(button, False)
    new_value = not current
    registry.bool_states[button] = new_value
    button.set_text("ON" if new_value else "OFF")


def update_value_entry(slider: t.Any, registry: SliderRegistry) -> None:
    """Update the value text entry for a slider that was just moved.

    Also marks the slider dirty in the registry, so apply_widgets_to_config
    knows the user actually touched it (see config_bridge._apply_sliders).

    Args:
        slider: The UIHorizontalSlider that fired the moved event.
        registry (SliderRegistry): Registry mapping sliders to descriptors and entries.
    """
    descriptor = registry.sliders.get(slider)
    value_entry = registry.value_entries.get(slider)
    if descriptor is None or value_entry is None:
        return
    registry.mark_dirty(descriptor.label)
    raw = slider.get_current_value()
    if descriptor.is_int:
        value_entry.set_text(str(int(round(raw))))
    else:
        value_entry.set_text(f"{raw:.3f}")


def apply_entry_to_slider(entry: t.Any, registry: SliderRegistry) -> None:
    """Parse a finished value text entry and update its linked slider.

    If the entry belongs to a slider's value entry, parse the text as a float
    (or int when the descriptor is flagged integer), clamp to the slider's
    range, and call set_current_value on the slider. Marks the slider dirty so
    apply_widgets_to_config knows the user changed it. Entries that are not a
    slider's value entry (e.g. plain string fields) are ignored here — they are
    written by apply_widgets_to_config directly.

    Args:
        entry: The UITextEntryLine that fired the finished event.
        registry (SliderRegistry): Registry with entry_to_slider reverse lookup.
    """
    slider = registry.entry_to_slider.get(entry)
    if slider is None:
        return
    descriptor = registry.sliders.get(slider)
    if descriptor is None:
        return

    text = entry.get_text().strip()
    try:
        raw = float(text)
    except ValueError:
        return

    registry.mark_dirty(descriptor.label)

    if descriptor.is_int:
        value = float(int(round(raw)))
    else:
        value = raw

    clamped = max(descriptor.min_val, min(descriptor.max_val, value))
    slider.set_current_value(clamped)

    if descriptor.is_int:
        entry.set_text(str(int(round(clamped))))
    else:
        entry.set_text(f"{clamped:.3f}")
