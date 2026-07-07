"""Shared slider/entry/toggle widget-sync handlers for devmode panels.

The UI Devmode tab imports these paired-widget update handlers. All three mark
the touched slider dirty so ``config_bridge`` only writes widgets the user
actually changed.
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
    (or int when the descriptor is flagged integer). A typed value outside the
    slider's current range EXPANDS the range to fit it rather than clamping —
    the UI must never silently cap a value the user explicitly entered. The
    descriptor and the live slider's ``value_range`` are widened together, then
    the value is set. Marks the slider dirty so apply_widgets_to_config knows
    the user changed it. Entries that are not a slider's value entry (e.g. plain
    string fields) are ignored here — they are written by apply_widgets_to_config
    directly.

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

    value = float(int(round(raw))) if descriptor.is_int else raw
    _fit_range_to_value(slider, descriptor, value)
    slider.set_current_value(value, warn=False)

    if descriptor.is_int:
        entry.set_text(str(int(round(value))))
    else:
        entry.set_text(f"{value:.3f}")


def _fit_range_to_value(slider: t.Any, descriptor: t.Any, value: float) -> None:
    """Widen the slider and descriptor range so value is representable.

    Only expands — a value inside the current range leaves it untouched. Keeps
    the descriptor (read by apply_widgets_to_config) and the live slider's
    ``value_range`` (read by set_current_value) in sync so the true typed value
    reaches config instead of a clamped one.

    Args:
        slider: The UIHorizontalSlider whose value_range to widen.
        descriptor: The SliderDescriptor whose min_val/max_val to widen.
        value (float): The value that must fit within the range.
    """
    min_val = min(descriptor.min_val, value)
    max_val = max(descriptor.max_val, value)
    if min_val == descriptor.min_val and max_val == descriptor.max_val:
        return
    descriptor.min_val = min_val
    descriptor.max_val = max_val
    slider.value_range = (min_val, max_val)
