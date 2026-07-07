# Radical
"""Tests for devmode.widget_sync: dirty marking, value formatting, clamping."""

from unittest import mock

from streets_at_night.devmode.slider_registry import SliderDescriptor, SliderRegistry
from streets_at_night.devmode.widget_sync import (
    apply_entry_to_slider,
    toggle_bool_button,
    update_value_entry,
)


def test_toggle_bool_button_flips_state_and_text():
    # Arrange
    registry = SliderRegistry()
    button = mock.MagicMock()
    registry.register_bool(button, ("flag",), initial_value=False)

    # Act
    toggle_bool_button(button, registry)

    # Assert
    assert registry.bool_states[button] is True
    button.set_text.assert_called_once_with("ON")


def test_update_value_entry_marks_dirty_and_formats_float():
    # Arrange
    registry = SliderRegistry()
    slider = mock.MagicMock()
    slider.get_current_value.return_value = 1.23456
    entry = mock.MagicMock()
    descriptor = SliderDescriptor("Cars: X:val", ("val",), 0.0, 10.0)
    registry.register_slider(slider, descriptor, entry)

    # Act
    update_value_entry(slider, registry)

    # Assert
    assert "Cars: X:val" in registry.dirty
    entry.set_text.assert_called_once_with("1.235")


def test_update_value_entry_formats_int():
    # Arrange
    registry = SliderRegistry()
    slider = mock.MagicMock()
    slider.get_current_value.return_value = 4.7
    entry = mock.MagicMock()
    descriptor = SliderDescriptor("A:n", ("n",), 0.0, 10.0, is_int=True)
    registry.register_slider(slider, descriptor, entry)

    # Act
    update_value_entry(slider, registry)

    # Assert
    entry.set_text.assert_called_once_with("5")


def test_apply_entry_to_slider_clamps_and_sets_slider():
    # Arrange
    registry = SliderRegistry()
    slider = mock.MagicMock()
    entry = mock.MagicMock()
    entry.get_text.return_value = "99.0"
    descriptor = SliderDescriptor("A:val", ("val",), 0.0, 10.0)
    registry.register_slider(slider, descriptor, entry)

    # Act
    apply_entry_to_slider(entry, registry)

    # Assert — clamped to max, slider updated, dirty marked.
    slider.set_current_value.assert_called_once_with(10.0)
    assert "A:val" in registry.dirty


def test_apply_entry_to_slider_ignores_non_numeric_text():
    # Arrange
    registry = SliderRegistry()
    slider = mock.MagicMock()
    entry = mock.MagicMock()
    entry.get_text.return_value = "not a number"
    descriptor = SliderDescriptor("A:val", ("val",), 0.0, 10.0)
    registry.register_slider(slider, descriptor, entry)

    # Act
    apply_entry_to_slider(entry, registry)

    # Assert — no slider update, not marked dirty.
    slider.set_current_value.assert_not_called()
    assert "A:val" not in registry.dirty


def test_apply_entry_to_slider_ignores_unregistered_entry():
    # Arrange
    registry = SliderRegistry()
    entry = mock.MagicMock()

    # Act / Assert — no exception, nothing happens.
    apply_entry_to_slider(entry, registry)
    assert registry.dirty == set()
