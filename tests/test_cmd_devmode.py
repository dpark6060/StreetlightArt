# Radical
"""Tests for streets_at_night.devmode.cmd_devmode render pass assembly and save handling."""

from pathlib import Path
from unittest import mock

import pytest

from streets_at_night.config.loader import load_config
from streets_at_night.devmode.cmd_devmode import (
    _DEBOUNCE_SECONDS,
    _build_render_passes,
    _handle_save,
    _should_render_full,
    compute_viewport_fit,
)
from streets_at_night.devmode.slider_registry import SliderDescriptor, SliderRegistry
from streets_at_night.rendering.compositor import render_composite


def test_build_render_passes_includes_composite():
    # Act
    passes = _build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=False)

    # Assert — without composite the accumulated light buffer never resolves and
    # only the dim substrate (ground/roads) shows; lights are invisible.
    assert render_composite in passes


def test_build_render_passes_composite_is_last_when_no_labels():
    # Act
    passes = _build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=False)

    # Assert — composite must run after the light passes that fill the buffer
    assert passes[-1] is render_composite


def test_build_render_passes_labels_come_after_composite():
    # Act
    passes = _build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=True)

    # Assert — labels overlay the final composited frame, so they run last
    assert passes.index(render_composite) < len(passes) - 1
    assert passes[-1] is not render_composite


# ---------------------------------------------------------------------------
# _handle_save — must apply widget state before writing, without Re-render
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config(tmp_path: Path):
    """A SceneConfig built from defaults (empty TOML overlay)."""
    empty = tmp_path / "empty.toml"
    empty.write_text("")
    return load_config(empty)


def test_handle_save_writes_changed_slider_value_without_rerender(base_config, tmp_path):
    """A slider dragged (dirty) but never re-rendered still lands in the saved file.

    Regression test: previously the save-confirm path called write_config
    directly on the untouched config object, so Save without a prior
    Re-render silently wrote stale (pre-drag) values.
    """
    registry = SliderRegistry()
    mock_slider = mock.MagicMock()
    mock_slider.get_current_value.return_value = 0.42
    descriptor = SliderDescriptor(
        label="headlight_brightness_min",
        config_path=("lights", "cars", "headlight_brightness_min"),
        min_val=0.0,
        max_val=1.0,
    )
    registry.register_slider(mock_slider, descriptor)
    registry.mark_dirty(descriptor.label)

    out_path = tmp_path / "saved.toml"
    save_entry = mock.MagicMock()
    save_entry.get_text.return_value = str(out_path)
    save_confirm_btn = mock.MagicMock()

    _handle_save(registry, base_config, save_entry, save_confirm_btn)

    result = load_config(out_path)
    assert result.lights.cars.headlight_brightness_min == 0.42
    save_entry.kill.assert_called_once()
    save_confirm_btn.kill.assert_called_once()


# ---------------------------------------------------------------------------
# compute_viewport_fit — pure viewport-scaling helper for issue A
# ---------------------------------------------------------------------------


def test_compute_viewport_fit_wider_render_letterboxes_top_and_bottom():
    # Arrange — render is wider (relative to viewport) than the viewport, so
    # it should be limited by width and centered vertically.
    render_width, render_height = 1600, 900
    viewport_width, viewport_height = 896, 720

    # Act
    fit = compute_viewport_fit(render_width, render_height, viewport_width, viewport_height)

    # Assert
    assert fit.width == viewport_width
    assert fit.height == pytest.approx(504, abs=1)
    assert fit.x == 0
    assert fit.y == (viewport_height - fit.height) // 2


def test_compute_viewport_fit_taller_render_letterboxes_left_and_right():
    # Arrange — render is taller (relative to viewport) than the viewport, so
    # it should be limited by height and centered horizontally.
    render_width, render_height = 600, 900
    viewport_width, viewport_height = 896, 720

    # Act
    fit = compute_viewport_fit(render_width, render_height, viewport_width, viewport_height)

    # Assert
    assert fit.height == viewport_height
    assert fit.width < viewport_width
    assert fit.y == 0
    assert fit.x == (viewport_width - fit.width) // 2


def test_compute_viewport_fit_matching_aspect_fills_viewport_exactly():
    # Arrange — render aspect ratio matches the viewport exactly.
    render_width, render_height = 896, 720
    viewport_width, viewport_height = 896, 720

    # Act
    fit = compute_viewport_fit(render_width, render_height, viewport_width, viewport_height)

    # Assert
    assert (fit.width, fit.height) == (viewport_width, viewport_height)
    assert (fit.x, fit.y) == (0, 0)


# ---------------------------------------------------------------------------
# _should_render_full — debounce state table for issue B
# ---------------------------------------------------------------------------


def test_should_render_full_returns_false_while_dirty():
    assert _should_render_full(last_move_t=1.0, now=5.0, dirty=True) is False


def test_should_render_full_returns_false_when_never_moved():
    assert _should_render_full(last_move_t=None, now=5.0, dirty=False) is False


def test_should_render_full_returns_false_before_debounce_window_elapses():
    last_move_t = 1.0
    now = last_move_t + _DEBOUNCE_SECONDS - 0.01

    assert _should_render_full(last_move_t=last_move_t, now=now, dirty=False) is False


def test_should_render_full_returns_true_after_debounce_window_elapses():
    last_move_t = 1.0
    now = last_move_t + _DEBOUNCE_SECONDS + 0.01

    assert _should_render_full(last_move_t=last_move_t, now=now, dirty=False) is True
