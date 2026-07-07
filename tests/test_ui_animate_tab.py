# Radical
"""Headless tests for the Animate tab: build/destroy, gate, preview, base change."""

import os
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.ui.app_state import AppState  # noqa: E402
from streets_at_night.ui.jobs import JobEvent, JobRunner  # noqa: E402
from streets_at_night.ui.tabs.animate_tab import AnimateTab, _GATE_STATUS  # noqa: E402


@pytest.fixture
def headless_manager():
    """Provide a headless pygame display + UIManager, torn down after the test."""
    pygame.init()
    pygame.display.set_mode((1440, 900))
    manager = pygame_gui.UIManager((1440, 900))
    yield manager
    pygame.quit()


@pytest.fixture
def config() -> SceneConfig:
    """Return a default SceneConfig (heat haze enabled) built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


@pytest.fixture
def content_rect() -> pygame.Rect:
    """Return a representative content rect below the tab/config bars."""
    return pygame.Rect(0, 84, 1440, 816)


def _make_tab(config: SceneConfig, tmp_path) -> tuple:
    """Build an AnimateTab against fresh state and a job runner."""
    state = AppState(config_path=None, config=config, cache_dir=tmp_path / "cache")
    jobs = JobRunner()
    tab = AnimateTab(state, jobs)
    return tab, state, jobs


def test_build_creates_widgets_and_destroy_clears_them_and_drops_haze(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)

    # Act
    tab.build(headless_manager, content_rect)
    tab._haze = mock.Mock()
    tab._playing = True

    # Assert build
    assert tab._base_picker is not None
    assert tab._output_picker is not None
    assert tab._preview_btn is not None
    assert tab._record_btn is not None
    assert tab._seamless_btn is not None
    assert tab._progress_bar is not None
    assert tab._status is not None
    assert tab._labels

    # Act destroy
    tab.destroy()

    # Assert destroy dropped widgets and the haze
    assert tab._base_picker is None
    assert tab._output_picker is None
    assert tab._preview_btn is None
    assert tab._record_btn is None
    assert tab._progress_bar is None
    assert tab._status is None
    assert tab._labels == []
    assert tab._haze is None
    assert tab._playing is False


def test_gate_disables_buttons_and_sets_status_when_haze_disabled(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    config.heat_haze.enabled = False
    tab, _, _ = _make_tab(config, tmp_path)

    # Act
    tab.build(headless_manager, content_rect)
    tab.update(0.016)

    # Assert
    assert tab._haze_enabled is False
    assert tab._status.text == _GATE_STATUS
    assert tab._preview_btn.is_enabled is False
    assert tab._record_btn.is_enabled is False


def test_gate_enables_buttons_when_haze_enabled(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)

    # Act
    tab.build(headless_manager, content_rect)
    tab.update(0.016)

    # Assert
    assert tab._haze_enabled is True
    assert tab._preview_btn.is_enabled is True
    assert tab._record_btn.is_enabled is True


def test_toggle_preview_submits_build_job(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    with mock.patch.object(jobs, "submit", return_value=True) as mock_submit:
        tab._toggle_preview()

    # Assert
    mock_submit.assert_called_once()
    assert tab._pending == "preview"


def test_preview_done_starts_playback(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._pending = "preview"
    haze = mock.Mock()

    # Act
    tab.handle_job_event(JobEvent("done", result=haze))

    # Assert
    assert tab._haze is haze
    assert tab._playing is True
    assert tab._elapsed == 0.0
    assert tab._preview_btn.text == "Stop"


def test_toggle_preview_while_playing_stops_playback(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._playing = True
    tab._haze = mock.Mock()

    # Act
    tab._toggle_preview()

    # Assert
    assert tab._playing is False
    assert tab._preview_btn.text == "Preview"


def test_base_path_change_stops_playback_and_clears_haze(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._playing = True
    tab._haze = mock.Mock()

    # Act — a base picker that reports a new path was picked.
    with mock.patch.object(tab._base_picker, "handle_event", return_value=True):
        tab.handle_event(pygame.event.Event(pygame.USEREVENT))

    # Assert
    assert tab._playing is False
    assert tab._haze is None


def test_record_done_reports_saved_path(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._pending = "record"
    out_path = Path("output/animate.mp4")

    # Act
    tab.handle_job_event(JobEvent("done", result=out_path))

    # Assert
    assert tab._status.text == "Saved output/animate.mp4"
    assert tab._progress_bar.current_progress == pytest.approx(100.0)


def test_run_record_submits_job_with_low_preset_dims(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    with mock.patch.object(jobs, "submit", return_value=True) as mock_submit:
        tab._run_record()

    # Assert
    mock_submit.assert_called_once()
    assert tab._pending == "record"


def test_run_record_ignored_while_playing(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._playing = True

    # Act
    with mock.patch.object(jobs, "submit") as mock_submit:
        tab._run_record()

    # Assert
    mock_submit.assert_not_called()
