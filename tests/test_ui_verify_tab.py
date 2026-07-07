# Radical
"""Headless tests for the Verify tab: bounds parsing, save, and the verify job."""

import os
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import BoundsConfig, SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.ui.app_state import AppState  # noqa: E402
from streets_at_night.ui.jobs import JobEvent, JobRunner  # noqa: E402
from streets_at_night.ui.tabs import verify_tab as verify_tab_module  # noqa: E402
from streets_at_night.ui.tabs.verify_tab import VerifyTab, run_verify_pipeline  # noqa: E402

_MODULE = "streets_at_night.ui.tabs.verify_tab"


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
    """Return a default SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


@pytest.fixture
def content_rect() -> pygame.Rect:
    """Return a representative content rect below the tab/config bars."""
    return pygame.Rect(0, 84, 1440, 816)


def _make_tab(config: SceneConfig, config_path, tmp_path) -> tuple:
    """Build a VerifyTab against fresh state and a job runner."""
    state = AppState(config_path=config_path, config=config, cache_dir=tmp_path / "cache")
    jobs = JobRunner()
    tab = VerifyTab(state, jobs)
    return tab, state, jobs


def test_build_creates_widgets_and_destroy_clears_them(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)

    # Act
    tab.build(headless_manager, content_rect)

    # Assert — four entries plus the two action buttons and status exist.
    assert set(tab._entries) == {"North", "South", "East", "West"}
    assert tab._rerender_btn is not None
    assert tab._save_btn is not None

    tab.destroy()
    assert tab._entries == {}
    assert tab._rerender_btn is None
    assert tab._save_btn is None


def test_refresh_from_state_blank_without_config_path(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    tab.refresh_from_state()

    # Assert
    assert all(entry.get_text() == "" for entry in tab._entries.values())


def test_refresh_from_state_populates_from_bounds_when_path_set(config, headless_manager, content_rect, tmp_path):
    # Arrange
    config.bounds = BoundsConfig(north=34.5, south=34.1, east=-117.2, west=-117.4)
    tab, _, _ = _make_tab(config, tmp_path / "scene.toml", tmp_path)
    tab.build(headless_manager, content_rect)

    # Assert
    assert tab._entries["North"].get_text() == "34.500000"
    assert tab._entries["West"].get_text() == "-117.400000"


def test_read_bounds_parses_valid_entries(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._entries["North"].set_text("34.5")
    tab._entries["South"].set_text("34.1")
    tab._entries["East"].set_text("-117.2")
    tab._entries["West"].set_text("-117.4")

    # Act
    bounds = tab._read_bounds()

    # Assert
    assert bounds == BoundsConfig(north=34.5, south=34.1, east=-117.2, west=-117.4)


def test_read_bounds_invalid_returns_none_and_sets_status(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._entries["North"].set_text("not a number")

    # Act
    bounds = tab._read_bounds()

    # Assert
    assert bounds is None
    assert tab._status.text == "Invalid bounds value"


def test_run_rerender_submits_job_and_adopts_bounds(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, jobs = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    for label, value in (("North", "34.5"), ("South", "34.1"), ("East", "-117.2"), ("West", "-117.4")):
        tab._entries[label].set_text(value)

    # Act
    with mock.patch.object(jobs, "submit", return_value=True) as mock_submit:
        tab._run_rerender()

    # Assert
    assert state.config.bounds == BoundsConfig(north=34.5, south=34.1, east=-117.2, west=-117.4)
    assert state.scene_dirty is True
    mock_submit.assert_called_once()


def test_run_rerender_ignored_when_busy(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    with mock.patch.object(type(jobs), "busy", new_callable=mock.PropertyMock, return_value=True):
        with mock.patch.object(jobs, "submit") as mock_submit:
            tab._run_rerender()

    # Assert
    mock_submit.assert_not_called()


def test_run_save_writes_bounds_to_existing_config(config, headless_manager, content_rect, tmp_path):
    # Arrange
    config_path = tmp_path / "scene.toml"
    tab, state, _ = _make_tab(config, config_path, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._entries["North"].set_text("34.5")
    tab._entries["South"].set_text("34.1")
    tab._entries["East"].set_text("-117.2")
    tab._entries["West"].set_text("-117.4")

    # Act
    with mock.patch(f"{_MODULE}.write_bounds_to_toml") as mock_write:
        tab._run_save()

    # Assert
    mock_write.assert_called_once_with(config_path, state.config.bounds)


def test_run_save_opens_dialog_when_no_config_path(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    for label, value in (("North", "34.5"), ("South", "34.1"), ("East", "-117.2"), ("West", "-117.4")):
        tab._entries[label].set_text(value)

    # Act
    tab._run_save()

    # Assert
    assert tab._save_dialog is not None


def test_save_to_path_new_file_writes_full_config(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    target = tmp_path / "new_config.toml"

    # Act
    with mock.patch(f"{_MODULE}.write_config") as mock_full, mock.patch(
        f"{_MODULE}.write_bounds_to_toml"
    ) as mock_bounds:
        tab._save_to_path(target)

    # Assert
    mock_full.assert_called_once_with(state.config, target)
    mock_bounds.assert_not_called()
    assert state.config_path == target


def test_save_to_path_existing_file_writes_bounds_only(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    target = tmp_path / "existing.toml"
    target.write_text("[bounds]\n")

    # Act
    with mock.patch(f"{_MODULE}.write_config") as mock_full, mock.patch(
        f"{_MODULE}.write_bounds_to_toml"
    ) as mock_bounds:
        tab._save_to_path(target)

    # Assert
    mock_bounds.assert_called_once_with(target, state.config.bounds)
    mock_full.assert_not_called()
    assert state.config_path == target


def test_handle_job_event_done_stores_surface_and_status(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)
    surface = pygame.Surface((10, 10))

    # Act
    with mock.patch(f"{_MODULE}.Image.open") as mock_open, mock.patch(
        f"{_MODULE}.fit_image_to_viewport"
    ), mock.patch(f"{_MODULE}.pil_to_pygame_surface", return_value=surface):
        tab.handle_job_event(JobEvent("done", result=Path("output/verify.png")))

    # Assert
    assert tab._plot_surface is surface
    assert tab._status.text == "Saved output/verify.png"
    mock_open.assert_called_once()


def test_handle_job_event_error_sets_status(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, None, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    tab.handle_job_event(JobEvent("error", message="boom"))

    # Assert
    assert tab._status.text == "Verify failed: boom"


def test_run_verify_pipeline_runs_stages_and_returns_path(config, tmp_path):
    # Arrange
    cache_dir = tmp_path / "cache"
    output_path = tmp_path / "output" / "verify.png"
    progress = mock.Mock()

    # Act
    with mock.patch.object(verify_tab_module, "load_roads") as mock_roads, mock.patch.object(
        verify_tab_module, "classify_roads"
    ) as mock_classify, mock.patch.object(
        verify_tab_module, "load_land_use"
    ) as mock_land, mock.patch.object(
        verify_tab_module, "load_buildings"
    ) as mock_buildings, mock.patch.object(
        verify_tab_module, "save_verification_plot"
    ) as mock_save:
        result = run_verify_pipeline(config, cache_dir, output_path, progress)

    # Assert
    assert result == output_path
    assert cache_dir.exists()
    mock_classify.assert_called_once_with(mock_roads.return_value)
    mock_save.assert_called_once_with(
        config,
        mock_classify.return_value,
        mock_land.return_value,
        mock_buildings.return_value,
        output_path,
    )
    assert progress.call_count == 4
