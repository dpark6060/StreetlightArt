# Radical
"""Headless tests for the Render tab: build/destroy, scene reuse, and the job."""

import os
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402
from PIL import Image  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.ui.app_state import AppState  # noqa: E402
from streets_at_night.ui.jobs import JobEvent, JobRunner  # noqa: E402
from streets_at_night.ui.tabs import render_tab as render_tab_module  # noqa: E402
from streets_at_night.ui.tabs.render_tab import (  # noqa: E402
    RenderResult,
    RenderTab,
    resolve_image_format,
    run_render,
)

_MODULE = "streets_at_night.ui.tabs.render_tab"


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


def _make_tab(config: SceneConfig, tmp_path) -> tuple:
    """Build a RenderTab against fresh state and a job runner."""
    state = AppState(config_path=None, config=config, cache_dir=tmp_path / "cache")
    jobs = JobRunner()
    tab = RenderTab(state, jobs)
    return tab, state, jobs


def test_build_creates_widgets_and_destroy_clears_them(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)

    # Act
    tab.build(headless_manager, content_rect)

    # Assert
    assert tab._output_picker is not None
    assert tab._quality_menu is not None
    assert tab._render_btn is not None
    assert tab._progress_bar is not None
    assert tab._status is not None
    assert tab._labels

    tab.destroy()
    assert tab._output_picker is None
    assert tab._quality_menu is None
    assert tab._render_btn is None
    assert tab._progress_bar is None
    assert tab._status is None
    assert tab._labels == []


def test_run_render_reuses_scene_when_quality_matches_and_clean(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, jobs = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    cached_scene = mock.MagicMock()
    state.scene = cached_scene
    state.scene_quality = "low"
    state.scene_dirty = False

    # Act
    with mock.patch.object(jobs, "submit", return_value=True) as mock_submit:
        tab._run_render()

    # Assert — the reusable scene should be the cached one (no rebuild needed).
    assert tab._reusable_scene("low") is cached_scene
    mock_submit.assert_called_once()


def test_run_render_rebuilds_when_quality_differs(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    state.scene = mock.MagicMock()
    state.scene_quality = "high"
    state.scene_dirty = False

    # Act / Assert — quality mismatch forces a rebuild.
    assert tab._reusable_scene("low") is None


def test_run_render_rebuilds_when_dirty(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    state.scene = mock.MagicMock()
    state.scene_quality = "low"
    state.scene_dirty = True

    # Act / Assert
    assert tab._reusable_scene("low") is None


def test_run_render_ignored_when_busy(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    with mock.patch.object(type(jobs), "busy", new_callable=mock.PropertyMock, return_value=True):
        with mock.patch.object(jobs, "submit") as mock_submit:
            tab._run_render()

    # Assert
    mock_submit.assert_not_called()


def test_run_render_ignored_when_output_empty(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    tab._output_picker._entry.set_text("")

    # Act
    with mock.patch.object(jobs, "submit") as mock_submit:
        tab._run_render()

    # Assert
    mock_submit.assert_not_called()


def test_handle_job_event_done_stores_surface_caches_scene_and_status(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    surface = pygame.Surface((10, 10))
    built = mock.MagicMock()
    result = RenderResult(Path("output/render.png"), Image.new("RGB", (4, 4)), built, "medium")

    # Act
    with mock.patch(f"{_MODULE}.fit_image_to_viewport"), mock.patch(
        f"{_MODULE}.pil_to_pygame_surface", return_value=surface
    ):
        tab.handle_job_event(JobEvent("done", result=result))

    # Assert
    assert tab._image_surface is surface
    assert tab._status.text == "Saved output/render.png"
    assert state.scene is built
    assert state.scene_quality == "medium"
    assert state.scene_dirty is False


def test_handle_job_event_done_reused_scene_leaves_cache_untouched(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)
    original = mock.MagicMock()
    state.scene = original
    state.scene_quality = "low"
    result = RenderResult(Path("output/render.png"), Image.new("RGB", (4, 4)), None, "low")

    # Act
    with mock.patch(f"{_MODULE}.fit_image_to_viewport"), mock.patch(
        f"{_MODULE}.pil_to_pygame_surface", return_value=pygame.Surface((4, 4))
    ):
        tab.handle_job_event(JobEvent("done", result=result))

    # Assert — built_scene is None, so the cached scene must be left alone.
    assert state.scene is original


def test_handle_job_event_progress_sets_bar_when_fraction_positive(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    tab.handle_job_event(JobEvent("progress", message="Rendering pass 3/7: illumination", fraction=0.5))

    # Assert
    assert tab._status.text == "Rendering pass 3/7: illumination"
    assert tab._progress_bar.current_progress == pytest.approx(50.0)


def test_handle_job_event_error_sets_status(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)
    tab.build(headless_manager, content_rect)

    # Act
    tab.handle_job_event(JobEvent("error", message="boom"))

    # Assert
    assert tab._status.text == "Render failed: boom"


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("render.tiff", "TIFF"),
        ("render.tif", "TIFF"),
        ("render.png", "PNG"),
        ("render.jpg", "PNG"),
    ],
)
def test_resolve_image_format_by_suffix(filename, expected):
    # Act / Assert
    assert resolve_image_format(Path(filename)) == expected


def test_run_render_reuses_existing_scene_without_load(config, tmp_path):
    # Arrange
    scene = mock.MagicMock()
    scene.car_pairs = []
    output_path = tmp_path / "out" / "render.png"
    progress = mock.Mock()

    # Act
    with mock.patch.object(render_tab_module, "load_scene") as mock_load, mock.patch.object(
        render_tab_module, "PillowBackend"
    ) as mock_backend, mock.patch.object(
        render_tab_module, "get_export_passes"
    ), mock.patch.object(
        render_tab_module, "get_export_pass_names"
    ), mock.patch.object(
        render_tab_module, "render_scene_frame_with_progress"
    ):
        result = run_render(config, "low", output_path, tmp_path / "cache", scene, progress)

    # Assert
    mock_load.assert_not_called()
    assert result.built_scene is None
    assert result.quality == "low"
    assert result.output_path == output_path
    mock_backend.return_value.get_image.return_value.save.assert_called_once()


def test_run_render_builds_scene_when_none_supplied(config, tmp_path):
    # Arrange
    built = mock.MagicMock()
    built.car_pairs = []
    output_path = tmp_path / "out" / "render.png"
    progress = mock.Mock()

    # Act
    with mock.patch.object(render_tab_module, "load_scene", return_value=built) as mock_load, mock.patch.object(
        render_tab_module, "PillowBackend"
    ), mock.patch.object(
        render_tab_module, "get_export_passes"
    ), mock.patch.object(
        render_tab_module, "get_export_pass_names"
    ), mock.patch.object(
        render_tab_module, "render_scene_frame_with_progress"
    ):
        result = run_render(config, "high", output_path, tmp_path / "cache", None, progress)

    # Assert
    mock_load.assert_called_once()
    assert result.built_scene is built
    assert result.quality == "high"
    assert output_path.parent.exists()
