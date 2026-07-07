# Radical
"""Headless tests for the Devmode tab: build/destroy, kind switching, save."""

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
from streets_at_night.ui.jobs import JobRunner  # noqa: E402
from streets_at_night.ui.tabs import devmode_tab as devmode_tab_module  # noqa: E402
from streets_at_night.ui.tabs.devmode_tab import DevmodeTab  # noqa: E402

_MODULE = "streets_at_night.ui.tabs.devmode_tab"


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
    """Build a DevmodeTab against fresh state and a job runner."""
    state = AppState(config_path=None, config=config, cache_dir=tmp_path / "cache")
    jobs = JobRunner()
    tab = DevmodeTab(state, jobs)
    return tab, state, jobs


def test_build_creates_widgets_and_destroy_clears_them(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)

    # Act — patch heavy render so build stays cheap.
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

    # Assert — panel widgets and an Atmosphere (static) section exist.
    assert tab._section_list is not None
    assert tab._scroll is not None
    assert tab._rerender_btn is not None
    assert tab._save_btn is not None
    assert tab._realtiles_btn is not None
    assert len(tab._static_registry.sliders) > 0

    tab.destroy()
    assert tab._section_list is None
    assert tab._scroll is None
    assert tab._rerender_btn is None
    assert tab._static_registry.sliders == {}
    assert tab._light_registry.sliders == {}


def test_section_names_lists_static_then_catalog(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _ = _make_tab(config, tmp_path)

    # Act
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)
    names = tab._section_names()

    # Assert
    assert names[0] == "Atmosphere"
    assert names[1] == "Devmode Camera"
    assert "Streetlight: Residential (pool 0)" in names
    assert "Cars: Headlights" in names
    assert "Streetlights: Global" in names


def test_switching_light_kinds_preserves_edited_value(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

        # Switch to a streetlight kind and edit brightness_max via a slider.
        tab._build_section("Streetlight: Highway (pool 0)")
        slider = next(
            s for s, d in tab._light_registry.sliders.items()
            if d.config_path == ("brightness_max",)
        )
        slider.set_current_value(0.42)
        tab._handle_slider_moved(slider)

        # Act — leave for House Lights, then return.
        tab._build_section("House Lights")
        tab._build_section("Streetlight: Highway (pool 0)")
        returned_slider = next(
            s for s, d in tab._light_registry.sliders.items()
            if d.config_path == ("brightness_max",)
        )

    # Assert — config kept the edit, and the rebuilt widget shows it (read from config).
    assert state.config.lights.streetlights.highway[0].profile.brightness_max == pytest.approx(0.42, abs=1e-2)
    assert returned_slider.get_current_value() == pytest.approx(0.42, abs=1e-2)


def test_save_writes_config_to_existing_path(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    state.config_path = tmp_path / "scene.toml"
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

        # Act
        with mock.patch.object(devmode_tab_module, "write_config") as mock_write:
            tab._on_save()

    # Assert
    mock_write.assert_called_once_with(state.config, state.config_path)


def test_save_opens_dialog_when_no_path(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    assert state.config_path is None
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

        # Act
        with mock.patch.object(devmode_tab_module, "write_config") as mock_write:
            tab._on_save()

    # Assert — no write yet; a save dialog is open instead.
    mock_write.assert_not_called()
    assert tab._save_dialog is not None


def test_toggle_real_tiles_submits_background_job(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, tmp_path)
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

        # Act
        with mock.patch.object(jobs, "submit", return_value=True) as mock_submit:
            tab._on_toggle_real_tiles()

    # Assert
    assert tab._use_real_tiles is True
    mock_submit.assert_called_once()
    assert tab._realtiles_btn.text == "Real Tiles: ON"
