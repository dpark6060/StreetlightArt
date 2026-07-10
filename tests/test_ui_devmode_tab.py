# Radical
"""Headless tests for the Devmode tab: build/destroy, kind switching, save."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
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


@pytest.fixture
def tall_content_rect() -> pygame.Rect:
    """A content rect tall enough that the whole Atmosphere section fits on screen.

    haze_color is the last row of the Atmosphere section; with the default
    content rect its swatch scrolls below the visible panel and cannot be
    clicked. A tall rect keeps every row (and its swatch) inside the panel.
    """
    return pygame.Rect(0, 0, 1440, 1100)


@pytest.fixture
def built_tab(config, headless_manager, tall_content_rect, tmp_path):
    """Build a DevmodeTab headless with heavy render stubbed and Atmosphere shown."""
    tab, _, _ = _make_tab(config, tmp_path)
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, tall_content_rect)
    return tab


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

        # Act — leave for Commercial: Signs, then return.
        tab._build_section("Commercial: Signs")
        tab._build_section("Streetlight: Highway (pool 0)")
        returned_slider = next(
            s for s, d in tab._light_registry.sliders.items()
            if d.config_path == ("brightness_max",)
        )

    # Assert — config kept the edit, and the rebuilt widget shows it (read from config).
    assert state.config.lights.streetlights.highway[0].profile.brightness_max == pytest.approx(0.42, abs=1e-2)
    assert returned_slider.get_current_value() == pytest.approx(0.42, abs=1e-2)


def test_finishing_color_hex_entry_updates_working_config(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

        # Switch to a colour-pool section and edit the first colour's hex.
        tab._build_section("Commercial: Parking")
        binding = tab._light_registry.color_pools[0]
        hex_entry = binding.rows[0][0]
        hex_entry.set_text("#0A0B0C")

        # Act — finishing the entry must reach _apply_active_section (early-return fix).
        tab._handle_entry_finished(hex_entry)

    # Assert — the working config's parking pool reflects the edit.
    assert state.config.lights.commercial.parking_colors.colors[0][0] == "#0A0B0C"
    assert isinstance(state.config.lights.commercial.parking_colors.colors[0], tuple)


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


def test_compute_world_extent_pads_bbox_by_ten_percent():
    # Arrange — a single segment spanning x 0..100, y 0..200.
    seg = SimpleNamespace(geometry=np.array([[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]]))
    scene = SimpleNamespace(road_segments=[seg])

    # Act
    (x_min, y_min), (x_max, y_max) = DevmodeTab._compute_world_extent(scene)

    # Assert — 10% pad each side.
    assert x_min == pytest.approx(-10.0)
    assert x_max == pytest.approx(110.0)
    assert y_min == pytest.approx(-20.0)
    assert y_max == pytest.approx(220.0)


def test_compute_world_extent_empty_scene_uses_fallback():
    scene = SimpleNamespace(road_segments=[])

    extent = DevmodeTab._compute_world_extent(scene)

    assert extent == ((-500.0, -500.0), (500.0, 500.0))


def test_minimap_drag_camera_writes_config_syncs_and_swallows(config, headless_manager, content_rect, tmp_path):
    # Arrange — build the tab (real synthetic scene + minimap, cheap render stubbed).
    tab, state, _ = _make_tab(config, tmp_path)
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

    cam_xy, _ = tab._dot_world_xy()
    cam_screen = tab._minimap.world_to_screen(*cam_xy)
    move_pos = (tab._minimap.layout.rect.centerx, tab._minimap.layout.rect.centery)
    down = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=cam_screen)
    motion = pygame.event.Event(pygame.MOUSEMOTION, pos=move_pos, rel=(1, 1), buttons=(1, 0, 0))
    tab.handle_event(down)
    tab.handle_event(motion)

    # Act — capture what the camera passthrough is fed; stub the cheap render.
    with mock.patch.object(devmode_tab_module, "sync_camera_from_config") as mock_sync, mock.patch.object(
        tab, "_render_cheap"
    ), mock.patch.object(tab, "_apply_camera_input", return_value=False) as mock_camera:
        tab.update(0.016)

    # Assert — config X/Y written to the drag world point, camera synced.
    expected_x, expected_y = tab._minimap.screen_to_world(*move_pos)
    assert state.config.devmode.camera.pos_x == pytest.approx(expected_x)
    assert state.config.devmode.camera.pos_y == pytest.approx(expected_y)
    mock_sync.assert_called_with(tab._scene, state.config)

    # The minimap events were swallowed, not forwarded to the scene camera.
    forwarded = mock_camera.call_args.args[0]
    assert down not in forwarded
    assert motion not in forwarded


def test_height_entry_finish_updates_pos_z_and_target_z(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, state, _ = _make_tab(config, tmp_path)
    with mock.patch.object(devmode_tab_module, "render_devmode_scene"), mock.patch.object(
        devmode_tab_module, "fit_image_to_viewport"
    ), mock.patch.object(devmode_tab_module, "pil_to_pygame_surface"):
        tab.build(headless_manager, content_rect)

    tab._cam_height_entry.set_text("123.5")
    tab._focal_height_entry.set_text("45.0")

    # Act
    with mock.patch.object(devmode_tab_module, "sync_camera_from_config") as mock_sync, mock.patch.object(
        tab, "_render_cheap"
    ):
        tab._handle_entry_finished(tab._cam_height_entry)

    # Assert
    assert state.config.devmode.camera.pos_z == pytest.approx(123.5)
    assert state.config.devmode.camera.target_z == pytest.approx(45.0)
    mock_sync.assert_called_once_with(tab._scene, state.config)


def test_apply_active_section_invalidates_shared_scene(config, tmp_path):
    # Arrange — a clean cached scene on the shared state.
    tab, state, _ = _make_tab(config, tmp_path)
    state.scene = mock.MagicMock()
    state.scene_quality = "low"
    state.scene_dirty = False

    # Act — a devmode edit writes working-config values.
    tab._apply_active_section()

    # Assert — shared scene cache is dropped so a later render rebuilds it.
    assert state.scene is None
    assert state.scene_dirty is True


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


def test_build_section_creates_swatches_for_hex_entries(built_tab):
    # Arrange — the Atmosphere section registers haze_color as a text entry.
    built_tab._build_section("Atmosphere")

    # Act
    registry = built_tab._active_registry()

    # Assert — at least the haze_color entry received a swatch.
    assert len(registry.swatches) >= 1


def test_swatch_click_opens_colour_picker_dialog(built_tab):
    # Arrange
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center

    # Act
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos})
    built_tab.handle_event(event)

    # Assert
    assert built_tab._color_dialog is not None
    assert built_tab._color_dialog_entry is entry


def test_second_swatch_click_does_not_stack_dialogs(built_tab):
    # Arrange
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    _entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos})
    built_tab.handle_event(event)
    first_dialog = built_tab._color_dialog

    # Act
    built_tab.handle_event(event)

    # Assert — same dialog instance, not a second one.
    assert built_tab._color_dialog is first_dialog


def test_colour_picked_sets_entry_applies_and_refreshes(built_tab):
    # Arrange — open the dialog for the haze_color entry.
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    built_tab.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos}))

    # Act
    picked = pygame.event.Event(
        pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED,
        {"colour": pygame.Color(18, 52, 86), "ui_element": built_tab._color_dialog},
    )
    built_tab.handle_event(picked)

    # Assert — entry text set, config applied through the normal path, dialog refs cleared.
    assert entry.get_text() == "#123456"
    assert built_tab.state.config.atmosphere.haze_color == (18, 52, 86)
    assert built_tab._color_dialog is None
    assert built_tab._color_dialog_entry is None


def test_camera_input_ignored_while_colour_dialog_open(built_tab):
    # Arrange — open the colour dialog via a swatch click.
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    _entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    built_tab.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos}))
    assert built_tab._is_modal_dialog_open() is True

    camera = built_tab._scene.camera
    pos_before = camera.position.copy()
    tgt_before = camera.target.copy()
    fov_before = camera.fov_deg

    # Act — a synthetic input frame that would otherwise reach handle_input.
    with mock.patch.object(devmode_tab_module, "handle_input") as mock_input:
        moved = built_tab._apply_camera_input([pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_w})], 0.016)

    # Assert — early return: no camera passthrough, camera untouched.
    assert moved is False
    mock_input.assert_not_called()
    assert np.array_equal(camera.position, pos_before)
    assert np.array_equal(camera.target, tgt_before)
    assert camera.fov_deg == fov_before


def test_section_switch_closes_open_colour_dialog(built_tab):
    # Arrange
    built_tab._build_section("Atmosphere")
    registry = built_tab._active_registry()
    _entry, swatch = next(iter(registry.swatches.items()))
    pos = swatch.get_abs_rect().center
    built_tab.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": pos}))

    # Act
    built_tab._build_section("Devmode Camera")

    # Assert
    assert built_tab._color_dialog is None
    assert built_tab._color_dialog_entry is None
