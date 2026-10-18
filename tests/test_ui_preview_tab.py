# Radical
"""Headless tests for the Preview tab: widgets, camera math, coord sync, load job."""

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from streets_at_night.camera.camera import create_camera  # noqa: E402
from streets_at_night.camera.controls import InputResult  # noqa: E402
from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.ui.app_state import AppState  # noqa: E402
from streets_at_night.ui.jobs import JobEvent, JobRunner  # noqa: E402
from streets_at_night.ui.tabs import preview_tab as preview_tab_module  # noqa: E402
from streets_at_night.ui.tabs.preview_tab import (  # noqa: E402
    PreviewTab,
    _load_preview_data,
    move_delta,
)

_MODULE = "streets_at_night.ui.tabs.preview_tab"


@pytest.fixture
def config() -> SceneConfig:
    """Return a default SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


@pytest.fixture
def content_rect() -> pygame.Rect:
    """Return a representative content rect below the tab/config bars."""
    return pygame.Rect(0, 84, 1440, 816)


def _make_tab(config, config_path, tmp_path):
    """Build a PreviewTab against fresh state and a job runner."""
    state = AppState(config_path=config_path, config=config, cache_dir=tmp_path / "cache")
    jobs = JobRunner()
    tab = PreviewTab(state, jobs)
    return tab, state, jobs


def _camera():
    """Return a camera in ENU meters looking north from altitude."""
    return create_camera(
        position=np.array([100.0, -500.0, 300.0]),
        target=np.array([100.0, 0.0, 100.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=100.0,
        far=60000.0,
    )


def _ready_tab(config, headless_manager, content_rect, tmp_path, config_path=None):
    """Build a tab and wire it into a loaded state with a mock scene + camera."""
    tab, state, jobs = _make_tab(config, config_path, tmp_path)
    with mock.patch.object(jobs, "submit", return_value=True):
        tab.build(headless_manager, content_rect)

    camera = _camera()
    scene = mock.MagicMock()
    scene.camera = camera
    scene.transformer = mock.Mock()
    state.scene = scene

    tab._loaded = True
    tab._flat_map = Image.new("RGB", (4, 4))
    tab._flat_meta = object()
    tab._flat_map_array = np.zeros((4, 4, 3), dtype=np.uint8)
    tab._original_position = camera.position.copy()
    tab._original_target = camera.target.copy()
    tab._original_fov = camera.fov_deg
    return tab, state, jobs, camera


# ---------------------------------------------------------------------------
# Widget lifecycle
# ---------------------------------------------------------------------------


def test_build_creates_widgets_and_destroy_kills_all_including_labels(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, _, jobs = _make_tab(config, None, tmp_path)

    # Act
    with mock.patch.object(jobs, "submit", return_value=True):
        tab.build(headless_manager, content_rect)

    # Assert — controls exist, labels captured for every labeled row.
    assert set(tab._coord_entries) == {"cam_lat", "cam_lon", "cam_alt", "tgt_lat", "tgt_lon", "tgt_alt"}
    assert len(tab._buttons) == 12  # 6 move + CCW/CW + Pitch+/Pitch- + Reset + Render
    assert len(tab._labels) == 8  # move step + rotate step + 6 coord labels
    labels = list(tab._labels)
    buttons = list(tab._buttons.keys())

    tab.destroy()

    # Assert — every widget (labels included) was killed and refs dropped.
    assert all(not lbl.alive() for lbl in labels)
    assert all(not btn.alive() for btn in buttons)
    assert tab._labels == []
    assert tab._buttons == {}
    assert tab._coord_entries == {}
    assert tab._status is None
    assert tab._flat_map_array is None


def test_build_submits_load_job(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, None, tmp_path)

    # Act
    with mock.patch.object(jobs, "submit", return_value=True) as mock_submit:
        tab.build(headless_manager, content_rect)

    # Assert
    mock_submit.assert_called_once()


# ---------------------------------------------------------------------------
# Movement math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "direction, expected",
    [
        ("N", [0.0, 50.0, 0.0]),
        ("S", [0.0, -50.0, 0.0]),
        ("E", [50.0, 0.0, 0.0]),
        ("W", [-50.0, 0.0, 0.0]),
        ("Up", [0.0, 0.0, 50.0]),
        ("Down", [0.0, 0.0, -50.0]),
    ],
)
def test_move_delta_returns_enu_vector_for_direction(direction, expected):
    # Act
    result = move_delta(direction, 50.0)

    # Assert
    assert result.tolist() == expected


def test_move_handler_pans_camera_by_typed_step(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, _, camera = _ready_tab(config, headless_manager, content_rect, tmp_path)
    tab._move_step_entry.set_text("250")
    pos_before = camera.position.copy()
    tgt_before = camera.target.copy()

    # Act — the North button handler pans position AND target by [0, step, 0].
    with mock.patch.object(tab, "_render_frame"), mock.patch.object(tab, "_refresh_coord_entries"):
        tab._make_move_handler("N")()

    # Assert
    assert camera.position.tolist() == (pos_before + np.array([0.0, 250.0, 0.0])).tolist()
    assert camera.target.tolist() == (tgt_before + np.array([0.0, 250.0, 0.0])).tolist()


def test_reset_rotation_restores_original_target_keeps_position(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, _, _, camera = _ready_tab(config, headless_manager, content_rect, tmp_path)
    pos_before = camera.position.copy()
    camera.target = np.array([9999.0, 9999.0, 9999.0])

    # Act
    with mock.patch.object(tab, "_render_frame"), mock.patch.object(tab, "_refresh_coord_entries"):
        tab._on_reset_rotation()

    # Assert
    assert camera.target.tolist() == tab._original_target.tolist()
    assert camera.position.tolist() == pos_before.tolist()


# ---------------------------------------------------------------------------
# Live render during held-key/mouse flight
# ---------------------------------------------------------------------------


def test_update_renders_immediately_while_camera_moves(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange — cursor over the view, handle_input moves the camera this frame.
    tab, _, _, camera = _ready_tab(config, headless_manager, content_rect, tmp_path)

    def _fake_input(events, cam, dt, held, drag, opos, otgt, ofov):
        cam.move(np.array([0.0, 10.0, 0.0]))
        return InputResult()

    # Act
    with mock.patch(f"{_MODULE}.pygame.mouse.get_pos", return_value=(500, 400)), mock.patch(
        f"{_MODULE}.handle_input", side_effect=_fake_input
    ), mock.patch.object(tab, "_render_frame") as mock_render, mock.patch.object(
        tab, "_refresh_coord_entries"
    ):
        tab.update(0.016)

    # Assert — a frame is produced in the same frame the camera moved (no settle wait).
    mock_render.assert_called_once()


def test_update_does_not_render_when_camera_still(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange — cursor over the view but handle_input reports no movement.
    tab, _, _, _ = _ready_tab(config, headless_manager, content_rect, tmp_path)

    # Act
    with mock.patch(f"{_MODULE}.pygame.mouse.get_pos", return_value=(500, 400)), mock.patch(
        f"{_MODULE}.handle_input", return_value=InputResult()
    ), mock.patch.object(tab, "_render_frame") as mock_render:
        tab.update(0.016)

    # Assert
    mock_render.assert_not_called()


# ---------------------------------------------------------------------------
# Render button: coordinate parse -> ENU
# ---------------------------------------------------------------------------


def test_on_render_converts_typed_coords_to_enu_and_assigns_camera(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, state, _, camera = _ready_tab(config, headless_manager, content_rect, tmp_path)
    state.scene.transformer.wgs84_to_enu.side_effect = [(11.0, 22.0, 33.0), (44.0, 55.0, 66.0)]
    texts = {
        "cam_lat": "34.2", "cam_lon": "-117.3", "cam_alt": "5000",
        "tgt_lat": "34.1", "tgt_lon": "-117.2", "tgt_alt": "1000",
    }
    for key, value in texts.items():
        tab._coord_entries[key].set_text(value)

    # Act
    with mock.patch.object(tab, "_render_frame"), mock.patch.object(
        tab, "_refresh_coord_entries"
    ), mock.patch.object(tab, "_persist_camera"):
        tab._on_render()

    # Assert — position/target come from the transformer's ENU output.
    assert camera.position.tolist() == [11.0, 22.0, 33.0]
    assert camera.target.tolist() == [44.0, 55.0, 66.0]
    first_call = state.scene.transformer.wgs84_to_enu.call_args_list[0]
    assert first_call.args[0] == pytest.approx(34.2)
    assert first_call.args[2] == pytest.approx(5000 * 0.3048)  # feet -> meters


def test_on_render_invalid_coordinate_sets_status_and_leaves_camera(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, _, _, camera = _ready_tab(config, headless_manager, content_rect, tmp_path)
    pos_before = camera.position.copy()
    tab._coord_entries["cam_lat"].set_text("not-a-number")

    # Act
    with mock.patch.object(tab, "_render_frame") as mock_render, mock.patch.object(
        tab, "_persist_camera"
    ) as mock_persist:
        tab._on_render()

    # Assert
    assert tab._status.text == "Invalid coordinate"
    assert camera.position.tolist() == pos_before.tolist()
    mock_render.assert_not_called()
    mock_persist.assert_not_called()


# ---------------------------------------------------------------------------
# Coordinate readout
# ---------------------------------------------------------------------------


def test_refresh_coord_entries_populates_from_camera(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, state, _, _ = _ready_tab(config, headless_manager, content_rect, tmp_path)
    state.scene.transformer.enu_to_wgs84.return_value = (34.123456, -117.654321, 30.48)

    # Act
    tab._refresh_coord_entries()

    # Assert — lat/lon at 6 dp, altitude converted to feet at 1 dp.
    assert tab._coord_entries["cam_lat"].get_text() == "34.123456"
    assert tab._coord_entries["cam_lon"].get_text() == "-117.654321"
    assert tab._coord_entries["cam_alt"].get_text() == "100.0"  # 30.48 m -> 100 ft


def test_refresh_coord_entries_skips_focused_entry(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, state, _, _ = _ready_tab(config, headless_manager, content_rect, tmp_path)
    state.scene.transformer.enu_to_wgs84.return_value = (34.0, -117.0, 0.0)
    tab._coord_entries["cam_lat"].set_text("EDITING")
    tab._coord_entries["cam_lat"].is_focused = True

    # Act
    tab._refresh_coord_entries()

    # Assert — focused entry left untouched, unfocused ones updated.
    assert tab._coord_entries["cam_lat"].get_text() == "EDITING"
    assert tab._coord_entries["cam_lon"].get_text() == "-117.000000"


# ---------------------------------------------------------------------------
# Load job (worker thread, no pygame/network)
# ---------------------------------------------------------------------------


def test_load_preview_data_builds_scene_when_none_cached(config, tmp_path):
    # Arrange
    progress = mock.Mock()
    flat = (object(), object())

    # Act
    with mock.patch(f"{_MODULE}.load_scene") as mock_load, mock.patch(
        f"{_MODULE}.get_flat_map", return_value=flat
    ) as mock_flat, mock.patch(
        f"{_MODULE}.save_flat_map_to_output"
    ) as mock_save:
        scene, flat_map, flat_meta = _load_preview_data(config, tmp_path, None, progress)

    # Assert
    mock_load.assert_called_once()
    assert mock_load.call_args.args[1] == "low"
    mock_flat.assert_called_once_with(mock_load.return_value, tmp_path)
    mock_save.assert_called_once_with(flat[0], config.region_name, Path("output"))
    assert scene is mock_load.return_value
    assert (flat_map, flat_meta) == flat


def test_load_preview_data_reuses_existing_scene(config, tmp_path):
    # Arrange
    progress = mock.Mock()
    existing = object()

    # Act
    with mock.patch(f"{_MODULE}.load_scene") as mock_load, mock.patch(
        f"{_MODULE}.get_flat_map", return_value=(object(), object())
    ) as mock_flat, mock.patch(f"{_MODULE}.save_flat_map_to_output"):
        scene, _, _ = _load_preview_data(config, tmp_path, existing, progress)

    # Assert
    mock_load.assert_not_called()
    mock_flat.assert_called_once_with(existing, tmp_path)
    assert scene is existing


def test_handle_job_event_done_adopts_scene_and_marks_clean(
    config, headless_manager, content_rect, tmp_path
):
    # Arrange
    tab, state, jobs = _make_tab(config, None, tmp_path)
    with mock.patch.object(jobs, "submit", return_value=True):
        tab.build(headless_manager, content_rect)
    scene = mock.MagicMock()
    scene.camera = _camera()
    flat_map = Image.new("RGB", (4, 4))
    flat_meta = object()

    # Act
    with mock.patch.object(tab, "_render_frame"), mock.patch.object(tab, "_refresh_coord_entries"):
        tab.handle_job_event(JobEvent("done", result=(scene, flat_map, flat_meta)))

    # Assert
    assert state.scene is scene
    assert state.scene_quality == "low"
    assert state.scene_dirty is False
    assert tab._loaded is True
    assert tab._flat_map_array.shape == (4, 4, 3)
    assert tab._original_target.tolist() == scene.camera.target.tolist()


def test_handle_job_event_error_sets_status(config, headless_manager, content_rect, tmp_path):
    # Arrange
    tab, _, jobs = _make_tab(config, None, tmp_path)
    with mock.patch.object(jobs, "submit", return_value=True):
        tab.build(headless_manager, content_rect)

    # Act
    tab.handle_job_event(JobEvent("error", message="boom"))

    # Assert
    assert tab._status.text == "Load failed: boom"
