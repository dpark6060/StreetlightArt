# Radical
"""Headless tests for streets_at_night.ui.widgets.file_picker.FilePickerRow."""

import os
from pathlib import Path
from unittest import mock

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.ui.widgets.file_picker import FilePickerRow  # noqa: E402

_ROW_RECT = pygame.Rect(0, 0, 180, 30)


def test_path_parses_initial_entry_text(headless_manager):
    # Act
    row = FilePickerRow(headless_manager, None, _ROW_RECT, initial_text="a.toml")

    # Assert
    assert row.path == Path("a.toml")
    row.kill()


def test_path_empty_text_returns_none(headless_manager):
    # Act
    row = FilePickerRow(headless_manager, None, _ROW_RECT, initial_text="")

    # Assert
    assert row.path is None
    row.kill()


def test_set_path_updates_the_entry_text(headless_manager):
    # Arrange
    row = FilePickerRow(headless_manager, None, _ROW_RECT, initial_text="a.toml")

    # Act
    row.set_path(Path("b.toml"))

    # Assert
    assert row.path == Path("b.toml")
    row.kill()


def test_handle_event_browse_press_opens_dialog_and_returns_false(headless_manager):
    # Arrange
    row = FilePickerRow(headless_manager, None, _ROW_RECT)
    event = pygame.event.Event(
        pygame_gui.UI_BUTTON_PRESSED, {"ui_element": row._browse_btn}
    )

    # Act — patch the dialog so no real file window is constructed.
    with mock.patch.object(pygame_gui.windows, "UIFileDialog") as dialog_cls:
        result = row.handle_event(event)

    # Assert — the press opens a dialog but reports no path picked yet.
    assert result is False
    assert row._dialog is dialog_cls.return_value
    row._dialog = None
    row.kill()


def test_handle_event_dialog_path_picked_sets_text_and_returns_true(headless_manager):
    # Arrange — simulate an already-open dialog, then feed its picked event.
    row = FilePickerRow(headless_manager, None, _ROW_RECT)
    row._dialog = mock.Mock()
    event = pygame.event.Event(
        pygame_gui.UI_FILE_DIALOG_PATH_PICKED,
        {"ui_element": row._dialog, "text": "/tmp/picked.toml"},
    )

    # Act
    result = row.handle_event(event)

    # Assert
    assert result is True
    assert row.path == Path("/tmp/picked.toml")
    assert row._dialog is None
    row.kill()


def test_handle_event_text_entry_finished_returns_true(headless_manager):
    # Arrange
    row = FilePickerRow(headless_manager, None, _ROW_RECT, initial_text="typed.toml")
    event = pygame.event.Event(
        pygame_gui.UI_TEXT_ENTRY_FINISHED, {"ui_element": row._entry}
    )

    # Act
    result = row.handle_event(event)

    # Assert
    assert result is True
    row.kill()


def test_handle_event_unrelated_event_returns_false(headless_manager):
    # Arrange
    row = FilePickerRow(headless_manager, None, _ROW_RECT)
    event = pygame.event.Event(pygame.USEREVENT, {"ui_element": None})

    # Act
    result = row.handle_event(event)

    # Assert
    assert result is False
    row.kill()


def test_kill_removes_entry_and_button_widgets(headless_manager):
    # Arrange
    row = FilePickerRow(headless_manager, None, _ROW_RECT, initial_text="a.toml")
    entry = row._entry
    button = row._browse_btn

    # Act
    row.kill()

    # Assert
    assert not entry.alive()
    assert not button.alive()
