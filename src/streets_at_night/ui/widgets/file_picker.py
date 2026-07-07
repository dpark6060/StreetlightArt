"""Text entry + Browse button that opens a UIFileDialog for path selection."""

import typing as t
from pathlib import Path

import pygame
import pygame_gui


class FilePickerRow:
    """Text entry + Browse button that opens a UIFileDialog.

    The dialog result lands in the text entry. Callers read ``.path``
    (Path | None, parsed from entry text) after handling events each frame.
    """

    def __init__(
        self,
        manager: pygame_gui.UIManager,
        container: t.Any,
        rect: pygame.Rect,
        *,
        initial_text: str = "",
        dialog_title: str = "Select file",
        allow_new_files: bool = False,
    ) -> None:
        """Build the entry and Browse button inside the given rect.

        The entry occupies the left portion of ``rect`` and an 80px Browse
        button sits at its right edge.

        Args:
            manager (pygame_gui.UIManager): The UI manager.
            container: Parent container for the widgets.
            rect (pygame.Rect): Bounding rect for the row.
            initial_text (str): Initial path text for the entry.
            dialog_title (str): Title for the file dialog window.
            allow_new_files (bool): When True, the dialog allows non-existent
                files (save mode); when False only existing files may be picked.
        """
        self._manager = manager
        self._dialog_title = dialog_title
        self._allow_new_files = allow_new_files
        self._dialog: t.Optional[pygame_gui.windows.UIFileDialog] = None

        button_width = 80
        entry_rect = pygame.Rect(rect.x, rect.y, rect.width - button_width, rect.height)
        button_rect = pygame.Rect(
            rect.x + rect.width - button_width, rect.y, button_width, rect.height
        )
        self._entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=entry_rect,
            manager=manager,
            container=container,
        )
        self._entry.set_text(initial_text)
        self._browse_btn = pygame_gui.elements.UIButton(
            relative_rect=button_rect,
            text="Browse",
            manager=manager,
            container=container,
        )

    @property
    def path(self) -> t.Optional[Path]:
        """Return the current entry text as a Path, or None when empty."""
        text = self._entry.get_text().strip()
        if not text:
            return None
        return Path(text)

    def set_path(self, path: Path) -> None:
        """Set the entry text to the given path.

        Args:
            path (Path): Path to display in the entry.
        """
        self._entry.set_text(str(path))

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process one event; report whether a new path was just picked.

        Args:
            event (pygame.event.Event): The event to handle.

        Returns:
            bool: True when the user picked a path via the dialog or finished
                typing in the entry.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element == self._browse_btn:
            self._open_dialog()
            return False
        if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED and event.ui_element == self._dialog:
            self._entry.set_text(event.text)
            self._dialog = None
            return True
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and event.ui_element == self._entry:
            return True
        return False

    def kill(self) -> None:
        """Kill the entry, Browse button, and any open dialog."""
        self._entry.kill()
        self._browse_btn.kill()
        if self._dialog is not None:
            self._dialog.kill()
            self._dialog = None

    def _open_dialog(self) -> None:
        """Open a UIFileDialog seeded with the current entry text or cwd."""
        initial = self._entry.get_text().strip() or str(Path.cwd())
        self._dialog = pygame_gui.windows.UIFileDialog(
            rect=pygame.Rect(150, 100, 600, 500),
            manager=self._manager,
            window_title=self._dialog_title,
            initial_file_path=initial,
            allow_existing_files_only=not self._allow_new_files,
        )
