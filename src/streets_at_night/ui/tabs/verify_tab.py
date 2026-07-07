"""Verify tab (stub — filled in by change 03)."""

import typing as t

import pygame
import pygame_gui

from .base import Tab


class VerifyTab(Tab):
    """Verify tab: OSM download + verification plot (stub for now)."""

    name = "Verify"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise the tab with no widgets built yet."""
        super().__init__(*args, **kwargs)
        self._label: t.Optional[pygame_gui.elements.UILabel] = None

    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Create a centered placeholder label inside content_rect.

        Args:
            manager (pygame_gui.UIManager): The UI manager.
            content_rect (pygame.Rect): Content area for the tab.
        """
        self._label = pygame_gui.elements.UILabel(
            relative_rect=content_rect,
            text=f"{self.name} tab: not implemented",
            manager=manager,
        )

    def destroy(self) -> None:
        """Kill the placeholder label."""
        if self._label is not None:
            self._label.kill()
            self._label = None
