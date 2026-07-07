"""Abstract base class for UI tabs."""

import abc

import pygame
import pygame_gui

from ..app_state import AppState
from ..jobs import JobEvent, JobRunner


class Tab(abc.ABC):
    """One tab of the UI. Widgets exist only while the tab is active."""

    name: str  # class attribute, e.g. "Verify"

    def __init__(self, state: AppState, jobs: JobRunner) -> None:
        """Store the shared state and job runner for this tab.

        Args:
            state (AppState): Shared application state.
            jobs (JobRunner): Shared background job runner.
        """
        self.state = state
        self.jobs = jobs

    @abc.abstractmethod
    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Create this tab's widgets inside content_rect.

        Args:
            manager (pygame_gui.UIManager): The UI manager to build widgets on.
            content_rect (pygame.Rect): Area below the tab and config bars.
        """

    @abc.abstractmethod
    def destroy(self) -> None:
        """Kill every widget this tab created."""

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle a pygame/pygame-gui event routed to the active tab.

        Args:
            event (pygame.event.Event): The event to handle.
        """

    def handle_job_event(self, event: JobEvent) -> None:
        """Handle one background-job event polled by the app.

        Args:
            event (JobEvent): The job progress/completion event.
        """

    def refresh_from_state(self) -> None:
        """Refresh config-derived widgets after the working config changed."""

    def update(self, dt: float) -> None:
        """Advance any per-frame tab logic.

        Args:
            dt (float): Seconds since the previous frame.
        """

    def draw(self, window: pygame.Surface) -> None:
        """Blit any render surfaces. Called before manager.draw_ui.

        Args:
            window (pygame.Surface): The window surface to draw onto.
        """
