"""UIApp: the tabbed pygame + pygame-gui window and its event loop."""

import typing as t
from pathlib import Path

import pygame
import pygame_gui

from ..config.loader import load_config
from ..config.schema import SceneConfig
from .app_state import AppState
from .jobs import JobRunner
from .tabs.animate_tab import AnimateTab
from .tabs.base import Tab
from .tabs.devmode_tab import DevmodeTab
from .tabs.preview_tab import PreviewTab
from .tabs.render_tab import RenderTab
from .tabs.verify_tab import VerifyTab
from .widgets.file_picker import FilePickerRow

_WINDOW_WIDTH = 1440
_WINDOW_HEIGHT = 900
_TAB_BAR_HEIGHT = 40
_CONFIG_BAR_HEIGHT = 44
_TAB_COUNT = 5


class UIApp:
    """Owns the window, tab bar, config bar, shared state, and event loop."""

    def __init__(
        self,
        config: SceneConfig,
        config_path: t.Optional[Path],
        cache_dir: Path,
        initial_tab: str = "Verify",
    ) -> None:
        """Initialise pygame, build the UI chrome, and construct the tabs.

        Args:
            config (SceneConfig): Initial working config.
            config_path (Path | None): Path the config came from, or None.
            cache_dir (Path): Resolved cache directory.
            initial_tab (str): Name of the tab to open first.
        """
        pygame.init()
        self._window = pygame.display.set_mode((_WINDOW_WIDTH, _WINDOW_HEIGHT))
        pygame.display.set_caption("Streets At Night")
        self._manager = pygame_gui.UIManager((_WINDOW_WIDTH, _WINDOW_HEIGHT))
        self._clock = pygame.time.Clock()

        self.state = AppState(config_path=config_path, config=config, cache_dir=cache_dir)
        self.jobs = JobRunner()

        self._content_rect = pygame.Rect(
            0,
            _TAB_BAR_HEIGHT + _CONFIG_BAR_HEIGHT,
            _WINDOW_WIDTH,
            _WINDOW_HEIGHT - _TAB_BAR_HEIGHT - _CONFIG_BAR_HEIGHT,
        )

        self._tabs: t.List[Tab] = [
            VerifyTab(self.state, self.jobs),
            PreviewTab(self.state, self.jobs),
            RenderTab(self.state, self.jobs),
            AnimateTab(self.state, self.jobs),
            DevmodeTab(self.state, self.jobs),
        ]
        self._tab_buttons: t.Dict[pygame_gui.elements.UIButton, Tab] = {}
        self._build_tab_bar()
        self._build_config_bar()

        self._active_tab = self._get_initial_tab(initial_tab)
        self._active_tab.build(self._manager, self._content_rect)
        self._refresh_tab_bar()

        self._displayed_config_path = self.state.config_path
        self._running = False

    def run(self) -> None:
        """Run the 60 fps main loop until the window is closed."""
        self._running = True
        while self._running:
            dt = self._clock.tick(60) / 1000.0
            self._process_frame_events()
            self._pump_jobs()
            self._refresh_tab_bar()
            self._sync_config_bar()
            self._active_tab.update(dt)
            self._manager.update(dt)
            self._draw(dt)
        pygame.quit()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_tab_bar(self) -> None:
        """Create one equal-width button per tab across the top of the window."""
        button_width = _WINDOW_WIDTH // _TAB_COUNT
        for i, tab in enumerate(self._tabs):
            button = pygame_gui.elements.UIButton(
                relative_rect=pygame.Rect(
                    i * button_width, 0, button_width, _TAB_BAR_HEIGHT
                ),
                text=tab.name,
                manager=self._manager,
            )
            self._tab_buttons[button] = tab

    def _build_config_bar(self) -> None:
        """Create the persistent config bar: label, file picker, and Load button."""
        y = _TAB_BAR_HEIGHT
        pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(8, y, 60, _CONFIG_BAR_HEIGHT),
            text="Config:",
            manager=self._manager,
        )
        initial = str(self.state.config_path) if self.state.config_path else ""
        self._config_picker = FilePickerRow(
            self._manager,
            None,
            pygame.Rect(72, y + 8, 700, _CONFIG_BAR_HEIGHT - 16),
            initial_text=initial,
            dialog_title="Select config file",
            allow_new_files=False,
        )
        self._load_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(784, y + 8, 80, _CONFIG_BAR_HEIGHT - 16),
            text="Load",
            manager=self._manager,
        )

    def _get_initial_tab(self, name: str) -> Tab:
        """Return the tab matching name, falling back to the first tab.

        Args:
            name (str): Requested tab name.

        Returns:
            Tab: The matching tab, or the first tab if none matches.
        """
        for tab in self._tabs:
            if tab.name == name:
                return tab
        return self._tabs[0]

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _process_frame_events(self) -> None:
        """Drain and dispatch this frame's pygame events."""
        for event in pygame.event.get():
            self._dispatch_event(event)
            self._manager.process_events(event)

    def _dispatch_event(self, event: pygame.event.Event) -> None:
        """Route one event to quit, tab bar, config bar, or the active tab.

        Args:
            event (pygame.event.Event): The event to dispatch.
        """
        if event.type == pygame.QUIT:
            self._running = False
            return
        if self._handle_tab_bar_event(event):
            return
        if self._handle_config_bar_event(event):
            return
        self._active_tab.handle_event(event)

    def _handle_tab_bar_event(self, event: pygame.event.Event) -> bool:
        """Switch tabs when a tab-bar button is pressed.

        Args:
            event (pygame.event.Event): The event to inspect.

        Returns:
            bool: True if this event was a tab-bar switch (consumed).
        """
        if event.type != pygame_gui.UI_BUTTON_PRESSED:
            return False
        tab = self._tab_buttons.get(event.ui_element)
        if tab is None or tab is self._active_tab:
            return False
        self._switch_tab(tab)
        return True

    def _handle_config_bar_event(self, event: pygame.event.Event) -> bool:
        """Handle config picker/Load events; load config on pick or Load press.

        Args:
            event (pygame.event.Event): The event to inspect.

        Returns:
            bool: True if the event belonged to the config bar (consumed).
        """
        picked = self._config_picker.handle_event(event)
        load_pressed = (
            event.type == pygame_gui.UI_BUTTON_PRESSED
            and event.ui_element == self._load_btn
        )
        if not picked and not load_pressed:
            return False
        self._load_config_from_picker()
        return True

    def _switch_tab(self, tab: Tab) -> None:
        """Destroy the active tab and build the incoming one.

        Args:
            tab (Tab): The tab to switch to.
        """
        self._active_tab.destroy()
        self._active_tab = tab
        self._active_tab.build(self._manager, self._content_rect)
        self._active_tab.refresh_from_state()
        self._refresh_tab_bar()

    def _refresh_tab_bar(self) -> None:
        """Set tab-bar button states from the active tab and the job-busy flag.

        While a job runs, every tab-bar button is disabled so no tab switch can
        start mid-job (the app runs one job at a time). When idle, the active
        tab's button stays disabled as the current-tab indicator and the rest
        are enabled.
        """
        busy = self.jobs.busy
        for button, tab in self._tab_buttons.items():
            if busy or tab is self._active_tab:
                button.disable()
            else:
                button.enable()

    def _load_config_from_picker(self) -> None:
        """Load the config at the picker path into state; keep the app alive on error."""
        path = self._config_picker.path
        if path is None:
            return
        previous = self.state.config_path
        try:
            new_config = load_config(path)
        except Exception as exc:  # noqa: BLE001 - a bad TOML must not crash the app
            print(f"Load config failed: {exc}")
            self._config_picker.set_path(previous if previous else Path(""))
            return
        self.state.set_config(new_config, path)
        self._active_tab.refresh_from_state()

    # ------------------------------------------------------------------
    # Per-frame updates
    # ------------------------------------------------------------------

    def _pump_jobs(self) -> None:
        """Forward each polled job event to the active tab."""
        for job_event in self.jobs.poll():
            self._active_tab.handle_job_event(job_event)

    def _sync_config_bar(self) -> None:
        """Refresh the config-bar entry when a tab changed state.config_path."""
        if self.state.config_path == self._displayed_config_path:
            return
        self._displayed_config_path = self.state.config_path
        if self.state.config_path is not None:
            self._config_picker.set_path(self.state.config_path)

    def _draw(self, dt: float) -> None:
        """Clear the window, draw the active tab, then the UI, and flip.

        Args:
            dt (float): Seconds since the previous frame (unused; kept for symmetry).
        """
        self._window.fill((20, 20, 20))
        self._active_tab.draw(self._window)
        self._manager.draw_ui(self._window)
        pygame.display.flip()


def run_ui(
    config: SceneConfig,
    config_path: t.Optional[Path],
    cache_dir: Path,
    initial_tab: str = "Verify",
) -> None:
    """Launch the tabbed UI window and run its event loop until quit.

    Args:
        config (SceneConfig): Initial working config.
        config_path (Path | None): Path the config came from, or None.
        cache_dir (Path): Resolved cache directory.
        initial_tab (str): Name of the tab to open first.
    """
    app = UIApp(config, config_path, cache_dir, initial_tab)
    app.run()
