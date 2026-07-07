"""Verify tab: edit bounds, run the OSM verify pipeline, show the plot."""

import typing as t
from pathlib import Path

import pygame
import pygame_gui
from PIL import Image

from ...config.schema import BoundsConfig
from ...data.osm_loader import load_buildings, load_land_use, load_roads
from ...data.road_classifier import classify_roads
from ...data.verify_plot import save_verification_plot
from ...devmode.config_writer import write_config
from ...config.camera_io import write_bounds_to_toml
from ...scene_loader import bounds_to_bbox
from ..imaging import fit_image_to_viewport, pil_to_pygame_surface
from ..jobs import JobEvent, ProgressFn
from ..widgets.labeled_entry import build_labeled_entry
from .base import Tab

_PANEL_WIDTH = 360
_PANEL_MARGIN = 8
_ROW_HEIGHT = 28
_BUTTON_HEIGHT = 36
_ROW_GAP = 8
_OUTPUT_PATH = Path("output/verify.png")
_BOUND_LABELS: t.Tuple[str, str, str, str] = ("North", "South", "East", "West")


def run_verify_pipeline(
    config: t.Any,
    cache_dir: Path,
    output_path: Path,
    progress: ProgressFn,
) -> Path:
    """Reproduce ``cmd_verify`` as a background job returning the plot path.

    Runs entirely on the worker thread and touches no pygame state: it only
    loads OSM data and writes the matplotlib figure (forced to the Agg backend
    inside ``save_verification_plot``).

    Args:
        config (SceneConfig): Working config whose bounds drive the download.
        cache_dir (Path): OSM cache directory.
        output_path (Path): Destination PNG for the verification plot.
        progress (ProgressFn): Callback for indeterminate progress messages.

    Returns:
        Path: The output_path the plot was written to.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bbox = bounds_to_bbox(config.bounds)

    progress("Loading road network", -1.0)
    roads = classify_roads(load_roads(bbox, cache_dir))

    progress("Loading land use", -1.0)
    land_use = load_land_use(bbox, cache_dir)

    progress("Loading buildings", -1.0)
    buildings = load_buildings(bbox, cache_dir)

    progress("Generating plot", -1.0)
    save_verification_plot(config, roads, land_use, buildings, output_path)
    return output_path


class VerifyTab(Tab):
    """Verify tab: bounds editor, OSM verify pipeline, and plot viewer."""

    name = "Verify"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise the tab with no widgets built yet."""
        super().__init__(*args, **kwargs)
        self._manager: t.Optional[pygame_gui.UIManager] = None
        self._view_rect = pygame.Rect(0, 0, 0, 0)
        self._entries: t.Dict[str, pygame_gui.elements.UITextEntryLine] = {}
        self._labels: t.List[pygame_gui.elements.UILabel] = []
        self._rerender_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._save_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._status: t.Optional[pygame_gui.elements.UILabel] = None
        self._save_dialog: t.Optional[pygame_gui.windows.UIFileDialog] = None
        self._plot_surface: t.Optional[pygame.Surface] = None
        self._job_pending = False

    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Create the view area, bounds entries, buttons, and status label.

        Args:
            manager (pygame_gui.UIManager): The UI manager.
            content_rect (pygame.Rect): Content area for the tab.
        """
        self._manager = manager
        self._view_rect = pygame.Rect(
            content_rect.x,
            content_rect.y,
            content_rect.width - _PANEL_WIDTH,
            content_rect.height,
        )
        panel_x = content_rect.x + content_rect.width - _PANEL_WIDTH + _PANEL_MARGIN
        panel_w = _PANEL_WIDTH - 2 * _PANEL_MARGIN
        y = content_rect.y + _PANEL_MARGIN

        for label in _BOUND_LABELS:
            label_widget, entry = build_labeled_entry(manager, None, panel_x, y, label)
            self._labels.append(label_widget)
            self._entries[label] = entry
            y += _ROW_HEIGHT + _ROW_GAP

        self._rerender_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
            text="Rerender",
            manager=manager,
        )
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._save_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
            text="Save",
            manager=manager,
        )
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._status = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            text="",
            manager=manager,
        )
        self.refresh_from_state()

    def destroy(self) -> None:
        """Kill every widget this tab created, including any open dialog."""
        widgets: t.List[t.Any] = [
            *self._labels,
            *self._entries.values(),
            self._rerender_btn,
            self._save_btn,
            self._status,
            self._save_dialog,
        ]
        for widget in widgets:
            if widget is not None:
                widget.kill()
        self._labels = []
        self._entries = {}
        self._rerender_btn = None
        self._save_btn = None
        self._status = None
        self._save_dialog = None
        self._manager = None

    def refresh_from_state(self) -> None:
        """Fill the bounds entries from the config, blank until a config loads."""
        if not self._entries:
            return
        if self.state.config_path is None:
            for entry in self._entries.values():
                entry.set_text("")
            return
        bounds = self.state.config.bounds
        values = {
            "North": bounds.north,
            "South": bounds.south,
            "East": bounds.east,
            "West": bounds.west,
        }
        for label, entry in self._entries.items():
            entry.set_text(f"{values[label]:.6f}")

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route button presses and the save-dialog result.

        Args:
            event (pygame.event.Event): The event to handle.
        """
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element is self._rerender_btn:
                self._run_rerender()
            elif event.ui_element is self._save_btn:
                self._run_save()
            return
        if (
            event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED
            and event.ui_element is self._save_dialog
        ):
            self._save_to_path(Path(event.text))
            self._save_dialog = None

    def handle_job_event(self, event: JobEvent) -> None:
        """Update the status line and store the plot surface on completion.

        Args:
            event (JobEvent): The job progress/completion event.
        """
        if event.kind in ("done", "error") and not self._job_pending:
            return
        if event.kind == "progress":
            self._set_status(event.message)
        elif event.kind == "done":
            self._job_pending = False
            self._show_plot(event.result)
        elif event.kind == "error":
            self._job_pending = False
            self._set_status(f"Verify failed: {event.message}")

    def update(self, dt: float) -> None:
        """Enable buttons only while no job is running.

        Args:
            dt (float): Seconds since the previous frame (unused).
        """
        if self._rerender_btn is None or self._save_btn is None:
            return
        for button in (self._rerender_btn, self._save_btn):
            if self.jobs.busy:
                button.disable()
            else:
                button.enable()

    def draw(self, window: pygame.Surface) -> None:
        """Blit the stored plot surface at the view-area origin.

        Args:
            window (pygame.Surface): The window surface to draw onto.
        """
        if self._plot_surface is not None:
            window.blit(self._plot_surface, (self._view_rect.x, self._view_rect.y))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _run_rerender(self) -> None:
        """Read bounds, adopt them, and submit the verify pipeline job."""
        if self.jobs.busy:
            return
        bounds = self._read_bounds()
        if bounds is None:
            return
        self.state.config.bounds = bounds
        self.state.invalidate_scene()
        config = self.state.config
        cache_dir = self.state.cache_dir
        self._job_pending = True
        self.jobs.submit(
            lambda progress: run_verify_pipeline(config, cache_dir, _OUTPUT_PATH, progress)
        )
        self._set_status("Loading road network")

    def _run_save(self) -> None:
        """Persist bounds to the config file, prompting for a path if needed."""
        bounds = self._read_bounds()
        if bounds is None:
            return
        self.state.config.bounds = bounds
        self.state.invalidate_scene()
        if self.state.config_path is not None:
            write_bounds_to_toml(self.state.config_path, bounds)
            self._set_status(f"Bounds saved to {self.state.config_path}")
            return
        self._open_save_dialog()

    def _open_save_dialog(self) -> None:
        """Open a save-mode file dialog seeded with scene_config.toml."""
        if self._manager is None:
            return
        self._save_dialog = pygame_gui.windows.UIFileDialog(
            rect=pygame.Rect(150, 100, 600, 500),
            manager=self._manager,
            window_title="Save config",
            initial_file_path="scene_config.toml",
            allow_existing_files_only=False,
        )

    def _save_to_path(self, path: Path) -> None:
        """Write bounds (or a full config for a new file) to the chosen path.

        Args:
            path (Path): Destination config file chosen in the save dialog.
        """
        if path.exists():
            write_bounds_to_toml(path, self.state.config.bounds)
        else:
            write_config(self.state.config, path)
        self.state.config_path = path
        self._set_status(f"Bounds saved to {path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_bounds(self) -> t.Optional[BoundsConfig]:
        """Parse the four bound entries into a BoundsConfig.

        Returns:
            BoundsConfig | None: The parsed bounds, or None if any entry is not
                a valid float (status label set to the error in that case).
        """
        try:
            values = {
                label: float(entry.get_text().strip())
                for label, entry in self._entries.items()
            }
        except ValueError:
            self._set_status("Invalid bounds value")
            return None
        return BoundsConfig(
            north=values["North"],
            south=values["South"],
            east=values["East"],
            west=values["West"],
        )

    def _show_plot(self, output_path: t.Any) -> None:
        """Load the verify PNG, letterbox it, and store it as a surface.

        Args:
            output_path (Path): Path the plot was written to.
        """
        image = Image.open(output_path)
        fitted = fit_image_to_viewport(image, self._view_rect.width, self._view_rect.height)
        self._plot_surface = pil_to_pygame_surface(fitted)
        self._set_status(f"Saved {output_path}")

    def _set_status(self, text: str) -> None:
        """Set the status label text if the label exists.

        Args:
            text (str): Status message to display.
        """
        if self._status is not None:
            self._status.set_text(text)
