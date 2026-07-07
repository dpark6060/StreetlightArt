"""Render tab: output picker, quality choice, and full HDR export with progress.

Minimal by design. The whole render runs on the JobRunner worker thread and
returns plain data (a :class:`RenderResult`); the worker never touches pygame or
mutates ``state``. Converting the PIL image to a surface and caching a
freshly-built scene both happen on the main thread in ``handle_job_event``.
"""

import typing as t
from dataclasses import dataclass
from pathlib import Path

import pygame
import pygame_gui
from PIL import Image

from ...config.schema import SceneConfig
from ...rendering.pillow_backend import PillowBackend
from ...rendering.render_pipeline import (
    get_export_pass_names,
    get_export_passes,
    render_scene_frame_with_progress,
)
from ...scene.scene import Scene
from ...scene_loader import load_scene
from ..imaging import fit_image_to_viewport, pil_to_pygame_surface
from ..jobs import JobEvent, ProgressFn
from ..widgets.file_picker import FilePickerRow
from .base import Tab

_PANEL_WIDTH = 360
_PANEL_MARGIN = 8
_ROW_HEIGHT = 28
_BUTTON_HEIGHT = 36
_ROW_GAP = 8
_QUALITIES: t.List[str] = ["low", "medium", "high"]
_INITIAL_OUTPUT = "output/render.png"


@dataclass
class RenderResult:
    """Plain-data result handed from the render job back to the main thread.

    Attributes:
        output_path: Path the image was written to.
        image: The rendered PIL image (converted to a surface on the main thread).
        built_scene: The freshly built scene when the job loaded one, else None
            (None means a cached scene was reused and must not be overwritten).
        quality: Quality preset the scene was built/rendered at.
    """

    output_path: Path
    image: Image.Image
    built_scene: t.Optional[Scene]
    quality: str


def resolve_image_format(output_path: Path) -> str:
    """Return the PIL format string for an output path by its suffix.

    Mirrors ``main._resolve_output_format`` without importing that private
    helper: ``.tiff``/``.tif`` map to TIFF, everything else to PNG.

    Args:
        output_path (Path): The chosen output file path.

    Returns:
        str: "TIFF" for .tiff/.tif suffixes, otherwise "PNG".
    """
    if output_path.suffix.lower() in (".tiff", ".tif"):
        return "TIFF"
    return "PNG"


def run_render(
    config: SceneConfig,
    quality: str,
    output_path: Path,
    cache_dir: Path,
    existing_scene: t.Optional[Scene],
    progress: ProgressFn,
) -> RenderResult:
    """Reproduce ``cmd_render`` as a background job returning plain data.

    Runs entirely on the worker thread and touches no pygame state. Reuses the
    supplied cached scene when one is given; otherwise loads a fresh scene at the
    requested quality (returned in the result so the main thread can cache it).

    Args:
        config (SceneConfig): Working config driving the render.
        quality (str): Quality preset name ("low"/"medium"/"high").
        output_path (Path): Destination image path.
        cache_dir (Path): OSM + tile cache directory.
        existing_scene (Scene | None): A clean cached scene to reuse, or None.
        progress (ProgressFn): Callback for (message, fraction) progress.

    Returns:
        RenderResult: The output path, rendered image, freshly built scene (or
            None when reused), and the quality it was rendered at.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    built_scene: t.Optional[Scene] = None
    if existing_scene is not None:
        scene = existing_scene
    else:
        scene = load_scene(config, quality, cache_dir, progress=lambda msg: progress(msg, -1.0))
        built_scene = scene

    preset = config.quality_presets[quality]
    backend = PillowBackend(preset.width, preset.height)
    for car_pair in scene.car_pairs:
        car_pair.resolve_facing(scene.camera.get_forward())

    passes = get_export_passes(cache_dir, no_lights=False, quality_preset=preset)
    names = get_export_pass_names(no_lights=False)
    render_scene_frame_with_progress(scene, backend, passes, names, progress=progress)

    image = backend.get_image()
    image.save(str(output_path), format=resolve_image_format(output_path))
    return RenderResult(output_path, image, built_scene, quality)


class RenderTab(Tab):
    """Render tab: output picker, quality dropdown, and progress-tracked export."""

    name = "Render"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise the tab with no widgets built yet."""
        super().__init__(*args, **kwargs)
        self._manager: t.Optional[pygame_gui.UIManager] = None
        self._view_rect = pygame.Rect(0, 0, 0, 0)
        self._labels: t.List[pygame_gui.elements.UILabel] = []
        self._output_picker: t.Optional[FilePickerRow] = None
        self._quality_menu: t.Optional[pygame_gui.elements.UIDropDownMenu] = None
        self._render_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._progress_bar: t.Optional[pygame_gui.elements.UIProgressBar] = None
        self._status: t.Optional[pygame_gui.elements.UILabel] = None
        self._image_surface: t.Optional[pygame.Surface] = None

    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Create the view area, output picker, quality menu, button, and bar.

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

        y = self._build_output_row(manager, panel_x, panel_w, y)
        y = self._build_quality_row(manager, panel_x, panel_w, y)
        y = self._build_action_widgets(manager, panel_x, panel_w, y)

    def _build_output_row(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the "Output" label and its FilePickerRow.

        Returns:
            int: The next free y coordinate.
        """
        self._labels.append(
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
                text="Output",
                manager=manager,
            )
        )
        y += _ROW_HEIGHT + _ROW_GAP
        self._output_picker = FilePickerRow(
            manager,
            None,
            pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            initial_text=_INITIAL_OUTPUT,
            dialog_title="Select output file",
            allow_new_files=True,
        )
        y += _ROW_HEIGHT + _ROW_GAP
        return y

    def _build_quality_row(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the "Quality" label and its dropdown menu.

        Returns:
            int: The next free y coordinate.
        """
        self._labels.append(
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
                text="Quality",
                manager=manager,
            )
        )
        y += _ROW_HEIGHT + _ROW_GAP
        self._quality_menu = pygame_gui.elements.UIDropDownMenu(
            options_list=_QUALITIES,
            starting_option=_QUALITIES[0],
            relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            manager=manager,
        )
        y += _ROW_HEIGHT + _ROW_GAP
        return y

    def _build_action_widgets(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the Render button, progress bar, and status label.

        Returns:
            int: The next free y coordinate.
        """
        self._render_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
            text="Render",
            manager=manager,
        )
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._progress_bar = pygame_gui.elements.UIProgressBar(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            manager=manager,
        )
        self._progress_bar.set_current_progress(0.0)
        y += _ROW_HEIGHT + _ROW_GAP
        self._status = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            text="",
            manager=manager,
        )
        return y + _ROW_HEIGHT + _ROW_GAP

    def destroy(self) -> None:
        """Kill every widget this tab created, including labels and the picker."""
        for label in self._labels:
            label.kill()
        widgets: t.List[t.Any] = [
            self._quality_menu,
            self._render_btn,
            self._progress_bar,
            self._status,
        ]
        for widget in widgets:
            if widget is not None:
                widget.kill()
        if self._output_picker is not None:
            self._output_picker.kill()
        self._labels = []
        self._output_picker = None
        self._quality_menu = None
        self._render_btn = None
        self._progress_bar = None
        self._status = None
        self._manager = None
        self._image_surface = None

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route the output picker's events and the Render button press.

        Args:
            event (pygame.event.Event): The event to handle.
        """
        if self._output_picker is not None:
            self._output_picker.handle_event(event)
        if event.type == pygame_gui.UI_BUTTON_PRESSED and event.ui_element is self._render_btn:
            self._run_render()

    def handle_job_event(self, event: JobEvent) -> None:
        """Advance the progress bar/status and store the finished image.

        Args:
            event (JobEvent): The job progress/completion event.
        """
        if event.kind == "progress":
            self._on_progress(event.message, event.fraction)
        elif event.kind == "done":
            self._on_done(event.result)
        elif event.kind == "error":
            self._on_error(event.message)

    def update(self, dt: float) -> None:
        """Disable the Render button while any job is running, else enable it.

        Args:
            dt (float): Seconds since the previous frame (unused).
        """
        if self._render_btn is None:
            return
        if self.jobs.busy:
            self._render_btn.disable()
        else:
            self._render_btn.enable()

    def draw(self, window: pygame.Surface) -> None:
        """Blit the last rendered image at the view-area origin.

        Args:
            window (pygame.Surface): The window surface to draw onto.
        """
        if self._image_surface is not None:
            window.blit(self._image_surface, (self._view_rect.x, self._view_rect.y))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _run_render(self) -> None:
        """Submit the render job with values captured at click time."""
        if self.jobs.busy or self._output_picker is None:
            return
        output_path = self._output_picker.path
        if output_path is None:
            return
        quality = self._selected_quality()
        config = self.state.config
        cache_dir = self.state.cache_dir
        existing = self._reusable_scene(quality)
        self.jobs.submit(
            lambda progress: run_render(
                config, quality, output_path, cache_dir, existing, progress
            )
        )
        self._set_status("Rendering...")

    def _reusable_scene(self, quality: str) -> t.Optional[Scene]:
        """Return the cached scene when it is safe to reuse for this quality.

        Reuse only when a scene is cached, its quality matches, and it is not
        dirty; the aspect ratio baked into the camera is quality-specific, so a
        quality mismatch always forces a rebuild.

        Args:
            quality (str): The quality preset the render will use.

        Returns:
            Scene | None: The reusable cached scene, or None to force a rebuild.
        """
        if (
            self.state.scene is not None
            and self.state.scene_quality == quality
            and not self.state.scene_dirty
        ):
            return self.state.scene
        return None

    def _selected_quality(self) -> str:
        """Return the currently selected quality name from the dropdown."""
        if self._quality_menu is None:
            return _QUALITIES[0]
        selected = self._quality_menu.selected_option
        if isinstance(selected, tuple):
            return selected[0]
        return selected

    # ------------------------------------------------------------------
    # Job-event handlers (main thread)
    # ------------------------------------------------------------------

    def _on_progress(self, message: str, fraction: float) -> None:
        """Update the status label and, for determinate stages, the progress bar.

        Args:
            message (str): Progress message from the worker.
            fraction (float): Progress fraction 0..1, or -1 for indeterminate.
        """
        self._set_status(message)
        if self._progress_bar is not None and fraction >= 0:
            self._progress_bar.set_current_progress(fraction * 100.0)

    def _on_done(self, result: RenderResult) -> None:
        """Cache a freshly built scene, fill the bar, and show the rendered image.

        Args:
            result (RenderResult): The completed render job's result.
        """
        if result.built_scene is not None:
            self.state.scene = result.built_scene
            self.state.scene_quality = result.quality
            self.state.scene_dirty = False
        if self._progress_bar is not None:
            self._progress_bar.set_current_progress(100.0)
        fitted = fit_image_to_viewport(
            result.image, self._view_rect.width, self._view_rect.height
        )
        self._image_surface = pil_to_pygame_surface(fitted)
        self._set_status(f"Saved {result.output_path}")

    def _on_error(self, message: str) -> None:
        """Report a render failure and reset the progress bar.

        Args:
            message (str): The error message from the worker.
        """
        if self._progress_bar is not None:
            self._progress_bar.set_current_progress(0.0)
        self._set_status(f"Render failed: {message}")

    def _set_status(self, text: str) -> None:
        """Set the status label text if the label exists.

        Args:
            text (str): Status message to display.
        """
        if self._status is not None:
            self._status.set_text(text)
