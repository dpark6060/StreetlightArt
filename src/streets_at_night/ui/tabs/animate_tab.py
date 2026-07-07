"""Animate tab: live heat-haze preview and offline MP4 record.

Two modes wrapping the CLI ``animate`` command:

* **Preview** plays the heat-haze shimmer in the view area at interactive rates.
  Building the haze (phase screens, seconds-long) runs on the JobRunner worker
  and returns a :class:`HeatHaze` instance as plain data. The per-frame
  ``HeatHaze.apply(t)`` warp runs on the **main thread** in ``update`` — exactly
  as the CLI pygame animate loop does — because it drives torch/MPS.
* **Record** runs entirely on the worker: load base frame, build haze, and
  stream frames to an MP4 with a progress callback.

Preview playback and record are mutually exclusive: Preview is disabled while a
job is running, and Record is disabled while a preview is playing (chosen over
force-stopping the preview because it keeps the button state purely
declarative in ``_update_buttons``).
"""

import typing as t
from pathlib import Path

import pygame
import pygame_gui

from ...config.schema import HeatHazeConfig, SceneConfig
from ...rendering.animate_service import (
    build_animate_haze,
    default_video_duration,
    load_animate_base_frame,
    record_haze,
    record_haze_seamless,
)
from ...rendering.heat_haze import HeatHaze
from ..imaging import array_to_pygame_surface
from ..jobs import JobEvent, ProgressFn
from ..widgets.file_picker import FilePickerRow
from ..widgets.labeled_entry import build_labeled_entry
from .base import Tab

_PANEL_WIDTH = 360
_PANEL_MARGIN = 8
_ROW_HEIGHT = 28
_BUTTON_HEIGHT = 36
_ROW_GAP = 8
_INITIAL_IMAGE = "output/render.png"
_INITIAL_OUTPUT = "output/animate.mp4"
_RECORD_QUALITY = "low"
_GATE_STATUS = "Enable [heat_haze] in the config to animate"

_DEFAULT_FPS = 30
_DEFAULT_CROSSFADE_S = 0.5


def build_preview_haze(
    config: SceneConfig,
    heat_haze_cfg: HeatHazeConfig,
    image_path: Path,
    view_w: int,
    view_h: int,
    progress: ProgressFn,
) -> HeatHaze:
    """Load the base frame and build a haze sized to the view area (worker job).

    Runs on the worker thread and returns the :class:`HeatHaze` as plain data;
    the main thread drives ``apply`` afterwards. Touches no pygame state.

    Args:
        config (SceneConfig): Working config (camera + heat haze).
        heat_haze_cfg (HeatHazeConfig): Active heat haze config.
        image_path (Path): Base image to animate.
        view_w (int): View-area width in pixels.
        view_h (int): View-area height in pixels.
        progress (ProgressFn): Progress callback for (message, fraction).

    Returns:
        HeatHaze: Ready-to-use haze instance sized to the view area.
    """
    base = load_animate_base_frame(image_path, view_w, view_h, progress)
    return build_animate_haze(config, heat_haze_cfg, base, view_w, view_h, progress)


def run_record(
    config: SceneConfig,
    heat_haze_cfg: HeatHazeConfig,
    image_path: Path,
    out_path: Path,
    out_w: int,
    out_h: int,
    fps: int,
    duration: t.Optional[float],
    crossfade_s: float,
    seamless: bool,
    progress: ProgressFn,
) -> Path:
    """Reproduce ``cmd_animate_record`` as a background job returning the path.

    Runs entirely on the worker thread. Mirrors the CLI: build the haze at the
    output resolution, derive the duration from the built haze when not given,
    then stream frames (seamless or plain) to the MP4.

    Args:
        config (SceneConfig): Working config.
        heat_haze_cfg (HeatHazeConfig): Active heat haze config.
        image_path (Path): Base image to animate.
        out_path (Path): Destination MP4 path.
        out_w (int): Output width in pixels.
        out_h (int): Output height in pixels.
        fps (int): Frame rate.
        duration (float | None): Duration in seconds, or None to auto-derive.
        crossfade_s (float): Crossfade tail length in seconds (seamless only).
        seamless (bool): Whether to crossfade the tail back to frame 0.
        progress (ProgressFn): Progress callback for (message, fraction).

    Returns:
        Path: The output path the MP4 was written to.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = load_animate_base_frame(image_path, out_w, out_h, progress)
    haze = build_animate_haze(config, heat_haze_cfg, base, out_w, out_h, progress)

    duration = duration if duration is not None else default_video_duration(haze)
    n_frames = max(1, int(round(duration * fps)))

    if seamless:
        crossfade_frames = max(1, int(round(crossfade_s * fps)))
        crossfade_frames = min(crossfade_frames, n_frames // 2)
        record_haze_seamless(haze, fps, n_frames, crossfade_frames, out_path, progress)
    else:
        record_haze(haze, fps, n_frames, out_path, progress)
    return out_path


class AnimateTab(Tab):
    """Animate tab: live heat-haze preview and progress-tracked MP4 record."""

    name = "Animate"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise the tab with no widgets built yet."""
        super().__init__(*args, **kwargs)
        self._manager: t.Optional[pygame_gui.UIManager] = None
        self._view_rect = pygame.Rect(0, 0, 0, 0)
        self._labels: t.List[pygame_gui.elements.UILabel] = []
        self._base_picker: t.Optional[FilePickerRow] = None
        self._output_picker: t.Optional[FilePickerRow] = None
        self._preview_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._seamless_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._record_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._entries: t.Dict[str, pygame_gui.elements.UITextEntryLine] = {}
        self._progress_bar: t.Optional[pygame_gui.elements.UIProgressBar] = None
        self._status: t.Optional[pygame_gui.elements.UILabel] = None

        self._haze: t.Optional[HeatHaze] = None
        self._playing = False
        self._elapsed = 0.0
        self._seamless = False
        self._haze_enabled = False
        self._pending: t.Optional[str] = None  # "preview" | "record" | None
        self._frame_surface: t.Optional[pygame.Surface] = None

    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Create the view area and the right-hand control panel.

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

        y = self._build_base_row(manager, panel_x, panel_w, y)
        y = self._build_preview_button(manager, panel_x, panel_w, y)
        y = self._build_record_section(manager, panel_x, panel_w, y)
        self.refresh_from_state()

    def _build_base_row(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the "Base image" label and its existing-file picker.

        Returns:
            int: The next free y coordinate.
        """
        self._labels.append(
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
                text="Base image",
                manager=manager,
            )
        )
        y += _ROW_HEIGHT + _ROW_GAP
        self._base_picker = FilePickerRow(
            manager,
            None,
            pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            initial_text=_INITIAL_IMAGE,
            dialog_title="Select base image",
        )
        return y + _ROW_HEIGHT + _ROW_GAP

    def _build_preview_button(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the Preview/Stop toggle button.

        Returns:
            int: The next free y coordinate.
        """
        self._preview_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
            text="Preview",
            manager=manager,
        )
        return y + _BUTTON_HEIGHT + _ROW_GAP

    def _build_record_section(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the Record separator, output picker, entries, and action widgets.

        Returns:
            int: The next free y coordinate.
        """
        self._labels.append(
            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
                text="Record",
                manager=manager,
            )
        )
        y += _ROW_HEIGHT + _ROW_GAP

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

        for label, default in (("FPS", "30"), ("Duration s", ""), ("Crossfade s", "0.5")):
            label_widget, entry = build_labeled_entry(manager, None, panel_x, y, label)
            entry.set_text(default)
            self._labels.append(label_widget)
            self._entries[label] = entry
            y += _ROW_HEIGHT + _ROW_GAP

        self._seamless_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
            text="Seamless: OFF",
            manager=manager,
        )
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._record_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
            text="Record",
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
        """Kill every widget, stop playback, and drop the haze (GPU tensors)."""
        self._playing = False
        self._haze = None
        self._frame_surface = None
        for label in self._labels:
            label.kill()
        widgets: t.List[t.Any] = [
            self._preview_btn,
            self._seamless_btn,
            self._record_btn,
            self._progress_bar,
            self._status,
            *self._entries.values(),
        ]
        for widget in widgets:
            if widget is not None:
                widget.kill()
        for picker in (self._base_picker, self._output_picker):
            if picker is not None:
                picker.kill()
        self._labels = []
        self._entries = {}
        self._base_picker = None
        self._output_picker = None
        self._preview_btn = None
        self._seamless_btn = None
        self._record_btn = None
        self._progress_bar = None
        self._status = None
        self._manager = None

    def refresh_from_state(self) -> None:
        """Re-check the heat-haze gate whenever the working config changes."""
        heat_haze = self.state.config.heat_haze
        self._haze_enabled = heat_haze is not None and heat_haze.enabled
        if not self._haze_enabled:
            self._stop_playback()
            self._set_status(_GATE_STATUS)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route picker events and the Preview/Seamless/Record button presses.

        Args:
            event (pygame.event.Event): The event to handle.
        """
        if self._output_picker is not None:
            self._output_picker.handle_event(event)
        if self._base_picker is not None and self._base_picker.handle_event(event):
            # A new base image invalidates any built haze and stops playback.
            self._stop_playback()
            self._haze = None
        if event.type != pygame_gui.UI_BUTTON_PRESSED:
            return
        if event.ui_element is self._preview_btn:
            self._toggle_preview()
        elif event.ui_element is self._seamless_btn:
            self._toggle_seamless()
        elif event.ui_element is self._record_btn:
            self._run_record()

    def handle_job_event(self, event: JobEvent) -> None:
        """Advance progress and dispatch completion to the right handler.

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
        """Advance the live preview frame and refresh button enable states.

        Args:
            dt (float): Seconds since the previous frame.
        """
        if self._playing and self._haze is not None:
            self._elapsed += dt
            frame = self._haze.apply(self._elapsed)
            self._frame_surface = array_to_pygame_surface(frame)
        self._update_buttons()

    def draw(self, window: pygame.Surface) -> None:
        """Blit the current preview frame at the view-area origin.

        Args:
            window (pygame.Surface): The window surface to draw onto.
        """
        if self._frame_surface is not None:
            window.blit(self._frame_surface, (self._view_rect.x, self._view_rect.y))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _toggle_preview(self) -> None:
        """Stop a playing preview, or submit a haze-build job to start one."""
        if self._playing:
            self._stop_playback()
            return
        if self.jobs.busy or not self._haze_enabled or self._base_picker is None:
            return
        image_path = self._base_picker.path
        if image_path is None:
            return
        config = self.state.config
        heat_haze_cfg = config.heat_haze
        view_w = max(1, self._view_rect.width)
        view_h = max(1, self._view_rect.height)
        self._pending = "preview"
        self.jobs.submit(
            lambda progress: build_preview_haze(
                config, heat_haze_cfg, image_path, view_w, view_h, progress
            )
        )
        self._set_status("Building haze...")

    def _toggle_seamless(self) -> None:
        """Flip the seamless flag and update the toggle button text."""
        self._seamless = not self._seamless
        if self._seamless_btn is not None:
            self._seamless_btn.set_text(f"Seamless: {'ON' if self._seamless else 'OFF'}")

    def _run_record(self) -> None:
        """Submit the MP4 record job with values captured at click time."""
        if self.jobs.busy or self._playing or not self._haze_enabled:
            return
        if self._base_picker is None or self._output_picker is None:
            return
        image_path = self._base_picker.path
        out_path = self._output_picker.path
        if image_path is None or out_path is None:
            return
        config = self.state.config
        heat_haze_cfg = config.heat_haze
        preset = config.quality_presets[_RECORD_QUALITY]
        out_w, out_h = int(preset.width), int(preset.height)
        fps = self._parse_fps()
        duration = self._parse_duration()
        crossfade_s = self._parse_crossfade()
        seamless = self._seamless
        self._pending = "record"
        self.jobs.submit(
            lambda progress: run_record(
                config, heat_haze_cfg, image_path, out_path, out_w, out_h,
                fps, duration, crossfade_s, seamless, progress,
            )
        )
        self._set_status("Recording...")

    def _stop_playback(self) -> None:
        """Freeze the preview on its last frame and reset the button text."""
        self._playing = False
        if self._preview_btn is not None:
            self._preview_btn.set_text("Preview")

    # ------------------------------------------------------------------
    # Job-event handlers (main thread)
    # ------------------------------------------------------------------

    def _on_progress(self, message: str, fraction: float) -> None:
        """Update the status label and, for determinate stages, the bar.

        Args:
            message (str): Progress message from the worker.
            fraction (float): Progress fraction 0..1, or -1 for indeterminate.
        """
        self._set_status(message)
        if self._progress_bar is not None and fraction >= 0:
            self._progress_bar.set_current_progress(fraction * 100.0)

    def _on_done(self, result: t.Any) -> None:
        """Route a completed job to the preview or record handler.

        Args:
            result (Any): The finished job's return value.
        """
        pending = self._pending
        self._pending = None
        if pending == "preview":
            self._on_preview_ready(result)
        elif pending == "record":
            self._on_record_done(result)

    def _on_preview_ready(self, haze: HeatHaze) -> None:
        """Store the built haze and start real-time playback.

        Args:
            haze (HeatHaze): The haze built on the worker thread.
        """
        self._haze = haze
        self._playing = True
        self._elapsed = 0.0
        if self._preview_btn is not None:
            self._preview_btn.set_text("Stop")
        self._set_status("Playing preview")

    def _on_record_done(self, out_path: Path) -> None:
        """Fill the progress bar and report the saved MP4 path.

        Args:
            out_path (Path): Path the MP4 was written to.
        """
        if self._progress_bar is not None:
            self._progress_bar.set_current_progress(100.0)
        self._set_status(f"Saved {out_path}")

    def _on_error(self, message: str) -> None:
        """Report a job failure and reset the progress bar.

        Args:
            message (str): The error message from the worker.
        """
        self._pending = None
        if self._progress_bar is not None:
            self._progress_bar.set_current_progress(0.0)
        self._set_status(f"Animate failed: {message}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_buttons(self) -> None:
        """Gate Preview/Record buttons on the haze config, busy, and playback."""
        if self._preview_btn is not None:
            if not self._haze_enabled or self.jobs.busy:
                self._preview_btn.disable()
            else:
                self._preview_btn.enable()
        if self._record_btn is not None:
            if not self._haze_enabled or self.jobs.busy or self._playing:
                self._record_btn.disable()
            else:
                self._record_btn.enable()

    def _parse_fps(self) -> int:
        """Parse the FPS entry as an int >= 1, falling back to 30."""
        try:
            return max(1, int(float(self._entries["FPS"].get_text().strip())))
        except (ValueError, KeyError):
            return _DEFAULT_FPS

    def _parse_duration(self) -> t.Optional[float]:
        """Parse the Duration entry as a float, or None when blank/invalid."""
        text = self._entries["Duration s"].get_text().strip() if "Duration s" in self._entries else ""
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_crossfade(self) -> float:
        """Parse the Crossfade entry as a float, falling back to 0.5."""
        try:
            return float(self._entries["Crossfade s"].get_text().strip())
        except (ValueError, KeyError):
            return _DEFAULT_CROSSFADE_S

    def _set_status(self, text: str) -> None:
        """Set the status label text if the label exists.

        Args:
            text (str): Status message to display.
        """
        if self._status is not None:
            self._status.set_text(text)
