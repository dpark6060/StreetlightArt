"""Preview tab: flat-map warp preview with increment-based camera controls.

Loads the scene + flat map on the background job runner, then renders the
warp preview synchronously on the main thread every time the camera moves.
Camera control is explicit: move-by-increment and rotate-by-increment buttons,
a live lat/lon/alt readout, and a Render button that applies typed coordinates
and persists them to the config file. Keyboard/mouse flight (WASD, arrows,
right-drag) still works while the cursor is over the view area.
"""

import typing as t
from pathlib import Path

import numpy as np
import pygame
import pygame_gui

from ...camera.controls import MouseDragState, handle_input, rotate_clamped
from ...camera.coordinates import (
    enu_to_wgs84,
    feet_to_meters,
    meters_to_feet,
    wgs84_to_enu,
)
from ...config.camera_io import (
    build_camera_toml_dict,
    save_camera_position,
    write_camera_to_toml,
)
from ...devmode.config_writer import write_config
from ...rendering.flat_map import get_flat_map
from ...rendering.pillow_backend import PillowBackend
from ...rendering.raycast import build_ground_gbuffer
from ...rendering.warp_preview import render_warp_preview
from ...scene_loader import load_scene
from ..imaging import pil_to_pygame_surface
from ..jobs import JobEvent, ProgressFn
from ..widgets.labeled_entry import build_labeled_entry, parse_float_entry
from .base import Tab

_PANEL_WIDTH = 360
_PANEL_MARGIN = 8
_ROW_HEIGHT = 28
_BUTTON_HEIGHT = 30
_ROW_GAP = 6
_BTN_W = 70
_DEFAULT_MOVE_STEP = 100.0
_DEFAULT_ROTATE_STEP = 5.0

# Coordinate entry key -> (label text, format spec). Order is display order.
_COORD_FIELDS: t.Tuple[t.Tuple[str, str, str], ...] = (
    ("cam_lat", "Cam lat", ".6f"),
    ("cam_lon", "Cam lon", ".6f"),
    ("cam_alt", "Cam alt (ft)", ".1f"),
    ("tgt_lat", "Tgt lat", ".6f"),
    ("tgt_lon", "Tgt lon", ".6f"),
    ("tgt_alt", "Tgt alt (ft)", ".1f"),
)

# ENU unit deltas per movement direction (x=east, y=north, z=up).
_MOVE_AXES: t.Dict[str, np.ndarray] = {
    "N": np.array([0.0, 1.0, 0.0]),
    "S": np.array([0.0, -1.0, 0.0]),
    "E": np.array([1.0, 0.0, 0.0]),
    "W": np.array([-1.0, 0.0, 0.0]),
    "Up": np.array([0.0, 0.0, 1.0]),
    "Down": np.array([0.0, 0.0, -1.0]),
}


def move_delta(direction: str, step: float) -> np.ndarray:
    """Return the ENU translation vector for one movement-button press.

    Args:
        direction (str): One of N/S/E/W/Up/Down.
        step (float): Move distance in meters.

    Returns:
        np.ndarray: (3,) ENU delta (x=east, y=north, z=up) scaled by step.
    """
    return _MOVE_AXES[direction] * step


class PreviewTab(Tab):
    """Preview tab: warp preview plus increment-based camera controls."""

    name = "Preview"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise the tab with no widgets built and no scene loaded yet."""
        super().__init__(*args, **kwargs)
        self._manager: t.Optional[pygame_gui.UIManager] = None
        self._view_rect = pygame.Rect(0, 0, 0, 0)

        self._labels: t.List[pygame_gui.elements.UILabel] = []
        self._buttons: t.Dict[pygame_gui.elements.UIButton, t.Callable[[], None]] = {}
        self._move_step_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None
        self._rotate_step_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None
        self._coord_entries: t.Dict[str, pygame_gui.elements.UITextEntryLine] = {}
        self._status: t.Optional[pygame_gui.elements.UILabel] = None
        self._save_dialog: t.Optional[pygame_gui.windows.UIFileDialog] = None

        # Render state (all set once the load job completes).
        self._flat_map: t.Any = None
        self._flat_meta: t.Any = None
        self._flat_map_array: t.Optional[np.ndarray] = None
        self._original_position: t.Optional[np.ndarray] = None
        self._original_target: t.Optional[np.ndarray] = None
        self._original_fov: t.Optional[float] = None
        self._frame_surface: t.Optional[pygame.Surface] = None
        self._loaded = False

        # Per-frame input bookkeeping.
        self._frame_events: t.List[pygame.event.Event] = []
        self._drag_state = MouseDragState()
        self._controls_state: t.Optional[t.Tuple[bool, bool]] = None

    # ------------------------------------------------------------------
    # Build / destroy
    # ------------------------------------------------------------------

    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Create the view area, control panel, and bootstrap the scene load.

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

        y = self._build_move_controls(manager, panel_x, panel_w, y)
        y = self._build_rotate_controls(manager, panel_x, panel_w, y)
        y = self._build_coord_controls(manager, panel_x, panel_w, y)
        self._build_status(manager, panel_x, panel_w, y)

        self._set_controls_enabled(False)
        self._bootstrap_scene()

    def destroy(self) -> None:
        """Kill every widget (including all labels) and drop the flat-map arrays."""
        widgets: t.List[t.Any] = [
            *self._labels,
            *self._buttons.keys(),
            self._move_step_entry,
            self._rotate_step_entry,
            *self._coord_entries.values(),
            self._status,
            self._save_dialog,
        ]
        for widget in widgets:
            if widget is not None:
                widget.kill()
        self._labels = []
        self._buttons = {}
        self._move_step_entry = None
        self._rotate_step_entry = None
        self._coord_entries = {}
        self._status = None
        self._save_dialog = None
        self._manager = None

        self._flat_map = None
        self._flat_meta = None
        self._flat_map_array = None
        self._frame_surface = None
        self._loaded = False
        self._controls_state = None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_move_controls(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the move-step entry and the N/W/E/S/Up/Down button cross.

        Returns:
            int: The next free y coordinate below this group.
        """
        label, self._move_step_entry = build_labeled_entry(
            manager, None, panel_x, y, "Move step (m)"
        )
        self._labels.append(label)
        self._move_step_entry.set_text("100")
        y += _ROW_HEIGHT + _ROW_GAP

        center_x = panel_x + (panel_w - _BTN_W) // 2
        left_x = center_x - _BTN_W - _ROW_GAP
        right_x = center_x + _BTN_W + _ROW_GAP

        self._add_button(manager, "N", pygame.Rect(center_x, y, _BTN_W, _BUTTON_HEIGHT),
                         self._make_move_handler("N"))
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._add_button(manager, "W", pygame.Rect(left_x, y, _BTN_W, _BUTTON_HEIGHT),
                         self._make_move_handler("W"))
        self._add_button(manager, "E", pygame.Rect(right_x, y, _BTN_W, _BUTTON_HEIGHT),
                         self._make_move_handler("E"))
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._add_button(manager, "S", pygame.Rect(center_x, y, _BTN_W, _BUTTON_HEIGHT),
                         self._make_move_handler("S"))
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._add_button(manager, "Up", pygame.Rect(left_x, y, _BTN_W, _BUTTON_HEIGHT),
                         self._make_move_handler("Up"))
        self._add_button(manager, "Down", pygame.Rect(right_x, y, _BTN_W, _BUTTON_HEIGHT),
                         self._make_move_handler("Down"))
        y += _BUTTON_HEIGHT + _ROW_GAP
        return y

    def _build_rotate_controls(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the rotate-step entry, yaw/pitch buttons, and reset button.

        Returns:
            int: The next free y coordinate below this group.
        """
        label, self._rotate_step_entry = build_labeled_entry(
            manager, None, panel_x, y, "Rotate step (deg)"
        )
        self._labels.append(label)
        self._rotate_step_entry.set_text("5")
        y += _ROW_HEIGHT + _ROW_GAP

        half = (panel_w - _ROW_GAP) // 2
        self._add_button(manager, "CCW", pygame.Rect(panel_x, y, half, _BUTTON_HEIGHT),
                         self._make_yaw_handler(1.0))
        self._add_button(manager, "CW", pygame.Rect(panel_x + half + _ROW_GAP, y, half, _BUTTON_HEIGHT),
                         self._make_yaw_handler(-1.0))
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._add_button(manager, "Pitch+", pygame.Rect(panel_x, y, half, _BUTTON_HEIGHT),
                         self._make_pitch_handler(1.0))
        self._add_button(manager, "Pitch-", pygame.Rect(panel_x + half + _ROW_GAP, y, half, _BUTTON_HEIGHT),
                         self._make_pitch_handler(-1.0))
        y += _BUTTON_HEIGHT + _ROW_GAP
        self._add_button(manager, "Reset rotation", pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
                         self._on_reset_rotation)
        y += _BUTTON_HEIGHT + _ROW_GAP
        return y

    def _build_coord_controls(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> int:
        """Build the six coordinate entries and the full-width Render button.

        Returns:
            int: The next free y coordinate below this group.
        """
        for key, text, _fmt in _COORD_FIELDS:
            label, entry = build_labeled_entry(manager, None, panel_x, y, text)
            self._labels.append(label)
            self._coord_entries[key] = entry
            y += _ROW_HEIGHT + _ROW_GAP

        self._add_button(manager, "Render", pygame.Rect(panel_x, y, panel_w, _BUTTON_HEIGHT),
                         self._on_render)
        y += _BUTTON_HEIGHT + _ROW_GAP
        return y

    def _build_status(
        self, manager: pygame_gui.UIManager, panel_x: int, panel_w: int, y: int
    ) -> None:
        """Build the status label."""
        self._status = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(panel_x, y, panel_w, _ROW_HEIGHT),
            text="Loading scene...",
            manager=manager,
        )

    def _add_button(
        self,
        manager: pygame_gui.UIManager,
        text: str,
        rect: pygame.Rect,
        handler: t.Callable[[], None],
    ) -> None:
        """Create a UIButton and register its click handler."""
        button = pygame_gui.elements.UIButton(
            relative_rect=rect, text=text, manager=manager
        )
        self._buttons[button] = handler

    # ------------------------------------------------------------------
    # Scene / flat-map bootstrap (background job)
    # ------------------------------------------------------------------

    def _bootstrap_scene(self) -> None:
        """Submit the scene + flat-map load job, reusing a clean cached scene."""
        config = self.state.config
        cache_dir = self.state.cache_dir
        existing = self.state.scene if not self.state.scene_dirty else None
        self.jobs.submit(lambda progress: _load_preview_data(config, cache_dir, existing, progress))
        self._set_status("Loading scene...")

    def handle_job_event(self, event: JobEvent) -> None:
        """Show progress, then adopt the loaded scene/flat map on completion.

        Args:
            event (JobEvent): The job progress/completion event.
        """
        if event.kind == "progress":
            self._set_status(event.message)
        elif event.kind == "done":
            self._adopt_loaded_data(event.result)
        elif event.kind == "error":
            self._set_status(f"Load failed: {event.message}")

    def _adopt_loaded_data(self, result: t.Tuple[t.Any, t.Any, t.Any]) -> None:
        """Store the scene + flat map, snapshot the camera, render the first frame.

        Args:
            result (tuple): (scene, flat_map, flat_meta) from the load job.
        """
        scene, flat_map, flat_meta = result
        self.state.scene = scene
        self.state.scene_quality = "low"
        self.state.scene_dirty = False
        self._flat_map = flat_map
        self._flat_meta = flat_meta
        self._flat_map_array = np.asarray(flat_map)

        camera = scene.camera
        self._original_position = camera.position.copy()
        self._original_target = camera.target.copy()
        self._original_fov = camera.fov_deg

        self._loaded = True
        self._render_frame()
        self._refresh_coord_entries()
        self._set_status("Ready")

    # ------------------------------------------------------------------
    # Per-frame update: input passthrough + debounced render
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        """Collect events for camera passthrough and route button/dialog events.

        Args:
            event (pygame.event.Event): The event to handle.
        """
        self._frame_events.append(event)
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            handler = self._buttons.get(event.ui_element)
            if handler is not None:
                handler()
            return
        if (
            event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED
            and event.ui_element is self._save_dialog
        ):
            self._save_to_path(Path(event.text))
            self._save_dialog = None

    def update(self, dt: float) -> None:
        """Gate controls and run keyboard/mouse passthrough with a live re-render.

        Args:
            dt (float): Seconds since the previous frame.
        """
        events = self._frame_events
        self._frame_events = []

        self._sync_controls_enabled()
        if not self._scene_ready():
            return
        self._process_camera_input(events, dt)

    def _process_camera_input(self, events: t.List[pygame.event.Event], dt: float) -> None:
        """Feed events to controls.handle_input only when the cursor is over the view.

        Renders every frame the camera actually moves, matching ``cmd_preview``'s
        "warp every frame" feel — the warp is cheap enough to be interactive at
        view size, so held-key/mouse flight stays live instead of freezing until
        movement settles.

        Args:
            events (list): This frame's collected pygame events.
            dt (float): Seconds since the previous frame.
        """
        camera = self.state.scene.camera
        if not self._view_rect.collidepoint(pygame.mouse.get_pos()):
            self._drag_state.clear()
            return

        pos_before = camera.position.copy()
        tgt_before = camera.target.copy()
        fov_before = camera.fov_deg
        held = pygame.key.get_pressed()
        result = handle_input(
            events, camera, dt, held, self._drag_state,
            self._original_position, self._original_target, self._original_fov,
        )
        if result.reset:
            camera.target = self._original_target.copy()

        moved = (
            not np.array_equal(camera.position, pos_before)
            or not np.array_equal(camera.target, tgt_before)
            or camera.fov_deg != fov_before
        )
        if moved:
            self._render_frame()
            self._refresh_coord_entries()

    # ------------------------------------------------------------------
    # Camera mutation handlers (buttons — immediate render)
    # ------------------------------------------------------------------

    def _make_move_handler(self, direction: str) -> t.Callable[[], None]:
        """Return a click handler that pans the camera in one ENU direction."""
        def handler() -> None:
            step = parse_float_entry(self._move_step_entry, _DEFAULT_MOVE_STEP)
            self.state.scene.camera.move(move_delta(direction, step))
            self._render_frame()
            self._refresh_coord_entries()
        return handler

    def _make_yaw_handler(self, sign: float) -> t.Callable[[], None]:
        """Return a click handler that yaws the camera (positive sign = CCW/left)."""
        def handler() -> None:
            step = parse_float_entry(self._rotate_step_entry, _DEFAULT_ROTATE_STEP)
            self.state.scene.camera.rotate(sign * step, 0.0)
            self._render_frame()
            self._refresh_coord_entries()
        return handler

    def _make_pitch_handler(self, sign: float) -> t.Callable[[], None]:
        """Return a click handler that pitches the camera with the ±85° clamp."""
        def handler() -> None:
            step = parse_float_entry(self._rotate_step_entry, _DEFAULT_ROTATE_STEP)
            rotate_clamped(self.state.scene.camera, 0.0, sign * step)
            self._render_frame()
            self._refresh_coord_entries()
        return handler

    def _on_reset_rotation(self) -> None:
        """Re-aim the camera at its original target without moving the position."""
        if self._original_target is None:
            return
        self.state.scene.camera.target = self._original_target.copy()
        self._render_frame()
        self._refresh_coord_entries()

    # ------------------------------------------------------------------
    # Render button (apply typed coordinates + persist)
    # ------------------------------------------------------------------

    def _on_render(self) -> None:
        """Apply the typed coordinates to the camera, re-render, and persist."""
        coords = self._parse_coord_entries()
        if coords is None:
            self._set_status("Invalid coordinate")
            return
        camera = self.state.scene.camera
        transformer = self.state.scene.transformer
        cam_e, cam_n, cam_u = wgs84_to_enu(
            transformer, coords["cam_lat"], coords["cam_lon"], feet_to_meters(coords["cam_alt"])
        )
        tgt_e, tgt_n, tgt_u = wgs84_to_enu(
            transformer, coords["tgt_lat"], coords["tgt_lon"], feet_to_meters(coords["tgt_alt"])
        )
        camera.position = np.array([cam_e, cam_n, cam_u])
        camera.target = np.array([tgt_e, tgt_n, tgt_u])

        self._render_frame()
        self._refresh_coord_entries()
        self._persist_camera()

    def _parse_coord_entries(self) -> t.Optional[t.Dict[str, float]]:
        """Parse all six coordinate entries as floats.

        Returns:
            dict[str, float] | None: Parsed values keyed by field, or None if
                any entry is not a valid float.
        """
        values: t.Dict[str, float] = {}
        for key, _text, _fmt in _COORD_FIELDS:
            try:
                values[key] = float(self._coord_entries[key].get_text().strip())
            except ValueError:
                return None
        return values

    def _persist_camera(self) -> None:
        """Save the camera to the config file, prompting for a path if needed."""
        if self.state.config_path is not None:
            save_camera_position(self.state.scene, self.state.config_path)
            self._sync_config_camera()
            self._set_status(f"Camera saved to {self.state.config_path}")
            return
        self._open_save_dialog()

    def _sync_config_camera(self) -> None:
        """Copy the current camera pose into the working config's [camera] section."""
        data = build_camera_toml_dict(self.state.scene)
        cam = self.state.config.camera
        cam.lat = data["lat"]
        cam.lon = data["lon"]
        cam.altitude_ft = data["altitude_ft"]
        cam.target_lat = data["target_lat"]
        cam.target_lon = data["target_lon"]
        cam.target_altitude_ft = data["target_altitude_ft"]
        cam.fov = data["fov"]

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
        """Write the camera (or a full config for a new file) to the chosen path.

        Args:
            path (Path): Destination config file chosen in the save dialog.
        """
        if path.exists():
            write_camera_to_toml(path, build_camera_toml_dict(self.state.scene))
        else:
            write_config(self.state.config, path)
        self.state.config_path = path
        self._sync_config_camera()
        self._set_status(f"Camera saved to {path}")

    # ------------------------------------------------------------------
    # Frame rendering + coordinate readout
    # ------------------------------------------------------------------

    def _render_frame(self) -> None:
        """Render one warp-preview frame synchronously and store it for draw()."""
        if not self._scene_ready():
            return
        camera = self.state.scene.camera
        width = max(1, self._view_rect.width)
        height = max(1, self._view_rect.height)
        gbuffer = build_ground_gbuffer(camera, width, height)
        backend = PillowBackend(width, height)
        render_warp_preview(
            self.state.scene, camera, backend, self._flat_map, self._flat_meta,
            flat_map_array=self._flat_map_array, gbuffer=gbuffer,
        )
        self._frame_surface = pil_to_pygame_surface(backend.get_image())

    def _refresh_coord_entries(self) -> None:
        """Update the coordinate entries from the camera, skipping focused ones."""
        if not self._scene_ready():
            return
        camera = self.state.scene.camera
        transformer = self.state.scene.transformer
        cam_lat, cam_lon, cam_alt_m = enu_to_wgs84(
            transformer, camera.position[0], camera.position[1], camera.position[2]
        )
        tgt_lat, tgt_lon, tgt_alt_m = enu_to_wgs84(
            transformer, camera.target[0], camera.target[1], camera.target[2]
        )
        values = {
            "cam_lat": cam_lat,
            "cam_lon": cam_lon,
            "cam_alt": meters_to_feet(cam_alt_m),
            "tgt_lat": tgt_lat,
            "tgt_lon": tgt_lon,
            "tgt_alt": meters_to_feet(tgt_alt_m),
        }
        for key, _text, fmt in _COORD_FIELDS:
            entry = self._coord_entries[key]
            if entry.is_focused:
                continue
            entry.set_text(format(values[key], fmt))

    def draw(self, window: pygame.Surface) -> None:
        """Blit the stored warp frame at the view-area origin.

        Args:
            window (pygame.Surface): The window surface to draw onto.
        """
        if self._frame_surface is not None:
            window.blit(self._frame_surface, (self._view_rect.x, self._view_rect.y))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scene_ready(self) -> bool:
        """Return True once the scene, flat map, and camera snapshot are loaded."""
        return (
            self._loaded
            and self.state.scene is not None
            and self._flat_map is not None
            and self._original_target is not None
        )

    def _sync_controls_enabled(self) -> None:
        """Apply the enabled/disabled state only when (loaded, busy) actually changed.

        Guards against re-running ``.enable()``/``.disable()`` on ~14 widgets every
        frame, which would force needless pygame_gui surface rebuilds.
        """
        state = (self._loaded, self.jobs.busy)
        if state == self._controls_state:
            return
        self._controls_state = state
        self._set_controls_enabled(self._loaded and not self.jobs.busy)

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable every button and entry in the control panel."""
        widgets: t.List[t.Any] = [
            *self._buttons.keys(),
            self._move_step_entry,
            self._rotate_step_entry,
            *self._coord_entries.values(),
        ]
        for widget in widgets:
            if widget is None:
                continue
            if enabled:
                widget.enable()
            else:
                widget.disable()

    def _set_status(self, text: str) -> None:
        """Set the status label text if the label exists.

        Args:
            text (str): Status message to display.
        """
        if self._status is not None:
            self._status.set_text(text)


def _load_preview_data(
    config: t.Any,
    cache_dir: Path,
    existing_scene: t.Any,
    progress: ProgressFn,
) -> t.Tuple[t.Any, t.Any, t.Any]:
    """Load (or reuse) the scene and generate its flat map on the worker thread.

    Runs entirely off the main thread and touches no pygame state. Reuses a
    clean cached scene when one is supplied; otherwise builds a low-quality
    scene from OSM data. The flat map is disk-cached, so it is cheap after the
    first run but a cache miss is slow — hence the background job.

    Args:
        config (SceneConfig): Working config driving the scene build.
        cache_dir (Path): OSM + flat-map cache directory.
        existing_scene (Scene | None): A clean cached scene to reuse, or None.
        progress (ProgressFn): Callback for indeterminate progress messages.

    Returns:
        tuple: (scene, flat_map, flat_meta) as plain data for the main thread.
    """
    if existing_scene is not None:
        scene = existing_scene
    else:
        scene = load_scene(config, "low", cache_dir, progress=lambda msg: progress(msg, -1.0))
    progress("Generating flat map...", -1.0)
    flat_map, flat_meta = get_flat_map(scene, cache_dir)
    return scene, flat_map, flat_meta
