"""Devmode tab: reflection-driven light-property editor over a live test scene.

Left: the devmode render (synthetic 10-row scene, or a clipped real OSM strip).
Right: a section list (Atmosphere, Devmode Camera, then every light kind the
catalog enumerates) over a scrolling property panel, plus Rerender / Save /
Real-Tiles buttons.

Edits apply immediately into the working config: each slider move / entry finish
/ toggle writes through ``config_bridge`` to the active section's root object.
For light kinds the root is ``LightKind.target`` — a reference into the live
config — so switching kinds and returning shows the edited value (read back from
config, never from stale widget state). Two registries keep the roots straight:
``static_registry`` (Atmosphere/Camera, root = full config) and
``light_registry`` (dynamic section, root = the current kind's target).

The scene rebuild + render run on the main thread (fast at devmode resolution),
matching the CLI devmode tool. Only the first real-tiles strip load — which can
hit OSM — runs on the background job runner.
"""

import typing as t
from pathlib import Path

import numpy as np
import pygame
import pygame_gui

from ...camera.controls import MouseDragState, handle_input
from ...devmode.config_bridge import apply_widgets_to_config, read_config_to_widgets
from ...devmode.config_writer import write_config
from ...devmode.dynamic_panel import build_dynamic_section
from ...devmode.light_catalog import LightKind, build_light_catalog
from ...devmode.panel_sections import (
    build_atmosphere_section,
    build_devmode_camera_section,
)
from ...devmode.real_scene import build_real_scene_strip
from ...devmode.render_helpers import (
    ensure_devmode_config,
    rebuild_real_lights,
    render_cheap,
    render_devmode_scene,
    should_render_full,
    sync_camera_from_config,
)
from ...devmode.minimap import Minimap, MinimapLayout
from ...devmode.slider_registry import SliderRegistry
from ...devmode.synthetic_scene import build_synthetic_scene
from ...devmode.widget_sync import (
    apply_entry_to_slider,
    toggle_bool_button,
    update_value_entry,
)
from ..imaging import fit_image_to_viewport, pil_to_pygame_surface
from ..jobs import JobEvent, ProgressFn
from ..widgets.labeled_entry import build_labeled_entry, parse_float_entry
from .base import Tab

_PANEL_WIDTH = 384
_HEIGHT_AREA_HEIGHT = 68
_SECTION_LIST_HEIGHT = 140
_BUTTON_AREA_HEIGHT = 44
_BUTTON_HEIGHT = 35
_STATIC_SECTIONS: t.Tuple[str, ...] = ("Atmosphere", "Devmode Camera")

# Minimap overlay: bottom-left of the view area, fixed size + margin.
_MINIMAP_WIDTH = 220
_MINIMAP_HEIGHT = 180
_MINIMAP_MARGIN = 12
_EMPTY_EXTENT = ((-500.0, -500.0), (500.0, 500.0))


class DevmodeTab(Tab):
    """Devmode tab: dynamic light-property panel over a live devmode scene."""

    name = "Devmode"

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise the tab with no widgets built and no scene loaded yet."""
        super().__init__(*args, **kwargs)
        self._manager: t.Optional[pygame_gui.UIManager] = None
        self._view_rect = pygame.Rect(0, 0, 0, 0)
        self._panel_rect = pygame.Rect(0, 0, 0, 0)

        # Panel widgets.
        self._section_list: t.Optional[pygame_gui.elements.UISelectionList] = None
        self._scroll: t.Optional[pygame_gui.elements.UIScrollingContainer] = None
        self._rerender_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._save_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._realtiles_btn: t.Optional[pygame_gui.elements.UIButton] = None
        self._save_dialog: t.Optional[pygame_gui.windows.UIFileDialog] = None

        # Height entries (camera / focal Z) above the section list.
        self._cam_height_label: t.Optional[pygame_gui.elements.UILabel] = None
        self._cam_height_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None
        self._focal_height_label: t.Optional[pygame_gui.elements.UILabel] = None
        self._focal_height_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None

        # Minimap overlay + active drag target ("camera" | "target" | None).
        self._minimap: t.Optional[Minimap] = None
        self._drag_dot: t.Optional[str] = None

        # Section state.
        self._static_registry = SliderRegistry()
        self._light_registry = SliderRegistry()
        self._catalog: t.List[LightKind] = []
        self._active_section = "Atmosphere"
        self._active_kind: t.Optional[LightKind] = None

        # Scene + camera state.
        self._scene: t.Any = None
        self._use_real_tiles = False
        self._original_position: t.Optional[np.ndarray] = None
        self._original_target: t.Optional[np.ndarray] = None
        self._original_fov: t.Optional[float] = None
        self._frame_surface: t.Optional[pygame.Surface] = None

        # Per-frame input + debounce bookkeeping.
        self._frame_events: t.List[pygame.event.Event] = []
        self._drag_state = MouseDragState()
        self._elapsed = 0.0
        self._last_move_time: t.Optional[float] = None

    # ------------------------------------------------------------------
    # Build / destroy
    # ------------------------------------------------------------------

    def build(self, manager: pygame_gui.UIManager, content_rect: pygame.Rect) -> None:
        """Build the panel, the synthetic scene, and the first rendered frame.

        Args:
            manager (pygame_gui.UIManager): The UI manager.
            content_rect (pygame.Rect): Content area for the tab.
        """
        self._manager = manager
        self._view_rect = pygame.Rect(
            content_rect.x, content_rect.y,
            content_rect.width - _PANEL_WIDTH, content_rect.height,
        )
        self._panel_rect = pygame.Rect(
            content_rect.x + content_rect.width - _PANEL_WIDTH, content_rect.y,
            _PANEL_WIDTH, content_rect.height,
        )

        ensure_devmode_config(self.state.config)
        self._catalog = build_light_catalog(self.state.config)
        self._build_panel(manager)

        self._scene = build_synthetic_scene(self.state.config)
        self._snapshot_camera()
        self._rebuild_minimap()
        self._render_full()
        self._build_section(self._active_section)

    def _build_panel(self, manager: pygame_gui.UIManager) -> None:
        """Create the height entries, section list, scroll container, and buttons."""
        self._build_height_entries(manager)
        list_y = self._panel_rect.y + _HEIGHT_AREA_HEIGHT
        self._section_list = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect(
                self._panel_rect.x, list_y,
                self._panel_rect.width, _SECTION_LIST_HEIGHT,
            ),
            item_list=self._section_names(),
            manager=manager,
        )
        scroll_y = list_y + _SECTION_LIST_HEIGHT
        scroll_h = (
            self._panel_rect.height
            - _HEIGHT_AREA_HEIGHT - _SECTION_LIST_HEIGHT - _BUTTON_AREA_HEIGHT
        )
        self._scroll = pygame_gui.elements.UIScrollingContainer(
            relative_rect=pygame.Rect(
                self._panel_rect.x, scroll_y, self._panel_rect.width, scroll_h,
            ),
            manager=manager,
        )
        self._build_buttons(manager)

    def _build_height_entries(self, manager: pygame_gui.UIManager) -> None:
        """Create the Cam height / Focal height labeled entries at the panel top."""
        x = self._panel_rect.x
        self._cam_height_label, self._cam_height_entry = build_labeled_entry(
            manager, None, x, self._panel_rect.y, "Cam height", label_width=110,
        )
        self._focal_height_label, self._focal_height_entry = build_labeled_entry(
            manager, None, x, self._panel_rect.y + 32, "Focal height", label_width=110,
        )
        self._sync_height_entries_from_config()

    def _sync_height_entries_from_config(self) -> None:
        """Set the height entry text from config.devmode.camera pos_z / target_z."""
        if self._cam_height_entry is None or self.state.config.devmode is None:
            return
        cam = self.state.config.devmode.camera
        self._cam_height_entry.set_text(f"{cam.pos_z:.1f}")
        self._focal_height_entry.set_text(f"{cam.target_z:.1f}")

    def _build_buttons(self, manager: pygame_gui.UIManager) -> None:
        """Create the Rerender, Save, and Real-Tiles buttons in the bottom row."""
        btn_y = self._panel_rect.y + self._panel_rect.height - _BUTTON_AREA_HEIGHT + 4
        btn_w = self._panel_rect.width // 3
        self._rerender_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(self._panel_rect.x, btn_y, btn_w, _BUTTON_HEIGHT),
            text="Rerender", manager=manager,
        )
        self._save_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(self._panel_rect.x + btn_w, btn_y, btn_w, _BUTTON_HEIGHT),
            text="Save", manager=manager,
        )
        self._realtiles_btn = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(self._panel_rect.x + btn_w * 2, btn_y, btn_w, _BUTTON_HEIGHT),
            text="Real Tiles: OFF", manager=manager,
        )

    def destroy(self) -> None:
        """Kill every widget from both registries, the panel, and any dialog."""
        self._reset_registry(self._static_registry)
        self._reset_registry(self._light_registry)
        widgets: t.List[t.Any] = [
            self._section_list, self._scroll,
            self._rerender_btn, self._save_btn, self._realtiles_btn,
            self._save_dialog,
            self._cam_height_label, self._cam_height_entry,
            self._focal_height_label, self._focal_height_entry,
        ]
        for widget in widgets:
            if widget is not None:
                widget.kill()
        self._section_list = None
        self._scroll = None
        self._rerender_btn = None
        self._save_btn = None
        self._realtiles_btn = None
        self._save_dialog = None
        self._cam_height_label = None
        self._cam_height_entry = None
        self._focal_height_label = None
        self._focal_height_entry = None
        self._minimap = None
        self._drag_dot = None
        self._manager = None
        self._scene = None
        self._frame_surface = None
        self._active_kind = None

    # ------------------------------------------------------------------
    # Section management
    # ------------------------------------------------------------------

    def _section_names(self) -> t.List[str]:
        """Return the section list items: the two static sections then all kinds."""
        return list(_STATIC_SECTIONS) + [kind.label for kind in self._catalog]

    def _build_section(self, name: str) -> None:
        """Kill the current section widgets and build the named one, populating values.

        Args:
            name (str): Section name — a static section or a LightKind label.
        """
        self._reset_registry(self._static_registry)
        self._reset_registry(self._light_registry)
        self._active_section = name
        container = self._scroll.scrollable_container

        if name == "Atmosphere":
            self._active_kind = None
            height = build_atmosphere_section(self._manager, container, 0, self._static_registry, self.state.config)
            read_config_to_widgets(self.state.config, self._static_registry)
        elif name == "Devmode Camera":
            self._active_kind = None
            height = build_devmode_camera_section(self._manager, container, 0, self._static_registry, self.state.config)
            read_config_to_widgets(self.state.config, self._static_registry)
        else:
            kind = self._kind_by_label(name)
            self._active_kind = kind
            height = build_dynamic_section(self._manager, container, 0, self._light_registry, kind)
            read_config_to_widgets(kind.target, self._light_registry)

        content_width = self._scroll.get_container().get_size()[0]
        self._scroll.set_scrollable_area_dimensions((content_width, max(height, 10)))

    def _kind_by_label(self, label: str) -> LightKind:
        """Return the catalog kind matching a section label.

        Args:
            label (str): A LightKind.label.

        Returns:
            LightKind: The matching kind.
        """
        return next(kind for kind in self._catalog if kind.label == label)

    def _reset_registry(self, registry: SliderRegistry) -> None:
        """Kill every tracked widget and clear all registry mappings.

        Args:
            registry (SliderRegistry): Registry to empty.
        """
        for widget in registry.all_widgets():
            widget.kill()
        registry.sliders.clear()
        registry.value_entries.clear()
        registry.entry_to_slider.clear()
        registry.bool_buttons.clear()
        registry.text_entries.clear()
        registry.color_pools.clear()
        registry.extra_widgets.clear()
        registry.dirty.clear()

    def _active_registry(self) -> SliderRegistry:
        """Return the registry backing the currently displayed section."""
        return self._light_registry if self._active_kind is not None else self._static_registry

    def _active_root(self) -> t.Any:
        """Return the root object edits for the active section write into."""
        return self._active_kind.target if self._active_kind is not None else self.state.config

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        """Collect events for camera passthrough and handle panel interactions.

        Args:
            event (pygame.event.Event): The event to handle.
        """
        self._frame_events.append(event)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            self._handle_button(event.ui_element)
        elif event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
            self._handle_section_selection(event.ui_element)
        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            self._handle_slider_moved(event.ui_element)
        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            self._handle_entry_finished(event.ui_element)
        elif event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED and event.ui_element is self._save_dialog:
            self._save_to_path(Path(event.text))
            self._save_dialog = None

    def _handle_button(self, element: t.Any) -> None:
        """Route a button press to Rerender / Save / Real-Tiles / a bool toggle."""
        if element is self._rerender_btn:
            self._on_rerender()
        elif element is self._save_btn:
            self._on_save()
        elif element is self._realtiles_btn:
            self._on_toggle_real_tiles()
        elif element in self._active_registry().bool_buttons:
            toggle_bool_button(element, self._active_registry())
            self._apply_active_section()

    def _handle_section_selection(self, element: t.Any) -> None:
        """Switch sections when the section list selection changes."""
        if element is not self._section_list:
            return
        selected = self._section_list.get_single_selection()
        if selected and selected != self._active_section:
            self._build_section(selected)

    def _handle_slider_moved(self, slider: t.Any) -> None:
        """Sync a moved slider's value entry, then apply the section immediately."""
        registry = self._active_registry()
        if slider not in registry.sliders:
            return
        update_value_entry(slider, registry)
        self._apply_active_section()

    def _handle_entry_finished(self, entry: t.Any) -> None:
        """Apply a finished text entry (height, value entry, or string field)."""
        if entry is self._cam_height_entry or entry is self._focal_height_entry:
            self._apply_height_entries()
            return
        registry = self._active_registry()
        if entry in registry.entry_to_slider:
            apply_entry_to_slider(entry, registry)
        elif entry not in registry.text_entries and entry not in registry.color_pool_widgets():
            return
        self._apply_active_section()

    def _apply_active_section(self) -> None:
        """Write the active section's widget values into its root object.

        Invalidates the shared scene cache so a later Render/Preview rebuilds
        with the edit instead of reusing a stale cached scene.
        """
        apply_widgets_to_config(self._active_registry(), self._active_root())
        self.state.invalidate_scene()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_rerender(self) -> None:
        """Apply both registries, rebuild the scene, and re-render at full quality."""
        if self.jobs.busy:
            return
        apply_widgets_to_config(self._static_registry, self.state.config)
        if self._active_kind is not None:
            apply_widgets_to_config(self._light_registry, self._active_kind.target)
        self._rebuild_scene_in_place()
        self._snapshot_camera()
        self._rebuild_minimap()
        self._render_full()

    def _rebuild_scene_in_place(self) -> None:
        """Rebuild the scene for the current mode without hitting OSM.

        Synthetic mode rebuilds from scratch; real-tiles mode re-places lights on
        the existing strip and re-syncs the camera (the OSM clip is untouched).
        """
        if self._use_real_tiles:
            if self._scene is None:
                return
            self._scene = rebuild_real_lights(self._scene, self.state.config)
            sync_camera_from_config(self._scene, self.state.config)
        else:
            self._scene = build_synthetic_scene(self.state.config)

    def _on_save(self) -> None:
        """Apply both registries, then write the full working config to disk."""
        apply_widgets_to_config(self._static_registry, self.state.config)
        if self._active_kind is not None:
            apply_widgets_to_config(self._light_registry, self._active_kind.target)
        if self.state.config_path is not None:
            write_config(self.state.config, self.state.config_path)
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
        """Write the full working config to the chosen path and adopt it.

        Args:
            path (Path): Destination config file chosen in the save dialog.
        """
        write_config(self.state.config, path)
        self.state.config_path = path

    def _on_toggle_real_tiles(self) -> None:
        """Flip real-tiles mode; ON loads the strip on a background job."""
        if self.jobs.busy:
            return
        self._use_real_tiles = not self._use_real_tiles
        if self._realtiles_btn is not None:
            self._realtiles_btn.set_text(
                "Real Tiles: ON" if self._use_real_tiles else "Real Tiles: OFF"
            )
        if self._use_real_tiles:
            config = self.state.config
            cache_dir = self.state.cache_dir
            self.jobs.submit(lambda progress: _load_real_strip(config, cache_dir, progress))
        else:
            self._scene = build_synthetic_scene(self.state.config)
            self._snapshot_camera()
            self._rebuild_minimap()
            self._render_full()

    # ------------------------------------------------------------------
    # Job events (real-tiles strip load)
    # ------------------------------------------------------------------

    def handle_job_event(self, event: JobEvent) -> None:
        """Adopt the loaded real strip scene on completion.

        Args:
            event (JobEvent): The job progress/completion event.
        """
        if event.kind == "done" and self._use_real_tiles:
            self._scene = event.result
            self._snapshot_camera()
            self._rebuild_minimap()
            self._render_full()

    # ------------------------------------------------------------------
    # Per-frame camera input + render
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Run camera passthrough with the cheap/full render debounce.

        Args:
            dt (float): Seconds since the previous frame.
        """
        events = self._frame_events
        self._frame_events = []
        self._elapsed += dt
        self._sync_buttons_enabled()
        if not self._scene_ready():
            return
        events, minimap_moved = self._process_minimap_events(events)
        moved = self._apply_camera_input(events, dt) or minimap_moved
        if moved:
            self._last_move_time = self._elapsed
            self._render_cheap()
        elif should_render_full(self._last_move_time, self._elapsed, moved):
            self._render_full()
            self._last_move_time = None

    def _apply_camera_input(self, events: t.List[pygame.event.Event], dt: float) -> bool:
        """Feed events to handle_input only when the cursor is over the view area.

        Ignores quit / screenshot / save_camera results (ESC/P/C must not quit
        the app or fire old CLI actions); honors reset by restoring the
        build-time camera snapshot.

        Args:
            events (list): This frame's collected pygame events.
            dt (float): Seconds since the previous frame.

        Returns:
            bool: True if the camera actually moved this frame.
        """
        camera = self._scene.camera
        if not self._view_rect.collidepoint(pygame.mouse.get_pos()):
            self._drag_state.clear()
            return False

        pos_before = camera.position.copy()
        tgt_before = camera.target.copy()
        fov_before = camera.fov_deg
        held = pygame.key.get_pressed()
        result = handle_input(
            events, camera, dt, held, self._drag_state,
            self._original_position, self._original_target, self._original_fov,
        )
        if result.reset:
            camera.position = self._original_position.copy()
            camera.target = self._original_target.copy()
            camera.fov_deg = self._original_fov
            self._drag_state.clear()

        return (
            not np.array_equal(camera.position, pos_before)
            or not np.array_equal(camera.target, tgt_before)
            or camera.fov_deg != fov_before
        )

    # ------------------------------------------------------------------
    # Minimap
    # ------------------------------------------------------------------

    def _process_minimap_events(
        self, events: t.List[pygame.event.Event]
    ) -> t.Tuple[t.List[pygame.event.Event], bool]:
        """Consume minimap drag events; return the rest and whether a drag moved.

        Runs before the camera passthrough and swallows any event the minimap
        consumes so a click/drag on the map never also moves the scene camera.

        Args:
            events (list): This frame's collected pygame events.

        Returns:
            tuple[list, bool]: (events not consumed, True if a drag moved the camera).
        """
        if self._minimap is None:
            return events, False
        remaining: t.List[pygame.event.Event] = []
        moved = False
        for event in events:
            consumed, did_move = self._handle_minimap_event(event)
            moved = moved or did_move
            if not consumed:
                remaining.append(event)
        return remaining, moved

    def _handle_minimap_event(self, event: pygame.event.Event) -> t.Tuple[bool, bool]:
        """Handle one event against the minimap.

        Args:
            event (pygame.event.Event): The event to consider.

        Returns:
            tuple[bool, bool]: (consumed by the minimap, moved the camera).
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self._minimap.layout.rect.collidepoint(event.pos):
                return False, False
            cam_xy, tgt_xy = self._dot_world_xy()
            self._drag_dot = self._minimap.hit_test(event.pos, cam_xy, tgt_xy)
            return True, False
        if event.type == pygame.MOUSEMOTION and self._drag_dot is not None:
            self._apply_minimap_drag(event.pos)
            return True, True
        if (
            event.type == pygame.MOUSEBUTTONUP
            and event.button == 1
            and self._drag_dot is not None
        ):
            self._drag_dot = None
            return True, False
        return False, False

    def _apply_minimap_drag(self, pos: t.Tuple[int, int]) -> None:
        """Write the dragged dot's new X/Y into the config and sync the camera.

        Args:
            pos (tuple[int, int]): Screen pixel the mouse is at.
        """
        world_x, world_y = self._minimap.screen_to_world(*pos)
        cam = self.state.config.devmode.camera
        if self._drag_dot == "camera":
            cam.pos_x = world_x
            cam.pos_y = world_y
        else:
            cam.target_x = world_x
            cam.target_y = world_y
        sync_camera_from_config(self._scene, self.state.config)

    def _dot_world_xy(self) -> t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]:
        """Return the live camera and target (x, y) shown as minimap dots."""
        camera = self._scene.camera
        cam_xy = (float(camera.position[0]), float(camera.position[1]))
        tgt_xy = (float(camera.target[0]), float(camera.target[1]))
        return cam_xy, tgt_xy

    def _rebuild_minimap(self) -> None:
        """Recompute the world extent from the current scene and rebuild the minimap."""
        if self._scene is None:
            return
        world_min, world_max = self._compute_world_extent(self._scene)
        rect = pygame.Rect(
            self._view_rect.x + _MINIMAP_MARGIN,
            self._view_rect.y + self._view_rect.height - _MINIMAP_MARGIN - _MINIMAP_HEIGHT,
            _MINIMAP_WIDTH, _MINIMAP_HEIGHT,
        )
        self._minimap = Minimap(MinimapLayout(rect, world_min, world_max))

    @staticmethod
    def _compute_world_extent(
        scene: t.Any,
    ) -> t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]:
        """Return the road bounding box padded 10%, or a fallback for empty scenes.

        Args:
            scene: The scene whose road_segments define the extent.

        Returns:
            tuple: ((x_min, y_min), (x_max, y_max)) ENU, padded 10% each side.
        """
        arrays = [
            seg.geometry[:, :2]
            for seg in scene.road_segments
            if seg.geometry is not None and len(seg.geometry) > 0
        ]
        if not arrays:
            return _EMPTY_EXTENT
        points = np.concatenate(arrays, axis=0)
        x_min, y_min = float(points[:, 0].min()), float(points[:, 1].min())
        x_max, y_max = float(points[:, 0].max()), float(points[:, 1].max())
        pad_x = (x_max - x_min) * 0.1
        pad_y = (y_max - y_min) * 0.1
        return (x_min - pad_x, y_min - pad_y), (x_max + pad_x, y_max + pad_y)

    def _apply_height_entries(self) -> None:
        """Parse the height entries into pos_z/target_z, sync, and debounce a render."""
        if self.state.config.devmode is None or self._scene is None:
            return
        cam = self.state.config.devmode.camera
        cam.pos_z = parse_float_entry(self._cam_height_entry, cam.pos_z)
        cam.target_z = parse_float_entry(self._focal_height_entry, cam.target_z)
        self.state.invalidate_scene()
        sync_camera_from_config(self._scene, self.state.config)
        self._last_move_time = self._elapsed
        self._render_cheap()

    def _snapshot_camera(self) -> None:
        """Record the current camera pose as the reset target."""
        camera = self._scene.camera
        self._original_position = camera.position.copy()
        self._original_target = camera.target.copy()
        self._original_fov = camera.fov_deg

    def _render_full(self) -> None:
        """Render the scene at full quality and store the fitted surface."""
        image = render_devmode_scene(self._scene, self.state.config, use_labels=not self._use_real_tiles)
        self._store_frame(image)

    def _render_cheap(self) -> None:
        """Render the cheap ground+roads preview frame while the camera moves."""
        image = render_cheap(self._scene, self.state.config)
        self._store_frame(image)

    def _store_frame(self, image: t.Any) -> None:
        """Letterbox a rendered PIL image into the view and store the surface."""
        width = max(1, self._view_rect.width)
        height = max(1, self._view_rect.height)
        fitted = fit_image_to_viewport(image, width, height)
        self._frame_surface = pil_to_pygame_surface(fitted)

    def draw(self, window: pygame.Surface) -> None:
        """Blit the stored devmode frame at the view-area origin.

        Args:
            window (pygame.Surface): The window surface to draw onto.
        """
        if self._frame_surface is not None:
            window.blit(self._frame_surface, (self._view_rect.x, self._view_rect.y))
        if self._minimap is not None and self._scene is not None:
            cam_xy, tgt_xy = self._dot_world_xy()
            self._minimap.draw(window, cam_xy, tgt_xy, self._scene.road_segments)

    def refresh_from_state(self) -> None:
        """Rebuild the catalog, section list, and scene after the config changed."""
        if self._manager is None:
            return
        ensure_devmode_config(self.state.config)
        self._catalog = build_light_catalog(self.state.config)
        if self._section_list is not None:
            self._section_list.set_item_list(self._section_names())
        if self._active_section not in self._section_names():
            self._active_section = "Atmosphere"
        self._rebuild_scene_in_place_or_synthetic()
        self._snapshot_camera()
        self._rebuild_minimap()
        self._sync_height_entries_from_config()
        self._render_full()
        self._build_section(self._active_section)

    def _rebuild_scene_in_place_or_synthetic(self) -> None:
        """Rebuild the scene for the current mode, falling back to synthetic.

        Real-tiles mode with an existing strip re-places lights without OSM;
        otherwise a fresh synthetic scene is built.
        """
        if self._use_real_tiles and self._scene is not None:
            self._scene = rebuild_real_lights(self._scene, self.state.config)
            sync_camera_from_config(self._scene, self.state.config)
        else:
            self._scene = build_synthetic_scene(self.state.config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scene_ready(self) -> bool:
        """Return True once a scene and its camera snapshot exist."""
        return self._scene is not None and self._original_target is not None

    def _sync_buttons_enabled(self) -> None:
        """Disable the action buttons while a background job runs, else enable them."""
        busy = self.jobs.busy
        for button in (self._rerender_btn, self._save_btn, self._realtiles_btn):
            if button is None:
                continue
            if busy:
                button.disable()
            else:
                button.enable()


def _load_real_strip(config: t.Any, cache_dir: Path, progress: ProgressFn) -> t.Any:
    """Build the real OSM strip scene on the worker thread.

    Args:
        config (SceneConfig): Working config driving the strip build.
        cache_dir (Path): OSM cache directory.
        progress (ProgressFn): Progress callback (indeterminate).

    Returns:
        Scene: The clipped real-strip devmode scene.
    """
    progress("Loading real tile strip...", -1.0)
    return build_real_scene_strip(config, cache_dir)
