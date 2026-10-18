"""Interactive GPU light-tuning window (`streets-at-night tune`).

A rendercanvas (glfw) window with a Dear ImGui panel over a live wgpu render
of the tuning scene. Controls: drag to orbit, shift + drag to pan, scroll to
zoom, R to reset the view. Config edits stay in memory until Save.
"""

import time
import typing as t
from pathlib import Path

from imgui_bundle import imgui
from rendercanvas.glfw import RenderCanvas, loop
from wgpu.utils.imgui import ImguiRenderer

from ..config.schema import CommercialLightConfig, FixtureProfileConfig, SceneConfig
from ..devmode.config_writer import write_config
from ..devmode.light_catalog import LightKind, build_light_catalog
from .field_editor import FieldEditor
from .orbit_camera import OrbitCamera
from .renderer import get_device
from .tuning_view import TuningView

_WINDOW_SIZE: t.Tuple[int, int] = (1600, 900)
_PANEL_WIDTH: float = 430.0

# Catalog labels split across the panel's tabs. Kinds not listed here (car
# density/separation, global color shift) stay in the 2D devmode.
_ROAD_PREFIXES: t.Tuple[str, ...] = (
    "Streetlight:", "Commercial: Parking", "Cars: Headlights", "Cars: Taillights",
)
_BUILDING_LABELS: t.Tuple[str, ...] = (
    "Buildings: Windows", "Commercial: Signs", "Commercial: Storefronts",
)
_SCENE_LABELS: t.Tuple[str, ...] = ("Illumination: Ground Pools", "Grade")


def run_tune(config: SceneConfig, config_path: t.Optional[Path]) -> None:
    """Open the tuning window and run until it is closed.

    Args:
        config (SceneConfig): Config to tune (edited in memory).
        config_path (Path | None): Where Save writes; None disables Save.
    """
    TuneApp(config, config_path).run()


class TuneApp:
    """Window, input handling, and the imgui control panel."""

    def __init__(self, config: SceneConfig, config_path: t.Optional[Path]) -> None:
        """Create the window, renderer, and panel state.

        Args:
            config (SceneConfig): Config to tune.
            config_path (Path | None): Save destination.
        """
        self._config = config
        self._config_path = config_path
        catalog = build_light_catalog(config)
        self._road_kinds = [k for k in catalog if k.label.startswith(_ROAD_PREFIXES)]
        self._building_kinds = [k for k in catalog if k.label in _BUILDING_LABELS]
        self._scene_kinds = [k for k in catalog if k.label in _SCENE_LABELS]
        self._scene_kinds.append(LightKind("Atmosphere: Glow", config.atmosphere, "glow_"))
        self._road_index = 0
        self._building_index = 0
        self._pole_kind = self._road_kinds[0]
        self._editor = FieldEditor()
        self._status = ""

        device = get_device()
        self._canvas = RenderCanvas(
            size=_WINDOW_SIZE, title="Streets at Night — light tuning",
            update_mode="continuous", max_fps=60,
        )
        self._context = self._canvas.get_context("wgpu")
        output_format = self._context.get_preferred_format(device.adapter)
        self._context.configure(device=device, format=output_format)
        self._view = TuningView(config, self._pole_kind.target, output_format, device)
        self._orbit = OrbitCamera()
        self._imgui = ImguiRenderer(device, self._canvas)
        self._imgui.set_gui(self._draw_panel)

        self._dragging = False
        self._last_pointer = (0.0, 0.0)
        self._frame_times: t.List[float] = []
        self._canvas.add_event_handler(self._on_pointer, "pointer_down", "pointer_up", "pointer_move")
        self._canvas.add_event_handler(self._on_wheel, "wheel")
        self._canvas.add_event_handler(self._on_key, "key_down")

    def run(self) -> None:
        """Start drawing and enter the event loop."""
        self._canvas.request_draw(self._draw_frame)
        loop.run()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        """Render the scene, then the imgui panel on top."""
        width, height = self._canvas.get_physical_size()
        if width == 0 or height == 0:
            return
        camera = self._orbit.to_camera(width / height)
        target = self._context.get_current_texture().create_view()
        self._view.draw(camera, target, (width, height))
        self._imgui.render()
        self._frame_times = [*self._frame_times[-59:], time.perf_counter()]

    def _draw_panel(self) -> None:
        """Build the imgui control panel for this frame."""
        imgui.set_next_window_pos((10, 10), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size((_PANEL_WIDTH, 860), imgui.Cond_.first_use_ever)
        imgui.begin("Light tuning")
        imgui.push_item_width(-200)

        changed = False
        toggled, self._view.road_lights = imgui.checkbox("Road lights", self._view.road_lights)
        changed |= toggled
        imgui.same_line()
        toggled, self._view.building_lights = imgui.checkbox(
            "Building lights", self._view.building_lights
        )
        changed |= toggled
        imgui.text_disabled(f"On poles: {self._pole_kind.label}")
        imgui.separator()

        if imgui.begin_tab_bar("tabs"):
            if imgui.begin_tab_item("Road / Commercial")[0]:
                changed |= self._draw_road_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item("Building")[0]:
                changed |= self._draw_kind_picker(self._building_kinds, "building")
                imgui.end_tab_item()
            if imgui.begin_tab_item("Scene")[0]:
                changed |= self._draw_scene_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()

        imgui.separator()
        self._draw_footer()
        imgui.pop_item_width()
        imgui.end()

        if changed:
            self._view.rebuild()

    def _draw_road_tab(self) -> bool:
        """Road / commercial kinds; picking a pole fixture mounts it on the poles.

        Returns:
            bool: True if anything changed.
        """
        changed = self._draw_kind_picker(self._road_kinds, "road")
        kind = self._road_kinds[self._road_index]
        if kind is not self._pole_kind and _is_pole_fixture(kind):
            self._pole_kind = kind
            self._view.pole_fixture = kind.target
            changed = True
        return changed

    def _draw_kind_picker(self, kinds: t.List[LightKind], which: str) -> bool:
        """A combo to pick a kind, then its field editor.

        Args:
            kinds (list[LightKind]): Kinds in this tab.
            which (str): "road" or "building" (which selection index to use).

        Returns:
            bool: True if a field value changed.
        """
        index = self._road_index if which == "road" else self._building_index
        if imgui.begin_combo(f"Light##{which}", kinds[index].label):
            for i, kind in enumerate(kinds):
                if imgui.selectable(kind.label, i == index)[0]:
                    index = i
            imgui.end_combo()
        if which == "road":
            self._road_index = index
        else:
            self._building_index = index
        imgui.spacing()
        return self._editor.draw(kinds[index])

    def _draw_scene_tab(self) -> bool:
        """Global knobs: ground-pool brightness and knee, grade, glow shape.

        Returns:
            bool: True if anything changed.
        """
        atmos = self._config.atmosphere
        changed, atmos.illumination_brightness = imgui.slider_int(
            "Illumination Brightness (0-255)", atmos.illumination_brightness, 0, 255
        )
        for kind in self._scene_kinds:
            if imgui.collapsing_header(kind.label, imgui.TreeNodeFlags_.default_open):
                changed |= self._editor.draw(kind)
        return changed

    def _draw_footer(self) -> None:
        """Save / reset buttons, status, frame rate, and control help."""
        if self._config_path is not None:
            if imgui.button(f"Save to {self._config_path.name}"):
                write_config(self._config, self._config_path)
                self._status = f"Saved {time.strftime('%H:%M:%S')}"
        else:
            imgui.text_disabled("No config file: Save disabled")
        imgui.same_line()
        if imgui.button("Reset view"):
            self._orbit.reset()
        if self._status:
            imgui.text(self._status)
        imgui.text_disabled(f"{self._fps():.0f} fps   drag: orbit   shift+drag: pan   scroll: zoom   R: reset")

    def _fps(self) -> float:
        """Average frames per second over the last 60 frames."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # ------------------------------------------------------------------
    # Input (imgui handlers run first and stop events it wants)
    # ------------------------------------------------------------------

    def _on_pointer(self, event: dict) -> None:
        """Orbit on drag, pan on shift + drag."""
        xy = (event["x"], event["y"])
        if event["event_type"] == "pointer_down" and event["button"] == 1:
            self._dragging = True
        elif event["event_type"] == "pointer_up":
            self._dragging = False
        elif self._dragging:
            dx, dy = xy[0] - self._last_pointer[0], xy[1] - self._last_pointer[1]
            if "Shift" in event["modifiers"]:
                self._orbit.pan(dx, dy, int(self._canvas.get_logical_size()[0]))
            else:
                self._orbit.rotate(dx, dy)
        self._last_pointer = xy

    def _on_wheel(self, event: dict) -> None:
        """Zoom toward the pivot on scroll up."""
        self._orbit.zoom(-event["dy"] / 100.0)

    def _on_key(self, event: dict) -> None:
        """R resets the view."""
        if event["key"].lower() == "r":
            self._orbit.reset()


def _is_pole_fixture(kind: LightKind) -> bool:
    """True for kinds that can be mounted on the poles (streetlights, parking).

    Args:
        kind (LightKind): A catalog kind.

    Returns:
        bool: Whether the kind is a pole fixture.
    """
    if isinstance(kind.target, FixtureProfileConfig):
        return True
    return isinstance(kind.target, CommercialLightConfig) and kind.field_prefix == "parking_"
