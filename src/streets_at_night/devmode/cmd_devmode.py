"""Main event loop for the devmode parameter tuning tool."""

import argparse
import random
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import pygame_gui
from PIL import Image

from ..camera.controls import MouseDragState, handle_input
from ..config.loader import _build_devmode_config, load_config
from ..config.schema import SceneConfig
from ..rendering.atmosphere_pass import render_atmosphere
from ..rendering.compositor import render_composite
from ..rendering.ground_pass import render_ground
from ..rendering.illumination_pass import render_illumination
from ..rendering.light_pass import render_lights_export
from ..rendering.pillow_backend import PillowBackend
from ..rendering.raycast import build_ground_gbuffer
from ..rendering.render_pipeline import render_scene_frame
from ..rendering.road_pass import render_roads_export
from .config_bridge import apply_widgets_to_config, read_config_to_widgets
from .config_writer import write_config
from .control_panel import build_control_panel, rebuild_section
from .label_pass import make_label_pass
from .real_scene import build_real_scene_strip
from .slider_registry import SliderRegistry
from .synthetic_scene import ROW_LABELS, build_synthetic_scene

_WINDOW_WIDTH = 1280
_WINDOW_HEIGHT = 720
_PANEL_WIDTH = 384
_RENDER_WIDTH = _WINDOW_WIDTH - _PANEL_WIDTH

# Full-quality re-render only fires once camera movement has been still for
# this many seconds; while moving, a cheap roads-only frame is shown instead.
_DEBOUNCE_SECONDS = 0.15


def cmd_devmode(args: argparse.Namespace, config: SceneConfig) -> None:
    """Run the devmode parameter tuning tool.

    Initialises pygame and pygame-gui, builds a test scene (synthetic or a real
    OSM strip when --real-tiles is given), renders it using the Pillow backend,
    and displays it alongside an interactive control panel. The user can adjust
    config parameters via sliders, re-render on demand, and save the modified
    config to a TOML file.

    Args:
        args (argparse.Namespace): Parsed CLI args (carries --config, --real-tiles,
            --cache-dir).
        config (SceneConfig): Loaded scene configuration.
    """
    _ensure_devmode_config(config)

    from ..main import _resolve_cache_dir

    use_real_tiles: bool = getattr(args, "real_tiles", False)
    cache_dir: Path = _resolve_cache_dir(getattr(args, "cache_dir", Path("cache")), config)

    pygame.init()
    window = pygame.display.set_mode((_WINDOW_WIDTH, _WINDOW_HEIGHT))
    pygame.display.set_caption("streets-at-night devmode")
    manager = pygame_gui.UIManager((_WINDOW_WIDTH, _WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    panel_rect = pygame.Rect(_RENDER_WIDTH, 0, _PANEL_WIDTH, _WINDOW_HEIGHT)

    if use_real_tiles:
        print("Loading real tile strip for devmode ...")
        scene = build_real_scene_strip(config, cache_dir)
    else:
        scene = build_synthetic_scene(config)

    render_surface = _render_and_convert(scene, config, use_labels=not use_real_tiles)

    registry, scroll, rerender_btn, save_btn, load_btn, section_list = build_control_panel(
        manager, panel_rect, config
    )
    read_config_to_widgets(config, registry)

    original_position = scene.camera.position.copy()
    original_target = scene.camera.target.copy()
    original_fov = scene.camera.fov_deg
    drag_state = MouseDragState()

    active_section = "Atmosphere"
    save_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None
    save_confirm_btn: t.Optional[pygame_gui.elements.UIButton] = None
    load_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None
    load_confirm_btn: t.Optional[pygame_gui.elements.UIButton] = None

    elapsed_time = 0.0
    last_move_time: t.Optional[float] = None

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0
        elapsed_time += time_delta
        events = pygame.event.get()
        held_keys = pygame.key.get_pressed()

        # Camera controls — snapshot before so we know if re-render is needed.
        cam_pos_before = scene.camera.position.copy()
        cam_tgt_before = scene.camera.target.copy()
        cam_fov_before = scene.camera.fov_deg

        input_result = handle_input(
            events, scene.camera, time_delta, held_keys, drag_state,
            original_position, original_target, original_fov,
        )

        if input_result.quit:
            running = False
        if input_result.reset:
            scene.camera.position = original_position.copy()
            scene.camera.target = original_target.copy()
            scene.camera.fov_deg = original_fov
            drag_state.clear()

        camera_moved = (
            not np.array_equal(scene.camera.position, cam_pos_before)
            or not np.array_equal(scene.camera.target, cam_tgt_before)
            or scene.camera.fov_deg != cam_fov_before
        )
        if camera_moved:
            last_move_time = elapsed_time

        # UI events
        for event in events:
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == rerender_btn:
                    apply_widgets_to_config(registry, config)
                    if use_real_tiles:
                        scene = _rebuild_real_lights(scene, config)
                        _sync_camera_from_config(scene, config)
                    else:
                        scene = build_synthetic_scene(config)
                    original_position = scene.camera.position.copy()
                    original_target = scene.camera.target.copy()
                    original_fov = scene.camera.fov_deg
                    render_surface = _render_and_convert(
                        scene, config, use_labels=not use_real_tiles
                    )
                    camera_moved = False
                    last_move_time = None

                elif event.ui_element == save_btn:
                    save_entry, save_confirm_btn = _show_save_dialog(manager)

                elif save_confirm_btn is not None and event.ui_element == save_confirm_btn:
                    _handle_save(registry, config, save_entry, save_confirm_btn)
                    save_entry = None
                    save_confirm_btn = None

                elif event.ui_element == load_btn:
                    load_entry, load_confirm_btn = _show_load_dialog(manager)

                elif load_confirm_btn is not None and event.ui_element == load_confirm_btn:
                    new_config = _handle_load(load_entry, load_confirm_btn, registry, config)
                    load_entry = None
                    load_confirm_btn = None
                    if new_config is not None:
                        config = new_config
                        if use_real_tiles:
                            scene = _rebuild_real_lights(scene, config)
                            _sync_camera_from_config(scene, config)
                        else:
                            scene = build_synthetic_scene(config)
                        original_position = scene.camera.position.copy()
                        original_target = scene.camera.target.copy()
                        original_fov = scene.camera.fov_deg
                        render_surface = _render_and_convert(
                            scene, config, use_labels=not use_real_tiles
                        )
                        camera_moved = False
                        last_move_time = None

                elif event.ui_element in registry.bool_buttons:
                    _toggle_bool_button(event.ui_element, registry)

            if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
                if event.ui_element == section_list:
                    selected = section_list.get_single_selection()
                    if selected and selected != active_section:
                        active_section = selected
                        rebuild_section(manager, scroll, registry, config, active_section)

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                _update_value_entry(event.ui_element, registry)

            if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                _apply_entry_to_slider(event.ui_element, registry)

            manager.process_events(event)

        if camera_moved:
            render_surface = _render_cheap(scene, config)
        elif _should_render_full(last_move_time, elapsed_time, camera_moved):
            render_surface = _render_and_convert(
                scene, config, use_labels=not use_real_tiles
            )
            last_move_time = None

        manager.update(time_delta)

        window.fill((30, 30, 30))
        window.blit(render_surface, (0, 0))
        manager.draw_ui(window)
        pygame.display.flip()

    pygame.quit()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_devmode_config(config: SceneConfig) -> None:
    """Initialise config.devmode from DEFAULTS if it is None.

    Args:
        config (SceneConfig): Config to populate in-place if devmode is absent.
    """
    if config.devmode is not None:
        return
    from ..config.defaults import DEFAULTS
    config.devmode = _build_devmode_config(DEFAULTS["devmode"])


def _sync_camera_from_config(scene: t.Any, config: SceneConfig) -> None:
    """Update scene.camera position, target, and FOV from config.devmode.camera.

    Used for real-tiles mode where the scene is not rebuilt on re-render —
    only the lights are replaced. Without this, camera slider changes have
    no effect because the Camera object on the scene is never touched.

    Args:
        scene: The scene whose camera to update.
        config (SceneConfig): Config with updated devmode.camera values.
    """
    if config.devmode is None:
        return
    cam = config.devmode.camera
    scene.camera.position = np.array([cam.pos_x, cam.pos_y, cam.pos_z])
    scene.camera.target = np.array([cam.target_x, cam.target_y, cam.target_z])
    scene.camera.fov_deg = cam.fov


def _rebuild_real_lights(scene: t.Any, config: SceneConfig) -> t.Any:
    """Clear and re-place all lights on the existing real strip scene.

    Keeps road geometry, zones, and camera fixed. Only the spatial index and
    light lists are rebuilt so that updated config parameters take effect.

    Args:
        scene: The real strip scene to update in-place.
        config (SceneConfig): Updated scene configuration.

    Returns:
        The same scene object with refreshed lights.
    """
    from ..placement.scene_builder import populate_lights

    scene.lights = []
    scene.car_pairs = []
    scene.build_spatial_index()
    populate_lights(scene, config, random.Random(config.seed))
    return scene


def _render_and_convert(
    scene: t.Any,
    config: SceneConfig,
    use_labels: bool = True,
) -> pygame.Surface:
    """Render the scene at full quality and return a pygame surface for display.

    Renders at config.devmode.render_width/render_height (making those config
    fields real render resolution, not just an aspect ratio), then scales the
    result to fit the fixed display viewport, preserving aspect ratio and
    letterboxing so the image is never squashed.

    Args:
        scene: The scene to render.
        config (SceneConfig): Current scene configuration.
        use_labels (bool): When True, overlay synthetic row labels.

    Returns:
        pygame.Surface: The rendered frame as a pygame surface, sized to the
            display viewport (_RENDER_WIDTH x _WINDOW_HEIGHT).
    """
    image = _render_scene(scene, config, use_labels=use_labels)
    fitted = _fit_image_to_viewport(image, _RENDER_WIDTH, _WINDOW_HEIGHT)
    return _pil_to_pygame(fitted)


def _render_cheap(scene: t.Any, config: SceneConfig) -> pygame.Surface:
    """Render a cheap ground+roads-only preview frame for use while the camera moves.

    Skips illumination, lights, atmosphere, and composite so navigation stays
    responsive. Renders at config.devmode.render_width/render_height — the
    same render resolution as the full-quality path — then letterboxes to the
    display viewport via _fit_image_to_viewport, so this transient frame has
    identical framing and aspect ratio to the settled full-quality frame and
    does not appear squashed or snap on settle.

    Args:
        scene: The scene to render.
        config (SceneConfig): Current scene configuration.

    Returns:
        pygame.Surface: The cheap preview frame, sized to the display viewport.
    """
    render_width = config.devmode.render_width
    render_height = config.devmode.render_height
    backend = PillowBackend(render_width, render_height)
    passes = [render_ground, render_roads_export]
    render_scene_frame(scene, backend, passes=passes)
    fitted = _fit_image_to_viewport(backend.get_image(), _RENDER_WIDTH, _WINDOW_HEIGHT)
    return _pil_to_pygame(fitted)


def _should_render_full(
    last_move_t: t.Optional[float],
    now: float,
    dirty: bool,
) -> bool:
    """Decide whether to run the full render pipeline this frame.

    While the camera is actively moving (dirty), a cheap preview frame is
    shown instead and this returns False. Once movement stops, this returns
    True exactly once movement has been still for at least _DEBOUNCE_SECONDS
    — callers should reset last_move_t to None after honoring a True result
    so the full render does not repeat every frame thereafter.

    Args:
        last_move_t (float | None): Elapsed-time timestamp of the last camera
            movement, or None if the camera has not moved since the last full
            render.
        now (float): Current elapsed-time timestamp.
        dirty (bool): True if the camera moved this frame.

    Returns:
        bool: True if a full-quality render should run this frame.
    """
    if dirty:
        return False
    if last_move_t is None:
        return False
    return (now - last_move_t) >= _DEBOUNCE_SECONDS


@dataclass
class ViewportFit:
    """Scaled size and centering offset for fitting a render into a viewport.

    Attributes:
        width: Scaled width in pixels.
        height: Scaled height in pixels.
        x: Horizontal offset in pixels for centering within the viewport.
        y: Vertical offset in pixels for centering within the viewport.
    """

    width: int
    height: int
    x: int
    y: int


def compute_viewport_fit(
    render_width: int,
    render_height: int,
    viewport_width: int,
    viewport_height: int,
) -> ViewportFit:
    """Compute the scaled size and letterbox offset to fit a render into a viewport.

    Preserves the render's aspect ratio: scales uniformly to the largest size
    that fits entirely within the viewport, then centers it.

    Args:
        render_width (int): Width of the rendered image in pixels.
        render_height (int): Height of the rendered image in pixels.
        viewport_width (int): Width of the target viewport in pixels.
        viewport_height (int): Height of the target viewport in pixels.

    Returns:
        ViewportFit: Scaled width/height and top-left offset within the viewport.
    """
    scale = min(viewport_width / render_width, viewport_height / render_height)
    width = max(1, round(render_width * scale))
    height = max(1, round(render_height * scale))
    x = (viewport_width - width) // 2
    y = (viewport_height - height) // 2
    return ViewportFit(width=width, height=height, x=x, y=y)


def _fit_image_to_viewport(
    image: Image.Image,
    viewport_width: int,
    viewport_height: int,
) -> Image.Image:
    """Scale a rendered image to fit a fixed-size viewport, preserving aspect ratio.

    Resizes the image to the largest size that fits within the viewport
    (via compute_viewport_fit) and pastes it onto a black canvas the size of
    the viewport, so the result is always exactly viewport_width x
    viewport_height regardless of the source image's aspect ratio.

    Args:
        image (PIL.Image.Image): The rendered image at its native resolution.
        viewport_width (int): Width of the display viewport in pixels.
        viewport_height (int): Height of the display viewport in pixels.

    Returns:
        PIL.Image.Image: Letterboxed image sized viewport_width x viewport_height.
    """
    fit = compute_viewport_fit(image.width, image.height, viewport_width, viewport_height)
    scaled = image.resize((fit.width, fit.height))
    canvas = Image.new("RGB", (viewport_width, viewport_height), (0, 0, 0))
    canvas.paste(scaled, (fit.x, fit.y))
    return canvas


def _render_scene(
    scene: t.Any,
    config: SceneConfig,
    use_labels: bool = True,
) -> Image.Image:
    """Render the scene using the Pillow backend and return the image.

    Mirrors the export pipeline so devmode shows the same result: ground →
    roads → illumination → lights → atmosphere → composite, then optional
    labels. The composite pass is essential — illumination and light passes
    only accumulate into the backend's linear buffer, and composite is what
    resolves that buffer (tone-map + sRGB encode) into the visible image.
    Renders at config.devmode.render_width/render_height so that config
    controls actual render resolution rather than just camera aspect ratio.

    Args:
        scene: The scene to render.
        config (SceneConfig): Current scene configuration.
        use_labels (bool): When True, append a label pass for synthetic rows.

    Returns:
        PIL.Image.Image: The rendered frame.
    """
    for car_pair in scene.car_pairs:
        car_pair.resolve_facing(scene.camera.get_forward())
    render_width = config.devmode.render_width
    render_height = config.devmode.render_height
    backend = PillowBackend(render_width, render_height)
    gbuffer = build_ground_gbuffer(scene.camera, render_width, render_height)
    passes = _build_render_passes(scene, gbuffer, use_labels)
    render_scene_frame(scene, backend, passes=passes)
    return backend.get_image()


def _build_render_passes(
    scene: t.Any,
    gbuffer: t.Any,
    use_labels: bool,
) -> list:
    """Build the devmode render pass list, matching the export pipeline order.

    Illumination and atmosphere are wrapped to receive the shared G-buffer.
    The composite pass must be present and after the light passes so the
    accumulated linear light buffer is resolved into the visible image; labels
    come last so they overlay the final composited frame.

    Args:
        scene: The scene to render (used for the label pass camera).
        gbuffer: Shared ground raycast for illumination and atmosphere.
        use_labels (bool): When True, append the synthetic-row label pass.

    Returns:
        list: Ordered render passes for render_scene_frame.
    """
    def _illumination_pass(s: t.Any, c: t.Any, b: t.Any) -> t.Any:
        return render_illumination(s, c, b, gbuffer=gbuffer)

    def _atmosphere_pass(s: t.Any, c: t.Any, b: t.Any) -> None:
        render_atmosphere(s, c, b, gbuffer=gbuffer)

    passes = [
        render_ground,
        render_roads_export,
        _illumination_pass,
        render_lights_export,
        _atmosphere_pass,
        render_composite,
    ]
    if use_labels:
        passes.append(make_label_pass(ROW_LABELS, scene.camera))
    return passes


def _pil_to_pygame(image: Image.Image) -> pygame.Surface:
    """Convert a PIL Image to a pygame Surface.

    Args:
        image (PIL.Image.Image): The image to convert.

    Returns:
        pygame.Surface: Equivalent pygame surface in RGB mode.
    """
    rgb_image = image.convert("RGB")
    raw = rgb_image.tobytes()
    return pygame.image.frombytes(raw, rgb_image.size, "RGB")


def _show_save_dialog(
    manager: pygame_gui.UIManager,
) -> t.Tuple[
    pygame_gui.elements.UITextEntryLine,
    pygame_gui.elements.UIButton,
]:
    """Create a text entry and confirm button for file-save input.

    Args:
        manager (pygame_gui.UIManager): The UI manager.

    Returns:
        tuple[UITextEntryLine, UIButton]: The entry field and confirm button.
    """
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(50, _WINDOW_HEIGHT - 60, 500, 35),
        manager=manager,
    )
    entry.set_text("scene_config_devmode.toml")

    confirm_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(560, _WINDOW_HEIGHT - 60, 100, 35),
        text="Save",
        manager=manager,
    )
    return entry, confirm_btn


def _show_load_dialog(
    manager: pygame_gui.UIManager,
) -> t.Tuple[
    pygame_gui.elements.UITextEntryLine,
    pygame_gui.elements.UIButton,
]:
    """Create a text entry and confirm button for file-load input.

    Args:
        manager (pygame_gui.UIManager): The UI manager.

    Returns:
        tuple[UITextEntryLine, UIButton]: The entry field and confirm button.
    """
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(50, _WINDOW_HEIGHT - 60, 500, 35),
        manager=manager,
    )
    entry.set_text("scene_config.toml")

    confirm_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(560, _WINDOW_HEIGHT - 60, 100, 35),
        text="Load",
        manager=manager,
    )
    return entry, confirm_btn


def _handle_load(
    load_entry: pygame_gui.elements.UITextEntryLine,
    load_confirm_btn: pygame_gui.elements.UIButton,
    registry: SliderRegistry,
    config: SceneConfig,
) -> t.Optional[SceneConfig]:
    """Load a config from the path in load_entry and apply it to the registry.

    Returns the new SceneConfig on success, None on failure (file not found,
    bad TOML, etc.). Kills the dialog widgets regardless of outcome.

    Args:
        load_entry: Text entry widget containing the file path.
        load_confirm_btn: Confirm button widget to kill after handling.
        registry: SliderRegistry to update with new config values.
        config: Current config (used as fallback on failure).

    Returns:
        SceneConfig | None: New config on success, None on failure.
    """
    filename = load_entry.get_text().strip()
    load_entry.kill()
    load_confirm_btn.kill()
    if not filename:
        return None
    try:
        new_config = load_config(Path(filename))
        read_config_to_widgets(new_config, registry)
        return new_config
    except Exception as e:
        print(f"Load Config failed: {e}")
        return None


def _handle_save(
    registry: SliderRegistry,
    config: SceneConfig,
    save_entry: t.Optional[pygame_gui.elements.UITextEntryLine],
    save_confirm_btn: t.Optional[pygame_gui.elements.UIButton],
) -> None:
    """Write config to the file path specified in the save entry widget.

    Applies current widget state to config before writing, so a slider moved
    since the last Re-render is not silently dropped — Save must not require
    Re-render first. Kills the entry and confirm button after saving.

    Args:
        registry (SliderRegistry): Widget registry to apply to config before saving.
        config (SceneConfig): Config to update in-place, then serialize.
        save_entry: Text entry holding the file path string.
        save_confirm_btn: Confirm button to kill after save.
    """
    if save_entry is None:
        return
    apply_widgets_to_config(registry, config)
    filename = save_entry.get_text().strip()
    if filename:
        write_config(config, Path(filename))
    save_entry.kill()
    if save_confirm_btn is not None:
        save_confirm_btn.kill()


def _toggle_bool_button(
    button: pygame_gui.elements.UIButton,
    registry: SliderRegistry,
) -> None:
    """Flip the tracked state of a toggle button and update its display text.

    Args:
        button (UIButton): The toggle button to flip.
        registry (SliderRegistry): Registry holding bool_states for all buttons.
    """
    current = registry.bool_states.get(button, False)
    new_value = not current
    registry.bool_states[button] = new_value
    button.set_text("ON" if new_value else "OFF")



def _update_value_entry(
    slider: t.Any,
    registry: SliderRegistry,
) -> None:
    """Update the value text entry for a slider that was just moved.

    Also marks the slider dirty in the registry, so apply_widgets_to_config
    knows the user actually touched it (see config_bridge._apply_sliders).

    Args:
        slider: The UIHorizontalSlider that fired the moved event.
        registry (SliderRegistry): Registry mapping sliders to descriptors and entries.
    """
    descriptor = registry.sliders.get(slider)
    value_entry = registry.value_entries.get(slider)
    if descriptor is None or value_entry is None:
        return
    registry.mark_dirty(descriptor.label)
    raw = slider.get_current_value()
    if descriptor.is_int:
        value_entry.set_text(str(int(round(raw))))
    else:
        value_entry.set_text(f"{raw:.3f}")


def _apply_entry_to_slider(
    entry: t.Any,
    registry: SliderRegistry,
) -> None:
    """Parse a finished text entry and update its linked slider.

    If the entry belongs to a slider's value entry, parse the text as a float
    (or int if the descriptor is flagged as integer), clamp to the slider's
    range, and call set_current_value on the slider. Also marks the slider
    dirty in the registry, so apply_widgets_to_config knows the user actually
    changed it (see config_bridge._apply_sliders).

    Args:
        entry: The UITextEntryLine that fired the finished event.
        registry (SliderRegistry): Registry with entry_to_slider reverse lookup.
    """
    slider = registry.entry_to_slider.get(entry)
    if slider is None:
        return
    descriptor = registry.sliders.get(slider)
    if descriptor is None:
        return

    text = entry.get_text().strip()
    try:
        raw = float(text)
    except ValueError:
        return

    registry.mark_dirty(descriptor.label)

    if descriptor.is_int:
        value = float(int(round(raw)))
    else:
        value = raw

    clamped = max(descriptor.min_val, min(descriptor.max_val, value))
    slider.set_current_value(clamped)

    if descriptor.is_int:
        entry.set_text(str(int(round(clamped))))
    else:
        entry.set_text(f"{clamped:.3f}")
