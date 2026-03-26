"""Main event loop for the devmode parameter tuning tool."""

import argparse
import typing as t
from pathlib import Path

import numpy as np
import pygame
import pygame_gui
from PIL import Image

from ..config.loader import _build_devmode_camera_config, _build_devmode_config
from ..config.schema import SceneConfig
from ..rendering.ground_pass import render_ground
from ..rendering.illumination_pass import render_illumination
from ..rendering.light_pass import render_lights_export
from ..rendering.pillow_backend import PillowBackend
from ..rendering.render_pipeline import render_scene_frame
from ..rendering.road_pass import render_roads_export
from .config_bridge import apply_widgets_to_config, read_config_to_widgets
from .config_writer import write_config
from .control_panel import build_control_panel, rebuild_section
from .label_pass import make_label_pass
from .slider_registry import SliderRegistry
from .synthetic_scene import ROW_LABELS, build_synthetic_scene

_WINDOW_WIDTH = 1280
_WINDOW_HEIGHT = 720
_PANEL_WIDTH = 384
_RENDER_WIDTH = _WINDOW_WIDTH - _PANEL_WIDTH
_SCROLL_ZOOM_FACTOR: float = 0.1


def cmd_devmode(args: argparse.Namespace, config: SceneConfig) -> None:
    """Run the devmode parameter tuning tool.

    Initialises pygame and pygame-gui, builds a synthetic test scene, renders
    it using the Pillow backend, and displays it alongside an interactive
    control panel. The user can adjust config parameters via sliders, re-render
    on demand, and save the modified config to a TOML file.

    Args:
        args (argparse.Namespace): Parsed CLI args (carries --config path).
        config (SceneConfig): Loaded scene configuration.
    """
    _ensure_devmode_config(config)

    pygame.init()
    window = pygame.display.set_mode((_WINDOW_WIDTH, _WINDOW_HEIGHT))
    pygame.display.set_caption("streets-at-night devmode")
    manager = pygame_gui.UIManager((_WINDOW_WIDTH, _WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    panel_rect = pygame.Rect(_RENDER_WIDTH, 0, _PANEL_WIDTH, _WINDOW_HEIGHT)
    scene = build_synthetic_scene(config)
    render_surface = _render_and_convert(scene, config)

    registry, scroll, rerender_btn, save_btn, section_list = build_control_panel(
        manager, panel_rect, config
    )
    read_config_to_widgets(config, registry)

    active_section = "Atmosphere"
    save_entry: t.Optional[pygame_gui.elements.UITextEntryLine] = None
    save_confirm_btn: t.Optional[pygame_gui.elements.UIButton] = None

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == rerender_btn:
                    render_surface = _handle_rerender(scene, config, registry)
                    scene = build_synthetic_scene(config)
                    render_surface = _render_and_convert(scene, config)

                elif event.ui_element == save_btn:
                    save_entry, save_confirm_btn = _show_save_dialog(manager)

                elif save_confirm_btn is not None and event.ui_element == save_confirm_btn:
                    _handle_save(config, save_entry, save_confirm_btn)
                    save_entry = None
                    save_confirm_btn = None

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

            if event.type == pygame.MOUSEWHEEL:
                _handle_scroll_zoom(scene, event.y)
                render_surface = _render_and_convert(scene, config)

            manager.process_events(event)

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


def _render_and_convert(scene: t.Any, config: SceneConfig) -> pygame.Surface:
    """Render the synthetic scene and return a pygame surface for display.

    Args:
        scene: The synthetic scene to render.
        config (SceneConfig): Current scene configuration.

    Returns:
        pygame.Surface: The rendered frame as a pygame surface.
    """
    image = _render_scene(scene, config)
    return _pil_to_pygame(image)


def _handle_rerender(
    scene: t.Any,
    config: SceneConfig,
    registry: SliderRegistry,
) -> pygame.Surface:
    """Apply widget values to config, rebuild the scene, and re-render.

    Args:
        scene: The current synthetic scene (unused — rebuilt inside).
        config (SceneConfig): Config to update from widgets.
        registry (SliderRegistry): Current slider registry.

    Returns:
        pygame.Surface: The newly rendered frame.
    """
    apply_widgets_to_config(registry, config)
    new_scene = build_synthetic_scene(config)
    image = _render_scene(new_scene, config)
    return _pil_to_pygame(image)


def _render_scene(scene: t.Any, config: SceneConfig) -> Image.Image:
    """Render the synthetic scene using the Pillow backend and return the image.

    Resolves car facing direction before rendering so car lights have non-zero
    brightness and glow radius. Uses the dark export color palette and standard
    widths. Pass order:
      1. render_ground — sky/ground fill and horizon glow
      2. render_roads_export — dark export colors and standard widths
      3. render_illumination — soft ground light pools
      4. render_lights_export — cores and glow halos
    Appends a label pass to identify each scene row.

    Args:
        scene: The synthetic scene to render.
        config (SceneConfig): Current scene configuration.

    Returns:
        PIL.Image.Image: The rendered frame.
    """
    for car_pair in scene.car_pairs:
        car_pair.resolve_facing(scene.camera.get_forward())
    backend = PillowBackend(_RENDER_WIDTH, _WINDOW_HEIGHT)
    passes = [
        render_ground,
        render_roads_export,
        render_illumination,
        render_lights_export,
    ]
    label_pass = make_label_pass(ROW_LABELS, scene.camera)
    passes.append(label_pass)
    render_scene_frame(scene, backend, passes=passes)
    return backend.get_image()


def _pil_to_pygame(image: Image.Image) -> pygame.Surface:
    """Convert a PIL Image to a pygame Surface.

    Args:
        image (PIL.Image.Image): The image to convert.

    Returns:
        pygame.Surface: Equivalent pygame surface in RGB mode.
    """
    rgb_image = image.convert("RGB")
    raw = rgb_image.tobytes()
    return pygame.image.fromstring(raw, rgb_image.size, "RGB")


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


def _handle_save(
    config: SceneConfig,
    save_entry: t.Optional[pygame_gui.elements.UITextEntryLine],
    save_confirm_btn: t.Optional[pygame_gui.elements.UIButton],
) -> None:
    """Write config to the file path specified in the save entry widget.

    Kills the entry and confirm button after saving.

    Args:
        config (SceneConfig): Config to serialize.
        save_entry: Text entry holding the file path string.
        save_confirm_btn: Confirm button to kill after save.
    """
    if save_entry is None:
        return
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

    Reads the current state from registry.bool_states, inverts it, stores the
    new state back, and updates the button text to "ON" or "OFF".

    Args:
        button (UIButton): The toggle button to flip.
        registry (SliderRegistry): Registry holding bool_states for all buttons.
    """
    current = registry.bool_states.get(button, False)
    new_value = not current
    registry.bool_states[button] = new_value
    button.set_text("ON" if new_value else "OFF")


def _handle_scroll_zoom(scene: t.Any, scroll_direction: int) -> None:
    """Move the camera along its forward vector in response to scroll input.

    Step size is proportional to the current camera-to-target distance so
    zoom speed feels consistent at any altitude.

    Args:
        scene: The scene containing the camera to adjust.
        scroll_direction (int): Positive for zoom in, negative for zoom out.
    """
    camera = scene.camera
    forward = camera.get_forward()
    distance = float(np.linalg.norm(camera.target - camera.position))
    step = distance * _SCROLL_ZOOM_FACTOR * scroll_direction
    camera.position = camera.position + forward * step


def _update_value_entry(
    slider: t.Any,
    registry: SliderRegistry,
) -> None:
    """Update the value text entry for a slider that was just moved.

    Args:
        slider: The UIHorizontalSlider that fired the moved event.
        registry (SliderRegistry): Registry mapping sliders to descriptors and entries.
    """
    descriptor = registry.sliders.get(slider)
    value_entry = registry.value_entries.get(slider)
    if descriptor is None or value_entry is None:
        return
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
    range, and call set_current_value on the slider.

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
