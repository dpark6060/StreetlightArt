"""Builds pygame-gui widget sections for each config category."""

import typing as t

import pygame
import pygame_gui

from ..config.schema import SceneConfig
from .config_bridge import get_nested
from .slider_registry import SliderDescriptor, SliderRegistry

# Layout constants
_ROW_HEIGHT = 50
_BOOL_BTN_HEIGHT = 35
_TEXT_ENTRY_HEIGHT = 35
_SCROLLBAR_GUTTER = 20
_LABEL_RATIO = 0.48
_SLIDER_RATIO = 0.33
# Value entry gets the remainder (1 - _LABEL_RATIO - _SLIDER_RATIO = 0.19)


def _compute_row_widths(container: t.Any) -> t.Tuple[int, int, int]:
    """Compute label, slider, and value-entry pixel widths from the container width.

    Subtracts a fixed scrollbar gutter from the container width before splitting
    proportionally, so long label strings are not clipped.

    Args:
        container: The scrollable container whose width drives the layout.

    Returns:
        tuple[int, int, int]: (label_width, slider_width, entry_width) in pixels.
    """
    usable = container.get_relative_rect().width - _SCROLLBAR_GUTTER
    label_width = int(usable * _LABEL_RATIO)
    slider_width = int(usable * _SLIDER_RATIO)
    entry_width = usable - label_width - slider_width
    return label_width, slider_width, entry_width


def build_atmosphere_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_offset: int,
    registry: SliderRegistry,
    config: SceneConfig,
) -> int:
    """Build slider widgets for atmosphere configuration.

    Adds sliders for the extinction coefficient (falloff_distance, which drives
    both light dimming and the airlight veil), scatter distance, the glow core
    and power-law-wing parameters, airlight intensity (haze_alpha), minimum
    brightness threshold, and illumination brightness; toggles for glow
    expansion and airlight (haze_enabled); and a text entry for the haze
    (airlight) color.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to add widgets to.
        y_offset (int): Starting Y position for the first widget row.
        registry (SliderRegistry): Registry to record all created widgets.
        config (SceneConfig): Current scene config for initial values.

    Returns:
        int: Total height consumed by this section.
    """
    y = y_offset
    sliders = [
        SliderDescriptor("falloff_distance", ("atmosphere", "falloff_distance"), 100.0, 50000.0),
        SliderDescriptor("scatter_distance", ("atmosphere", "scatter_distance"), 1000.0, 50000.0),
        SliderDescriptor("glow_max_alpha", ("atmosphere", "glow_max_alpha"), 0.0, 255.0, is_int=True),
        SliderDescriptor("glow_falloff_steepness", ("atmosphere", "glow_falloff_steepness"), 0.1, 5.0),
        SliderDescriptor("glow_wing_exponent", ("atmosphere", "glow_wing_exponent"), 1.0, 5.0),
        SliderDescriptor("glow_tail_strength", ("atmosphere", "glow_tail_strength"), 0.0, 1.0),
        SliderDescriptor("glow_cutoff_mult", ("atmosphere", "glow_cutoff_mult"), 2.0, 20.0),
        SliderDescriptor("haze_alpha", ("atmosphere", "haze_alpha"), 0.0, 255.0, is_int=True),
        SliderDescriptor("min_brightness_threshold", ("atmosphere", "min_brightness_threshold"), 0.0, 1.0),
        SliderDescriptor("illumination_brightness", ("atmosphere", "illumination_brightness"), 0.0, 255.0, is_int=True),
    ]

    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

    y = _add_bool_button_row(
        manager, container, y, registry, config,
        "glow_expansion_enabled", ("atmosphere", "glow_expansion_enabled"),
    )
    y = _add_bool_button_row(
        manager, container, y, registry, config,
        "haze_enabled", ("atmosphere", "haze_enabled"),
    )
    y = _add_text_entry_row(
        manager, container, y, registry,
        "haze_color", ("atmosphere", "haze_color"),
    )

    return y - y_offset


def build_devmode_camera_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_offset: int,
    registry: SliderRegistry,
    config: SceneConfig,
) -> int:
    """Build slider widgets for devmode camera configuration.

    Accesses config.devmode.camera. Callers must ensure config.devmode is not
    None before invoking this function.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to add widgets to.
        y_offset (int): Starting Y position for the first widget row.
        registry (SliderRegistry): Registry to record all created widgets.
        config (SceneConfig): Current scene config for initial values.
            config.devmode must not be None.

    Returns:
        int: Total height consumed by this section.
    """
    sliders = [
        SliderDescriptor("pos_x", ("devmode", "camera", "pos_x"), -5000.0, 15000.0),
        SliderDescriptor("pos_y", ("devmode", "camera", "pos_y"), -5000.0, 15000.0),
        SliderDescriptor("pos_z", ("devmode", "camera", "pos_z"), -5000.0, 15000.0),
        SliderDescriptor("target_x", ("devmode", "camera", "target_x"), -5000.0, 15000.0),
        SliderDescriptor("target_y", ("devmode", "camera", "target_y"), -5000.0, 15000.0),
        SliderDescriptor("target_z", ("devmode", "camera", "target_z"), -5000.0, 15000.0),
        SliderDescriptor("fov", ("devmode", "camera", "fov"), 10.0, 120.0),
    ]

    y = y_offset
    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

    return y - y_offset


# ---------------------------------------------------------------------------
# Internal widget builders
# ---------------------------------------------------------------------------


def _add_slider_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    descriptor: SliderDescriptor,
) -> int:
    """Create a label, slider, and value text entry for one config parameter row.

    Widths are computed proportionally from the container width so long label
    strings are not clipped. Row height is fixed at _ROW_HEIGHT (50px).

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to parent widgets to.
        y (int): Top Y coordinate for this row.
        registry (SliderRegistry): Registry to record the new slider.
        descriptor (SliderDescriptor): Slider configuration and binding.

    Returns:
        int: Y coordinate of the next row (y + _ROW_HEIGHT).
    """
    label_width, slider_width, entry_width = _compute_row_widths(container)
    row_y = y + (_ROW_HEIGHT - 20) // 2

    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, row_y, label_width, 20),
        text=descriptor.label,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)

    slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect(label_width, y + (_ROW_HEIGHT - 30) // 2, slider_width, 30),
        start_value=descriptor.min_val,
        value_range=(descriptor.min_val, descriptor.max_val),
        manager=manager,
        container=container,
    )

    entry_y = y + (_ROW_HEIGHT - 30) // 2
    value_entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(label_width + slider_width, entry_y, entry_width, 30),
        manager=manager,
        container=container,
    )

    registry.register_slider(slider, descriptor, value_entry)
    return y + _ROW_HEIGHT


def _add_bool_button_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    config: SceneConfig,
    label_text: str,
    config_path: tuple,
) -> int:
    """Create a label and ON/OFF toggle button for a boolean config field.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to parent widgets to.
        y (int): Top Y coordinate for this row.
        registry (SliderRegistry): Registry to record the new button.
        config (SceneConfig): Current scene config for reading initial value.
        label_text (str): Human-readable label for the boolean field.
        config_path (tuple): Path to the boolean field in SceneConfig.

    Returns:
        int: Y coordinate of the next row.
    """
    label_width, slider_width, entry_width = _compute_row_widths(container)
    initial_value: bool = bool(get_nested(config, config_path))
    row_y = y + (_ROW_HEIGHT - _BOOL_BTN_HEIGHT) // 2

    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, row_y, label_width, _BOOL_BTN_HEIGHT),
        text=label_text,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)

    button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(label_width, row_y, 80, _BOOL_BTN_HEIGHT),
        text="ON" if initial_value else "OFF",
        manager=manager,
        container=container,
    )

    registry.register_bool(button, config_path, initial_value)
    return y + _ROW_HEIGHT


def _add_text_entry_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    label_text: str,
    config_path: tuple,
) -> int:
    """Create a label and text entry field for a string or color config field.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to parent widgets to.
        y (int): Top Y coordinate for this row.
        registry (SliderRegistry): Registry to record the new entry.
        label_text (str): Human-readable label for the field.
        config_path (tuple): Path to the field in SceneConfig.

    Returns:
        int: Y coordinate of the next row.
    """
    label_width, slider_width, entry_width = _compute_row_widths(container)
    row_y = y + (_ROW_HEIGHT - _TEXT_ENTRY_HEIGHT) // 2

    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, row_y, label_width, _TEXT_ENTRY_HEIGHT),
        text=label_text,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)

    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(label_width, row_y, slider_width + entry_width, _TEXT_ENTRY_HEIGHT),
        manager=manager,
        container=container,
    )

    registry.register_text(entry, config_path)
    return y + _ROW_HEIGHT
