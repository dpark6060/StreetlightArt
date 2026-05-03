"""Builds pygame-gui widget sections for each config category."""

import typing as t

import pygame
import pygame_gui

from ..config.schema import FixturePoolEntry, SceneConfig
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

    Adds sliders for falloff, max distance, scatter, glow parameters,
    minimum brightness threshold, illumination brightness, a toggle button for
    glow expansion, and a text entry for haze color.

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
        SliderDescriptor("max_distance", ("atmosphere", "max_distance"), 1000.0, 100000.0),
        SliderDescriptor("scatter_distance", ("atmosphere", "scatter_distance"), 1000.0, 50000.0),
        SliderDescriptor("glow_max_alpha", ("atmosphere", "glow_max_alpha"), 0.0, 255.0, is_int=True),
        SliderDescriptor("glow_falloff_steepness", ("atmosphere", "glow_falloff_steepness"), 0.1, 5.0),
        SliderDescriptor("min_brightness_threshold", ("atmosphere", "min_brightness_threshold"), 0.0, 1.0),
        SliderDescriptor("illumination_brightness", ("atmosphere", "illumination_brightness"), 0.0, 255.0, is_int=True),
    ]

    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

    y = _add_bool_button_row(
        manager, container, y, registry, config,
        "glow_expansion_enabled", ("atmosphere", "glow_expansion_enabled"),
    )
    y = _add_text_entry_row(
        manager, container, y, registry,
        "haze_color", ("atmosphere", "haze_color"),
    )

    return y - y_offset


def build_streetlight_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_offset: int,
    registry: SliderRegistry,
    config: SceneConfig,
    road_type_name: str,
    road_type_path_key: str,
) -> int:
    """Build slider widgets for a single streetlight road-type configuration.

    Targets the first FixturePoolEntry that has a non-None profile. If no
    entry with a profile exists, a placeholder label is shown instead.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to add widgets to.
        y_offset (int): Starting Y position for the first widget row.
        registry (SliderRegistry): Registry to record all created widgets.
        config (SceneConfig): Current scene config for initial values.
        road_type_name (str): Display name for the road type (e.g. "Highway").
        road_type_path_key (str): Attribute name on StreetLightConfig (e.g. "highway").

    Returns:
        int: Total height consumed by this section.
    """
    entries: t.List[FixturePoolEntry] = getattr(
        config.lights.streetlights, road_type_path_key
    )
    profile_index = _find_first_profile_index(entries)

    if profile_index is None:
        label_width, slider_width, entry_width = _compute_row_widths(container)
        label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(0, y_offset, label_width + slider_width + entry_width, 30),
            text=f"No fixture profiles for {road_type_name}",
            manager=manager,
            container=container,
        )
        registry.extra_widgets.append(label)
        return 30

    base_path = ("lights", "streetlights", road_type_path_key, profile_index, "profile")
    all_profile_indices = _get_all_profile_indices(entries)
    extra_indices = [i for i in all_profile_indices if i != profile_index]

    def _make_extra_paths(field_name: str) -> t.List[tuple]:
        """Build extra config paths for field_name across all non-primary profile entries."""
        return [
            ("lights", "streetlights", road_type_path_key, i, "profile", field_name)
            for i in extra_indices
        ]

    sliders = [
        SliderDescriptor("glow_radius", base_path + ("glow_radius",), 0.0, 100.0,
                         extra_config_paths=_make_extra_paths("glow_radius")),
        SliderDescriptor("illumination_radius", base_path + ("illumination_radius",), 0.0, 200.0,
                         extra_config_paths=_make_extra_paths("illumination_radius")),
        SliderDescriptor("illumination_intensity", base_path + ("illumination_intensity",), 0.0, 5.0,
                         extra_config_paths=_make_extra_paths("illumination_intensity")),
        SliderDescriptor("illumination_falloff", base_path + ("illumination_falloff",), 0.5, 6.0,
                         extra_config_paths=_make_extra_paths("illumination_falloff")),
        SliderDescriptor("illumination_inner_fraction", base_path + ("illumination_inner_fraction",), 0.0, 1.0,
                         extra_config_paths=_make_extra_paths("illumination_inner_fraction")),
        SliderDescriptor("brightness_min", base_path + ("brightness_min",), 0.0, 1.0,
                         extra_config_paths=_make_extra_paths("brightness_min")),
        SliderDescriptor("brightness_max", base_path + ("brightness_max",), 0.0, 1.0,
                         extra_config_paths=_make_extra_paths("brightness_max")),
        SliderDescriptor("height_min", base_path + ("height_min",), 1.0, 30.0,
                         extra_config_paths=_make_extra_paths("height_min")),
        SliderDescriptor("height_max", base_path + ("height_max",), 1.0, 30.0,
                         extra_config_paths=_make_extra_paths("height_max")),
        SliderDescriptor("spacing_min", base_path + ("spacing_min",), 10.0, 200.0,
                         extra_config_paths=_make_extra_paths("spacing_min")),
        SliderDescriptor("spacing_max", base_path + ("spacing_max",), 10.0, 200.0,
                         extra_config_paths=_make_extra_paths("spacing_max")),
    ]

    y = y_offset
    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

    return y - y_offset


def build_cars_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_offset: int,
    registry: SliderRegistry,
    config: SceneConfig,
) -> int:
    """Build slider widgets for car light configuration.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to add widgets to.
        y_offset (int): Starting Y position for the first widget row.
        registry (SliderRegistry): Registry to record all created widgets.
        config (SceneConfig): Current scene config for initial values.

    Returns:
        int: Total height consumed by this section.
    """
    sliders = [
        SliderDescriptor("separation_min", ("lights", "cars", "separation_min"), 0.5, 5.0),
        SliderDescriptor("separation_max", ("lights", "cars", "separation_max"), 0.5, 5.0),
        SliderDescriptor("headlight_brightness_min", ("lights", "cars", "headlight_brightness_min"), 0.0, 1.0),
        SliderDescriptor("headlight_brightness_max", ("lights", "cars", "headlight_brightness_max"), 0.0, 1.0),
        SliderDescriptor("headlight_glow_radius", ("lights", "cars", "headlight_glow_radius"), 0.0, 30.0),
        SliderDescriptor("headlight_illum_radius", ("lights", "cars", "headlight_illumination_radius"), 0.0, 50.0),
        SliderDescriptor("headlight_illum_intensity", ("lights", "cars", "headlight_illumination_intensity"), 0.0, 5.0),
        SliderDescriptor("headlight_illum_falloff", ("lights", "cars", "headlight_illumination_falloff"), 0.5, 6.0),
        SliderDescriptor("headlight_illum_inner", ("lights", "cars", "headlight_illumination_inner_fraction"), 0.0, 1.0),
        SliderDescriptor("taillight_brightness_min", ("lights", "cars", "taillight_brightness_min"), 0.0, 1.0),
        SliderDescriptor("taillight_brightness_max", ("lights", "cars", "taillight_brightness_max"), 0.0, 1.0),
        SliderDescriptor("taillight_glow_radius", ("lights", "cars", "taillight_glow_radius"), 0.0, 15.0),
        SliderDescriptor("taillight_illum_radius", ("lights", "cars", "taillight_illumination_radius"), 0.0, 20.0),
        SliderDescriptor("taillight_illum_intensity", ("lights", "cars", "taillight_illumination_intensity"), 0.0, 5.0),
        SliderDescriptor("taillight_illum_falloff", ("lights", "cars", "taillight_illumination_falloff"), 0.5, 6.0),
        SliderDescriptor("taillight_illum_inner", ("lights", "cars", "taillight_illumination_inner_fraction"), 0.0, 1.0),
    ]

    y = y_offset
    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

    return y - y_offset


def build_house_lights_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_offset: int,
    registry: SliderRegistry,
    config: SceneConfig,
) -> int:
    """Build slider widgets for house light configuration.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to add widgets to.
        y_offset (int): Starting Y position for the first widget row.
        registry (SliderRegistry): Registry to record all created widgets.
        config (SceneConfig): Current scene config for initial values.

    Returns:
        int: Total height consumed by this section.
    """
    sliders = [
        SliderDescriptor("brightness_min", ("lights", "house_lights", "brightness_min"), 0.0, 1.0),
        SliderDescriptor("brightness_max", ("lights", "house_lights", "brightness_max"), 0.0, 1.0),
        SliderDescriptor("glow_radius", ("lights", "house_lights", "glow_radius"), 0.0, 50.0),
        SliderDescriptor("illumination_radius", ("lights", "house_lights", "illumination_radius"), 0.0, 100.0),
        SliderDescriptor("illumination_intensity", ("lights", "house_lights", "illumination_intensity"), 0.0, 5.0),
        SliderDescriptor("illumination_falloff", ("lights", "house_lights", "illumination_falloff"), 0.5, 6.0),
        SliderDescriptor("illumination_inner_fraction", ("lights", "house_lights", "illumination_inner_fraction"), 0.0, 1.0),
    ]

    y = y_offset
    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

    return y - y_offset


def build_commercial_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_offset: int,
    registry: SliderRegistry,
    config: SceneConfig,
    label: str,
    prefix_path_key: str,
) -> int:
    """Build slider widgets for a commercial light sub-type.

    Handles sign, parking, and storefront sub-sections of CommercialLightConfig.
    Field names are derived by prepending the prefix key (e.g. "sign_brightness_min").

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to add widgets to.
        y_offset (int): Starting Y position for the first widget row.
        registry (SliderRegistry): Registry to record all created widgets.
        config (SceneConfig): Current scene config for initial values.
        label (str): Display prefix for this section (e.g. "Sign").
        prefix_path_key (str): Config field prefix (e.g. "sign", "parking", "storefront").

    Returns:
        int: Total height consumed by this section.
    """
    p = prefix_path_key
    sliders = [
        SliderDescriptor("brightness_min", ("lights", "commercial", f"{p}_brightness_min"), 0.0, 1.0),
        SliderDescriptor("brightness_max", ("lights", "commercial", f"{p}_brightness_max"), 0.0, 1.0),
        SliderDescriptor("glow_radius", ("lights", "commercial", f"{p}_glow_radius"), 0.0, 100.0),
        SliderDescriptor("illumination_radius", ("lights", "commercial", f"{p}_illumination_radius"), 0.0, 200.0),
        SliderDescriptor("illumination_intensity", ("lights", "commercial", f"{p}_illumination_intensity"), 0.0, 5.0),
        SliderDescriptor("illumination_falloff", ("lights", "commercial", f"{p}_illumination_falloff"), 0.5, 6.0),
        SliderDescriptor("illumination_inner_fraction", ("lights", "commercial", f"{p}_illumination_inner_fraction"), 0.0, 1.0),
    ]

    y = y_offset
    for descriptor in sliders:
        y = _add_slider_row(manager, container, y, registry, descriptor)

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


def _find_first_profile_index(entries: t.List[FixturePoolEntry]) -> t.Optional[int]:
    """Return the index of the first FixturePoolEntry with a non-None profile.

    Args:
        entries (list[FixturePoolEntry]): Pool entries to search.

    Returns:
        int or None: Index of first entry with a profile, or None if all are empty.
    """
    for i, entry in enumerate(entries):
        if entry.profile is not None:
            return i
    return None


def _get_all_profile_indices(entries: t.List[FixturePoolEntry]) -> t.List[int]:
    """Return the indices of all FixturePoolEntry items with a non-None profile.

    Args:
        entries (list[FixturePoolEntry]): Pool entries to search.

    Returns:
        list[int]: Indices of all entries that have a profile. Empty if none exist.
    """
    return [i for i, entry in enumerate(entries) if entry.profile is not None]
