"""Creates and lays out the full devmode control panel."""

import typing as t
from dataclasses import dataclass

import pygame
import pygame_gui

from ..config.schema import SceneConfig
from .config_bridge import read_config_to_widgets
from .panel_sections import (
    build_atmosphere_section,
    build_cars_section,
    build_commercial_section,
    build_devmode_camera_section,
    build_house_lights_section,
    build_streetlight_section,
)
from .slider_registry import SliderRegistry

_SECTION_NAMES = [
    "Atmosphere",
    "Streetlights: Highway",
    "Streetlights: Major Road",
    "Streetlights: Collector",
    "Streetlights: Residential",
    "Streetlights: Service",
    "Cars",
    "House Lights",
    "Sign Lights",
    "Parking Lights",
    "Storefront Lights",
    "Devmode Camera",
]

_STREETLIGHT_ROAD_TYPES = {
    "Streetlights: Highway":     ("Highway",    "highway"),
    "Streetlights: Major Road":  ("Major Road", "major_road"),
    "Streetlights: Collector":   ("Collector",  "collector"),
    "Streetlights: Residential": ("Residential","residential"),
    "Streetlights: Service":     ("Service",    "service"),
}

_COMMERCIAL_TYPES = {
    "Sign Lights":       ("Sign",      "sign"),
    "Parking Lights":    ("Parking",   "parking"),
    "Storefront Lights": ("Storefront","storefront"),
}

_SELECTION_HEIGHT = 120
_BUTTON_AREA_HEIGHT = 80


@dataclass
class ControlPanel:
    """Widgets making up the full devmode control panel.

    Attributes:
        registry: Registry for the initially-displayed section.
        scroll: The scroll area for sliders.
        rerender_btn: The "Re-render" button.
        save_btn: The "Save Config" button.
        load_btn: The "Load Config" button.
        section_list: The section navigation list.
    """

    registry: SliderRegistry
    scroll: pygame_gui.elements.UIScrollingContainer
    rerender_btn: pygame_gui.elements.UIButton
    save_btn: pygame_gui.elements.UIButton
    load_btn: pygame_gui.elements.UIButton
    section_list: pygame_gui.elements.UISelectionList


def build_control_panel(
    manager: pygame_gui.UIManager,
    panel_rect: pygame.Rect,
    config: SceneConfig,
) -> ControlPanel:
    """Create and lay out the full devmode control panel.

    Layout (top to bottom within panel_rect):
        - 120px  UISelectionList for section navigation.
        - Middle UIScrollingContainer holding all parameter sliders.
        - 80px   Three equal-width action buttons (Re-render, Save Config, Load Config).

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        panel_rect (pygame.Rect): Bounding rectangle for the entire panel.
        config (SceneConfig): Current scene configuration.

    Returns:
        ControlPanel: The registry and widgets making up the panel.
    """
    section_list = _build_section_list(manager, panel_rect)
    scroll = _build_scroll_container(manager, panel_rect)
    rerender_btn, save_btn, load_btn = _build_action_buttons(manager, panel_rect)

    registry = SliderRegistry()
    _build_section_widgets(manager, scroll, registry, config, _SECTION_NAMES[0])

    return ControlPanel(
        registry=registry,
        scroll=scroll,
        rerender_btn=rerender_btn,
        save_btn=save_btn,
        load_btn=load_btn,
        section_list=section_list,
    )


def rebuild_section(
    manager: pygame_gui.UIManager,
    scroll: pygame_gui.elements.UIScrollingContainer,
    registry: SliderRegistry,
    config: SceneConfig,
    section_name: str,
) -> None:
    """Kill all current section widgets, clear the registry, and build the new section.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        scroll (UIScrollingContainer): The scroll container holding section widgets.
        registry (SliderRegistry): Registry to clear and repopulate.
        config (SceneConfig): Current scene configuration.
        section_name (str): The name of the section to build (must be in _SECTION_NAMES).
    """
    _kill_registry_widgets(registry)
    _clear_registry(registry)
    _build_section_widgets(manager, scroll, registry, config, section_name)
    read_config_to_widgets(config, registry)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_section_list(
    manager: pygame_gui.UIManager,
    panel_rect: pygame.Rect,
) -> pygame_gui.elements.UISelectionList:
    """Create the section navigation selection list.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        panel_rect (pygame.Rect): Panel bounding rectangle.

    Returns:
        UISelectionList: The created selection list.
    """
    return pygame_gui.elements.UISelectionList(
        relative_rect=pygame.Rect(
            panel_rect.x, panel_rect.y,
            panel_rect.width, _SELECTION_HEIGHT,
        ),
        item_list=_SECTION_NAMES,
        manager=manager,
    )


def _build_scroll_container(
    manager: pygame_gui.UIManager,
    panel_rect: pygame.Rect,
) -> pygame_gui.elements.UIScrollingContainer:
    """Create the scrollable container for parameter sliders.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        panel_rect (pygame.Rect): Panel bounding rectangle.

    Returns:
        UIScrollingContainer: The created scroll container.
    """
    scroll_y = panel_rect.y + _SELECTION_HEIGHT
    scroll_height = panel_rect.height - _SELECTION_HEIGHT - _BUTTON_AREA_HEIGHT
    return pygame_gui.elements.UIScrollingContainer(
        relative_rect=pygame.Rect(
            panel_rect.x, scroll_y,
            panel_rect.width, scroll_height,
        ),
        manager=manager,
    )


def _build_action_buttons(
    manager: pygame_gui.UIManager,
    panel_rect: pygame.Rect,
) -> t.Tuple[
    pygame_gui.elements.UIButton,
    pygame_gui.elements.UIButton,
    pygame_gui.elements.UIButton,
]:
    """Create the Re-render, Save Config, and Load Config action buttons.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        panel_rect (pygame.Rect): Panel bounding rectangle.

    Returns:
        tuple[UIButton, UIButton, UIButton]: (rerender_button, save_button, load_button).
    """
    btn_y = panel_rect.y + panel_rect.height - _BUTTON_AREA_HEIGHT + 10
    btn_width = panel_rect.width // 3

    rerender_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(panel_rect.x, btn_y, btn_width, 35),
        text="Re-render",
        manager=manager,
    )
    save_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(panel_rect.x + btn_width, btn_y, btn_width, 35),
        text="Save Config",
        manager=manager,
    )
    load_btn = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(panel_rect.x + btn_width * 2, btn_y, btn_width, 35),
        text="Load Config",
        manager=manager,
    )
    return rerender_btn, save_btn, load_btn


def _build_section_widgets(
    manager: pygame_gui.UIManager,
    scroll: pygame_gui.elements.UIScrollingContainer,
    registry: SliderRegistry,
    config: SceneConfig,
    section_name: str,
) -> None:
    """Dispatch to the correct section builder and update the scroll area height.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        scroll (UIScrollingContainer): The scroll container for new widgets.
        registry (SliderRegistry): Registry to populate with new widgets.
        config (SceneConfig): Current scene configuration.
        section_name (str): Name of the section to build.
    """
    container = scroll.scrollable_container
    height = _dispatch_section_builder(manager, container, registry, config, section_name)
    content_width = scroll.get_container().get_size()[0]
    scroll.set_scrollable_area_dimensions((content_width, max(height, 10)))


def _dispatch_section_builder(
    manager: pygame_gui.UIManager,
    container: t.Any,
    registry: SliderRegistry,
    config: SceneConfig,
    section_name: str,
) -> int:
    """Call the appropriate section builder for the given section name.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container for widget parenting.
        registry (SliderRegistry): Registry to populate.
        config (SceneConfig): Current scene configuration.
        section_name (str): Name of the section to build.

    Returns:
        int: Total height (in pixels) consumed by the new section widgets.
    """
    if section_name == "Atmosphere":
        return build_atmosphere_section(manager, container, 0, registry, config)

    if section_name in _STREETLIGHT_ROAD_TYPES:
        road_type_name, road_type_key = _STREETLIGHT_ROAD_TYPES[section_name]
        return build_streetlight_section(
            manager, container, 0, registry, config, road_type_name, road_type_key
        )

    if section_name == "Cars":
        return build_cars_section(manager, container, 0, registry, config)

    if section_name == "House Lights":
        return build_house_lights_section(manager, container, 0, registry, config)

    if section_name in _COMMERCIAL_TYPES:
        label, prefix_key = _COMMERCIAL_TYPES[section_name]
        return build_commercial_section(
            manager, container, 0, registry, config, label, prefix_key
        )

    if section_name == "Devmode Camera":
        return build_devmode_camera_section(manager, container, 0, registry, config)

    return 0


def _kill_registry_widgets(registry: SliderRegistry) -> None:
    """Call .kill() on every widget tracked by the registry.

    Args:
        registry (SliderRegistry): Registry whose widgets should be destroyed.
    """
    for widget in registry.all_widgets():
        widget.kill()


def _clear_registry(registry: SliderRegistry) -> None:
    """Empty all mapping dicts, and the dirty set, on the registry.

    Args:
        registry (SliderRegistry): Registry to clear.
    """
    registry.sliders.clear()
    registry.value_entries.clear()
    registry.entry_to_slider.clear()
    registry.bool_buttons.clear()
    registry.text_entries.clear()
    registry.extra_widgets.clear()
    registry.dirty.clear()
