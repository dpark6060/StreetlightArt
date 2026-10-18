"""Reflection-driven property editor for a single LightKind.

Builds slider / toggle / text-entry rows for every editable field of a light
config dataclass, using ``dataclasses.fields`` to discover them and
``field_meta`` to pick slider ranges. Reuses ``SliderRegistry`` so the tab's
slider/entry/toggle sync handlers work identically for dynamic and hand-written
sections.

config_path on each descriptor is RELATIVE to ``kind.target`` (a single-element
tuple), not to SceneConfig — the tab passes ``kind.target`` as the root object
to ``config_bridge`` for dynamic sections. Descriptors deliberately do NOT set
``extra_config_paths``: each pool entry gets its own section here, which is the
"better fix" the mix-flattening footgun comment in ``config_bridge`` defers to.
"""

import dataclasses
import typing as t

import pygame
import pygame_gui

from ..config.schema import ColorPoolConfig
from .field_meta import get_field_meta, get_field_unit
from .light_catalog import LightKind
from .slider_registry import SliderDescriptor, SliderRegistry

# Layout constants (mirrors panel_sections so dynamic rows line up with the
# hand-written Atmosphere / Devmode Camera sections).
_ROW_HEIGHT = 50
_LABEL_HEIGHT = 20
_SLIDER_HEIGHT = 30
_TEXT_HEIGHT = 30
_BTN_HEIGHT = 35
_SKIP_HEIGHT = 30
_SCROLLBAR_GUTTER = 20
_LABEL_RATIO = 0.48
_SLIDER_RATIO = 0.33


def build_dynamic_section(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y_start: int,
    registry: SliderRegistry,
    kind: LightKind,
) -> int:
    """Create widgets for every editable field of kind.target; return height used.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container to parent widgets to.
        y_start (int): Starting Y position for the first row.
        registry (SliderRegistry): Registry to record all created widgets.
        kind (LightKind): The light kind whose target dataclass is reflected.

    Returns:
        int: Total pixel height consumed by this section.
    """
    label_width, slider_width, entry_width = _compute_row_widths(container)
    y = y_start
    for field in dataclasses.fields(type(kind.target)):
        if not _field_matches_prefix(field.name, kind.field_prefix):
            continue
        value = getattr(kind.target, field.name)
        display = _display_name(field.name, kind.field_prefix)
        y = _add_field_row(
            manager, container, y, registry, kind, field.name, value, display,
            label_width, slider_width, entry_width,
        )
    return y - y_start


def _field_matches_prefix(field_name: str, prefix: t.Optional[str]) -> bool:
    """Return True when the field should be shown for the given prefix.

    Args:
        field_name (str): Dataclass field name.
        prefix (str | None): Kind prefix, or None to show every field.

    Returns:
        bool: True if the field passes the prefix filter.
    """
    if prefix is None:
        return True
    return field_name.startswith(prefix)


def _display_name(field_name: str, prefix: t.Optional[str]) -> str:
    """Build a human label: strip prefix, title-case, append the unit suffix.

    The unit is looked up from the full field name via field_meta.get_field_unit.
    A trailing name token that spells the unit ("Color Temp K", "Beam Angle
    Deg") is dropped before the suffix is appended, so units are never doubled.

    Args:
        field_name (str): Dataclass field name.
        prefix (str | None): Prefix to strip, or None.

    Returns:
        str: Display label (e.g. "Glow Radius (m)", "Color Temp (K)").
    """
    name = field_name
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    label = name.replace("_", " ").title()
    unit = get_field_unit(field_name)
    if not unit:
        return label
    if label.lower().endswith(f" {unit.lower()}"):
        label = label[: -(len(unit) + 1)]
    return f"{label} ({unit})"


def _add_field_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    kind: LightKind,
    field_name: str,
    value: t.Any,
    display: str,
    label_width: int,
    slider_width: int,
    entry_width: int,
) -> int:
    """Dispatch one field to the slider/bool/text/skip builder by its value type.

    bool is checked before int/float because ``bool`` is a subclass of ``int``.
    ColorPoolConfig fields get an editable hex + weight row per colour.
    Lists and other nested dataclasses are skipped with a visible
    "(colors: edit TOML)" label so the omission is not silent.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container.
        y (int): Top Y coordinate for this row.
        registry (SliderRegistry): Registry to record widgets.
        kind (LightKind): Owning kind (for the unique descriptor label).
        field_name (str): Dataclass field name.
        value: Current field value (drives the widget type).
        display (str): Human display label.
        label_width (int): Label column width.
        slider_width (int): Slider column width.
        entry_width (int): Value-entry column width.

    Returns:
        int: Y coordinate of the next row.
    """
    if isinstance(value, bool):
        return _add_bool_row(manager, container, y, registry, field_name, value, display, label_width)
    if isinstance(value, (int, float)):
        return _add_slider_row(
            manager, container, y, registry, kind, field_name, float(value), display,
            label_width, slider_width, entry_width,
        )
    if isinstance(value, str):
        return _add_text_row(manager, container, y, registry, field_name, display, label_width, slider_width, entry_width)
    if isinstance(value, ColorPoolConfig):
        return _add_color_pool_rows(
            manager, container, y, registry, field_name, value, display,
            label_width, slider_width, entry_width,
        )
    return _add_skip_label(manager, container, y, registry, display, label_width, slider_width, entry_width)


def _add_slider_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    kind: LightKind,
    field_name: str,
    value: float,
    display: str,
    label_width: int,
    slider_width: int,
    entry_width: int,
) -> int:
    """Create a label + slider + value-entry row for a numeric field.

    The descriptor label is unique per section (``f"{kind.label}:{field_name}"``)
    so the dirty set does not collide across sections; the displayed text is the
    pretty ``display`` name. config_path is relative to kind.target.

    Returns:
        int: Y coordinate of the next row.
    """
    meta = get_field_meta(field_name, value)
    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, y + (_ROW_HEIGHT - _LABEL_HEIGHT) // 2, label_width, _LABEL_HEIGHT),
        text=display,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)

    slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect(
            label_width, y + (_ROW_HEIGHT - _SLIDER_HEIGHT) // 2, slider_width, _SLIDER_HEIGHT
        ),
        start_value=meta.min_val,
        value_range=(meta.min_val, meta.max_val),
        manager=manager,
        container=container,
    )
    value_entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(
            label_width + slider_width, y + (_ROW_HEIGHT - _SLIDER_HEIGHT) // 2, entry_width, _SLIDER_HEIGHT
        ),
        manager=manager,
        container=container,
    )

    descriptor = SliderDescriptor(
        label=f"{kind.label}:{field_name}",
        config_path=(field_name,),
        min_val=meta.min_val,
        max_val=meta.max_val,
        is_int=meta.is_int,
    )
    registry.register_slider(slider, descriptor, value_entry)
    return y + _ROW_HEIGHT


def _add_bool_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    field_name: str,
    value: bool,
    display: str,
    label_width: int,
) -> int:
    """Create a label + ON/OFF toggle row for a boolean field.

    Returns:
        int: Y coordinate of the next row.
    """
    row_y = y + (_ROW_HEIGHT - _BTN_HEIGHT) // 2
    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, row_y, label_width, _BTN_HEIGHT),
        text=display,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)

    button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect(label_width, row_y, 80, _BTN_HEIGHT),
        text="ON" if value else "OFF",
        manager=manager,
        container=container,
    )
    registry.register_bool(button, (field_name,), value)
    return y + _ROW_HEIGHT


def _add_text_row(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    field_name: str,
    display: str,
    label_width: int,
    slider_width: int,
    entry_width: int,
) -> int:
    """Create a label + text-entry row for a string field.

    Returns:
        int: Y coordinate of the next row.
    """
    row_y = y + (_ROW_HEIGHT - _TEXT_HEIGHT) // 2
    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, row_y, label_width, _TEXT_HEIGHT),
        text=display,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)

    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(label_width, row_y, slider_width + entry_width, _TEXT_HEIGHT),
        manager=manager,
        container=container,
    )
    registry.register_text(entry, (field_name,))
    return y + _ROW_HEIGHT


def _add_color_pool_rows(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    field_name: str,
    pool: ColorPoolConfig,
    display: str,
    label_width: int,
    slider_width: int,
    entry_width: int,
) -> int:
    """Create a header plus one editable (hex, weight) row per colour in a pool.

    The colour count is fixed (v1): add/remove colours via the TOML. Each row is
    an index label, a hex entry, and a weight entry laid out on the shared
    label/slider/entry columns. The whole pool is registered as one
    ColorPoolBinding so config_bridge can rebuild the immutable tuple list.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: The scrollable container.
        y (int): Top Y coordinate for the header row.
        registry (SliderRegistry): Registry to record widgets and the binding.
        field_name (str): Dataclass field name of the ColorPoolConfig.
        pool (ColorPoolConfig): The colour pool whose colours are edited.
        display (str): Human display label used as the header.
        label_width (int): Label column width.
        slider_width (int): Slider column width (holds the hex entry).
        entry_width (int): Value-entry column width (holds the weight entry).

    Returns:
        int: Total pixel height consumed (header + one row per colour).
    """
    header = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, y + (_ROW_HEIGHT - _LABEL_HEIGHT) // 2, label_width + slider_width + entry_width, _LABEL_HEIGHT),
        text=display,
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(header)
    y += _ROW_HEIGHT

    rows: t.List[t.Tuple[t.Any, t.Any]] = []
    for index, (hex_str, weight) in enumerate(pool.colors):
        row_y = y + (_ROW_HEIGHT - _TEXT_HEIGHT) // 2
        index_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(0, row_y, label_width, _TEXT_HEIGHT),
            text=str(index),
            manager=manager,
            container=container,
        )
        registry.extra_widgets.append(index_label)
        hex_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(label_width, row_y, slider_width, _TEXT_HEIGHT),
            manager=manager,
            container=container,
        )
        hex_entry.set_text(hex_str)
        weight_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(label_width + slider_width, row_y, entry_width, _TEXT_HEIGHT),
            manager=manager,
            container=container,
        )
        weight_entry.set_text(f"{weight:.3f}")
        rows.append((hex_entry, weight_entry))
        y += _ROW_HEIGHT

    registry.register_color_pool((field_name, "colors"), rows)
    return y


def _add_skip_label(
    manager: pygame_gui.UIManager,
    container: t.Any,
    y: int,
    registry: SliderRegistry,
    display: str,
    label_width: int,
    slider_width: int,
    entry_width: int,
) -> int:
    """Add a small "(colors: edit TOML)" label for a skipped complex field.

    Covers lists and nested dataclasses that v1 does not edit in the panel, so
    the omission is visible rather than silent.

    Returns:
        int: Y coordinate of the next row.
    """
    label = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(0, y + 5, label_width + slider_width + entry_width, _SKIP_HEIGHT - 10),
        text=f"{display} (colors: edit TOML)",
        manager=manager,
        container=container,
    )
    registry.extra_widgets.append(label)
    return y + _SKIP_HEIGHT


def _compute_row_widths(container: t.Any) -> t.Tuple[int, int, int]:
    """Compute label, slider, and value-entry pixel widths from the container.

    Subtracts a fixed scrollbar gutter before splitting proportionally, so long
    label strings are not clipped.

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
