"""Colour-preview swatches for hex-valued panel entries.

A post-build pass over a section's SliderRegistry: every text entry whose
current text is a #RRGGBB colour gets a small UIImage swatch to its left
(the entry is shrunk to make room). Swatches live in
``SliderRegistry.swatches`` so section teardown kills them with everything
else. Click routing and the colour-picker dialog live in the Devmode tab;
this module only answers "which entry's swatch is at this screen position".
"""

import re
import typing as t

import pygame
import pygame_gui

from .slider_registry import SliderRegistry

SWATCH_SIZE = 22
SWATCH_GAP = 4

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_hex_color(text: str) -> bool:
    """Check whether text is a #RRGGBB hex colour string."""
    return bool(_HEX_COLOR_RE.match(text))


def color_to_hex(colour: pygame.Color) -> str:
    """Format a pygame Color as an uppercase #RRGGBB string (alpha dropped).

    Args:
        colour (pygame.Color): The colour to format.

    Returns:
        str: "#RRGGBB", matching config_bridge's hex format.
    """
    return f"#{colour.r:02X}{colour.g:02X}{colour.b:02X}"


def add_swatches(manager: pygame_gui.UIManager, registry: SliderRegistry) -> None:
    """Create a colour swatch beside every hex-valued entry in the registry.

    Must run AFTER the section's entries have their text populated (i.e.
    after read_config_to_widgets), because hex detection reads entry text.
    Non-hex entries are left untouched.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        registry (SliderRegistry): The section's registry; swatches are
            recorded in registry.swatches.
    """
    for entry in _iter_hex_candidates(registry):
        if entry in registry.swatches or not is_hex_color(entry.get_text()):
            continue
        registry.swatches[entry] = _create_swatch(manager, entry)


def refresh_swatch(registry: SliderRegistry, entry: t.Any) -> None:
    """Repaint an entry's swatch from its current text, if valid hex.

    Invalid hex leaves the swatch on its last valid colour. Entries without
    a swatch are ignored.

    Args:
        registry (SliderRegistry): The section's registry.
        entry: The UITextEntryLine whose swatch should refresh.
    """
    swatch = registry.swatches.get(entry)
    if swatch is None or not is_hex_color(entry.get_text()):
        return
    swatch.set_image(_fill_surface(entry.get_text()))


def find_swatch_entry_at(
    registry: SliderRegistry,
    pos: t.Tuple[int, int],
    panel_rect: pygame.Rect,
) -> t.Optional[t.Any]:
    """Return the entry whose swatch is under a screen position, if any.

    Only the part of a swatch inside ``panel_rect`` counts, so swatches
    scrolled outside the visible panel area don't take clicks.

    Args:
        registry (SliderRegistry): The section's registry.
        pos (tuple[int, int]): Screen position of the click.
        panel_rect (pygame.Rect): Absolute rect of the visible panel area.

    Returns:
        The hex entry owning the clicked swatch, or None.
    """
    for entry, swatch in registry.swatches.items():
        visible_part = swatch.get_abs_rect().clip(panel_rect)
        if visible_part.collidepoint(pos):
            return entry
    return None


def _iter_hex_candidates(registry: SliderRegistry) -> t.Iterator[t.Any]:
    """Yield every entry that may hold a hex colour: text entries + pool hex entries."""
    yield from registry.text_entries.keys()
    for binding in registry.color_pools:
        for hex_entry, _weight_entry in binding.rows:
            yield hex_entry


def _create_swatch(manager: pygame_gui.UIManager, entry: t.Any) -> pygame_gui.elements.UIImage:
    """Shrink an entry and place a colour-filled UIImage in the freed slot.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        entry: The UITextEntryLine to attach a swatch to.

    Returns:
        UIImage: The created swatch, vertically centered on the entry row.
    """
    rect = entry.get_relative_rect()
    swatch_y = rect.y + (rect.height - SWATCH_SIZE) // 2
    swatch_rect = pygame.Rect(rect.x, swatch_y, SWATCH_SIZE, SWATCH_SIZE)

    entry.set_relative_position((rect.x + SWATCH_SIZE + SWATCH_GAP, rect.y))
    entry.set_dimensions((rect.width - SWATCH_SIZE - SWATCH_GAP, rect.height))

    return pygame_gui.elements.UIImage(
        relative_rect=swatch_rect,
        image_surface=_fill_surface(entry.get_text()),
        manager=manager,
        container=entry.ui_container,
    )


def _fill_surface(hex_text: str) -> pygame.Surface:
    """Build a SWATCH_SIZE square surface filled with the given hex colour."""
    surface = pygame.Surface((SWATCH_SIZE, SWATCH_SIZE))
    surface.fill(pygame.Color(hex_text))
    return surface
