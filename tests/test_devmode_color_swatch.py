# Radical
"""Headless tests for the colour-swatch post-build pass over a SliderRegistry."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402
import pytest  # noqa: E402

from streets_at_night.devmode.color_swatch import (  # noqa: E402
    SWATCH_GAP,
    SWATCH_SIZE,
    add_swatches,
    color_to_hex,
    find_swatch_entry_at,
    is_hex_color,
    refresh_swatch,
)
from streets_at_night.devmode.slider_registry import SliderRegistry  # noqa: E402


@pytest.fixture
def scroll(headless_manager):
    """Provide a scrolling container to parent widgets into."""
    return pygame_gui.elements.UIScrollingContainer(
        relative_rect=pygame.Rect(0, 0, 384, 600),
        manager=headless_manager,
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#FFAA00", True),
        ("#ffaa00", True),
        ("#0D0A08", True),
        ("FFAA00", False),
        ("#FFAA0", False),
        ("#FFAA000", False),
        ("#GGAA00", False),
        ("", False),
        ("not a color", False),
    ],
)
def test_is_hex_color_pattern_table(text, expected):
    assert is_hex_color(text) == expected


def test_color_to_hex_formats_uppercase_rgb_dropping_alpha():
    assert color_to_hex(pygame.Color(255, 170, 0, 128)) == "#FFAA00"


def _make_entry(headless_manager, container, text):
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(100, 10, 120, 30),
        manager=headless_manager,
        container=container,
    )
    entry.set_text(text)
    return entry


def test_add_swatches_creates_swatch_for_hex_text_entry(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))

    # Act
    add_swatches(headless_manager, registry)

    # Assert — swatch exists, entry shrank and shifted right to make room
    assert entry in registry.swatches
    swatch = registry.swatches[entry]
    assert swatch.get_relative_rect().width == SWATCH_SIZE
    assert entry.get_relative_rect().x == 100 + SWATCH_SIZE + SWATCH_GAP
    assert entry.get_relative_rect().width == 120 - SWATCH_SIZE - SWATCH_GAP


def test_add_swatches_skips_non_hex_text_entry(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "some string")
    registry.register_text(entry, ("region_name",))

    # Act
    add_swatches(headless_manager, registry)

    # Assert
    assert registry.swatches == {}


def test_add_swatches_covers_color_pool_hex_entries(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    hex_entry = _make_entry(headless_manager, scroll.scrollable_container, "#0D0A08")
    weight_entry = _make_entry(headless_manager, scroll.scrollable_container, "1.000")
    registry.register_color_pool(("colors", "colors"), [(hex_entry, weight_entry)])

    # Act
    add_swatches(headless_manager, registry)

    # Assert — hex entry got a swatch, weight entry did not
    assert hex_entry in registry.swatches
    assert weight_entry not in registry.swatches


def test_refresh_swatch_updates_image_on_valid_hex(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FF0000")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)

    # Act
    entry.set_text("#00FF00")
    refresh_swatch(registry, entry)

    # Assert — swatch surface repainted to the new colour
    swatch = registry.swatches[entry]
    assert swatch.image.get_at((SWATCH_SIZE // 2, SWATCH_SIZE // 2))[:3] == (0, 255, 0)


def test_refresh_swatch_keeps_last_color_on_invalid_hex(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FF0000")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)

    # Act
    entry.set_text("#nope")
    refresh_swatch(registry, entry)

    # Assert
    swatch = registry.swatches[entry]
    assert swatch.image.get_at((SWATCH_SIZE // 2, SWATCH_SIZE // 2))[:3] == (255, 0, 0)


def test_find_swatch_entry_at_hit_and_miss(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)
    swatch = registry.swatches[entry]
    panel_rect = scroll.get_abs_rect()
    inside = swatch.get_abs_rect().center

    # Act / Assert
    assert find_swatch_entry_at(registry, inside, panel_rect) is entry
    assert find_swatch_entry_at(registry, (panel_rect.right + 50, panel_rect.bottom + 50), panel_rect) is None


def test_find_swatch_entry_at_ignores_swatch_outside_panel_rect(headless_manager, scroll):
    # Arrange — a panel rect that excludes the swatch simulates it being scrolled out of view
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)
    swatch = registry.swatches[entry]
    point = swatch.get_abs_rect().center
    off_panel = pygame.Rect(-500, -500, 10, 10)

    # Act / Assert
    assert find_swatch_entry_at(registry, point, off_panel) is None


def test_all_widgets_includes_swatches(headless_manager, scroll):
    # Arrange
    registry = SliderRegistry()
    entry = _make_entry(headless_manager, scroll.scrollable_container, "#FFAA00")
    registry.register_text(entry, ("haze_color",))
    add_swatches(headless_manager, registry)

    # Act
    widgets = registry.all_widgets()

    # Assert
    assert registry.swatches[entry] in widgets
