# Radical
"""Tests for streets_at_night.ui.widgets.labeled_entry.parse_float_entry."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.ui.widgets.labeled_entry import (  # noqa: E402
    build_labeled_entry,
    parse_float_entry,
)


def test_parse_float_entry_valid_text_returns_float(headless_manager):
    # Arrange
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(0, 0, 100, 28), manager=headless_manager
    )
    entry.set_text("3.5")

    # Act
    result = parse_float_entry(entry, fallback=1.0)

    # Assert
    assert result == 3.5


def test_parse_float_entry_invalid_text_returns_fallback_and_resets(headless_manager):
    # Arrange
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(0, 0, 100, 28), manager=headless_manager
    )
    entry.set_text("not-a-number")

    # Act
    result = parse_float_entry(entry, fallback=2.5)

    # Assert
    assert result == 2.5
    assert entry.get_text() == "2.5"


def test_parse_float_entry_resets_to_trimmed_fallback(headless_manager):
    # Arrange
    entry = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(0, 0, 100, 28), manager=headless_manager
    )
    entry.set_text("junk")

    # Act
    parse_float_entry(entry, fallback=10.0)

    # Assert — trailing zeros and dot stripped, so "10" not "10.000000".
    assert entry.get_text() == "10"


def test_build_labeled_entry_returns_label_and_entry(headless_manager):
    # Act
    label, entry = build_labeled_entry(headless_manager, None, 0, 0, "Speed")

    # Assert
    assert isinstance(label, pygame_gui.elements.UILabel)
    assert isinstance(entry, pygame_gui.elements.UITextEntryLine)
