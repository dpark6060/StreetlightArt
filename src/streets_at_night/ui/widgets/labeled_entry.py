"""Label + text-entry row helper and a tolerant float parser."""

import typing as t

import pygame
import pygame_gui


def build_labeled_entry(
    manager: pygame_gui.UIManager,
    container: t.Any,
    x: int,
    y: int,
    label: str,
    entry_width: int = 120,
    label_width: int = 140,
    height: int = 28,
) -> t.Tuple[pygame_gui.elements.UILabel, pygame_gui.elements.UITextEntryLine]:
    """Create a UILabel and UITextEntryLine side by side.

    Args:
        manager (pygame_gui.UIManager): The UI manager.
        container: Parent container for both widgets.
        x (int): Left edge of the label, relative to the container.
        y (int): Top edge of both widgets, relative to the container.
        label (str): Label text.
        entry_width (int): Width of the text entry in pixels.
        label_width (int): Width of the label in pixels.
        height (int): Height of both widgets in pixels.

    Returns:
        tuple[UILabel, UITextEntryLine]: The label and entry widgets.
    """
    label_widget = pygame_gui.elements.UILabel(
        relative_rect=pygame.Rect(x, y, label_width, height),
        text=label,
        manager=manager,
        container=container,
    )
    entry_widget = pygame_gui.elements.UITextEntryLine(
        relative_rect=pygame.Rect(x + label_width, y, entry_width, height),
        manager=manager,
        container=container,
    )
    return label_widget, entry_widget


def parse_float_entry(
    entry: pygame_gui.elements.UITextEntryLine,
    fallback: float,
) -> float:
    """Parse the entry's text as a float, resetting to the fallback on failure.

    On a ValueError the entry text is rewritten to a trimmed representation of
    the fallback (``f"{fallback:.6f}"`` with trailing zeros and dot stripped).

    Args:
        entry (UITextEntryLine): The entry to read (and possibly reset).
        fallback (float): Value to return and display when parsing fails.

    Returns:
        float: The parsed value, or the fallback on failure.
    """
    text = entry.get_text().strip()
    try:
        return float(text)
    except ValueError:
        entry.set_text(f"{fallback:.6f}".rstrip("0").rstrip("."))
        return fallback
