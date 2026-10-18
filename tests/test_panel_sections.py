# Radical
"""Headless tests for panel_sections: hand-written section label unit suffixes."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.devmode.panel_sections import (  # noqa: E402
    build_atmosphere_section,
    build_devmode_camera_section,
)
from streets_at_night.devmode.slider_registry import SliderRegistry  # noqa: E402


@pytest.fixture
def scroll(headless_manager):
    """Provide a scrolling container to parent widgets into."""
    return pygame_gui.elements.UIScrollingContainer(
        relative_rect=pygame.Rect(0, 0, 384, 600),
        manager=headless_manager,
    )


@pytest.fixture
def config() -> SceneConfig:
    """Return a default SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


def test_build_atmosphere_section_metered_labels_get_suffix(headless_manager, scroll, config):
    # Arrange
    registry = SliderRegistry()

    # Act
    build_atmosphere_section(headless_manager, scroll.scrollable_container, 0, registry, config)
    texts = {w.text for w in registry.extra_widgets if isinstance(w, pygame_gui.elements.UILabel)}

    # Assert — distance fields metered, unitless atmosphere fields untouched.
    assert "falloff_distance (m)" in texts
    assert "scatter_distance (m)" in texts
    assert "glow_max_alpha" in texts


def test_build_devmode_camera_section_labels_get_units(headless_manager, scroll, config):
    # Arrange
    registry = SliderRegistry()

    # Act
    build_devmode_camera_section(headless_manager, scroll.scrollable_container, 0, registry, config)
    texts = {w.text for w in registry.extra_widgets if isinstance(w, pygame_gui.elements.UILabel)}

    # Assert — camera positions are meters, fov is degrees.
    assert "pos_x (m)" in texts
    assert "target_z (m)" in texts
    assert "fov (deg)" in texts
