# Radical
"""Headless tests for build_dynamic_section: reflection, prefix filtering, skips."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pygame_gui  # noqa: E402

from streets_at_night.config.defaults import DEFAULTS  # noqa: E402
from streets_at_night.config.schema import SceneConfig  # noqa: E402
from streets_at_night.config.serde import dataclass_from_dict  # noqa: E402
from streets_at_night.devmode.dynamic_panel import build_dynamic_section  # noqa: E402
from streets_at_night.devmode.light_catalog import LightKind  # noqa: E402
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


def test_build_dynamic_section_fixture_profile_sliders_and_text(headless_manager, scroll, config):
    # Arrange — a FixtureProfileConfig has 11 float fields, one str (side).
    profile = config.lights.streetlights.highway[0].profile
    kind = LightKind("Streetlight: Highway (pool 0)", profile, None)
    registry = SliderRegistry()

    # Act
    height = build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)

    # Assert
    assert len(registry.sliders) == 11
    assert len(registry.text_entries) == 1
    assert len(registry.bool_buttons) == 0
    assert height > 0


def test_build_dynamic_section_slider_config_path_relative_to_target(headless_manager, scroll, config):
    # Arrange
    profile = config.lights.streetlights.highway[0].profile
    kind = LightKind("Streetlight: Highway (pool 0)", profile, None)
    registry = SliderRegistry()

    # Act
    build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)
    paths = {d.config_path for d in registry.sliders.values()}

    # Assert — single-element paths relative to kind.target, not the full config.
    assert ("brightness_max",) in paths
    assert all(len(p) == 1 for p in paths)


def test_build_dynamic_section_prefix_filters_and_strips(headless_manager, scroll, config):
    # Arrange — only headlight_ fields; headlight_colors (ColorPool) is skipped.
    kind = LightKind("Cars: Headlights", config.lights.cars, "headlight_")
    registry = SliderRegistry()

    # Act
    build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)
    labels = {d.label for d in registry.sliders.values()}
    paths = {d.config_path[0] for d in registry.sliders.values()}

    # Assert — 7 numeric headlight fields, no taillight/density/separation.
    assert len(registry.sliders) == 7
    assert all(p.startswith("headlight_") for p in paths)
    # Descriptor labels are unique per section (kind.label:field_name).
    assert "Cars: Headlights:headlight_brightness_min" in labels
    # No slider for the ColorPool field.
    assert not any("colors" in p for p in paths)


def test_build_dynamic_section_skips_color_pool_with_visible_label(headless_manager, scroll, config):
    # Arrange
    kind = LightKind("Cars: Headlights", config.lights.cars, "headlight_")
    registry = SliderRegistry()

    # Act
    build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)
    label_texts = [w.text for w in registry.extra_widgets if hasattr(w, "text")]

    # Assert — the color pool omission is surfaced as a label.
    assert any("edit TOML" in text for text in label_texts)


def test_build_dynamic_section_global_kind_only_shift_field(headless_manager, scroll, config):
    # Arrange — StreetLightConfig has list road-type fields (skipped) + the shift.
    kind = LightKind("Streetlights: Global", config.lights.streetlights, "color_temp_shift")
    registry = SliderRegistry()

    # Act
    build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)

    # Assert — only color_temp_shift_k matches the prefix; the lists are skipped.
    assert len(registry.sliders) == 1
    descriptor = next(iter(registry.sliders.values()))
    assert descriptor.config_path == ("color_temp_shift_k",)
    assert descriptor.min_val == -3000.0
