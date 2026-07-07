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
    # Arrange — only headlight_ fields; headlight_colors (ColorPool) is not a slider.
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


def test_build_dynamic_section_renders_color_pool_rows_and_binding(headless_manager, scroll, config):
    # Arrange — Commercial: Parking exposes parking_colors (a ColorPoolConfig).
    kind = LightKind("Commercial: Parking", config.lights.commercial, "parking_")
    pool = config.lights.commercial.parking_colors
    registry = SliderRegistry()

    # Act
    height = build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)

    # Assert — one binding whose rows match the colour count, positive height.
    assert len(registry.color_pools) == 1
    binding = registry.color_pools[0]
    assert binding.config_path == ("parking_colors", "colors")
    assert len(binding.rows) == len(pool.colors)
    assert height > 0


def test_build_dynamic_section_color_pool_entries_prefilled(headless_manager, scroll, config):
    # Arrange
    kind = LightKind("House Lights", config.lights.house_lights, None)
    pool = config.lights.house_lights.colors
    registry = SliderRegistry()

    # Act
    build_dynamic_section(headless_manager, scroll.scrollable_container, 0, registry, kind)
    binding = registry.color_pools[0]

    # Assert — hex + weight entries pre-filled from the config colours.
    first_hex, first_weight = binding.rows[0]
    assert first_hex.get_text() == pool.colors[0][0]
    assert first_weight.get_text() == f"{pool.colors[0][1]:.3f}"


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
