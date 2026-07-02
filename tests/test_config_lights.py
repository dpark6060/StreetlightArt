# Radical
"""Tests for Stage 5 light configuration in streets_at_night.config.

Covers: LightConfig, StreetLightConfig, CarConfig, CommercialLightConfig,
HouseLightConfig, FixturePoolEntry, and FixtureProfileConfig loading
from defaults and from an empty TOML overlay.
"""

from pathlib import Path

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import load_config
from streets_at_night.config.schema import (
    CarConfig,
    CommercialLightConfig,
    FixturePoolEntry,
    FixtureProfileConfig,
    HouseLightConfig,
    LightConfig,
    SceneConfig,
    StreetLightConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_toml(tmp_path: Path) -> Path:
    """A valid but empty TOML file — all values come from defaults."""
    toml_file = tmp_path / "empty.toml"
    toml_file.write_text("")
    return toml_file


@pytest.fixture
def loaded_config(empty_toml: Path) -> SceneConfig:
    """SceneConfig built entirely from defaults via an empty TOML."""
    return load_config(empty_toml)


# ---------------------------------------------------------------------------
# SceneConfig includes lights
# ---------------------------------------------------------------------------


def test_load_config_includes_lights(loaded_config: SceneConfig):
    # Act — config already loaded via fixture

    # Assert
    assert hasattr(loaded_config, "lights")
    assert loaded_config.lights is not None


def test_load_config_lights_is_light_config_instance(loaded_config: SceneConfig):
    # Act — config already loaded via fixture

    # Assert
    assert isinstance(loaded_config.lights, LightConfig)


# ---------------------------------------------------------------------------
# LightConfig sub-section types
# ---------------------------------------------------------------------------


def test_light_config_has_streetlights(loaded_config: SceneConfig):
    # Assert
    assert isinstance(loaded_config.lights.streetlights, StreetLightConfig)


def test_light_config_has_cars(loaded_config: SceneConfig):
    # Assert
    assert isinstance(loaded_config.lights.cars, CarConfig)


def test_light_config_has_commercial(loaded_config: SceneConfig):
    # Assert
    assert isinstance(loaded_config.lights.commercial, CommercialLightConfig)


def test_light_config_has_house_lights(loaded_config: SceneConfig):
    # Assert
    assert isinstance(loaded_config.lights.house_lights, HouseLightConfig)


# ---------------------------------------------------------------------------
# StreetLightConfig — highway fixture pool
# ---------------------------------------------------------------------------


def test_streetlight_highway_pool_has_entries(loaded_config: SceneConfig):
    # Assert — highway pool must be a non-empty list of FixturePoolEntry
    highway_pool = loaded_config.lights.streetlights.highway

    assert isinstance(highway_pool, list)
    assert len(highway_pool) > 0
    assert all(isinstance(entry, FixturePoolEntry) for entry in highway_pool)


def test_streetlight_highway_pool_entries_all_have_profiles(loaded_config: SceneConfig):
    # Assert — highway is always lit: no entry should have profile=None
    highway_pool = loaded_config.lights.streetlights.highway

    assert all(entry.profile is not None for entry in highway_pool)


# ---------------------------------------------------------------------------
# StreetLightConfig — residential pool has a no-lights entry
# ---------------------------------------------------------------------------


def test_streetlight_residential_has_no_lights_entry(loaded_config: SceneConfig):
    # Assert — at least one residential entry must have profile=None (unlit stretch)
    residential_pool = loaded_config.lights.streetlights.residential

    none_profiles = [entry for entry in residential_pool if entry.profile is None]
    assert len(none_profiles) > 0


# ---------------------------------------------------------------------------
# CarConfig — color pools
# ---------------------------------------------------------------------------


def test_car_headlight_colors_non_empty(loaded_config: SceneConfig):
    # Assert
    headlight_colors = loaded_config.lights.cars.headlight_colors.colors

    assert isinstance(headlight_colors, list)
    assert len(headlight_colors) > 0


def test_car_taillight_colors_non_empty(loaded_config: SceneConfig):
    # Assert
    taillight_colors = loaded_config.lights.cars.taillight_colors.colors

    assert isinstance(taillight_colors, list)
    assert len(taillight_colors) > 0


# ---------------------------------------------------------------------------
# CarConfig — density min < max and positive values
# ---------------------------------------------------------------------------


def test_car_density_values_valid(loaded_config: SceneConfig):
    # Arrange
    cars = loaded_config.lights.cars
    density_pairs = [
        (cars.density_highway_min, cars.density_highway_max),
        (cars.density_major_min, cars.density_major_max),
        (cars.density_collector_min, cars.density_collector_max),
        (cars.density_residential_min, cars.density_residential_max),
        (cars.density_service_min, cars.density_service_max),
    ]

    # Assert
    for min_val, max_val in density_pairs:
        assert min_val > 0, f"density min must be positive, got {min_val}"
        assert max_val > 0, f"density max must be positive, got {max_val}"
        assert min_val < max_val, f"density min ({min_val}) must be less than max ({max_val})"


# ---------------------------------------------------------------------------
# CommercialLightConfig — sign color pool
# ---------------------------------------------------------------------------


def test_commercial_sign_colors_non_empty(loaded_config: SceneConfig):
    # Assert
    sign_colors = loaded_config.lights.commercial.sign_colors.colors

    assert isinstance(sign_colors, list)
    assert len(sign_colors) > 0


# ---------------------------------------------------------------------------
# HouseLightConfig — color pool
# ---------------------------------------------------------------------------


def test_house_light_colors_non_empty(loaded_config: SceneConfig):
    # Assert
    house_colors = loaded_config.lights.house_lights.colors.colors

    assert isinstance(house_colors, list)
    assert len(house_colors) > 0


# ---------------------------------------------------------------------------
# FixtureProfileConfig — highway pool entry has all required fields
# ---------------------------------------------------------------------------


def test_fixture_profile_config_has_all_fields(loaded_config: SceneConfig):
    # Arrange — grab first highway pool entry, which always has a profile
    highway_pool = loaded_config.lights.streetlights.highway
    profile = highway_pool[0].profile

    # Assert
    assert isinstance(profile, FixtureProfileConfig)
    assert isinstance(profile.color_temp_k, float)
    assert isinstance(profile.saturation, float)
    assert isinstance(profile.height_min, float)
    assert isinstance(profile.height_max, float)
    assert isinstance(profile.brightness_min, float)
    assert isinstance(profile.brightness_max, float)
    assert isinstance(profile.glow_radius, float)
    assert isinstance(profile.beam_angle_deg, float)
    assert isinstance(profile.illumination_intensity, float)
    assert isinstance(profile.spacing_min, float)
    assert isinstance(profile.spacing_max, float)
    assert isinstance(profile.side, str)


# ---------------------------------------------------------------------------
# FixturePoolEntry — weights are positive floats
# ---------------------------------------------------------------------------


def test_fixture_pool_entry_weights_are_positive(loaded_config: SceneConfig):
    # Arrange — collect all entries across all road type pools
    streetlights = loaded_config.lights.streetlights
    all_pools = [
        streetlights.highway,
        streetlights.major_road,
        streetlights.collector,
        streetlights.residential,
        streetlights.service,
    ]

    # Assert
    for pool in all_pools:
        for entry in pool:
            assert entry.weight > 0, f"fixture pool entry weight must be positive, got {entry.weight}"


# ---------------------------------------------------------------------------
# _build_fixture_pool_entries — missing profile key yields profile=None
# ---------------------------------------------------------------------------


def test_build_fixture_pool_entries_handles_missing_profile_key(tmp_path: Path):
    # Arrange — a TOML fixture pool entry with no profile sub-table
    toml_content = (
        "[[lights.streetlights.residential]]\n"
        "weight = 0.25\n"
    )
    toml_file = tmp_path / "no_profile.toml"
    toml_file.write_text(toml_content)

    # Act
    result = load_config(toml_file)

    # Assert — the entry with no profile key produces a FixturePoolEntry with profile=None
    residential_pool = result.lights.streetlights.residential
    no_lights_entries = [e for e in residential_pool if e.profile is None]
    assert len(no_lights_entries) == 1
    assert no_lights_entries[0].weight == 0.25


# ---------------------------------------------------------------------------
# TOML override — lights section deep-merges correctly
# ---------------------------------------------------------------------------


def test_toml_override_streetlight_preserves_defaults(tmp_path: Path):
    # Arrange — override only the highway pool weight of the first entry
    toml_content = (
        "[[lights.streetlights.highway]]\n"
        'weight = 0.8\n'
        "[lights.streetlights.highway.profile]\n"
        "color_temp_k = 3000.0\n"
        "saturation = 1.0\n"
        "height_min = 12.0\n"
        "height_max = 15.0\n"
        "brightness_min = 0.85\n"
        "brightness_max = 1.0\n"
        "glow_radius = 25.0\n"
        "beam_angle_deg = 150.0\n"
        "illumination_intensity = 0.9\n"
        "spacing_min = 40.0\n"
        "spacing_max = 60.0\n"
        'side = "both"\n'
    )
    toml_file = tmp_path / "partial_override.toml"
    toml_file.write_text(toml_content)

    # Act
    result = load_config(toml_file)

    # Assert — the overridden highway pool reflects new values
    highway_pool = result.lights.streetlights.highway
    assert len(highway_pool) > 0
    assert highway_pool[0].profile.color_temp_k == 3000.0

    # Assert — non-overridden sub-sections still come from defaults
    assert isinstance(result.lights.cars, CarConfig)
    assert len(result.lights.cars.headlight_colors.colors) > 0


# ---------------------------------------------------------------------------
# color_temp_shift_k — city-wide master lever
# ---------------------------------------------------------------------------


def test_color_temp_shift_k_defaults_to_zero(loaded_config: SceneConfig):
    # Assert — absent from config, the master shift defaults to neutral
    assert loaded_config.lights.streetlights.color_temp_shift_k == 0.0


def test_color_temp_shift_k_loads_from_toml(tmp_path: Path):
    # Arrange
    toml_content = (
        "[lights.streetlights]\n"
        "color_temp_shift_k = 1500.0\n"
    )
    toml_file = tmp_path / "shift.toml"
    toml_file.write_text(toml_content)

    # Act
    result = load_config(toml_file)

    # Assert
    assert result.lights.streetlights.color_temp_shift_k == 1500.0
