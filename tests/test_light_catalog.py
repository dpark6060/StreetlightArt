# Radical
"""Tests for the devmode light catalog: kinds enumerated and live target identity."""

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import FixtureProfileConfig, SceneConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.devmode.light_catalog import LightKind, build_light_catalog


@pytest.fixture
def config() -> SceneConfig:
    """Return a default SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


def test_build_light_catalog_streetlight_pools_one_kind_per_profile(config):
    # Act
    catalog = build_light_catalog(config)
    labels = [k.label for k in catalog]

    # Assert — default pools: highway 2, major_road 3, collector 3,
    # residential 4 (first entry is no-lights), service 2 (first is no-lights).
    assert "Streetlight: Highway (pool 0)" in labels
    assert "Streetlight: Highway (pool 1)" in labels
    assert "Streetlight: Major Road (pool 2)" in labels
    assert "Streetlight: Residential (pool 0)" in labels
    assert "Streetlight: Residential (pool 3)" in labels
    assert "Streetlight: Service (pool 1)" in labels
    street = [label for label in labels if label.startswith("Streetlight:") and "Global" not in label]
    assert len(street) == 14


def test_build_light_catalog_streetlight_target_is_fixture_profile(config):
    # Act
    catalog = build_light_catalog(config)
    highway0 = next(k for k in catalog if k.label == "Streetlight: Highway (pool 0)")

    # Assert
    assert isinstance(highway0.target, FixtureProfileConfig)
    assert highway0.field_prefix is None


def test_build_light_catalog_car_kinds_present_with_prefixes(config):
    # Act
    catalog = build_light_catalog(config)
    by_label = {k.label: k for k in catalog}

    # Assert
    assert by_label["Cars: Headlights"].field_prefix == "headlight_"
    assert by_label["Cars: Taillights"].field_prefix == "taillight_"
    assert by_label["Cars: Density"].field_prefix == "density_"
    assert by_label["Cars: Separation"].field_prefix == "separation_"
    for label in ("Cars: Headlights", "Cars: Taillights", "Cars: Density", "Cars: Separation"):
        assert by_label[label].target is config.lights.cars


def test_build_light_catalog_commercial_kinds(config):
    # Act
    catalog = build_light_catalog(config)
    by_label = {k.label: k for k in catalog}

    # Assert
    assert by_label["Commercial: Signs"].field_prefix == "sign_"
    assert by_label["Commercial: Parking"].field_prefix == "parking_"
    assert by_label["Commercial: Storefronts"].field_prefix == "storefront_"
    for label in ("Commercial: Signs", "Commercial: Parking", "Commercial: Storefronts"):
        assert by_label[label].target is config.lights.commercial


def test_build_light_catalog_global_streetlight_kind(config):
    # Act
    catalog = build_light_catalog(config)
    by_label = {k.label: k for k in catalog}

    # Assert
    glob = by_label["Streetlights: Global"]
    assert glob.target is config.lights.streetlights
    assert glob.field_prefix == "color_temp_shift"


def test_build_light_catalog_includes_illumination_kind(config):
    # Act
    kinds = build_light_catalog(config)
    by_label = {kind.label: kind for kind in kinds}

    # Assert
    assert by_label["Illumination: Ground Pools"].target is config.lights.illumination
    assert by_label["Illumination: Ground Pools"].field_prefix is None


def test_build_light_catalog_buildings_windows_kind_present(config):
    # Act
    catalog = build_light_catalog(config)
    by_label = {k.label: k for k in catalog}

    # Assert — the window section targets the buildings config, scoped to window_ fields.
    windows = by_label["Buildings: Windows"]
    assert windows.target is config.buildings
    assert windows.field_prefix == "window_"


def test_build_light_catalog_omits_buildings_windows_when_unconfigured(config):
    # Arrange — an older config with no [buildings] section.
    config.buildings = None

    # Act
    labels = [k.label for k in build_light_catalog(config)]

    # Assert
    assert "Buildings: Windows" not in labels


def test_build_light_catalog_target_identity_mutation_reaches_config(config):
    # Arrange
    catalog = build_light_catalog(config)
    highway0 = next(k for k in catalog if k.label == "Streetlight: Highway (pool 0)")

    # Act — mutating a catalog target mutates the live config.
    highway0.target.brightness_max = 0.123

    # Assert
    assert config.lights.streetlights.highway[0].profile.brightness_max == 0.123
