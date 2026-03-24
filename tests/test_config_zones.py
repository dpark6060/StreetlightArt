# Radical
"""Tests for Stage 4 zone config extensions in streets_at_night.config.

Covers: ColorPoolConfig, HouseConfig, ZoneColorsConfig, ZoneConfig loading
from defaults and from TOML overrides.
"""

import tempfile
from pathlib import Path

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import load_config
from streets_at_night.config.schema import (
    ColorPoolConfig,
    HouseConfig,
    SceneConfig,
    ZoneColorsConfig,
    ZoneConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENE_CONFIG_TOML = Path(__file__).parent.parent / "scene_config.toml"


@pytest.fixture
def empty_toml(tmp_path: Path) -> Path:
    """A valid but empty TOML file — all values come from defaults."""
    toml_file = tmp_path / "empty.toml"
    toml_file.write_text("")
    return toml_file


@pytest.fixture
def commercial_color_override_toml(tmp_path: Path) -> Path:
    """TOML that overrides only zones.colors.commercial."""
    toml_file = tmp_path / "commercial_override.toml"
    toml_file.write_text('[zones.colors]\ncommercial = "#FFFFFF"\n')
    return toml_file


# ---------------------------------------------------------------------------
# SceneConfig includes zones
# ---------------------------------------------------------------------------


def test_load_config_includes_zones(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert hasattr(result, "zones")
    assert result.zones is not None


def test_load_config_zones_is_zone_config_instance(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result.zones, ZoneConfig)


# ---------------------------------------------------------------------------
# ZoneConfig structure
# ---------------------------------------------------------------------------


def test_zone_config_has_house(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result.zones.house, HouseConfig)


def test_zone_config_has_colors(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result.zones.colors, ZoneColorsConfig)


# ---------------------------------------------------------------------------
# HouseConfig defaults
# ---------------------------------------------------------------------------


def test_house_config_defaults_yard_base_color(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.house.yard_base_color == DEFAULTS["zones"]["house"]["yard_base_color"]


def test_house_config_defaults_yard_variation(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.house.yard_variation == DEFAULTS["zones"]["house"]["yard_variation"]


def test_house_config_defaults_min_max_lights(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.house.min_lights == DEFAULTS["zones"]["house"]["min_lights"]
    assert result.zones.house.max_lights == DEFAULTS["zones"]["house"]["max_lights"]


def test_house_config_defaults_light_weights(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.house.light_weights == DEFAULTS["zones"]["house"]["light_weights"]


def test_house_config_defaults_footprint_and_lot(empty_toml: Path):
    # Act
    result = load_config(empty_toml)
    house = result.zones.house

    # Assert
    assert house.footprint_coverage == DEFAULTS["zones"]["house"]["footprint_coverage"]
    assert house.lot_width == DEFAULTS["zones"]["house"]["lot_width"]
    assert house.lot_depth == DEFAULTS["zones"]["house"]["lot_depth"]
    assert house.setback == DEFAULTS["zones"]["house"]["setback"]


def test_house_config_exterior_colors_is_color_pool_config(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result.zones.house.exterior_colors, ColorPoolConfig)


# ---------------------------------------------------------------------------
# ColorPoolConfig
# ---------------------------------------------------------------------------


def test_color_pool_config_has_colors(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert — colors list is non-empty
    assert len(result.zones.house.exterior_colors.colors) > 0


def test_color_pool_config_colors_are_tuples(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert — each entry is a tuple of (str, float)
    for entry in result.zones.house.exterior_colors.colors:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        assert isinstance(entry[0], str)
        assert isinstance(entry[1], float)


# ---------------------------------------------------------------------------
# ZoneColorsConfig defaults
# ---------------------------------------------------------------------------


def test_zone_colors_defaults_commercial(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.colors.commercial == DEFAULTS["zones"]["colors"]["commercial"]


def test_zone_colors_defaults_industrial(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.colors.industrial == DEFAULTS["zones"]["colors"]["industrial"]


def test_zone_colors_defaults_park_forest(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.colors.park_forest == DEFAULTS["zones"]["colors"]["park_forest"]


def test_zone_colors_defaults_water(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.colors.water == DEFAULTS["zones"]["colors"]["water"]


def test_zone_colors_defaults_road_surface(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.colors.road_surface == DEFAULTS["zones"]["colors"]["road_surface"]


def test_zone_colors_defaults_default_ground(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.colors.default_ground == DEFAULTS["zones"]["colors"]["default_ground"]


# ---------------------------------------------------------------------------
# ZoneConfig tile size and resolution defaults
# ---------------------------------------------------------------------------


def test_tile_size_default(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.tile_size_meters == 256.0


def test_tile_resolution_default(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.zones.tile_resolution_px == 128


# ---------------------------------------------------------------------------
# TOML override for zone colors
# ---------------------------------------------------------------------------


def test_toml_override_zone_color_commercial(commercial_color_override_toml: Path):
    # Act
    result = load_config(commercial_color_override_toml)

    # Assert — overridden value is reflected
    assert result.zones.colors.commercial == "#FFFFFF"


def test_toml_override_zone_color_preserves_other_colors(commercial_color_override_toml: Path):
    # Act
    result = load_config(commercial_color_override_toml)

    # Assert — non-overridden colors still come from defaults
    assert result.zones.colors.industrial == DEFAULTS["zones"]["colors"]["industrial"]
    assert result.zones.colors.water == DEFAULTS["zones"]["colors"]["water"]


# ---------------------------------------------------------------------------
# Load from real scene_config.toml (integration smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SCENE_CONFIG_TOML.exists(),
    reason="scene_config.toml not present in repo root",
)
def test_load_config_real_toml_includes_zones():
    # Act
    result = load_config(SCENE_CONFIG_TOML)

    # Assert — zones section is populated from defaults (not in the TOML file)
    assert isinstance(result, SceneConfig)
    assert isinstance(result.zones, ZoneConfig)
    assert isinstance(result.zones.house, HouseConfig)
    assert isinstance(result.zones.colors, ZoneColorsConfig)
