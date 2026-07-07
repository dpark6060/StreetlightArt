# Radical
"""Tests for streets_at_night.config.loader.load_config.

NOTE on FileNotFoundError: The source code (_load_merged) silently returns
defaults when the path does not exist — it does NOT raise FileNotFoundError.
test_load_config_nonexistent_file_returns_defaults reflects the actual
behavior. If raising on a missing file is a requirement, that is a source-code
bug to fix in loader.py, not a test bug.
"""

import tempfile
from pathlib import Path

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import load_config
from streets_at_night.config.schema import HeatHazeConfig, RoadsConfig, SceneConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_toml(tmp_path: Path) -> Path:
    """A valid but empty TOML file."""
    toml_file = tmp_path / "empty.toml"
    toml_file.write_text("")
    return toml_file


@pytest.fixture
def camera_lat_toml(tmp_path: Path) -> Path:
    """TOML with only camera.lat overridden."""
    toml_file = tmp_path / "camera_lat.toml"
    toml_file.write_text("[camera]\nlat = 1.23\n")
    return toml_file


@pytest.fixture
def bounds_toml(tmp_path: Path) -> Path:
    """TOML with a full bounds section."""
    toml_file = tmp_path / "bounds.toml"
    toml_file.write_text(
        "[bounds]\nnorth = 35.0\nsouth = 34.0\neast = -117.0\nwest = -118.0\n"
    )
    return toml_file


@pytest.fixture
def quality_low_override_toml(tmp_path: Path) -> Path:
    """TOML that overrides only the low quality width."""
    toml_file = tmp_path / "quality_low.toml"
    toml_file.write_text("[quality.low]\nwidth = 1280\n")
    return toml_file


@pytest.fixture
def seed_toml(tmp_path: Path) -> Path:
    """TOML with a seed value."""
    toml_file = tmp_path / "seed.toml"
    toml_file.write_text("seed = 99\n")
    return toml_file


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_load_config_returns_scene_config_type(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result, SceneConfig)


# ---------------------------------------------------------------------------
# Camera section
# ---------------------------------------------------------------------------

def test_load_config_camera_lat_from_toml(camera_lat_toml: Path):
    # Act
    result = load_config(camera_lat_toml)

    # Assert
    assert result.camera.lat == 1.23


def test_load_config_camera_uses_default_when_not_in_toml(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.camera.lat == DEFAULTS["camera"]["lat"]


def test_load_config_partial_camera_override_preserves_other_camera_fields(
    camera_lat_toml: Path,
):
    # Act
    result = load_config(camera_lat_toml)

    # Assert — lat is overridden
    assert result.camera.lat == 1.23
    # All other camera fields come from defaults
    assert result.camera.lon == DEFAULTS["camera"]["lon"]
    assert result.camera.altitude_ft == DEFAULTS["camera"]["altitude_ft"]
    assert result.camera.fov == DEFAULTS["camera"]["fov"]


# ---------------------------------------------------------------------------
# Bounds section
# ---------------------------------------------------------------------------

def test_load_config_bounds_from_toml(bounds_toml: Path):
    # Act
    result = load_config(bounds_toml)

    # Assert
    assert result.bounds.north == 35.0
    assert result.bounds.south == 34.0
    assert result.bounds.east == -117.0
    assert result.bounds.west == -118.0


# ---------------------------------------------------------------------------
# Quality presets section
# ---------------------------------------------------------------------------

def test_load_config_quality_presets_contains_low_medium_high(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert "low" in result.quality_presets
    assert "medium" in result.quality_presets
    assert "high" in result.quality_presets


def test_load_config_quality_preset_width_from_toml(quality_low_override_toml: Path):
    # Act
    result = load_config(quality_low_override_toml)

    # Assert
    assert result.quality_presets["low"].width == 1280


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def test_load_config_seed_from_toml(seed_toml: Path):
    # Act
    result = load_config(seed_toml)

    # Assert
    assert result.seed == 99


def test_load_config_seed_uses_default_when_missing(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.seed == DEFAULTS["seed"]


# ---------------------------------------------------------------------------
# Missing file behavior
#
# The source returns defaults silently for a missing file rather than raising.
# This test documents the actual behavior. If FileNotFoundError is required,
# that is a source-level fix needed in loader._load_merged.
# ---------------------------------------------------------------------------

def test_load_config_nonexistent_file_returns_defaults(tmp_path: Path):
    # Arrange
    missing_path = tmp_path / "does_not_exist.toml"

    # Act
    result = load_config(missing_path)

    # Assert — defaults are returned without error
    assert isinstance(result, SceneConfig)
    assert result.seed == DEFAULTS["seed"]
    assert result.camera.lat == DEFAULTS["camera"]["lat"]


# ---------------------------------------------------------------------------
# Roads config section
# ---------------------------------------------------------------------------

def test_load_config_roads_returns_roads_config(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result.roads, RoadsConfig)


def test_load_config_roads_export_uses_defaults(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.roads.export.highway.color == DEFAULTS["roads"]["export"]["highway"]["color"]
    assert result.roads.export.highway.width == DEFAULTS["roads"]["export"]["highway"]["width"]


def test_load_config_roads_preview_uses_defaults(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.roads.preview.highway.color == DEFAULTS["roads"]["preview"]["highway"]["color"]
    assert result.roads.preview.highway.width == DEFAULTS["roads"]["preview"]["highway"]["width"]


@pytest.fixture
def road_color_override_toml(tmp_path: Path) -> Path:
    """TOML that overrides only the export highway color."""
    toml_file = tmp_path / "road_colors.toml"
    toml_file.write_text(
        '[roads.export.highway]\ncolor = "#FF0000"\n'
    )
    return toml_file


def test_load_config_roads_partial_override(road_color_override_toml: Path):
    # Act
    result = load_config(road_color_override_toml)

    # Assert — highway color overridden
    assert result.roads.export.highway.color == "#FF0000"
    # Width preserved from default
    assert result.roads.export.highway.width == DEFAULTS["roads"]["export"]["highway"]["width"]
    # Other road types preserved
    assert result.roads.export.minor.color == DEFAULTS["roads"]["export"]["minor"]["color"]
    # Preview unchanged
    assert result.roads.preview.highway.color == DEFAULTS["roads"]["preview"]["highway"]["color"]


# ---------------------------------------------------------------------------
# Atmosphere section
# ---------------------------------------------------------------------------


def test_load_config_atmosphere_defaults_are_loaded(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert — falloff_distance matches the default value
    assert result.atmosphere.falloff_distance == DEFAULTS["atmosphere"]["falloff_distance"]


@pytest.fixture
def atmosphere_override_toml(tmp_path: Path) -> Path:
    """TOML with atmosphere.falloff_distance overridden."""
    toml_file = tmp_path / "atmosphere.toml"
    toml_file.write_text("[atmosphere]\nfalloff_distance = 5000.0\n")
    return toml_file


def test_load_config_atmosphere_can_be_overridden(atmosphere_override_toml: Path):
    # Act
    result = load_config(atmosphere_override_toml)

    # Assert — override applied
    assert result.atmosphere.falloff_distance == 5000.0


# ---------------------------------------------------------------------------
# Heat haze config section
# ---------------------------------------------------------------------------


def test_load_config_heat_haze_returns_heat_haze_config(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert isinstance(result.heat_haze, HeatHazeConfig)


def test_load_config_heat_haze_defaults_match_defaults_dict(empty_toml: Path):
    # Act
    result = load_config(empty_toml)

    # Assert
    assert result.heat_haze.enabled == DEFAULTS["heat_haze"]["enabled"]
    assert result.heat_haze.amplitude_px == DEFAULTS["heat_haze"]["amplitude_px"]
    assert result.heat_haze.max_distance_m == DEFAULTS["heat_haze"]["max_distance_m"]
    assert result.heat_haze.distance_exponent == DEFAULTS["heat_haze"]["distance_exponent"]
    assert result.heat_haze.sky_distance_factor == DEFAULTS["heat_haze"]["sky_distance_factor"]
    assert result.heat_haze.screen_width_px == DEFAULTS["heat_haze"]["screen_width_px"]
    assert result.heat_haze.screen_height_px == DEFAULTS["heat_haze"]["screen_height_px"]
    assert len(result.heat_haze.layers) == len(DEFAULTS["heat_haze"]["layers"])


@pytest.fixture
def heat_haze_override_toml(tmp_path: Path) -> Path:
    """TOML with heat_haze.amplitude_px overridden."""
    toml_file = tmp_path / "heat_haze.toml"
    toml_file.write_text("[heat_haze]\namplitude_px = 1.25\n")
    return toml_file


def test_load_config_heat_haze_amplitude_override(heat_haze_override_toml: Path):
    # Act
    result = load_config(heat_haze_override_toml)

    # Assert — overridden field changed
    assert result.heat_haze.amplitude_px == 1.25
    # Non-overridden fields stay at defaults
    assert result.heat_haze.distance_exponent == DEFAULTS["heat_haze"]["distance_exponent"]


def test_load_config_heat_haze_absent_section_uses_defaults(empty_toml: Path):
    # Act — no [heat_haze] section in the TOML at all
    result = load_config(empty_toml)

    # Assert — still gets a populated config object, not None
    assert result.heat_haze is not None
    assert result.heat_haze.sky_distance_factor == DEFAULTS["heat_haze"]["sky_distance_factor"]


@pytest.fixture
def heat_haze_layers_override_toml(tmp_path: Path) -> Path:
    """TOML with the heat_haze.layers list overridden."""
    toml_file = tmp_path / "heat_haze_layers.toml"
    toml_file.write_text(
        "[[heat_haze.layers]]\n"
        "r0 = 0.2\n"
        "L0 = 1.0\n"
        "pixel_scale_m = 0.1\n"
        "weight = 0.5\n"
        "wind_x_pxs = 50.0\n"
        "wind_y_pxs = 5.0\n"
    )
    return toml_file


def test_load_config_heat_haze_layers_override(heat_haze_layers_override_toml: Path):
    # Act
    result = load_config(heat_haze_layers_override_toml)

    # Assert — single user-defined layer replaces the multi-layer default list
    assert len(result.heat_haze.layers) == 1
    layer = result.heat_haze.layers[0]
    # The legacy 'r0' key in the TOML is silently ignored — the field was
    # removed (it was a mathematical no-op under screen renormalization).
    assert not hasattr(layer, "r0")
    assert layer.L0 == 1.0
    assert layer.pixel_scale_m == 0.1
    assert layer.weight == 0.5
    assert layer.wind_x_pxs == 50.0
    assert layer.wind_y_pxs == 5.0


# ---------------------------------------------------------------------------
# Pre-migration streetlight profiles (missing color_temp_k / beam_angle_deg)
# ---------------------------------------------------------------------------


@pytest.fixture
def pre_migration_streetlight_toml(tmp_path: Path) -> Path:
    """TOML with a highway streetlight profile predating the Kelvin/beam-angle
    migration — no color_temp_k or beam_angle_deg field."""
    toml_file = tmp_path / "pre_migration.toml"
    toml_file.write_text(
        "[[lights.streetlights.highway]]\n"
        "weight = 1.0\n"
        "[lights.streetlights.highway.profile]\n"
        "saturation = 1.0\n"
        "height_min = 8.0\n"
        "height_max = 10.0\n"
        "brightness_min = 0.5\n"
        "brightness_max = 0.8\n"
        "glow_radius = 20.0\n"
        "illumination_intensity = 1.0\n"
        "spacing_min = 30.0\n"
        "spacing_max = 40.0\n"
        "side = \"both\"\n"
    )
    return toml_file


def test_load_config_pre_migration_streetlight_profile_raises_value_error(
    pre_migration_streetlight_toml: Path,
):
    # Act & Assert — a bare KeyError would give the user no idea what to do
    with pytest.raises(ValueError, match="color_temp_k.*beam_angle_deg"):
        load_config(pre_migration_streetlight_toml)
