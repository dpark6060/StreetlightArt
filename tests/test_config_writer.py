# Radical
"""Tests for streets_at_night.devmode.config_writer round-trip serialization."""

from pathlib import Path

import pytest

from streets_at_night.config.loader import load_config
from streets_at_night.config.schema import SceneConfig
from streets_at_night.devmode.config_writer import write_config


@pytest.fixture
def base_config(tmp_path: Path) -> SceneConfig:
    """Return a SceneConfig built from defaults (empty TOML overlay)."""
    empty = tmp_path / "empty.toml"
    empty.write_text("")
    return load_config(empty)


def _round_trip(config: SceneConfig, tmp_path: Path) -> SceneConfig:
    """Write config to TOML and load it back."""
    out = tmp_path / "round_trip.toml"
    write_config(config, out)
    return load_config(out)


def test_write_config_round_trips_color_temp_shift_k(base_config, tmp_path):
    # Arrange — the city-wide master lever
    base_config.lights.streetlights.color_temp_shift_k = 1500.0

    # Act
    result = _round_trip(base_config, tmp_path)

    # Assert
    assert result.lights.streetlights.color_temp_shift_k == 1500.0


def test_write_config_round_trips_streetlight_profile_kelvin_and_beam(base_config, tmp_path):
    # Arrange — mutate the first profiled highway entry
    profile = next(e.profile for e in base_config.lights.streetlights.highway if e.profile)
    profile.color_temp_k = 3300.0
    profile.saturation = 1.45
    profile.beam_angle_deg = 138.0

    # Act
    result = _round_trip(base_config, tmp_path)

    # Assert
    out_profile = next(e.profile for e in result.lights.streetlights.highway if e.profile)
    assert out_profile.color_temp_k == 3300.0
    assert out_profile.saturation == 1.45
    assert out_profile.beam_angle_deg == 138.0


def test_write_config_round_trips_glow_wing_fields(base_config, tmp_path):
    # Arrange
    base_config.atmosphere.glow_wing_exponent = 3.1
    base_config.atmosphere.glow_tail_strength = 0.22
    base_config.atmosphere.glow_cutoff_mult = 6.0

    # Act
    result = _round_trip(base_config, tmp_path)

    # Assert
    assert result.atmosphere.glow_wing_exponent == 3.1
    assert result.atmosphere.glow_tail_strength == 0.22
    assert result.atmosphere.glow_cutoff_mult == 6.0


def test_write_config_round_trips_airlight_fields(base_config, tmp_path):
    # Arrange — haze_alpha is the airlight veil strength; haze_enabled its toggle
    base_config.atmosphere.haze_alpha = 45
    base_config.atmosphere.haze_enabled = False

    # Act
    result = _round_trip(base_config, tmp_path)

    # Assert
    assert result.atmosphere.haze_alpha == 45
    assert result.atmosphere.haze_enabled is False


def test_write_config_omits_removed_max_distance(base_config, tmp_path):
    # Act
    out = tmp_path / "no_max.toml"
    write_config(base_config, out)
    text = out.read_text()

    # Assert — the retired knob is not written
    assert "max_distance =" not in text
