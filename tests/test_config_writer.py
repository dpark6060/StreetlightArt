# Radical
"""Tests for streets_at_night.devmode.config_writer round-trip serialization."""

from dataclasses import asdict
from pathlib import Path

import pytest

from streets_at_night.config.loader import load_config
from streets_at_night.config.schema import SceneConfig
from streets_at_night.devmode.config_writer import config_to_dict, write_config


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


def test_config_writer_roundtrip_preserves_everything(tmp_path):
    """write(load(defaults)) is self-consistent under config_to_dict.

    This only verifies that the writer's own dict view of a config survives a
    round trip — it compares ``config_to_dict(cfg)`` on both sides, so a field
    the serializer omits entirely is absent from both sides and cannot show up
    as a diff. It does NOT catch a dropped field on its own; that guardrail is
    ``test_config_writer_roundtrip_preserves_mutated_fields``, which compares
    ``dataclasses.asdict()`` instead and would have caught the car brightness
    and grade/stars/heat_haze/region_name omissions.
    """
    cfg = load_config(Path("nonexistent.toml"))  # pure defaults
    write_config(cfg, tmp_path / "rt.toml")
    cfg2 = load_config(tmp_path / "rt.toml")
    assert config_to_dict(cfg) == config_to_dict(cfg2)


def test_config_writer_roundtrip_preserves_mutated_fields(base_config, tmp_path):
    """Mutating one field per section still round-trips exactly.

    Compares full dataclass state via ``asdict`` rather than ``config_to_dict``
    output — a field silently missing from the writer's own dict would never
    show up as a diff if both sides are filtered through the same buggy
    serializer (neither side has the key). ``asdict`` reads the dataclasses
    directly so a value lost in the TOML round trip (and quietly replaced by
    the loader's default) is caught.
    """
    base_config.lights.cars.headlight_brightness_min = 0.42
    base_config.lights.cars.taillight_brightness_max = 0.61
    base_config.grade.exposure = 1.25
    base_config.grade.tonemap = "linear"
    base_config.stars.stars_enabled = True
    base_config.stars.count = 500
    base_config.heat_haze.enabled = False
    base_config.heat_haze.layers[0].weight = 0.99
    base_config.region_name = "san_bernardino"

    result = _round_trip(base_config, tmp_path)

    assert asdict(result) == asdict(base_config)
