# Radical
"""Tests for place/look layered configs: loading (config.layers) and saving (config_writer)."""

import tomllib
from pathlib import Path

import pytest

from streets_at_night.config.layers import PLACE_KEYS, load_layered, split_for_save
from streets_at_night.config.loader import load_config
from streets_at_night.devmode.config_writer import write_config


@pytest.fixture
def layered(tmp_path: Path) -> Path:
    """Write a look file and a place file that layers over it; return the place path."""
    (tmp_path / "looks").mkdir()
    (tmp_path / "looks" / "night.toml").write_text(
        "[atmosphere]\nfalloff_distance = 5000.0\nillumination_brightness = 70\n"
        "[grade]\nexposure = 0.5\n"
    )
    place = tmp_path / "rochester.toml"
    place.write_text(
        'look = "looks/night.toml"\nregion_name = "rochester"\n'
        "[bounds]\nnorth = 43.3\nsouth = 42.9\neast = -77.3\nwest = -77.8\n"
        "[atmosphere]\nillumination_brightness = 90\n"
    )
    return place


def _read(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# load_layered / load_config
# ---------------------------------------------------------------------------


def test_load_config_uses_look_values(layered):
    # Act
    config = load_config(layered)

    # Assert
    assert config.atmosphere.falloff_distance == 5000.0
    assert config.grade.exposure == 0.5


def test_load_config_place_overrides_look(layered):
    # Act
    config = load_config(layered)

    # Assert
    assert config.atmosphere.illumination_brightness == 90


def test_load_config_keeps_look_reference(layered):
    # Act
    config = load_config(layered)

    # Assert
    assert config.look == "looks/night.toml"
    assert config.region_name == "rochester"


def test_load_layered_missing_look_raises(tmp_path):
    # Arrange
    place = tmp_path / "city.toml"
    place.write_text('look = "looks/nope.toml"\n')

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="nope.toml"):
        load_layered(place)


def test_load_layered_nested_look_raises(tmp_path):
    # Arrange
    (tmp_path / "a.toml").write_text('look = "b.toml"\n')
    place = tmp_path / "city.toml"
    place.write_text('look = "a.toml"\n')

    # Act / Assert
    with pytest.raises(ValueError, match="can't nest"):
        load_layered(place)


def test_load_config_without_look_is_unchanged(tmp_path):
    # Arrange
    single = tmp_path / "single.toml"
    single.write_text("[grade]\nexposure = 1.5\n")

    # Act
    config = load_config(single)

    # Assert
    assert config.look is None
    assert config.grade.exposure == 1.5


# ---------------------------------------------------------------------------
# split_for_save
# ---------------------------------------------------------------------------


def test_split_for_save_routes_place_keys_to_place():
    # Arrange
    full = {"look": "l.toml", "camera": {"fov": 60}, "bounds": {"north": 1.0}, "grade": {"exposure": 1.0}}

    # Act
    place, look = split_for_save(full, place_raw={}, look_raw={})

    # Assert
    assert set(place) == {"look", "camera", "bounds"}
    assert set(look) == {"grade"}
    assert set(place) <= PLACE_KEYS


def test_split_for_save_place_override_stays_in_place_and_look_keeps_its_value():
    # Arrange — the city overrides exposure; the shared look says 0.5
    full = {"look": "l.toml", "grade": {"exposure": 2.0, "tonemap": "aces"}}
    place_raw = {"look": "l.toml", "grade": {"exposure": 1.0}}
    look_raw = {"grade": {"exposure": 0.5}}

    # Act
    place, look = split_for_save(full, place_raw, look_raw)

    # Assert — edited override lands in the city file, shared look untouched
    assert place["grade"] == {"exposure": 2.0}
    assert look["grade"]["exposure"] == 0.5
    assert look["grade"]["tonemap"] == "aces"


def test_split_for_save_override_of_default_restores_default_in_look():
    # Arrange — the look file doesn't set exposure, so its value is the default
    full = {"grade": {"exposure": 2.0}}
    place_raw = {"grade": {"exposure": 1.0}}

    # Act
    _, look = split_for_save(full, place_raw, look_raw={})

    # Assert
    assert look["grade"]["exposure"] == 0.0


# ---------------------------------------------------------------------------
# write_config with a layered config
# ---------------------------------------------------------------------------


def test_write_config_light_edit_goes_to_look_file(layered):
    # Arrange
    config = load_config(layered)
    config.lights.streetlights.color_temp_shift_k = 750.0

    # Act
    write_config(config, layered)

    # Assert
    look = _read(layered.parent / "looks" / "night.toml")
    assert look["lights"]["streetlights"]["color_temp_shift_k"] == 750.0
    assert "lights" not in _read(layered)


def test_write_config_camera_edit_goes_to_place_file(layered):
    # Arrange
    config = load_config(layered)
    config.camera.fov = 42.0

    # Act
    write_config(config, layered)

    # Assert
    assert _read(layered)["camera"]["fov"] == 42.0
    assert "camera" not in _read(layered.parent / "looks" / "night.toml")


def test_write_config_keeps_city_override_out_of_shared_look(layered):
    # Arrange
    config = load_config(layered)

    # Act
    write_config(config, layered)

    # Assert
    assert _read(layered)["atmosphere"] == {"illumination_brightness": 90}
    look = _read(layered.parent / "looks" / "night.toml")
    assert look["atmosphere"]["illumination_brightness"] == 70


def test_write_config_layered_round_trips(layered):
    # Arrange
    config = load_config(layered)
    config.grade.exposure = -1.25
    config.bounds.north = 43.31

    # Act
    write_config(config, layered)
    reloaded = load_config(layered)

    # Assert
    assert reloaded == config
