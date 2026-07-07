# Radical
"""Unit tests for config.camera_io TOML read/write helpers."""

import tomllib
from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.camera_io import (
    build_camera_toml_dict,
    save_camera_position,
    write_bounds_to_toml,
    write_camera_to_toml,
)
from streets_at_night.config.schema import BoundsConfig
from streets_at_night.scene.scene import Scene


@pytest.fixture
def bounds():
    return BoundsConfig(north=34.3, south=34.1, east=-117.2, west=-117.4)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "scene.toml"
    path.write_text(
        "[camera]\n"
        "lat = 1.0\n"
        "[bounds]\n"
        "north = 1.0\n"
        "south = 2.0\n"
        "east = 3.0\n"
        "west = 4.0\n"
        "[atmosphere]\n"
        "intensity = 0.5\n"
    )
    return path


# ---------------------------------------------------------------------------
# write_bounds_to_toml
# ---------------------------------------------------------------------------


def test_write_bounds_to_toml_preserves_unrelated_sections(config_file, bounds):
    write_bounds_to_toml(config_file, bounds)

    with open(config_file, "rb") as f:
        data = tomllib.load(f)
    assert data["atmosphere"] == {"intensity": 0.5}
    assert data["camera"] == {"lat": 1.0}


def test_write_bounds_to_toml_replaces_bounds_values(config_file, bounds):
    write_bounds_to_toml(config_file, bounds)

    with open(config_file, "rb") as f:
        data = tomllib.load(f)
    assert data["bounds"] == {
        "north": 34.3,
        "south": 34.1,
        "east": -117.2,
        "west": -117.4,
    }


def test_write_bounds_to_toml_creates_bounds_section_when_absent(tmp_path, bounds):
    path = tmp_path / "scene.toml"
    path.write_text("[camera]\nlat = 1.0\n")

    write_bounds_to_toml(path, bounds)

    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data["bounds"]["north"] == 34.3
    assert data["camera"] == {"lat": 1.0}


def test_write_bounds_to_toml_missing_file_raises(tmp_path, bounds):
    with pytest.raises(FileNotFoundError):
        write_bounds_to_toml(tmp_path / "nope.toml", bounds)


# ---------------------------------------------------------------------------
# write_camera_to_toml
# ---------------------------------------------------------------------------


def test_write_camera_to_toml_updates_existing_camera(config_file):
    write_camera_to_toml(config_file, {"lat": 9.0, "fov": 60.0})

    with open(config_file, "rb") as f:
        data = tomllib.load(f)
    assert data["camera"]["lat"] == 9.0
    assert data["camera"]["fov"] == 60.0
    assert data["atmosphere"] == {"intensity": 0.5}


def test_write_camera_to_toml_creates_camera_section_when_absent(tmp_path):
    path = tmp_path / "scene.toml"
    path.write_text("[bounds]\nnorth = 1.0\n")

    write_camera_to_toml(path, {"lat": 9.0})

    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data["camera"] == {"lat": 9.0}
    assert data["bounds"] == {"north": 1.0}


# ---------------------------------------------------------------------------
# save_camera_position
# ---------------------------------------------------------------------------


def test_save_camera_position_no_transformer_prints_and_skips_write(tmp_path, capsys):
    missing = tmp_path / "nope.toml"
    scene = mock.MagicMock(spec=Scene)
    scene.transformer = None

    save_camera_position(scene, missing)

    captured = capsys.readouterr()
    assert "no coordinate transformer" in captured.out
    assert not missing.exists()


@mock.patch("streets_at_night.config.camera_io.write_camera_to_toml")
@mock.patch("streets_at_night.config.camera_io.build_camera_toml_dict")
def test_save_camera_position_writes_when_transformer_present(
    mock_build, mock_write, config_file
):
    scene = mock.MagicMock(spec=Scene)
    scene.transformer = object()

    save_camera_position(scene, config_file)

    mock_build.assert_called_once_with(scene)
    mock_write.assert_called_once_with(config_file, mock_build.return_value)


# ---------------------------------------------------------------------------
# build_camera_toml_dict
# ---------------------------------------------------------------------------


def test_build_camera_toml_dict_converts_camera_and_target(tmp_path):
    scene = mock.MagicMock(spec=Scene)
    scene.transformer = object()
    scene.camera = mock.MagicMock()
    scene.camera.position = np.array([10.0, 20.0, 30.0])
    scene.camera.target = np.array([40.0, 50.0, 60.0])
    scene.camera.fov_deg = 55.0

    with mock.patch(
        "streets_at_night.config.camera_io.enu_to_wgs84",
        side_effect=[(34.2, -117.3, 1500.0), (34.1, -117.2, 300.0)],
    ), mock.patch(
        "streets_at_night.config.camera_io.meters_to_feet",
        side_effect=[4921.0, 984.0],
    ):
        result = build_camera_toml_dict(scene)

    assert result == {
        "lat": 34.2,
        "lon": -117.3,
        "altitude_ft": 4921.0,
        "target_lat": 34.1,
        "target_lon": -117.2,
        "target_altitude_ft": 984.0,
        "fov": 55.0,
    }
