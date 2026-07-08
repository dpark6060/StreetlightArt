# Radical
"""Unit tests for scene_loader.load_scene orchestration and helpers."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from streets_at_night import scene_loader
from streets_at_night.config.schema import BoundsConfig


@pytest.fixture
def config():
    cfg = mock.MagicMock()
    preset = mock.MagicMock()
    preset.width = 1920
    preset.height = 1080
    cfg.quality_presets = {"low": preset}
    cfg.bounds = BoundsConfig(north=34.3, south=34.1, east=-117.2, west=-117.4)
    cfg.camera.lat = 34.23
    cfg.camera.lon = -117.31
    cfg.camera.altitude_ft = 5186.0
    cfg.camera.target_lat = 34.11
    cfg.camera.target_lon = -117.30
    cfg.camera.target_altitude_ft = 1000.0
    cfg.camera.fov = 60.0
    cfg.camera.near = 100.0
    cfg.camera.far = 60000.0
    return cfg


@pytest.fixture
def patched_boundary():
    """Patch every module-boundary call load_scene makes."""
    with mock.patch("streets_at_night.scene_loader.load_roads") as load_roads, \
        mock.patch("streets_at_night.scene_loader.classify_roads") as classify_roads, \
        mock.patch("streets_at_night.scene_loader.load_land_use") as load_land_use, \
        mock.patch("streets_at_night.scene_loader.load_buildings") as load_buildings, \
        mock.patch("streets_at_night.scene_loader.create_transformer") as create_transformer, \
        mock.patch("streets_at_night.scene_loader.create_camera") as create_camera, \
        mock.patch("streets_at_night.scene_loader.build_scene") as build_scene:
        classify_roads.return_value = ["r1", "r2"]
        load_land_use.return_value = ["lu"]
        load_buildings.return_value = ["b1", "b2", "b3"]
        create_transformer.return_value.wgs84_to_enu.return_value = (1.0, 2.0, 3.0)
        build_scene.return_value.road_segments = ["s1", "s2"]
        yield {
            "load_roads": load_roads,
            "classify_roads": classify_roads,
            "load_land_use": load_land_use,
            "load_buildings": load_buildings,
            "create_transformer": create_transformer,
            "create_camera": create_camera,
            "build_scene": build_scene,
        }


def test_load_scene_calls_each_boundary_with_right_args(config, patched_boundary):
    cache_dir = Path("cache")
    bbox = (34.3, 34.1, -117.2, -117.4)

    result = scene_loader.load_scene(config, "low", cache_dir, use_scene_cache=False)

    patched_boundary["load_roads"].assert_called_once_with(bbox, cache_dir)
    patched_boundary["classify_roads"].assert_called_once_with(
        patched_boundary["load_roads"].return_value
    )
    patched_boundary["load_land_use"].assert_called_once_with(bbox, cache_dir)
    patched_boundary["load_buildings"].assert_called_once_with(bbox, cache_dir)
    patched_boundary["create_transformer"].assert_called_once()
    patched_boundary["create_camera"].assert_called_once()
    patched_boundary["build_scene"].assert_called_once_with(
        ["r1", "r2"],
        patched_boundary["create_transformer"].return_value,
        patched_boundary["create_camera"].return_value,
        config,
        ["lu"],
        ["b1", "b2", "b3"],
    )
    assert result is patched_boundary["build_scene"].return_value


def test_load_scene_passes_preset_aspect_ratio_to_camera(config, patched_boundary):
    scene_loader.load_scene(config, "low", Path("cache"), use_scene_cache=False)

    _, kwargs = patched_boundary["create_camera"].call_args
    assert kwargs["fov_deg"] == 60.0
    assert kwargs["aspect_ratio"] == 1920 / 1080
    assert kwargs["near"] == 100.0
    assert kwargs["far"] == 60000.0


def test_load_scene_with_progress_calls_callback_and_prints_nothing(
    config, patched_boundary, capsys
):
    messages = []

    scene_loader.load_scene(
        config, "low", Path("cache"), progress=messages.append, use_scene_cache=False
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert any("Loading road network" in m for m in messages)


def test_load_scene_without_progress_prints_messages(config, patched_boundary, capsys):
    scene_loader.load_scene(config, "low", Path("cache"), use_scene_cache=False)

    captured = capsys.readouterr()
    assert "Loading road network" in captured.out


def test_load_scene_cache_hit_skips_build(config, patched_boundary):
    # A cache hit rehydrates from content and must not touch OSM or build_scene.
    with mock.patch(
        "streets_at_night.scene_loader.compute_scene_cache_key", return_value="k"
    ), mock.patch(
        "streets_at_night.scene_loader.load_scene_content", return_value={"lights": []}
    ), mock.patch(
        "streets_at_night.scene_loader._scene_from_content", return_value="cached_scene"
    ) as from_content:
        result = scene_loader.load_scene(config, "low", Path("cache"))

    assert result == "cached_scene"
    from_content.assert_called_once()
    patched_boundary["load_roads"].assert_not_called()
    patched_boundary["build_scene"].assert_not_called()


def test_load_scene_cache_miss_saves_content(config, patched_boundary):
    # A cache miss builds normally and persists the built content.
    with mock.patch(
        "streets_at_night.scene_loader.compute_scene_cache_key", return_value="k"
    ), mock.patch(
        "streets_at_night.scene_loader.load_scene_content", return_value=None
    ), mock.patch(
        "streets_at_night.scene_loader.save_scene_content"
    ) as save, mock.patch(
        "streets_at_night.scene_loader._content_from_scene", return_value={"c": 1}
    ):
        scene_loader.load_scene(config, "low", Path("cache"))

    patched_boundary["build_scene"].assert_called_once()
    save.assert_called_once_with(Path("cache"), "k", {"c": 1})


# ---------------------------------------------------------------------------
# bounds_to_bbox
# ---------------------------------------------------------------------------


def test_bounds_to_bbox_returns_nsew_tuple():
    bounds = BoundsConfig(north=1.0, south=2.0, east=3.0, west=4.0)

    assert scene_loader.bounds_to_bbox(bounds) == (1.0, 2.0, 3.0, 4.0)


# ---------------------------------------------------------------------------
# camera_position_enu / target_position_enu
# ---------------------------------------------------------------------------


def test_camera_position_enu_converts_via_transformer(config):
    transformer = mock.MagicMock()
    transformer.wgs84_to_enu.return_value = (10.0, 20.0, 30.0)

    result = scene_loader.camera_position_enu(config, transformer)

    assert np.array_equal(result, np.array([10.0, 20.0, 30.0]))


def test_target_position_enu_converts_via_transformer(config):
    transformer = mock.MagicMock()
    transformer.wgs84_to_enu.return_value = (40.0, 50.0, 60.0)

    result = scene_loader.target_position_enu(config, transformer)

    assert np.array_equal(result, np.array([40.0, 50.0, 60.0]))
