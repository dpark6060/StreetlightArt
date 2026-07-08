# Radical
"""Unit tests for streets_at_night.scene.scene_cache."""

import copy

import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import SceneConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.scene.scene_cache import (
    _MAX_CACHED_SCENES,
    compute_scene_cache_key,
    load_scene_content,
    save_scene_content,
)


@pytest.fixture
def default_config() -> SceneConfig:
    """Return a full SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


def _bbox(config: SceneConfig) -> tuple:
    b = config.bounds
    return (b.north, b.south, b.east, b.west)


def test_compute_scene_cache_key_stable_for_same_inputs(default_config, tmp_path):
    key1 = compute_scene_cache_key(default_config, _bbox(default_config), tmp_path)
    key2 = compute_scene_cache_key(default_config, _bbox(default_config), tmp_path)

    assert key1 == key2


def test_compute_scene_cache_key_changes_with_seed(default_config, tmp_path):
    key1 = compute_scene_cache_key(default_config, _bbox(default_config), tmp_path)
    other = copy.deepcopy(default_config)
    other.seed = default_config.seed + 1

    key2 = compute_scene_cache_key(other, _bbox(other), tmp_path)

    assert key1 != key2


def test_compute_scene_cache_key_changes_with_camera_origin(default_config, tmp_path):
    # The ENU origin is the camera/target lat-lon midpoint, so moving the
    # camera latitude shifts every scene coordinate and must change the key.
    key1 = compute_scene_cache_key(default_config, _bbox(default_config), tmp_path)
    other = copy.deepcopy(default_config)
    other.camera.lat = default_config.camera.lat + 0.01

    key2 = compute_scene_cache_key(other, _bbox(other), tmp_path)

    assert key1 != key2


def test_compute_scene_cache_key_changes_with_lights(default_config, tmp_path):
    key1 = compute_scene_cache_key(default_config, _bbox(default_config), tmp_path)
    other = copy.deepcopy(default_config)
    other.lights.cars.separation_min += 1.0

    key2 = compute_scene_cache_key(other, _bbox(other), tmp_path)

    assert key1 != key2


def test_compute_scene_cache_key_unchanged_by_render_only_config(default_config, tmp_path):
    # Atmosphere/exposure do not affect scene content, so tuning them must not
    # invalidate the cache — that is the whole point of the cache.
    key1 = compute_scene_cache_key(default_config, _bbox(default_config), tmp_path)
    other = copy.deepcopy(default_config)
    other.grade.exposure += 1.0
    other.atmosphere.haze_alpha += 5

    key2 = compute_scene_cache_key(other, _bbox(other), tmp_path)

    assert key1 == key2


def test_save_and_load_scene_content_roundtrip(tmp_path):
    content = {"lights": [1, 2, 3], "index_bounds": (0.0, 0.0, 10.0, 10.0)}

    save_scene_content(tmp_path, "abc123", content)
    loaded = load_scene_content(tmp_path, "abc123")

    assert loaded == content


def test_load_scene_content_returns_none_on_miss(tmp_path):
    assert load_scene_content(tmp_path, "does_not_exist") is None


def test_save_scene_content_evicts_oldest_beyond_cap(tmp_path):
    for i in range(_MAX_CACHED_SCENES + 3):
        save_scene_content(tmp_path, f"key_{i}", {"n": i})

    remaining = list((tmp_path / "scene_cache").glob("*.pkl"))
    assert len(remaining) == _MAX_CACHED_SCENES
