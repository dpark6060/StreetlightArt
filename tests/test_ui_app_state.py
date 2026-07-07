# Radical
"""Unit tests for streets_at_night.ui.app_state.AppState."""

from pathlib import Path
from unittest import mock

import pytest

from streets_at_night.config.schema import SceneConfig
from streets_at_night.scene.scene import Scene
from streets_at_night.ui.app_state import AppState


@pytest.fixture
def state() -> AppState:
    """Return an AppState with a cached scene already populated."""
    app_state = AppState(
        config_path=Path("scene_config.toml"),
        config=mock.MagicMock(spec=SceneConfig),
        cache_dir=Path("cache"),
    )
    app_state.scene = mock.MagicMock(spec=Scene)
    app_state.scene_quality = "low"
    app_state.scene_dirty = False
    return app_state


def test_invalidate_scene_clears_cached_scene(state: AppState):
    # Act
    state.invalidate_scene()

    # Assert
    assert state.scene is None
    assert state.scene_dirty is True


def test_set_config_replaces_config_and_path(state: AppState):
    # Arrange
    new_config = mock.MagicMock(spec=SceneConfig)
    new_path = Path("other.toml")

    # Act
    state.set_config(new_config, new_path)

    # Assert
    assert state.config is new_config
    assert state.config_path == new_path


def test_set_config_invalidates_scene(state: AppState):
    # Arrange
    new_config = mock.MagicMock(spec=SceneConfig)

    # Act
    state.set_config(new_config, None)

    # Assert
    assert state.config_path is None
    assert state.scene is None
    assert state.scene_dirty is True
