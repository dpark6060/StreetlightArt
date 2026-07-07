# Radical
"""Tests for devmode.render_helpers: debounce, pass assembly, config helpers."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import SceneConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.devmode.render_helpers import (
    DEBOUNCE_SECONDS,
    build_render_passes,
    ensure_devmode_config,
    rebuild_real_lights,
    should_render_full,
    sync_camera_from_config,
)
from streets_at_night.rendering.compositor import render_composite


@pytest.fixture
def config() -> SceneConfig:
    """Return a default SceneConfig built from DEFAULTS."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


def test_should_render_full_false_while_moving():
    # Act / Assert
    assert should_render_full(1.0, 5.0, dirty=True) is False


def test_should_render_full_false_before_debounce_elapses():
    # Act / Assert
    assert should_render_full(1.0, 1.0 + DEBOUNCE_SECONDS / 2, dirty=False) is False


def test_should_render_full_true_after_debounce():
    # Act / Assert
    assert should_render_full(1.0, 1.0 + DEBOUNCE_SECONDS + 0.01, dirty=False) is True


def test_should_render_full_false_when_no_pending_move():
    # Act / Assert
    assert should_render_full(None, 10.0, dirty=False) is False


def test_build_render_passes_includes_composite_last_without_labels():
    # Act
    passes = build_render_passes(mock.MagicMock(), gbuffer=None, use_labels=False)

    # Assert — composite must resolve the light buffer and be the final pass.
    assert render_composite in passes
    assert passes[-1] is render_composite


def test_build_render_passes_appends_label_pass_when_requested():
    # Arrange
    scene = mock.MagicMock()

    # Act
    without = build_render_passes(scene, gbuffer=None, use_labels=False)
    with_labels = build_render_passes(scene, gbuffer=None, use_labels=True)

    # Assert — labels add exactly one trailing pass after composite.
    assert len(with_labels) == len(without) + 1
    assert with_labels[-2] is render_composite


def test_ensure_devmode_config_populates_when_none(config):
    # Arrange
    config.devmode = None

    # Act
    ensure_devmode_config(config)

    # Assert
    assert config.devmode is not None


def test_ensure_devmode_config_leaves_existing_untouched(config):
    # Arrange
    existing = config.devmode

    # Act
    ensure_devmode_config(config)

    # Assert
    assert config.devmode is existing


def test_sync_camera_from_config_sets_camera_pose(config):
    # Arrange
    scene = mock.MagicMock()
    config.devmode.camera.pos_x = 11.0
    config.devmode.camera.pos_y = 22.0
    config.devmode.camera.pos_z = 33.0
    config.devmode.camera.fov = 55.0

    # Act
    sync_camera_from_config(scene, config)

    # Assert
    assert np.array_equal(scene.camera.position, np.array([11.0, 22.0, 33.0]))
    assert scene.camera.fov_deg == 55.0


def test_rebuild_real_lights_clears_and_repopulates(config):
    # Arrange
    scene = mock.MagicMock()
    scene.lights = ["stale"]
    scene.car_pairs = ["stale"]

    # Act — populate_lights is imported inside the function from scene_builder.
    with mock.patch("streets_at_night.placement.scene_builder.populate_lights") as mock_pop:
        result = rebuild_real_lights(scene, config)

    # Assert — lights reset, spatial index rebuilt, placer invoked, scene returned.
    assert scene.lights == []
    assert scene.car_pairs == []
    scene.build_spatial_index.assert_called_once()
    mock_pop.assert_called_once()
    assert result is scene
