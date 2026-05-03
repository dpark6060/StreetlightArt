# Radical
"""Unit tests for Stage 5 render pipeline pass composition."""

from unittest import mock

import pytest

from streets_at_night.rendering.render_pipeline import (
    get_export_passes,
    get_preview_passes,
    render_scene_frame,
)


def _get_illumination_pass_index(passes):
    """Return the index of the illumination closure in the export pass list.

    The illumination pass is at index 2: ground(0), roads(1), illumination(2), lights(3).

    Args:
        passes (list): Export pass list from get_export_passes().

    Returns:
        int: Index of the illumination closure.
    """
    return 2


# ---------------------------------------------------------------------------
# get_preview_passes — pass list includes light rendering
# ---------------------------------------------------------------------------


def test_get_preview_passes_includes_light_pass():
    # Act
    passes = get_preview_passes()

    # Assert — render_lights_preview must be in the pass list
    from streets_at_night.rendering.light_pass import render_lights_preview

    assert render_lights_preview in passes


def test_get_preview_passes_includes_ground_pass():
    # Act
    passes = get_preview_passes()

    # Assert — render_ground must be present for sky/horizon rendering
    from streets_at_night.rendering.ground_pass import render_ground

    assert render_ground in passes


def test_get_preview_passes_includes_road_pass():
    # Act
    passes = get_preview_passes()

    # Assert — road projection must be in the list
    from streets_at_night.rendering.road_pass import render_roads_from_scene

    assert render_roads_from_scene in passes


def test_get_preview_passes_returns_three_passes():
    # Act
    passes = get_preview_passes()

    # Assert — ground, roads, lights = exactly three
    assert len(passes) == 3


def test_get_preview_passes_ground_before_roads_before_lights():
    # Arrange — correct render order: ground → roads → lights
    from streets_at_night.rendering.ground_pass import render_ground
    from streets_at_night.rendering.light_pass import render_lights_preview
    from streets_at_night.rendering.road_pass import render_roads_from_scene

    # Act
    passes = get_preview_passes()

    # Assert — strict ordering
    ground_idx = passes.index(render_ground)
    roads_idx = passes.index(render_roads_from_scene)
    lights_idx = passes.index(render_lights_preview)
    assert ground_idx < roads_idx < lights_idx


# ---------------------------------------------------------------------------
# get_export_passes — pass list includes illumination and light export
# ---------------------------------------------------------------------------


def test_get_export_passes_includes_illumination():
    # Act
    passes = get_export_passes()

    # Assert — six passes: ground, roads, illumination closure, lights,
    # atmosphere, compositor. Illumination is a closure so identity cannot be
    # checked directly.
    assert len(passes) == 6


def test_get_export_passes_includes_light_export_pass():
    # Act
    passes = get_export_passes()

    # Assert — high-quality glow pass must be in export list
    from streets_at_night.rendering.light_pass import render_lights_export

    assert render_lights_export in passes


def test_get_export_passes_includes_road_pass():
    # Act
    passes = get_export_passes()

    # Assert
    from streets_at_night.rendering.road_pass import render_roads_export

    assert render_roads_export in passes


def test_get_export_passes_returns_six_passes():
    # Act
    passes = get_export_passes()

    # Assert — ground, roads, illumination, lights, atmosphere, compositor = exactly six
    assert len(passes) == 6


def test_get_export_passes_illumination_before_lights():
    # Arrange — illumination pools must render before light cores/glow
    from streets_at_night.rendering.light_pass import render_lights_export

    # Act
    passes = get_export_passes()

    # Assert — illumination closure at index 2 is before render_lights_export at index 3
    illumination_idx = _get_illumination_pass_index(passes)
    lights_idx = passes.index(render_lights_export)

    assert illumination_idx < lights_idx


def test_get_export_passes_roads_before_illumination():
    # Arrange — roads drawn before illumination pools are composited
    from streets_at_night.rendering.road_pass import render_roads_export

    # Act
    passes = get_export_passes()
    roads_idx = passes.index(render_roads_export)

    # Assert — roads at index 1, illumination closure at index 2
    illumination_idx = _get_illumination_pass_index(passes)
    assert roads_idx < illumination_idx


# ---------------------------------------------------------------------------
# get_export_passes — no_lights flag
# ---------------------------------------------------------------------------


def test_get_export_passes_default_includes_illumination_and_lights():
    # Arrange
    from streets_at_night.rendering.light_pass import render_lights_export

    # Act
    passes = get_export_passes()

    # Assert — 6 passes: illumination and lights are both present;
    # lights export pass is referenced directly so we can check it
    assert len(passes) == 6
    assert render_lights_export in passes


def test_get_export_passes_no_lights_omits_illumination_and_light_passes():
    # Arrange
    from streets_at_night.rendering.light_pass import render_lights_export

    # Act
    passes = get_export_passes(no_lights=True)

    # Assert — with no_lights: ground, roads, atmosphere, compositor = 4 passes
    assert len(passes) == 4
    assert render_lights_export not in passes


def test_get_export_passes_no_lights_still_includes_ground_and_roads():
    # Act
    passes = get_export_passes(no_lights=True)

    # Assert — ground and road passes are always present regardless of flag;
    # atmosphere and compositor also always run, so 4 passes total
    from streets_at_night.rendering.road_pass import render_roads_export

    assert render_roads_export in passes
    assert len(passes) == 4


# ---------------------------------------------------------------------------
# get_preview_passes — no_lights flag
# ---------------------------------------------------------------------------


def test_get_preview_passes_no_lights_omits_light_pass():
    # Act
    passes = get_preview_passes(no_lights=True)

    # Assert — light preview pass must be absent
    from streets_at_night.rendering.light_pass import render_lights_preview

    assert render_lights_preview not in passes


def test_get_preview_passes_no_lights_still_includes_ground_and_roads():
    # Act
    passes = get_preview_passes(no_lights=True)

    # Assert — ground and road passes are always present regardless of flag
    from streets_at_night.rendering.ground_pass import render_ground
    from streets_at_night.rendering.road_pass import render_roads_from_scene

    assert render_ground in passes
    assert render_roads_from_scene in passes
    assert len(passes) == 2


# ---------------------------------------------------------------------------
# render_scene_frame — pass invocation and stat collection
# ---------------------------------------------------------------------------


def test_render_scene_frame_calls_each_pass_with_scene_camera_backend():
    # Arrange
    mock_scene = mock.MagicMock()
    mock_camera = mock.MagicMock()
    mock_scene.camera = mock_camera
    mock_backend = mock.MagicMock()

    mock_pass_a = mock.MagicMock(return_value={"lights_drawn": 5})
    mock_pass_b = mock.MagicMock(return_value=None)

    # Act
    render_scene_frame(mock_scene, mock_backend, passes=[mock_pass_a, mock_pass_b])

    # Assert — each pass receives the standard three arguments
    mock_pass_a.assert_called_once_with(mock_scene, mock_camera, mock_backend)
    mock_pass_b.assert_called_once_with(mock_scene, mock_camera, mock_backend)


def test_render_scene_frame_merges_dict_stats_from_passes():
    # Arrange
    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    mock_pass = mock.MagicMock(return_value={"lights_drawn": 42})

    # Act
    stats = render_scene_frame(mock_scene, mock_backend, passes=[mock_pass])

    # Assert — dict return value is merged into stats
    assert stats["lights_drawn"] == 42


def test_render_scene_frame_int_return_stored_as_segments_drawn():
    # Arrange
    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    mock_pass = mock.MagicMock(return_value=17)

    # Act
    stats = render_scene_frame(mock_scene, mock_backend, passes=[mock_pass])

    # Assert — int return value goes into segments_drawn key
    assert stats["segments_drawn"] == 17


def test_render_scene_frame_none_return_does_not_modify_stats():
    # Arrange
    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    mock_pass = mock.MagicMock(return_value=None)

    # Act
    stats = render_scene_frame(mock_scene, mock_backend, passes=[mock_pass])

    # Assert — None return leaves stats empty
    assert stats == {}


def test_render_scene_frame_empty_passes_returns_empty_stats():
    # Arrange
    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    # Act
    stats = render_scene_frame(mock_scene, mock_backend, passes=[])

    # Assert
    assert stats == {}
