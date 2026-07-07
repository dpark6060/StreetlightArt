# Radical
"""Unit tests for Stage 6 additions to the render pipeline."""

import pytest

from streets_at_night.rendering.render_pipeline import (
    get_export_pass_names,
    get_export_passes,
    render_scene_frame_with_progress,
)


# ---------------------------------------------------------------------------
# get_export_passes — atmosphere and compositor always present
# ---------------------------------------------------------------------------


def test_get_export_passes_includes_atmosphere_pass():
    # Act — atmosphere is wrapped in a gbuffer-sharing closure, so check by name
    names = get_export_pass_names()

    # Assert — atmosphere pass must be present
    assert "atmosphere" in names


def test_get_export_passes_includes_compositor_pass():
    # Arrange
    from streets_at_night.rendering.compositor import render_composite

    # Act
    passes = get_export_passes()

    # Assert — compositor pass must be in the list
    assert render_composite in passes


def test_get_export_passes_atmosphere_after_lights():
    # Act — atmosphere runs after light glow (check by name; both are wrapped)
    names = get_export_pass_names()

    # Assert
    assert names.index("lights") < names.index("atmosphere")


def test_get_export_passes_compositor_after_atmosphere():
    # Act — compositor is the last pass (check by name; atmosphere is wrapped)
    names = get_export_pass_names()

    # Assert
    assert names.index("atmosphere") < names.index("compositor")


def test_get_export_passes_no_lights_still_includes_atmosphere():
    # Act — atmosphere always runs regardless of no_lights (check by name)
    names = get_export_pass_names(no_lights=True)

    # Assert
    assert "atmosphere" in names


def test_get_export_passes_no_lights_still_includes_compositor():
    # Arrange
    from streets_at_night.rendering.compositor import render_composite

    # Act
    passes = get_export_passes(no_lights=True)

    # Assert — compositor always runs regardless of no_lights
    assert render_composite in passes


# ---------------------------------------------------------------------------
# get_export_pass_names — matches pass list length and order
# ---------------------------------------------------------------------------


def test_get_export_pass_names_length_matches_passes_with_lights():
    # Act
    names = get_export_pass_names(no_lights=False)
    passes = get_export_passes(no_lights=False)

    # Assert — one name per pass
    assert len(names) == len(passes)


def test_get_export_pass_names_length_matches_passes_without_lights():
    # Act
    names = get_export_pass_names(no_lights=True)
    passes = get_export_passes(no_lights=True)

    # Assert
    assert len(names) == len(passes)


def test_get_export_pass_names_includes_atmosphere():
    # Act
    names = get_export_pass_names()

    # Assert
    assert "atmosphere" in names


def test_get_export_pass_names_includes_compositor():
    # Act
    names = get_export_pass_names()

    # Assert
    assert "compositor" in names


def test_get_export_pass_names_no_lights_omits_illumination_and_lights():
    # Act
    names = get_export_pass_names(no_lights=True)

    # Assert
    assert "illumination" not in names
    assert "lights" not in names


def test_get_export_pass_names_with_lights_includes_illumination_and_lights():
    # Act
    names = get_export_pass_names(no_lights=False)

    # Assert
    assert "illumination" in names
    assert "lights" in names


# ---------------------------------------------------------------------------
# render_scene_frame_with_progress — progress printing and stats
# ---------------------------------------------------------------------------


def test_render_scene_frame_with_progress_prints_pass_info(capsys):
    # Arrange
    from unittest import mock

    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    mock_pass = mock.MagicMock(return_value=None)
    passes = [mock_pass]
    pass_names = ["ground"]

    # Act
    render_scene_frame_with_progress(mock_scene, mock_backend, passes, pass_names)

    # Assert — progress line was printed
    captured = capsys.readouterr()
    assert "1/1" in captured.out
    assert "ground" in captured.out


def test_render_scene_frame_with_progress_calls_each_pass():
    # Arrange
    from unittest import mock

    mock_scene = mock.MagicMock()
    mock_camera = mock.MagicMock()
    mock_scene.camera = mock_camera
    mock_backend = mock.MagicMock()

    mock_pass_a = mock.MagicMock(return_value=None)
    mock_pass_b = mock.MagicMock(return_value=None)

    # Act
    render_scene_frame_with_progress(
        mock_scene, mock_backend,
        passes=[mock_pass_a, mock_pass_b],
        pass_names=["ground", "roads"],
    )

    # Assert — both passes called with correct args
    mock_pass_a.assert_called_once_with(mock_scene, mock_camera, mock_backend)
    mock_pass_b.assert_called_once_with(mock_scene, mock_camera, mock_backend)


def test_render_scene_frame_with_progress_merges_dict_stats():
    # Arrange
    from unittest import mock

    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    mock_pass = mock.MagicMock(return_value={"lights_drawn": 7})

    # Act
    stats = render_scene_frame_with_progress(
        mock_scene, mock_backend,
        passes=[mock_pass],
        pass_names=["lights"],
    )

    # Assert
    assert stats["lights_drawn"] == 7


def test_render_scene_frame_with_progress_prints_correct_total(capsys):
    # Arrange
    from unittest import mock

    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    passes = [mock.MagicMock(return_value=None) for _ in range(3)]
    pass_names = ["ground", "roads", "atmosphere"]

    # Act
    render_scene_frame_with_progress(mock_scene, mock_backend, passes, pass_names)

    # Assert — total shown as 3
    captured = capsys.readouterr()
    assert "1/3" in captured.out
    assert "2/3" in captured.out
    assert "3/3" in captured.out


def test_render_scene_frame_with_progress_callback_fractions_and_completion(capsys):
    # Arrange
    from unittest import mock

    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    passes = [mock.MagicMock(return_value=None) for _ in range(2)]
    pass_names = ["ground", "roads"]
    calls = []

    # Act
    render_scene_frame_with_progress(
        mock_scene, mock_backend, passes, pass_names,
        progress=lambda msg, frac: calls.append((msg, frac)),
    )

    # Assert — one call per pass with ascending fractions, plus final completion
    captured = capsys.readouterr()
    assert captured.out == ""
    fractions = [frac for _, frac in calls]
    assert fractions == [0.0, 0.5, 1.0]
    assert calls[-1] == ("Render complete", 1.0)


def test_render_scene_frame_with_progress_callback_suppresses_print(capsys):
    # Arrange
    from unittest import mock

    mock_scene = mock.MagicMock()
    mock_scene.camera = mock.MagicMock()
    mock_backend = mock.MagicMock()

    mock_pass = mock.MagicMock(return_value=None)

    # Act
    render_scene_frame_with_progress(
        mock_scene, mock_backend, [mock_pass], ["ground"],
        progress=mock.MagicMock(),
    )

    # Assert — nothing printed when a callback is supplied
    captured = capsys.readouterr()
    assert captured.out == ""
