# Radical
"""Tests for devmode.scale_marker_pass: marker geometry and draw behavior."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import Camera
from streets_at_night.devmode.scale_marker_pass import (
    CAR_LENGTH_M,
    _MARKER_COLOR,
    _build_marker_pairs,
    make_scale_marker_pass,
)
from streets_at_night.devmode.synthetic_scene import ROW_LABELS
from streets_at_night.rendering.backend import RenderBackend


@pytest.fixture
def devmode_camera() -> Camera:
    """Camera matching the default devmode pose: looks down +X along the rows."""
    return Camera(
        position=np.array([-500.0, 0.0, 800.0]),
        target=np.array([5000.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=896.0 / 720.0,
        near=100.0,
        far=60000.0,
    )


@pytest.fixture
def mock_backend() -> mock.MagicMock:
    """Backend mock with a real surface size for projection."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (896, 720)
    return backend


def test_build_marker_pairs_one_line_per_row_with_car_length():
    # Act
    pairs = _build_marker_pairs(ROW_LABELS)

    # Assert — one marker per row, exactly CAR_LENGTH_M long, on the ground plane.
    assert len(pairs) == len(ROW_LABELS)
    for p1, p2 in pairs:
        assert np.linalg.norm(p2 - p1) == pytest.approx(CAR_LENGTH_M)
        assert p1[2] == 0.0 and p2[2] == 0.0


def test_build_marker_pairs_centered_on_row_centerline():
    # Act
    pairs = _build_marker_pairs(ROW_LABELS)

    # Assert — each line's Y midpoint is the row's y offset.
    for (p1, p2), (_, y_offset) in zip(pairs, ROW_LABELS):
        assert (p1[1] + p2[1]) / 2.0 == pytest.approx(y_offset)


def test_scale_marker_pass_draws_lines_when_rows_in_front(devmode_camera, mock_backend):
    # Arrange
    render_pass = make_scale_marker_pass(ROW_LABELS)

    # Act
    render_pass(mock.MagicMock(), devmode_camera, mock_backend)

    # Assert — markers in front of the camera are drawn in the marker style.
    assert mock_backend.draw_line.call_count > 0
    for call in mock_backend.draw_line.call_args_list:
        assert call.args[2] == _MARKER_COLOR


def test_scale_marker_pass_skips_rows_behind_camera(mock_backend):
    # Arrange — camera at the far end looking away from all markers (x=30).
    camera = Camera(
        position=np.array([20000.0, 0.0, 800.0]),
        target=np.array([30000.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=896.0 / 720.0,
        near=100.0,
        far=60000.0,
    )
    render_pass = make_scale_marker_pass(ROW_LABELS)

    # Act
    render_pass(mock.MagicMock(), camera, mock_backend)

    # Assert
    mock_backend.draw_line.assert_not_called()
