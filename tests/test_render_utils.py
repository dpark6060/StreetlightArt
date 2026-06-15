# Radical
"""Unit tests for streets_at_night.rendering.render_utils module."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.rendering.render_utils import (
    apply_brightness_falloff,
    camera_distance,
    compute_focal_length,
    compute_horizon_y,
    extract_positions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_camera():
    """Return a mock Camera with known position and far distance."""
    camera = mock.MagicMock()
    camera.position = np.array([0.0, 0.0, 500.0])
    camera.far = 10000.0
    camera.fov_deg = 60.0
    camera.get_forward.return_value = np.array([0.0, 1.0, 0.0])
    return camera


# ---------------------------------------------------------------------------
# compute_horizon_y — valid projection returns screen y
# ---------------------------------------------------------------------------


def test_compute_horizon_y_valid_projection_returns_screen_y(mock_camera):
    # Arrange
    width, height = 640, 360
    view_matrix = np.eye(4)
    proj_matrix = np.eye(4)

    # project_points returns a coordinate at screen y=100 and valid=True
    mock_coords = np.array([[320.0, 100.0]])
    mock_valid = np.array([True])

    with mock.patch(
        "streets_at_night.camera.projection.project_points",
        return_value=(mock_coords, mock_valid),
    ) as mock_proj:
        # Act
        result = compute_horizon_y(mock_camera, view_matrix, proj_matrix, width, height)

    # Assert
    mock_proj.assert_called_once()
    assert result == 100


def test_compute_horizon_y_valid_projection_clips_to_screen_bounds(mock_camera):
    # Arrange
    width, height = 640, 360
    view_matrix = np.eye(4)
    proj_matrix = np.eye(4)

    # project_points returns y beyond the image height
    mock_coords = np.array([[320.0, 500.0]])
    mock_valid = np.array([True])

    with mock.patch(
        "streets_at_night.camera.projection.project_points",
        return_value=(mock_coords, mock_valid),
    ):
        # Act
        result = compute_horizon_y(mock_camera, view_matrix, proj_matrix, width, height)

    # Assert — clipped to height - 1
    assert result == height - 1


def test_compute_horizon_y_invalid_projection_returns_fallback(mock_camera):
    # Arrange — far point is behind the camera: valid[0] == False
    width, height = 640, 360
    view_matrix = np.eye(4)
    proj_matrix = np.eye(4)

    mock_coords = np.array([[320.0, 999.0]])
    mock_valid = np.array([False])

    with mock.patch(
        "streets_at_night.camera.projection.project_points",
        return_value=(mock_coords, mock_valid),
    ):
        # Act
        result = compute_horizon_y(mock_camera, view_matrix, proj_matrix, width, height)

    # Assert — fallback is height // 4
    assert result == height // 4


# ---------------------------------------------------------------------------
# extract_positions — stacks light positions
# ---------------------------------------------------------------------------


def test_extract_positions_returns_correct_shape():
    # Arrange
    mock_lights = [mock.MagicMock(), mock.MagicMock()]
    mock_lights[0].position = np.array([1.0, 2.0, 3.0])
    mock_lights[1].position = np.array([4.0, 5.0, 6.0])

    # Act
    result = extract_positions(mock_lights)

    # Assert
    assert result.shape == (2, 3)
    assert result[0, 0] == 1.0
    assert result[1, 2] == 6.0


def test_extract_positions_empty_list_returns_empty_array():
    # Act
    result = extract_positions([])

    # Assert — empty input produces an empty array with zero elements
    assert result.size == 0


# ---------------------------------------------------------------------------
# camera_distance — Euclidean distance clamped to at least 1.0
# ---------------------------------------------------------------------------


def test_camera_distance_returns_euclidean_distance():
    # Arrange
    light_pos = np.array([3.0, 4.0, 0.0])
    cam_pos = np.array([0.0, 0.0, 0.0])

    # Act
    result = camera_distance(light_pos, cam_pos)

    # Assert
    assert abs(result - 5.0) < 1e-6


def test_camera_distance_clamps_to_minimum_of_one():
    # Arrange — same position: distance would be 0
    pos = np.array([1.0, 1.0, 1.0])

    # Act
    result = camera_distance(pos, pos)

    # Assert
    assert result == 1.0


# ---------------------------------------------------------------------------
# compute_focal_length — derived from FOV and screen width
# ---------------------------------------------------------------------------


def test_compute_focal_length_90_degree_fov_equals_half_width(mock_camera):
    # Arrange — tan(45deg) = 1.0, so focal_length = width / (2 * 1.0) = width / 2
    mock_camera.fov_deg = 90.0

    # Act
    result = compute_focal_length(mock_camera, 640)

    # Assert
    assert abs(result - 320.0) < 1e-6


def test_compute_focal_length_scales_with_width(mock_camera):
    # Arrange
    mock_camera.fov_deg = 60.0

    # Act
    result_640 = compute_focal_length(mock_camera, 640)
    result_1280 = compute_focal_length(mock_camera, 1280)

    # Assert — doubling width doubles focal length
    assert abs(result_1280 - 2.0 * result_640) < 1e-6


# ---------------------------------------------------------------------------
# apply_brightness_falloff — exponential attenuation
# ---------------------------------------------------------------------------


def test_apply_brightness_falloff_returns_input_at_zero_distance():
    # At distance 0, falloff = exp(0) = 1.0, so result == brightness
    result = apply_brightness_falloff(1.0, 0.0, 1000.0)
    assert abs(result - 1.0) < 1e-6


def test_apply_brightness_falloff_decays_with_distance():
    # At distance = falloff_distance, exp(-1) ≈ 0.368
    result = apply_brightness_falloff(1.0, 1000.0, 1000.0)
    import math
    assert abs(result - math.exp(-1.0)) < 1e-6
