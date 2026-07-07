# Radical
"""Unit tests for streets_at_night.camera.projection."""

import numpy as np
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.camera.projection import (
    _camera_z,
    clip_line_to_near_plane,
    clip_polygon_to_near_plane,
    project_points,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def camera():
    """Return a camera at (0, -1000, 500) looking at the origin."""
    return create_camera(
        position=np.array([0.0, -1000.0, 500.0]),
        target=np.array([0.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


@pytest.fixture
def view_matrix(camera):
    """View matrix for the standard test camera."""
    return camera.get_view_matrix()


@pytest.fixture
def proj_matrix(camera):
    """Projection matrix for the standard test camera."""
    return camera.get_projection_matrix()


@pytest.fixture
def screen_size():
    """Standard 1920x1080 screen size."""
    return (1920, 1080)


# ---------------------------------------------------------------------------
# project_points — output shapes
# ---------------------------------------------------------------------------

def test_project_points_returns_correct_shapes(view_matrix, proj_matrix, screen_size):
    # Arrange
    points = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [-100.0, 0.0, 0.0],
    ])

    # Act
    screen_coords, valid_mask = project_points(points, view_matrix, proj_matrix, screen_size)

    # Assert
    assert screen_coords.shape == (3, 2)
    assert valid_mask.shape == (3,)


# ---------------------------------------------------------------------------
# project_points — validity
# ---------------------------------------------------------------------------

def test_project_points_point_in_front_is_valid(camera, view_matrix, proj_matrix, screen_size):
    # Arrange — the camera looks from (0, -1000, 500) toward (0, 0, 0).
    # A point at (0, -400, 250) is between the camera and the target, within frustum.
    points = np.array([[0.0, -400.0, 250.0]])

    # Act
    _, valid_mask = project_points(points, view_matrix, proj_matrix, screen_size)

    # Assert
    assert valid_mask[0] is np.bool_(True) or bool(valid_mask[0]) is True


def test_project_points_point_behind_camera_is_invalid(camera, view_matrix, proj_matrix, screen_size):
    # Arrange — a point directly behind the camera (further along -Y from the camera position)
    # Camera is at (0, -1000, 500) looking toward +Y direction (roughly).
    # A point at (0, -2000, 500) is behind the camera.
    points = np.array([[0.0, -2000.0, 500.0]])

    # Act
    _, valid_mask = project_points(points, view_matrix, proj_matrix, screen_size)

    # Assert
    assert bool(valid_mask[0]) is False


def test_project_points_center_projects_near_screen_center(camera, view_matrix, proj_matrix, screen_size):
    # Arrange — a point on the camera's forward axis should project near the screen centre.
    # Camera at (0, -1000, 500) looking at (0, 0, 0).
    # The midpoint (0, -500, 250) lies exactly on the look-at vector.
    points = np.array([[0.0, -500.0, 250.0]])
    width, height = screen_size

    # Act
    screen_coords, _ = project_points(points, view_matrix, proj_matrix, screen_size)

    # Assert — within ±5% of screen dimensions from centre
    cx, cy = width / 2.0, height / 2.0
    assert screen_coords[0, 0] == pytest.approx(cx, abs=width * 0.05)
    assert screen_coords[0, 1] == pytest.approx(cy, abs=height * 0.05)


# ---------------------------------------------------------------------------
# clip_line_to_near_plane
# ---------------------------------------------------------------------------

def test_clip_line_to_near_plane_both_in_front_returns_unchanged(camera, view_matrix):
    # Arrange — both endpoints between camera and target; clearly in front of near plane.
    near = camera.near
    p1 = np.array([0.0, -800.0, 400.0])
    p2 = np.array([0.0, -600.0, 300.0])

    # Act
    result = clip_line_to_near_plane(p1, p2, view_matrix, near)

    # Assert — segment returned unmodified
    assert result is not None
    np.testing.assert_array_equal(result[0], p1)
    np.testing.assert_array_equal(result[1], p2)


def test_clip_line_to_near_plane_both_behind_returns_none(camera, view_matrix):
    # Arrange — both points behind the camera (further in the -Y direction).
    near = camera.near
    p1 = np.array([0.0, -1500.0, 500.0])
    p2 = np.array([0.0, -2000.0, 500.0])

    # Act
    result = clip_line_to_near_plane(p1, p2, view_matrix, near)

    # Assert
    assert result is None


def test_clip_line_to_near_plane_one_behind_returns_clipped(camera, view_matrix):
    # Arrange — p1 is in front, p2 is behind the near plane.
    # We need one endpoint whose camera-space z < -near (in front) and one >= -near (behind).
    # The near plane is at camera-space z = -near = -10.
    # We find two points by checking camera-space z via the view matrix.
    near = camera.near

    # A point at the target (origin) is definitely in front.
    p1_in_front = np.array([0.0, 0.0, 0.0])

    # A point behind the camera: move further back than camera position along -Y.
    p2_behind = np.array([0.0, -2000.0, 500.0])

    # Act
    result = clip_line_to_near_plane(p1_in_front, p2_behind, view_matrix, near)

    # Assert — result is a 2-tuple (not None)
    assert result is not None
    clipped_p1, clipped_p2 = result

    # One endpoint should be the unchanged in-front point, and the other should
    # be the clipped intersection near the near plane.
    assert clipped_p1.shape == (3,)
    assert clipped_p2.shape == (3,)

    # The clipped endpoint (p2 was behind) should now be at camera-space z ≈ -near
    clipped_z = _camera_z(clipped_p2, view_matrix)
    assert clipped_z == pytest.approx(-near, abs=1.0)


# ---------------------------------------------------------------------------
# clip_polygon_to_near_plane
# ---------------------------------------------------------------------------


def test_clip_polygon_to_near_plane_all_in_front_returns_all_vertices(camera, view_matrix):
    # Arrange — a small triangle between the camera and its target; fully in front.
    # Camera is at (0, -1000, 500) looking toward origin. Points near midpoint are safe.
    near = camera.near
    vertices = [
        np.array([-50.0, -500.0, 250.0]),
        np.array([50.0, -500.0, 250.0]),
        np.array([0.0, -400.0, 200.0]),
    ]

    # Act
    result = clip_polygon_to_near_plane(vertices, view_matrix, near)

    # Assert — all three vertices survive; no closing duplicate added
    assert len(result) == 3


def test_clip_polygon_to_near_plane_all_behind_returns_empty(camera, view_matrix):
    # Arrange — all vertices are behind the camera (further than position along -Y).
    near = camera.near
    vertices = [
        np.array([-50.0, -1500.0, 500.0]),
        np.array([50.0, -1500.0, 500.0]),
        np.array([0.0, -2000.0, 500.0]),
    ]

    # Act
    result = clip_polygon_to_near_plane(vertices, view_matrix, near)

    # Assert
    assert result == []


def test_clip_polygon_to_near_plane_straddling_returns_clipped_polygon(camera, view_matrix):
    # Arrange — a quad that straddles the near plane: two vertices clearly in
    # front, two clearly behind the camera.
    near = camera.near
    vertices = [
        np.array([-100.0, -500.0, 250.0]),   # in front
        np.array([100.0, -500.0, 250.0]),    # in front
        np.array([100.0, -2000.0, 500.0]),   # behind camera
        np.array([-100.0, -2000.0, 500.0]),  # behind camera
    ]

    # Act
    result = clip_polygon_to_near_plane(vertices, view_matrix, near)

    # Assert — result is a valid polygon (>=3 vertices) and all are in front of near
    assert len(result) >= 3
    for v in result:
        cam_z = _camera_z(v, view_matrix)
        assert cam_z <= -near + 0.01


def test_clip_polygon_to_near_plane_degenerate_after_clip_returns_empty(camera, view_matrix):
    # Arrange — a contrived degenerate "polygon" with only 2 vertices.
    # A 2-vertex input is a line segment, not a real polygon. Clipping
    # can produce at most 2 output vertices, triggering the < 3 guard.
    near = camera.near
    vertices = [
        np.array([0.0, -500.0, 250.0]),   # in front of near plane
        np.array([0.0, -400.0, 200.0]),   # also in front
    ]

    # Act
    result = clip_polygon_to_near_plane(vertices, view_matrix, near)

    # Assert — fewer than 3 vertices → degenerate polygon → empty list
    assert result == []
