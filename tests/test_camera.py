# Radical
"""Unit tests for streets_at_night.camera.camera."""

import numpy as np
import pytest

from streets_at_night.camera.camera import Camera, create_camera

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def camera() -> Camera:
    """Return a simple camera positioned behind and above, looking toward origin."""
    position = np.array([0.0, -5000.0, 1500.0])
    target = np.array([0.0, 0.0, 300.0])
    return create_camera(
        position=position,
        target=target,
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


@pytest.fixture
def north_facing_camera() -> Camera:
    """Return a camera whose target is directly north (positive Y) of position."""
    position = np.array([0.0, 0.0, 500.0])
    target = np.array([0.0, 1000.0, 500.0])
    return create_camera(
        position=position,
        target=target,
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


# ---------------------------------------------------------------------------
# get_forward
# ---------------------------------------------------------------------------

def test_get_forward_is_normalized(camera):
    # Act
    forward = camera.get_forward()

    # Assert
    assert np.linalg.norm(forward) == pytest.approx(1.0, abs=1e-9)


def test_get_forward_points_toward_target(north_facing_camera):
    # Act
    forward = north_facing_camera.get_forward()

    # Assert — target is directly north; forward Y component should dominate
    assert forward[1] > 0.0
    assert abs(forward[1]) > abs(forward[0])
    assert abs(forward[1]) > abs(forward[2])


# ---------------------------------------------------------------------------
# get_right
# ---------------------------------------------------------------------------

def test_get_right_is_normalized(camera):
    # Act
    right = camera.get_right()

    # Assert
    assert np.linalg.norm(right) == pytest.approx(1.0, abs=1e-9)


def test_get_right_is_perpendicular_to_forward(camera):
    # Act
    forward = camera.get_forward()
    right = camera.get_right()

    # Assert — dot product of perpendicular vectors is 0
    assert np.dot(forward, right) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# get_view_matrix
# ---------------------------------------------------------------------------

def test_get_view_matrix_shape(camera):
    # Act
    view = camera.get_view_matrix()

    # Assert
    assert view.shape == (4, 4)


# ---------------------------------------------------------------------------
# get_projection_matrix
# ---------------------------------------------------------------------------

def test_get_projection_matrix_shape(camera):
    # Act
    proj = camera.get_projection_matrix()

    # Assert
    assert proj.shape == (4, 4)
