# Radical
"""Tests for streets_at_night.gpu.orbit_camera."""

import numpy as np
import pytest

from streets_at_night.gpu.orbit_camera import OrbitCamera


@pytest.fixture
def orbit() -> OrbitCamera:
    """Return an orbit camera at a known pose."""
    return OrbitCamera(pivot=np.array([0.0, 0.0, 0.0]), yaw_deg=0.0, pitch_deg=30.0, distance=10.0)


def test_eye_position_distance_matches_orbit_distance(orbit):
    # Act
    eye = orbit.eye_position()

    # Assert
    assert np.linalg.norm(eye - orbit.pivot) == pytest.approx(10.0)


def test_eye_position_yaw_zero_is_east_of_pivot(orbit):
    # Act
    eye = orbit.eye_position()

    # Assert
    assert eye[0] > 0.0
    assert eye[1] == pytest.approx(0.0)
    assert eye[2] == pytest.approx(10.0 * np.sin(np.radians(30.0)))


def test_rotate_pitch_clamped_above_horizon(orbit):
    # Act
    orbit.rotate(0.0, -10000.0)

    # Assert
    assert orbit.pitch_deg == pytest.approx(2.0)


def test_rotate_pitch_clamped_below_straight_down(orbit):
    # Act
    orbit.rotate(0.0, 10000.0)

    # Assert
    assert orbit.pitch_deg == pytest.approx(89.0)


def test_pan_keeps_pivot_on_ground(orbit):
    # Act
    orbit.pan(120.0, -45.0, 1000)

    # Assert
    assert orbit.pivot[2] == 0.0


def test_pan_drag_right_moves_pivot_against_camera_right(orbit):
    # Arrange
    right = orbit.to_camera(1.5).get_right()

    # Act
    orbit.pan(100.0, 0.0, 1000)

    # Assert — scene follows the mouse, so the pivot slides the other way
    assert float(np.dot(orbit.pivot, right)) < 0.0


def test_pan_drag_down_moves_pivot_away_from_camera(orbit):
    # Arrange
    forward = orbit.to_camera(1.5).get_forward()

    # Act
    orbit.pan(0.0, 100.0, 1000)

    # Assert
    assert float(np.dot(orbit.pivot, forward)) > 0.0


def test_zoom_in_reduces_distance(orbit):
    # Act
    orbit.zoom(1.0)

    # Assert
    assert orbit.distance == pytest.approx(10.0 / 1.1)


def test_zoom_clamped_to_minimum_distance(orbit):
    # Act
    orbit.zoom(1000.0)

    # Assert
    assert orbit.distance == pytest.approx(2.0)


def test_reset_restores_defaults(orbit):
    # Arrange
    default = OrbitCamera()
    orbit.pan(300.0, 300.0, 1000)
    orbit.zoom(-5.0)

    # Act
    orbit.reset()

    # Assert
    np.testing.assert_allclose(orbit.eye_position(), default.eye_position())


def test_to_camera_looks_at_pivot(orbit):
    # Act
    camera = orbit.to_camera(16 / 9)

    # Assert
    np.testing.assert_allclose(camera.target, orbit.pivot)
    assert camera.near == pytest.approx(0.1)
