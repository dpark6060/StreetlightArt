# Radical
"""Unit tests for streets_at_night.rendering.raycast module."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.rendering.raycast import (
    GroundHit,
    build_ground_gbuffer,
    build_ray_directions,
)


def _identity_camera(position) -> mock.MagicMock:
    """Return a mock camera with identity view/projection and the given position."""
    cam = mock.MagicMock()
    cam.get_projection_matrix.return_value = np.eye(4)
    cam.get_view_matrix.return_value = np.eye(4)
    cam.position = np.array(position, dtype=float)
    return cam


def test_build_ray_directions_returns_correct_shape():
    rays = build_ray_directions(8, 6, np.eye(4), np.eye(4))

    assert rays.shape == (6, 8, 3)


def test_build_ray_directions_are_normalized():
    rays = build_ray_directions(8, 6, np.eye(4), np.eye(4))

    norms = np.linalg.norm(rays, axis=2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_build_ray_directions_center_pixel_points_straight_down():
    # With identity matrices, the (4,4) pixel of an 8x8 grid sits at NDC (0,0),
    # so its direction is the forward axis [0, 0, -1].
    rays = build_ray_directions(8, 8, np.eye(4), np.eye(4))

    np.testing.assert_allclose(rays[4, 4], [0.0, 0.0, -1.0], atol=1e-6)


def test_build_ground_gbuffer_returns_groundhit():
    cam = _identity_camera([0.0, 0.0, 10.0])

    gb = build_ground_gbuffer(cam, 8, 8)

    assert isinstance(gb, GroundHit)
    assert gb.ground_mask.shape == (8, 8)
    assert gb.hit_x.shape == (8, 8)


def test_build_ground_gbuffer_downward_camera_hits_ground():
    cam = _identity_camera([0.0, 0.0, 10.0])

    gb = build_ground_gbuffer(cam, 8, 8)

    assert gb.ground_mask.all()


def test_build_ground_gbuffer_center_pixel_hits_below_camera():
    # Camera at (0,0,10) looking down; the NDC-center pixel hits z=0 at the
    # camera's ground projection (0, 0) after travelling t=10 along [0,0,-1].
    cam = _identity_camera([0.0, 0.0, 10.0])

    gb = build_ground_gbuffer(cam, 8, 8)

    assert gb.hit_x[4, 4] == pytest.approx(0.0, abs=1e-6)
    assert gb.hit_y[4, 4] == pytest.approx(0.0, abs=1e-6)
