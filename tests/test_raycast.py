# Radical
"""Unit tests for streets_at_night.rendering.raycast module."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.rendering.raycast import (
    GroundHit,
    build_ground_gbuffer,
    build_ray_directions,
    visible_ground_aabb,
    _far_horizontal_point,
    _ground_point_for_ray,
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


# ---------------------------------------------------------------------------
# GroundHit.hit_t
# ---------------------------------------------------------------------------


def test_build_ground_gbuffer_hit_t_equals_metric_distance_on_ground():
    cam_pos = [0.0, 0.0, 10.0]
    cam = _identity_camera(cam_pos)

    gb = build_ground_gbuffer(cam, 8, 8)

    expected = np.hypot(
        gb.hit_x - cam_pos[0],
        np.hypot(gb.hit_y - cam_pos[1], cam_pos[2]),
    )
    np.testing.assert_allclose(
        gb.hit_t[gb.ground_mask], expected[gb.ground_mask], atol=1e-4,
    )


def test_build_ground_gbuffer_hit_t_is_minus_one_on_sky():
    # Camera looking dead level at the horizon — the upper half of the
    # frustum never reaches the ground.
    from streets_at_night.camera.camera import create_camera

    cam = create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 10000.0, 990.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )

    gb = build_ground_gbuffer(cam, 64, 36)

    assert not gb.ground_mask.all()
    np.testing.assert_array_equal(gb.hit_t[~gb.ground_mask], -1.0)


# ---------------------------------------------------------------------------
# _ground_point_for_ray
# ---------------------------------------------------------------------------


def test_ground_point_for_ray_intersects_ground_when_pointing_down():
    cam_pos = np.array([0.0, 0.0, 1000.0])
    ray_dir = np.array([0.0, 0.0, -1.0])

    point = _ground_point_for_ray(cam_pos, ray_dir, far=2000.0)

    assert point == pytest.approx((0.0, 0.0))


def test_ground_point_for_ray_caps_at_far_when_pointing_above_horizon():
    cam_pos = np.array([0.0, 0.0, 1000.0])
    ray_dir = np.array([1.0, 0.0, 0.5])
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    point = _ground_point_for_ray(cam_pos, ray_dir, far=2000.0)

    # The ray never reaches z=0 in front of the camera — the horizontal
    # direction is capped at `far`, not dropped.
    assert point == pytest.approx((2000.0, 0.0))


def test_ground_point_for_ray_caps_at_far_when_hit_would_be_behind_camera():
    # dz slightly negative but the intersection distance would be negative
    # (ray points down but away from the ground given cam_pos[2]) is not
    # physically reachable here; instead test the dz >= 0 boundary directly.
    cam_pos = np.array([0.0, 0.0, 1000.0])
    ray_dir = np.array([0.0, 1.0, 0.0])  # exactly horizontal, dz == 0

    point = _ground_point_for_ray(cam_pos, ray_dir, far=2000.0)

    assert point == pytest.approx((0.0, 2000.0))


# ---------------------------------------------------------------------------
# _far_horizontal_point
# ---------------------------------------------------------------------------


def test_far_horizontal_point_projects_along_horizontal_ray_direction():
    cam_pos = np.array([100.0, 200.0, 1000.0])
    ray_dir = np.array([1.0, 0.0, 0.0])

    point = _far_horizontal_point(cam_pos, ray_dir, far=500.0)

    assert point == pytest.approx((600.0, 200.0))


def test_far_horizontal_point_falls_back_to_camera_xy_for_vertical_ray():
    cam_pos = np.array([50.0, 60.0, 1000.0])
    ray_dir = np.array([0.0, 0.0, 1.0])

    point = _far_horizontal_point(cam_pos, ray_dir, far=500.0)

    assert point == pytest.approx((50.0, 60.0))


# ---------------------------------------------------------------------------
# visible_ground_aabb
# ---------------------------------------------------------------------------


def test_visible_ground_aabb_straight_down_camera_matches_tan_half_fov():
    # Camera at (0, 0, 1000), looking almost straight down (tiny forward
    # offset avoids the degenerate zero-cross-product case), FOV 90 aspect 1:
    # tan(45 deg) == 1, so the ground footprint half-width equals the
    # camera height (1000m) on both axes.
    camera = create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 0.001, 0.0]),
        fov_deg=90.0,
        aspect_ratio=1.0,
        near=10.0,
        far=20000.0,
    )

    min_x, min_y, max_x, max_y = visible_ground_aabb(camera)

    assert min_x == pytest.approx(-1000.0, abs=1.0)
    assert max_x == pytest.approx(1000.0, abs=1.0)
    assert min_y == pytest.approx(-1000.0, abs=1.0)
    assert max_y == pytest.approx(1000.0, abs=1.0)


def test_visible_ground_aabb_horizon_camera_is_capped_by_far():
    # Camera looking dead level at the horizon — the upper half of the
    # frustum never reaches the ground, so those corners must fall back to
    # the far-plane cap rather than being dropped or diverging.
    camera = create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 10000.0, 1000.0]),
        fov_deg=90.0,
        aspect_ratio=1.0,
        near=10.0,
        far=5000.0,
    )

    min_x, min_y, max_x, max_y = visible_ground_aabb(camera)

    assert np.isfinite([min_x, min_y, max_x, max_y]).all()
    # Every bound stays within the far-clamped range around the camera.
    assert min_x >= -5000.0 - 1e-6
    assert max_x <= 5000.0 + 1e-6
    assert min_y >= -5000.0 - 1e-6
    assert max_y <= 5000.0 + 1e-6
    # The forward (north) extent is dominated by the far-capped top corners,
    # not the shallow downward-looking bottom corners.
    assert max_y == pytest.approx(5000.0 / np.sqrt(2), abs=1.0)


def test_visible_ground_aabb_includes_camera_position():
    camera = create_camera(
        position=np.array([500.0, -300.0, 1000.0]),
        target=np.array([500.0, -299.999, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )

    min_x, min_y, max_x, max_y = visible_ground_aabb(camera)

    assert min_x <= 500.0 <= max_x
    assert min_y <= -300.0 <= max_y
