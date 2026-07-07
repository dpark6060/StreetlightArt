# Radical
"""Pure-logic tests for the devmode Minimap coordinate math and hit testing."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from streets_at_night.devmode.minimap import Minimap, MinimapLayout  # noqa: E402


@pytest.fixture
def square_map() -> Minimap:
    """A 200x200 screen rect over a square [0,1000]x[0,1000] world."""
    layout = MinimapLayout(
        rect=pygame.Rect(50, 30, 200, 200),
        world_min=(0.0, 0.0),
        world_max=(1000.0, 1000.0),
    )
    return Minimap(layout)


@pytest.fixture
def wide_map() -> Minimap:
    """A square 200x200 rect over a wide 1000x100 world (non-square)."""
    layout = MinimapLayout(
        rect=pygame.Rect(0, 0, 200, 200),
        world_min=(0.0, 0.0),
        world_max=(1000.0, 100.0),
    )
    return Minimap(layout)


def test_screen_to_world_round_trips_within_one_pixel(square_map):
    rect = square_map.layout.rect
    screen_points = [
        (rect.x, rect.y),
        (rect.right - 1, rect.y),
        (rect.x, rect.bottom - 1),
        (rect.right - 1, rect.bottom - 1),
        (rect.centerx, rect.centery),
    ]

    for px, py in screen_points:
        wx, wy = square_map.screen_to_world(px, py)
        rx, ry = square_map.world_to_screen(wx, wy)

        assert abs(rx - px) <= 1
        assert abs(ry - py) <= 1


def test_world_to_screen_flips_north_to_smaller_y(square_map):
    center_x = 500.0

    high_north = square_map.world_to_screen(center_x, 900.0)
    low_south = square_map.world_to_screen(center_x, 100.0)

    assert high_north[1] < low_south[1]


def test_world_to_screen_preserves_aspect_and_centers_short_axis(wide_map):
    rect = wide_map.layout.rect

    left_edge = wide_map.world_to_screen(0.0, 50.0)
    right_edge = wide_map.world_to_screen(1000.0, 50.0)
    center = wide_map.world_to_screen(500.0, 50.0)

    # Wide world fills the x span edge to edge.
    assert left_edge[0] == rect.x
    assert right_edge[0] == rect.right
    # Center of the world maps to the center of the rect (short y axis centered).
    assert center[0] == pytest.approx(rect.centerx, abs=1)
    assert center[1] == pytest.approx(rect.centery, abs=1)
    # The short (y) axis does not touch the rect top/bottom: it is inset.
    top = wide_map.world_to_screen(500.0, 100.0)
    assert top[1] > rect.y


def test_hit_test_inside_radius_returns_camera(square_map):
    camera_xy = (500.0, 500.0)
    target_xy = (100.0, 100.0)
    cam_screen = square_map.world_to_screen(*camera_xy)

    result = square_map.hit_test((cam_screen[0] + 5, cam_screen[1]), camera_xy, target_xy)

    assert result == "camera"


def test_hit_test_overlapping_dots_prefers_camera(square_map):
    shared_xy = (500.0, 500.0)
    screen = square_map.world_to_screen(*shared_xy)

    result = square_map.hit_test(screen, shared_xy, shared_xy)

    assert result == "camera"


def test_hit_test_near_target_only_returns_target(square_map):
    camera_xy = (100.0, 100.0)
    target_xy = (900.0, 900.0)
    tgt_screen = square_map.world_to_screen(*target_xy)

    result = square_map.hit_test((tgt_screen[0] + 4, tgt_screen[1]), camera_xy, target_xy)

    assert result == "target"


def test_hit_test_far_from_both_returns_none(square_map):
    camera_xy = (100.0, 100.0)
    target_xy = (900.0, 900.0)
    cam_screen = square_map.world_to_screen(*camera_xy)

    result = square_map.hit_test((cam_screen[0] + 40, cam_screen[1] + 40), camera_xy, target_xy)

    assert result is None


def test_screen_to_world_clamps_outside_rect_to_extent(square_map):
    # Far above-left of the rect: should clamp to (world_min_x, world_max_y).
    wx, wy = square_map.screen_to_world(-500, -500)

    assert wx == pytest.approx(square_map.layout.world_min[0])
    assert wy == pytest.approx(square_map.layout.world_max[1])

    # Far below-right: clamps to (world_max_x, world_min_y).
    wx2, wy2 = square_map.screen_to_world(5000, 5000)

    assert wx2 == pytest.approx(square_map.layout.world_max[0])
    assert wy2 == pytest.approx(square_map.layout.world_min[1])
