# Radical
"""Unit tests for streets_at_night.rendering.building_pass."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import BuildingConfig
from streets_at_night.models.building import Building
from streets_at_night.rendering.building_pass import (
    _get_outward_normal,
    _is_below_pixel_threshold,
    _is_wall_facing_camera,
    render_buildings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def unit_square_footprint() -> np.ndarray:
    """Return a CCW unit-square-like footprint (10m x 10m), no closing duplicate."""
    return np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


@pytest.fixture
def building_config() -> BuildingConfig:
    """Return an enabled BuildingConfig with known colors and thresholds."""
    return BuildingConfig(
        enabled=True,
        floor_height_m=3.2,
        height_jitter=0.0,
        building_max_distance_m=12000.0,
        window_max_distance_m=6000.0,
        wall_color="#0B0A09",
        roof_color="#080807",
        face_shade_strength=0.2,
        window_lit_fraction=0.12,
        window_width_m=1.4,
        window_height_m=1.1,
        window_pitch_m=2.4,
        window_side_margin_m=0.8,
        window_sill_margin_m=0.9,
        window_roof_margin_m=0.6,
        window_color_temp_min_k=1900.0,
        window_color_temp_max_k=5500.0,
        window_saturation=0.85,
        window_brightness_min=0.15,
        window_brightness_max=0.5,
        window_glow_px=0.0,
        height_defaults={"default": 5.0},
    )


@pytest.fixture
def overhead_camera():
    """Return a camera positioned above and south of the origin, looking down."""
    return create_camera(
        position=np.array([5.0, -500.0, 400.0]),
        target=np.array([5.0, 5.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


@pytest.fixture
def mock_backend():
    """Return a mock render backend with a fixed 1920x1080 surface size."""
    backend = mock.MagicMock()
    backend.get_surface_size.return_value = (1920, 1080)
    return backend


# ---------------------------------------------------------------------------
# _get_outward_normal / _is_wall_facing_camera — backface culling
# ---------------------------------------------------------------------------

def test_get_outward_normal_points_away_from_square_interior_for_bottom_edge(
    unit_square_footprint,
):
    # Act — bottom edge, from (0,0) to (10,0), interior is at +y
    normal = _get_outward_normal(unit_square_footprint[0], unit_square_footprint[1])

    # Assert — outward normal points in -y
    assert np.allclose(normal, [0.0, -1.0])


def test_get_outward_normal_points_away_from_square_interior_for_right_edge(
    unit_square_footprint,
):
    # Act — right edge, from (10,0) to (10,10), interior is at -x
    normal = _get_outward_normal(unit_square_footprint[1], unit_square_footprint[2])

    # Assert — outward normal points in +x
    assert np.allclose(normal, [1.0, 0.0])


def test_is_wall_facing_camera_true_for_wall_between_camera_and_footprint(
    unit_square_footprint,
):
    # Arrange — camera due south of the square; bottom edge faces south
    p0, p1 = unit_square_footprint[0], unit_square_footprint[1]
    normal = _get_outward_normal(p0, p1)
    cam_xy = np.array([5.0, -100.0])

    # Act
    facing = _is_wall_facing_camera(normal, p0, p1, cam_xy)

    # Assert
    assert facing is True


def test_is_wall_facing_camera_false_for_wall_facing_away_from_camera(
    unit_square_footprint,
):
    # Arrange — camera due south of the square; top edge faces north, away from camera
    p0, p1 = unit_square_footprint[2], unit_square_footprint[3]
    normal = _get_outward_normal(p0, p1)
    cam_xy = np.array([5.0, -100.0])

    # Act
    facing = _is_wall_facing_camera(normal, p0, p1, cam_xy)

    # Assert
    assert facing is False


# ---------------------------------------------------------------------------
# _is_below_pixel_threshold
# ---------------------------------------------------------------------------

def test_is_below_pixel_threshold_true_for_subpixel_polygon():
    # Act
    result = _is_below_pixel_threshold([(10.0, 10.0), (10.5, 10.0), (10.5, 10.5)])

    # Assert
    assert result is True


def test_is_below_pixel_threshold_false_for_large_polygon():
    # Act
    result = _is_below_pixel_threshold([(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])

    # Assert
    assert result is False


# ---------------------------------------------------------------------------
# render_buildings — config gating
# ---------------------------------------------------------------------------

def test_render_buildings_returns_zero_when_disabled(
    overhead_camera, mock_backend, building_config, unit_square_footprint
):
    # Arrange
    building_config.enabled = False
    scene = mock.MagicMock()
    scene.config.buildings = building_config

    # Act
    stats = render_buildings(scene, overhead_camera, mock_backend)

    # Assert
    assert stats == {"buildings_drawn": 0}
    mock_backend.draw_polygon.assert_not_called()


def test_render_buildings_returns_zero_when_buildings_config_none(
    overhead_camera, mock_backend
):
    # Arrange
    scene = mock.MagicMock()
    scene.config.buildings = None

    # Act
    stats = render_buildings(scene, overhead_camera, mock_backend)

    # Assert
    assert stats == {"buildings_drawn": 0}
    mock_backend.draw_polygon.assert_not_called()


def test_render_buildings_returns_zero_when_no_candidates_in_view(
    overhead_camera, mock_backend, building_config
):
    # Arrange
    scene = mock.MagicMock()
    scene.config.buildings = building_config
    scene.get_buildings_in_rect.return_value = []

    # Act
    stats = render_buildings(scene, overhead_camera, mock_backend)

    # Assert
    assert stats == {"buildings_drawn": 0}


# ---------------------------------------------------------------------------
# render_buildings — draws visible building geometry
# ---------------------------------------------------------------------------

def test_render_buildings_draws_roof_and_walls_for_visible_building(
    overhead_camera, mock_backend, building_config, unit_square_footprint
):
    # Arrange
    building = Building(
        footprint=unit_square_footprint, height=6.0, building_kind="retail", window_seed=1
    )
    scene = mock.MagicMock()
    scene.config.buildings = building_config
    scene.get_buildings_in_rect.return_value = [building]

    # Act
    stats = render_buildings(scene, overhead_camera, mock_backend)

    # Assert — the building is drawn (roof always, plus any facing walls)
    assert stats == {"buildings_drawn": 1}
    assert mock_backend.draw_polygon.call_count >= 1


def test_render_buildings_drops_buildings_beyond_max_distance(
    overhead_camera, mock_backend, building_config
):
    # Arrange — a building far outside building_max_distance_m
    building_config.building_max_distance_m = 100.0
    far_footprint = np.array(
        [[20000.0, 20000.0], [20010.0, 20000.0], [20010.0, 20010.0], [20000.0, 20010.0]]
    )
    building = Building(
        footprint=far_footprint, height=6.0, building_kind="retail", window_seed=1
    )
    scene = mock.MagicMock()
    scene.config.buildings = building_config
    scene.get_buildings_in_rect.return_value = [building]

    # Act
    stats = render_buildings(scene, overhead_camera, mock_backend)

    # Assert
    assert stats == {"buildings_drawn": 0}
    mock_backend.draw_polygon.assert_not_called()
