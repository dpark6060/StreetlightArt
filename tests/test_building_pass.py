# Radical
"""Unit tests for streets_at_night.rendering.building_pass."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import BuildingConfig
from streets_at_night.models.building import Building
from streets_at_night.rendering.building_pass import (
    FacingWall,
    WindowCell,
    _accumulate_building_windows,
    _accumulate_subpixel_window,
    _get_facing_walls,
    _get_lit_fraction,
    _get_outward_normal,
    _get_window_col_count,
    _get_window_row_count,
    _is_below_pixel_threshold,
    _is_wall_facing_camera,
    _is_window_subpixel,
    _project_window,
    build_window_grid,
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
        window_lit_fraction_residential=0.4,
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


# ---------------------------------------------------------------------------
# build_window_grid — pure geometry, tested heavily
# ---------------------------------------------------------------------------

@pytest.fixture
def south_wall_edge() -> tuple:
    """Return a 10m south-facing wall edge (p0, p1, outward normal)."""
    p0 = np.array([0.0, 0.0])
    p1 = np.array([10.0, 0.0])
    normal = np.array([0.0, -1.0])
    return p0, p1, normal


def test_build_window_grid_returns_expected_cell_count(south_wall_edge, building_config):
    # Arrange — 6m tall building, 10m wide wall, default margins/pitch
    p0, p1, normal = south_wall_edge

    # Act
    cells = build_window_grid(p0, p1, 6.0, normal, edge_index=2, window_seed=7, config=building_config)

    # Assert — usable height 4.5m / 3.2m floor height -> 1 row;
    # usable width 8.4m / 2.4m pitch -> 3 columns
    assert len(cells) == 3


def test_build_window_grid_rng_keys_carry_seed_edge_row_col(south_wall_edge, building_config):
    # Arrange
    p0, p1, normal = south_wall_edge

    # Act
    cells = build_window_grid(p0, p1, 6.0, normal, edge_index=2, window_seed=7, config=building_config)

    # Assert — each cell's key is (window_seed, edge_index, row, col), unique per cell
    keys = [cell.rng_key for cell in cells]
    assert keys == [(7, 2, 0, 0), (7, 2, 0, 1), (7, 2, 0, 2)]


def test_build_window_grid_first_cell_corners_match_expected_position(
    south_wall_edge, building_config
):
    # Arrange — hand-computed expected center for row 0, col 0 (see plan math):
    # row center z = 0.9 + (4.5 - 3.2)/2 + 3.2*0.5 = 3.15
    # col offset   = 0.8 + (8.4 - 7.2)/2 + 2.4*0.5 = 2.6
    p0, p1, normal = south_wall_edge

    # Act
    cells = build_window_grid(p0, p1, 6.0, normal, edge_index=0, window_seed=1, config=building_config)
    first = cells[0]

    # Assert — bottom-left, bottom-right, top-right, top-left, inset 0.05m along normal
    expected = [
        np.array([1.9, -0.05, 2.6]),
        np.array([3.3, -0.05, 2.6]),
        np.array([3.3, -0.05, 3.7]),
        np.array([1.9, -0.05, 3.7]),
    ]
    for actual_corner, expected_corner in zip(first.corners, expected):
        assert np.allclose(actual_corner, expected_corner)


def test_build_window_grid_returns_empty_when_wall_too_short_for_a_column(building_config):
    # Arrange — 1m wide wall: usable width 1 - 1.6 = negative
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    normal = np.array([0.0, -1.0])

    # Act
    cells = build_window_grid(p0, p1, 6.0, normal, edge_index=0, window_seed=1, config=building_config)

    # Assert
    assert cells == []


def test_build_window_grid_returns_empty_when_building_too_short_for_a_row(
    south_wall_edge, building_config
):
    # Arrange — 1m tall building: usable height 1 - 1.5 = negative
    p0, p1, normal = south_wall_edge

    # Act
    cells = build_window_grid(p0, p1, 1.0, normal, edge_index=0, window_seed=1, config=building_config)

    # Assert
    assert cells == []


def test_get_window_row_count_returns_zero_for_negative_usable_height(building_config):
    # Act
    rows = _get_window_row_count(1.0, building_config)

    # Assert
    assert rows == 0


def test_get_window_row_count_returns_fallback_row_for_short_house(building_config):
    # Act — 4.5m house: usable 3.0m is less than one 3.2m floor, but a 1.1m window fits
    rows = _get_window_row_count(4.5, building_config)

    # Assert
    assert rows == 1


def test_get_window_row_count_returns_zero_when_no_window_fits(building_config):
    # Act — 2.0m wall: usable 0.5m is smaller than window_height_m 1.1m
    rows = _get_window_row_count(2.0, building_config)

    # Assert
    assert rows == 0


def test_get_window_row_count_multi_floor_heights_unaffected_by_fallback(building_config):
    # Act — 10m building: floor(8.5 / 3.2) = 2 rows, fallback path not involved
    rows = _get_window_row_count(10.0, building_config)

    # Assert
    assert rows == 2


def test_build_window_grid_short_house_gets_single_centered_row(south_wall_edge, building_config):
    # Arrange
    p0, p1, normal = south_wall_edge

    # Act — 4.5m house on a 10m wall
    cells = build_window_grid(
        p0, p1, 4.5, normal, edge_index=0, window_seed=1, config=building_config
    )

    # Assert — 1 row x 3 cols; row center z = 0.9 + (3.0 - 3.2)/2 + 1.6 = 2.4,
    # so windows span 1.85..2.95, below the 3.9m roofline
    assert len(cells) == 3
    for cell in cells:
        assert cell.corners[0][2] == pytest.approx(1.85)
        assert cell.corners[2][2] == pytest.approx(2.95)


def test_get_window_col_count_returns_zero_for_negative_usable_width(building_config):
    # Act
    cols = _get_window_col_count(1.0, building_config)

    # Assert
    assert cols == 0


# ---------------------------------------------------------------------------
# _get_facing_walls — reused for both wall drawing and window placement
# ---------------------------------------------------------------------------

def test_get_facing_walls_returns_only_the_camera_facing_edge(unit_square_footprint):
    # Arrange — camera due south; only the bottom edge (index 0) should face it
    building = Building(
        footprint=unit_square_footprint, height=6.0, building_kind="retail", window_seed=1
    )
    camera = create_camera(
        position=np.array([5.0, -100.0, 50.0]),
        target=np.array([5.0, 5.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=1.0,
        far=5000.0,
    )

    # Act
    facing_walls = _get_facing_walls(building, camera)

    # Assert
    assert [wall.edge_index for wall in facing_walls] == [0]
    assert isinstance(facing_walls[0], FacingWall)


# ---------------------------------------------------------------------------
# _get_lit_fraction — per-building-kind lit fraction selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kind,expected",
    [
        ("house", 0.4),
        ("detached", 0.4),
        ("residential", 0.4),
        ("apartments", 0.12),
        ("retail", 0.12),
        ("yes", 0.12),
    ],
)
def test_get_lit_fraction_selects_fraction_by_building_kind(kind, expected, building_config):
    # Act
    fraction = _get_lit_fraction(kind, building_config)

    # Assert
    assert fraction == expected


# ---------------------------------------------------------------------------
# _accumulate_building_windows / render_buildings — lit selection + accumulation
# ---------------------------------------------------------------------------

@pytest.fixture
def facing_wall_for_windows() -> FacingWall:
    """Return a single south-facing 10m wall edge as a FacingWall."""
    return FacingWall(
        edge_index=0,
        p0=np.array([0.0, 0.0]),
        p1=np.array([10.0, 0.0]),
        normal=np.array([0.0, -1.0]),
    )


@pytest.fixture
def window_test_building() -> Building:
    """Return a 6m-tall building whose south wall fits a 1x3 window grid."""
    footprint = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    return Building(footprint=footprint, height=6.0, building_kind="retail", window_seed=1)


@pytest.fixture
def near_view_and_proj():
    """Return (view_matrix, proj_matrix) for a camera 100m south of the wall."""
    camera = create_camera(
        position=np.array([5.0, -100.0, 3.0]),
        target=np.array([5.0, 0.0, 3.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=1.0,
        far=5000.0,
    )
    return camera.get_view_matrix(), camera.get_projection_matrix()


def test_accumulate_building_windows_zero_emission_when_lit_fraction_zero(
    window_test_building, facing_wall_for_windows, building_config, near_view_and_proj
):
    # Arrange
    building_config.window_lit_fraction = 0.0
    view_matrix, proj_matrix = near_view_and_proj
    window_buffer = np.zeros((1080, 1920, 3), dtype=np.float32)

    # Act
    lit_count = _accumulate_building_windows(
        window_test_building, [facing_wall_for_windows], 100.0,
        view_matrix, proj_matrix, 1920, 1080, building_config, window_buffer,
    )

    # Assert
    assert lit_count == 0
    assert np.array_equal(window_buffer, np.zeros_like(window_buffer))


def test_accumulate_building_windows_every_cell_emits_when_lit_fraction_one(
    window_test_building, facing_wall_for_windows, building_config, near_view_and_proj
):
    # Arrange
    building_config.window_lit_fraction = 1.0
    view_matrix, proj_matrix = near_view_and_proj
    window_buffer = np.zeros((1080, 1920, 3), dtype=np.float32)
    expected_cells = build_window_grid(
        facing_wall_for_windows.p0, facing_wall_for_windows.p1, window_test_building.height,
        facing_wall_for_windows.normal, facing_wall_for_windows.edge_index,
        window_test_building.window_seed, building_config,
    )

    # Act
    lit_count = _accumulate_building_windows(
        window_test_building, [facing_wall_for_windows], 100.0,
        view_matrix, proj_matrix, 1920, 1080, building_config, window_buffer,
    )

    # Assert — every grid cell was lit, and some emission landed in the buffer
    assert lit_count == len(expected_cells)
    assert window_buffer.sum() > 0.0


def test_accumulate_building_windows_deterministic_across_repeated_runs(
    window_test_building, facing_wall_for_windows, building_config, near_view_and_proj
):
    # Arrange
    view_matrix, proj_matrix = near_view_and_proj

    # Act — run twice with fresh buffers
    buffer_a = np.zeros((1080, 1920, 3), dtype=np.float32)
    count_a = _accumulate_building_windows(
        window_test_building, [facing_wall_for_windows], 100.0,
        view_matrix, proj_matrix, 1920, 1080, building_config, buffer_a,
    )
    buffer_b = np.zeros((1080, 1920, 3), dtype=np.float32)
    count_b = _accumulate_building_windows(
        window_test_building, [facing_wall_for_windows], 100.0,
        view_matrix, proj_matrix, 1920, 1080, building_config, buffer_b,
    )

    # Assert — identical seeds produce an identical result
    assert count_a == count_b
    assert np.array_equal(buffer_a, buffer_b)


def test_render_buildings_does_not_add_linear_when_lit_fraction_zero(
    overhead_camera, mock_backend, building_config, unit_square_footprint
):
    # Arrange
    building_config.window_lit_fraction = 0.0
    building = Building(
        footprint=unit_square_footprint, height=6.0, building_kind="retail", window_seed=1
    )
    scene = mock.MagicMock()
    scene.config.buildings = building_config
    scene.get_buildings_in_rect.return_value = [building]

    # Act
    render_buildings(scene, overhead_camera, mock_backend)

    # Assert
    mock_backend.add_linear.assert_not_called()


def test_render_buildings_adds_linear_window_buffer_when_lit_fraction_one(
    overhead_camera, mock_backend, building_config, unit_square_footprint
):
    # Arrange
    building_config.window_lit_fraction = 1.0
    building = Building(
        footprint=unit_square_footprint, height=6.0, building_kind="retail", window_seed=1
    )
    scene = mock.MagicMock()
    scene.config.buildings = building_config
    scene.get_buildings_in_rect.return_value = [building]

    # Act
    render_buildings(scene, overhead_camera, mock_backend)

    # Assert
    mock_backend.add_linear.assert_called_once()
    window_buffer = mock_backend.add_linear.call_args[0][0]
    assert window_buffer.sum() > 0.0


# ---------------------------------------------------------------------------
# Sub-pixel accumulation path — distant windows still add energy
# ---------------------------------------------------------------------------

def test_is_window_subpixel_true_for_tiny_projected_quad():
    # Act
    result = _is_window_subpixel(np.array([[10.0, 10.0], [10.4, 10.0], [10.4, 10.3], [10.0, 10.3]]))

    # Assert
    assert result is True


def test_is_window_subpixel_false_for_large_projected_quad():
    # Act
    result = _is_window_subpixel(np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]]))

    # Assert
    assert result is False


def test_accumulate_subpixel_window_adds_energy_at_rounded_pixel():
    # Arrange
    buffer = np.zeros((10, 10, 3), dtype=np.float32)

    # Act
    _accumulate_subpixel_window(buffer, cx=4.6, cy=2.2, linear_color=(1.0, 0.5, 0.0), brightness=2.0)

    # Assert — energy lands only at the rounded-to-nearest pixel (5, 2)
    assert np.allclose(buffer[2, 5], [2.0, 1.0, 0.0])
    buffer[2, 5] = 0.0
    assert np.array_equal(buffer, np.zeros_like(buffer))


def test_accumulate_subpixel_window_ignores_points_outside_buffer():
    # Arrange
    buffer = np.zeros((10, 10, 3), dtype=np.float32)

    # Act
    _accumulate_subpixel_window(buffer, cx=-5.0, cy=-5.0, linear_color=(1.0, 1.0, 1.0), brightness=1.0)

    # Assert — no exception, buffer untouched
    assert np.array_equal(buffer, np.zeros_like(buffer))


def test_accumulate_building_windows_uses_subpixel_path_beyond_distance_threshold(
    window_test_building, facing_wall_for_windows, building_config,
):
    # Arrange — a camera far enough away that the projected window quad is
    # sub-pixel, past the forced sub-pixel distance threshold.
    building_config.window_lit_fraction = 1.0
    camera = create_camera(
        position=np.array([5.0, -5000.0, 3.0]),
        target=np.array([5.0, 0.0, 3.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=1.0,
        far=20000.0,
    )
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    window_buffer = np.zeros((1080, 1920, 3), dtype=np.float32)

    # Act
    lit_count = _accumulate_building_windows(
        window_test_building, [facing_wall_for_windows], 5000.0,
        view_matrix, proj_matrix, 1920, 1080, building_config, window_buffer,
    )

    # Assert — windows were lit and energy was added somewhere in the buffer
    assert lit_count > 0
    assert window_buffer.sum() > 0.0
    non_zero_pixels = np.count_nonzero(window_buffer.sum(axis=2))
    assert non_zero_pixels <= lit_count


# ---------------------------------------------------------------------------
# _project_window — behind-camera rejection
# ---------------------------------------------------------------------------

def test_project_window_returns_none_when_corner_is_behind_camera(near_view_and_proj):
    # Arrange — a window with one corner far behind the camera
    view_matrix, proj_matrix = near_view_and_proj
    corners = [
        np.array([5.0, 0.0, 2.6]),
        np.array([6.0, 0.0, 2.6]),
        np.array([6.0, -500.0, 3.7]),
        np.array([5.0, 0.0, 3.7]),
    ]

    # Act
    result = _project_window(corners, view_matrix, proj_matrix, 1920, 1080)

    # Assert
    assert result is None
