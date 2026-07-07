# Radical
"""Unit tests for streets_at_night.rendering.road_pass."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import (
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
)
from streets_at_night.models.road_types import RoadType
from streets_at_night.rendering.road_pass import (
    _build_style_table,
    _draw_segment,
    _FALLBACK_EXPORT_STYLE,
    _FALLBACK_PREVIEW_STYLE,
    render_roads_export,
    render_roads_from_scene,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def style_config() -> RoadStyleConfig:
    """Return a test road style config with known colors."""
    return RoadStyleConfig(
        highway=RoadStyleEntry(color="#FF0000", width=5),
        major_road=RoadStyleEntry(color="#00FF00", width=4),
        collector=RoadStyleEntry(color="#0000FF", width=3),
        residential=RoadStyleEntry(color="#FFFFFF", width=2),
        service=RoadStyleEntry(color="#808080", width=1),
        minor=RoadStyleEntry(color="#000000", width=1),
    )


@pytest.fixture
def mock_scene():
    """Return a mock scene with no roads and a config with roads."""
    scene = mock.MagicMock()
    scene.get_roads_in_rect.return_value = []
    _style = RoadStyleConfig(
        highway=RoadStyleEntry(color="#141210", width=4),
        major_road=RoadStyleEntry(color="#121010", width=3),
        collector=RoadStyleEntry(color="#100E0E", width=2),
        residential=RoadStyleEntry(color="#0E0C0C", width=1),
        service=RoadStyleEntry(color="#0C0A0A", width=1),
        minor=RoadStyleEntry(color="#0A0908", width=1),
    )
    scene.config.roads = RoadsConfig(export=_style, preview=_style)
    return scene


@pytest.fixture
def mock_camera():
    """Return a mock camera."""
    camera = mock.MagicMock()
    camera.position = np.array([0.0, 0.0, 1000.0])
    camera.far = 60000.0
    camera.near = 100.0
    camera.get_view_matrix.return_value = np.eye(4)
    camera.get_projection_matrix.return_value = np.eye(4)
    return camera


@pytest.fixture
def mock_backend():
    """Return a mock render backend."""
    backend = mock.MagicMock()
    backend.get_surface_size.return_value = (640, 360)
    return backend


# ---------------------------------------------------------------------------
# _build_style_table
# ---------------------------------------------------------------------------

def test_build_style_table_maps_all_road_types(style_config: RoadStyleConfig):
    # Act
    table = _build_style_table(style_config)

    # Assert
    assert set(table.keys()) == {
        RoadType.HIGHWAY,
        RoadType.MAJOR_ROAD,
        RoadType.COLLECTOR,
        RoadType.RESIDENTIAL,
        RoadType.SERVICE,
        RoadType.MINOR,
    }


def test_build_style_table_converts_hex_to_rgb(style_config: RoadStyleConfig):
    # Act
    table = _build_style_table(style_config)

    # Assert
    color, width = table[RoadType.HIGHWAY]
    assert color == (255, 0, 0)
    assert width == 5


def test_build_style_table_preserves_widths(style_config: RoadStyleConfig):
    # Act
    table = _build_style_table(style_config)

    # Assert
    assert table[RoadType.MAJOR_ROAD][1] == 4
    assert table[RoadType.COLLECTOR][1] == 3
    assert table[RoadType.RESIDENTIAL][1] == 2


# ---------------------------------------------------------------------------
# render_roads_export — reads export config
# ---------------------------------------------------------------------------

def test_render_roads_export_returns_zero_when_no_roads(
    mock_scene, mock_camera, mock_backend
):
    # Act
    drawn = render_roads_export(mock_scene, mock_camera, mock_backend)

    # Assert
    assert drawn == 0


def test_render_roads_export_queries_spatial_index(
    mock_scene, mock_camera, mock_backend
):
    # Act
    render_roads_export(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_scene.get_roads_in_rect.assert_called_once()


def test_render_roads_export_falls_back_without_roads_config(
    mock_camera, mock_backend
):
    # Arrange — scene without roads config attribute
    scene = mock.MagicMock()
    scene.get_roads_in_rect.return_value = []
    del scene.config.roads

    # Act
    drawn = render_roads_export(scene, mock_camera, mock_backend)

    # Assert — should not crash, uses fallback
    assert drawn == 0


# ---------------------------------------------------------------------------
# render_roads_from_scene — reads preview config
# ---------------------------------------------------------------------------

def test_render_roads_from_scene_returns_zero_when_no_roads(
    mock_scene, mock_camera, mock_backend
):
    # Act
    drawn = render_roads_from_scene(mock_scene, mock_camera, mock_backend)

    # Assert
    assert drawn == 0


def test_render_roads_from_scene_falls_back_without_roads_config(
    mock_camera, mock_backend
):
    # Arrange
    scene = mock.MagicMock()
    scene.get_roads_in_rect.return_value = []
    del scene.config.roads

    # Act
    drawn = render_roads_from_scene(scene, mock_camera, mock_backend)

    # Assert
    assert drawn == 0


# ---------------------------------------------------------------------------
# Fallback style tables
# ---------------------------------------------------------------------------

def test_fallback_export_style_has_dark_colors():
    # Assert — export colors should be very dark (max channel < 30)
    for road_type, (color, _width) in _FALLBACK_EXPORT_STYLE.items():
        assert max(color) <= 30, f"{road_type} export color too bright: {color}"


def test_fallback_preview_style_has_bright_colors():
    # Assert — preview colors should be visible (max channel > 80)
    for road_type, (color, _width) in _FALLBACK_PREVIEW_STYLE.items():
        assert max(color) > 80, f"{road_type} preview color too dark: {color}"


# ---------------------------------------------------------------------------
# _draw_segment — screen-edge crossing edges are still drawn
# ---------------------------------------------------------------------------

def test_draw_segment_edge_crossing_screen_border_is_drawn():
    # Arrange — camera at (0, 0, 100) looking down +X. The segment runs from
    # (500, -50, 0), which projects near screen center, to (500, 5000, 0),
    # which is far off to one side. Both endpoints are in front of the
    # camera, so the edge must still be drawn even though one endpoint
    # projects well outside the screen bounds — backends clip lines
    # themselves.
    camera = create_camera(
        position=np.array([0.0, 0.0, 100.0]),
        target=np.array([1000.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    segment_geom = np.array([[500.0, -50.0, 0.0], [500.0, 5000.0, 0.0]])
    style_table = {RoadType.HIGHWAY: ((255, 215, 0), 4)}
    backend = mock.MagicMock()

    # Act
    was_drawn = _draw_segment(
        segment_geom,
        RoadType.HIGHWAY,
        style_table,
        view_matrix,
        proj_matrix,
        camera.near,
        camera.far,
        (1920, 1080),
        backend,
    )

    # Assert
    assert was_drawn is True
    backend.draw_line.assert_called_once()
