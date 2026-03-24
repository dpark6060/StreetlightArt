# Radical
"""Unit tests for streets_at_night.rendering.light_pass module."""

from unittest import mock

import numpy as np
import pytest
from PIL import Image

from streets_at_night.models.light_source import LightSource
from streets_at_night.rendering.backend import RenderBackend
from streets_at_night.rendering.light_pass import (
    _preview_radius,
    render_lights_export,
    render_lights_preview,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> mock.MagicMock:
    """Return a mock RenderBackend with spec and a default surface size."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (640, 360)
    return backend


@pytest.fixture
def sample_light() -> LightSource:
    """Return a LightSource at a known position inside the frustum."""
    return LightSource(
        position=np.array([0.0, 100.0, 5.0]),
        color=(255, 200, 100),
        brightness=1.0,
        size=0.5,
        glow_radius=10.0,
        illumination_radius=20.0,
        illumination_intensity=0.8,
    )


@pytest.fixture
def mock_camera() -> mock.MagicMock:
    """Return a mock Camera that returns identity-like matrices."""
    cam = mock.MagicMock()
    cam.fov_deg = 60.0
    cam.position = np.array([0.0, 0.0, 1000.0])
    cam.get_view_matrix.return_value = np.eye(4)
    cam.get_projection_matrix.return_value = np.eye(4)
    return cam


def _make_scene_with_lights(lights: list) -> mock.MagicMock:
    """Create a mock scene carrying the given lights list."""
    scene = mock.MagicMock()
    scene.lights = lights
    return scene


# ---------------------------------------------------------------------------
# render_lights_preview — no lights
# ---------------------------------------------------------------------------


def test_render_lights_preview_no_lights_skips_draw(mock_backend, mock_camera):
    # Arrange
    scene = _make_scene_with_lights([])

    # Act
    result = render_lights_preview(scene, mock_camera, mock_backend)

    # Assert — no draw calls, count is zero
    mock_backend.draw_circle.assert_not_called()
    assert result["lights_drawn"] == 0


# ---------------------------------------------------------------------------
# render_lights_preview — with lights
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_preview_draws_circles_for_visible_lights(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    scene = _make_scene_with_lights([sample_light])

    # Act
    render_lights_preview(scene, mock_camera, mock_backend)

    # Assert — one circle drawn for the visible light
    mock_backend.draw_circle.assert_called_once()


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_preview_returns_lights_drawn_count(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    scene = _make_scene_with_lights([sample_light])

    # Act
    result = render_lights_preview(scene, mock_camera, mock_backend)

    # Assert
    assert result["lights_drawn"] == 1


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_preview_invalid_mask_skips_circle(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange — valid_mask is False; light is behind the camera
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([False])
    mock_project.return_value = (screen_coords, valid_mask)
    scene = _make_scene_with_lights([sample_light])

    # Act
    result = render_lights_preview(scene, mock_camera, mock_backend)

    # Assert
    mock_backend.draw_circle.assert_not_called()
    assert result["lights_drawn"] == 0


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_preview_uses_light_color_for_draw(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    scene = _make_scene_with_lights([sample_light])

    # Act
    render_lights_preview(scene, mock_camera, mock_backend)

    # Assert — the correct color is passed to draw_circle
    _, call_kwargs = mock_backend.draw_circle.call_args
    call_args = mock_backend.draw_circle.call_args[0]
    assert call_args[2] == sample_light.color


# ---------------------------------------------------------------------------
# render_lights_export — no lights
# ---------------------------------------------------------------------------


def test_render_lights_export_no_lights_skips_draw(mock_backend, mock_camera):
    # Arrange
    scene = _make_scene_with_lights([])
    mock_backend.create_layer.return_value = Image.new("RGBA", (640, 360), (0, 0, 0, 0))

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert
    mock_backend.draw_circle.assert_not_called()
    assert result["lights_drawn"] == 0


# ---------------------------------------------------------------------------
# render_lights_export — with lights
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_draws_core_and_composites_glow(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    glow_layer = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
    mock_backend.create_layer.return_value = glow_layer
    scene = _make_scene_with_lights([sample_light])

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert — core circle drawn and glow composited
    mock_backend.draw_circle.assert_called_once()
    mock_backend.composite_additive.assert_called_once()
    assert result["lights_drawn"] == 1


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_always_composites_even_with_no_visible_lights(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange — light exists but is off-screen (invalid mask)
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([False])
    mock_project.return_value = (screen_coords, valid_mask)
    glow_layer = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
    mock_backend.create_layer.return_value = glow_layer
    scene = _make_scene_with_lights([sample_light])

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert — composite is still called (empty glow layer), but no core drawn
    mock_backend.draw_circle.assert_not_called()
    mock_backend.composite_additive.assert_called_once()
    assert result["lights_drawn"] == 0


# ---------------------------------------------------------------------------
# _preview_radius
# ---------------------------------------------------------------------------


def test_preview_radius_normal_brightness_returns_three():
    # Act
    radius = _preview_radius(1.0)

    # Assert — brightness 1.0 → 3.0 pixels
    assert radius == pytest.approx(3.0)


def test_preview_radius_zero_brightness_returns_minimum():
    # Act
    radius = _preview_radius(0.0)

    # Assert — minimum is 1.0 even when brightness is zero
    assert radius == pytest.approx(1.0)


def test_preview_radius_high_brightness_scales_up():
    # Act
    radius_low = _preview_radius(0.5)
    radius_high = _preview_radius(2.0)

    # Assert — brighter lights have larger dots
    assert radius_high > radius_low
