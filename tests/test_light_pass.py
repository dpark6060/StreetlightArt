# Radical
"""Unit tests for streets_at_night.rendering.light_pass module."""

import math
from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_atmosphere_config
from streets_at_night.config.schema import AtmosphereConfig
from streets_at_night.models.light_source import LightSource
from streets_at_night.rendering.backend import RenderBackend
from streets_at_night.rendering.light_pass import (
    _accumulate_glow,
    _apply_brightness_falloff,
    _apply_glow_expansion,
    _compute_effective_brightness,
    _compute_light_distances,
    _preview_radius,
    render_lights_export,
    render_lights_preview,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_atmosphere() -> AtmosphereConfig:
    """Return a default AtmosphereConfig built from DEFAULTS."""
    return _build_atmosphere_config(DEFAULTS["atmosphere"])


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
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
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


def _make_scene_with_lights(lights: list, atmosphere: AtmosphereConfig = None) -> mock.MagicMock:
    """Create a mock scene carrying the given lights list and atmosphere config."""
    scene = mock.MagicMock()
    scene.lights = lights
    if atmosphere is not None:
        scene.config.atmosphere = atmosphere
    else:
        scene.config.atmosphere = _build_atmosphere_config(DEFAULTS["atmosphere"])
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

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert
    mock_backend.draw_circle.assert_not_called()
    assert result["lights_drawn"] == 0


# ---------------------------------------------------------------------------
# render_lights_export — with lights
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_accumulates_linear_for_visible_light(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    scene = _make_scene_with_lights([sample_light])

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert — emission (core + glow) added to the linear buffer; cores no
    # longer drawn opaque, so draw_circle/draw_point are never called.
    mock_backend.add_linear.assert_called_once()
    mock_backend.draw_circle.assert_not_called()
    mock_backend.draw_point.assert_not_called()
    assert result["lights_drawn"] == 1


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_adds_linear_even_with_no_visible_lights(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange — light exists but is off-screen (invalid mask)
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([False])
    mock_project.return_value = (screen_coords, valid_mask)
    scene = _make_scene_with_lights([sample_light])

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert — add_linear still called (with an all-zero buffer), nothing drawn
    mock_backend.draw_circle.assert_not_called()
    mock_backend.add_linear.assert_called_once()
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


# ---------------------------------------------------------------------------
# _apply_brightness_falloff
# ---------------------------------------------------------------------------


def test_apply_brightness_falloff_at_zero_distance_returns_full_brightness():
    # Act
    result = _apply_brightness_falloff(1.0, 0.0, 8000.0)

    # Assert — exp(0) = 1, so brightness is unchanged
    assert result == pytest.approx(1.0)


def test_apply_brightness_falloff_at_falloff_distance_returns_roughly_37_percent():
    # Arrange
    brightness = 1.0
    falloff_distance = 8000.0

    # Act
    result = _apply_brightness_falloff(brightness, falloff_distance, falloff_distance)

    # Assert — exp(-1) ≈ 0.368
    assert result == pytest.approx(brightness * math.exp(-1.0), rel=1e-6)


# ---------------------------------------------------------------------------
# _apply_glow_expansion
# ---------------------------------------------------------------------------


def test_apply_glow_expansion_at_zero_distance_returns_base_radius():
    # Act
    result = _apply_glow_expansion(10.0, 0.0, 10000.0)

    # Assert — no expansion at zero distance
    assert result == pytest.approx(10.0)


def test_apply_glow_expansion_at_scatter_distance_doubles_radius():
    # Arrange
    base_radius = 10.0
    scatter_distance = 10000.0

    # Act
    result = _apply_glow_expansion(base_radius, scatter_distance, scatter_distance)

    # Assert — radius doubles at scatter_distance
    assert result == pytest.approx(base_radius * 2.0)


# ---------------------------------------------------------------------------
# _compute_light_distances / _compute_effective_brightness
# ---------------------------------------------------------------------------


def test_compute_light_distances_clamps_to_minimum_one():
    # Arrange — camera sitting exactly on top of the light
    positions = np.array([[0.0, 0.0, 0.0]])
    camera_position = np.array([0.0, 0.0, 0.0])

    # Act
    distances = _compute_light_distances(positions, camera_position)

    # Assert
    assert distances[0] == pytest.approx(1.0)


def test_compute_light_distances_matches_euclidean_norm():
    # Arrange
    positions = np.array([[3.0, 4.0, 0.0]])
    camera_position = np.array([0.0, 0.0, 0.0])

    # Act
    distances = _compute_light_distances(positions, camera_position)

    # Assert — 3-4-5 triangle
    assert distances[0] == pytest.approx(5.0)


def test_compute_effective_brightness_matches_scalar_falloff(sample_light):
    # Arrange
    distances = np.array([2000.0])
    falloff_distance = 8000.0

    # Act
    result = _compute_effective_brightness([sample_light], distances, falloff_distance)

    # Assert — matches the scalar formula for a single light
    expected = sample_light.brightness * math.exp(-2000.0 / falloff_distance)
    assert result[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# render_lights_export — brightness falloff applied
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_applies_brightness_falloff(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange — camera far enough away that falloff measurably dims the light
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    mock_camera.position = np.array([0.0, 100.0, 5.0 + 4000.0])
    scene = _make_scene_with_lights([sample_light])

    # Act
    render_lights_export(scene, mock_camera, mock_backend)

    # Assert — light is still drawn and accumulated into the linear buffer
    mock_backend.add_linear.assert_called_once()


# ---------------------------------------------------------------------------
# New tests — Change 4
# ---------------------------------------------------------------------------


@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_skips_light_below_brightness_threshold(
    mock_project, mock_backend, mock_camera, sample_light
):
    # Arrange — threshold set impossibly high so all lights are culled
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    atmos = AtmosphereConfig(
        falloff_distance=8000.0,
        haze_color=(58, 32, 16),
        scatter_distance=10000.0,
        glow_expansion_enabled=False,
        glow_max_alpha=80,
        glow_falloff_steepness=1.0,
        min_brightness_threshold=1.0,
        illumination_brightness=60,
    )
    scene = _make_scene_with_lights([sample_light], atmosphere=atmos)

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert — nothing drawn because brightness threshold culls the light
    mock_backend.draw_circle.assert_not_called()
    mock_backend.draw_point.assert_not_called()
    assert result["lights_drawn"] == 0



@mock.patch("streets_at_night.rendering.light_pass._apply_glow_expansion")
@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_glow_expansion_disabled_uses_base_radius(
    mock_project, mock_expansion, mock_backend, mock_camera, sample_light
):
    # Arrange — expansion disabled
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    atmos = AtmosphereConfig(
        falloff_distance=8000.0,
        haze_color=(58, 32, 16),
        scatter_distance=10000.0,
        glow_expansion_enabled=False,
        glow_max_alpha=80,
        glow_falloff_steepness=1.0,
        min_brightness_threshold=0.0,
        illumination_brightness=60,
    )
    scene = _make_scene_with_lights([sample_light], atmosphere=atmos)

    # Act
    render_lights_export(scene, mock_camera, mock_backend)

    # Assert — expansion function never called when disabled
    mock_expansion.assert_not_called()


@mock.patch("streets_at_night.rendering.light_pass._apply_glow_expansion")
@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_glow_expansion_enabled_calls_expansion(
    mock_project, mock_expansion, mock_backend, mock_camera, sample_light
):
    # Arrange — expansion enabled
    mock_expansion.return_value = sample_light.glow_radius * 2.0
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    atmos = AtmosphereConfig(
        falloff_distance=8000.0,
        haze_color=(58, 32, 16),
        scatter_distance=10000.0,
        glow_expansion_enabled=True,
        glow_max_alpha=80,
        glow_falloff_steepness=1.0,
        min_brightness_threshold=0.0,
        illumination_brightness=60,
    )
    scene = _make_scene_with_lights([sample_light], atmosphere=atmos)

    # Act
    render_lights_export(scene, mock_camera, mock_backend)

    # Assert — expansion function called once for the visible light
    mock_expansion.assert_called_once()



@mock.patch("streets_at_night.rendering.light_pass.project_points")
def test_render_lights_export_accumulates_tiny_far_light(
    mock_project, mock_backend, mock_camera
):
    # Arrange — light very far away so its core is sub-pixel. The linear path
    # accumulates it via the coverage term rather than a separate point-draw.
    far_light = LightSource(
        position=np.array([0.0, 60000.0, 5.0]),
        color=(255, 200, 100),
        brightness=1.0,
        size=0.01,
        glow_radius=1.0,
        illumination_radius=2.0,
        illumination_intensity=0.5,
        illumination_falloff=2.0,
        illumination_inner_fraction=0.0,
    )
    screen_coords = np.array([[320.0, 180.0]])
    valid_mask = np.array([True])
    mock_project.return_value = (screen_coords, valid_mask)
    atmos = AtmosphereConfig(
        falloff_distance=8000.0,
        haze_color=(58, 32, 16),
        scatter_distance=10000.0,
        glow_expansion_enabled=False,
        glow_max_alpha=80,
        glow_falloff_steepness=1.0,
        min_brightness_threshold=0.0,
        illumination_brightness=60,
    )
    scene = _make_scene_with_lights([far_light], atmosphere=atmos)

    # Act
    result = render_lights_export(scene, mock_camera, mock_backend)

    # Assert — accumulated into the linear buffer, no opaque draw calls
    mock_backend.add_linear.assert_called_once()
    mock_backend.draw_point.assert_not_called()
    mock_backend.draw_circle.assert_not_called()
    assert result["lights_drawn"] == 1


# ---------------------------------------------------------------------------
# _accumulate_glow
# ---------------------------------------------------------------------------


def test_accumulate_glow_adds_to_buffer_at_center():
    # Arrange
    buffer = np.zeros((100, 100, 3), dtype=np.float32)
    cx, cy = 50.0, 50.0

    # Act
    _accumulate_glow(buffer, cx, cy, (1.0, 1.0, 1.0), radius_px=10.0, peak=0.8, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)

    # Assert — center pixels are greater than zero
    assert buffer[50, 50, 0] > 0.0
    assert buffer[50, 50, 1] > 0.0
    assert buffer[50, 50, 2] > 0.0


def test_accumulate_glow_falls_off_with_distance():
    # Arrange
    buffer = np.zeros((100, 100, 3), dtype=np.float32)

    # Act
    _accumulate_glow(buffer, 50.0, 50.0, (1.0, 0.0, 0.0), radius_px=10.0, peak=0.8, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)

    # Assert — pixel near center brighter than pixel near edge of glow
    center_val = buffer[50, 50, 0]
    edge_val = buffer[50, 70, 0]
    assert center_val > edge_val


def test_accumulate_glow_two_lights_accumulate_additively():
    # Arrange
    buffer_single = np.zeros((100, 100, 3), dtype=np.float32)
    buffer_double = np.zeros((100, 100, 3), dtype=np.float32)

    # Act — same position called once vs twice
    _accumulate_glow(buffer_single, 50.0, 50.0, (0.4, 0.0, 0.0), radius_px=10.0, peak=0.8, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)
    _accumulate_glow(buffer_double, 50.0, 50.0, (0.4, 0.0, 0.0), radius_px=10.0, peak=0.8, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)
    _accumulate_glow(buffer_double, 50.0, 50.0, (0.4, 0.0, 0.0), radius_px=10.0, peak=0.8, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)

    # Assert — double accumulation produces roughly double the value
    assert buffer_double[50, 50, 0] == pytest.approx(buffer_single[50, 50, 0] * 2.0, rel=1e-5)


def test_accumulate_glow_respects_peak_zero():
    # Arrange
    buffer = np.zeros((100, 100, 3), dtype=np.float32)

    # Act — peak=0 should contribute nothing
    _accumulate_glow(buffer, 50.0, 50.0, (1.0, 1.0, 1.0), radius_px=10.0, peak=0.0, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)

    # Assert — buffer remains all zeros
    assert np.all(buffer == 0.0)


def test_accumulate_glow_center_equals_peak():
    # Arrange
    buffer = np.zeros((100, 100, 3), dtype=np.float32)

    # Act — core and tail both equal 1.0 at the center, so center == peak
    _accumulate_glow(buffer, 50.0, 50.0, (1.0, 1.0, 1.0), radius_px=10.0, peak=0.8, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)

    # Assert
    assert buffer[50, 50, 0] == pytest.approx(0.8, rel=1e-5)


def test_accumulate_glow_wings_exceed_pure_gaussian():
    # Arrange — a point 40px out, far beyond a sigma=5 gaussian's reach
    buffer = np.zeros((100, 100, 3), dtype=np.float32)

    # Act
    _accumulate_glow(buffer, 50.0, 50.0, (1.0, 0.0, 0.0), radius_px=10.0, peak=1.0, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=8.0)

    # Assert — the power-law wing carries real energy where a gaussian is ~0
    sigma = 10.0 / 2.0
    gaussian_only = np.exp(-(40.0 ** 2) / (2.0 * sigma * sigma))
    far_val = buffer[50, 90, 0]
    assert far_val > 0.0
    assert far_val > gaussian_only


def test_accumulate_glow_zero_beyond_cutoff():
    # Arrange — large buffer so the cutoff is the limiting factor, not the edge
    buffer = np.zeros((200, 200, 3), dtype=np.float32)

    # Act — cutoff_mult=4 with radius 5 bounds the paint to ~22px of center
    _accumulate_glow(buffer, 100.0, 100.0, (1.0, 1.0, 1.0), radius_px=5.0, peak=1.0, steepness=1.0, wing_exponent=2.5, tail_strength=0.15, cutoff_mult=4.0)

    # Assert — a pixel 50px away is outside the bounded region
    assert buffer[100, 150, 0] == 0.0
