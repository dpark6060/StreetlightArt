# Radical
"""Unit tests for streets_at_night.rendering.atmosphere_pass module."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_atmosphere_config
from streets_at_night.config.schema import AtmosphereConfig
from streets_at_night.rendering.atmosphere_pass import (
    _column_alpha_to_linear_layer,
    _render_airlight,
    _render_horizon_glow,
    render_atmosphere,
)
from streets_at_night.rendering.backend import RenderBackend
from streets_at_night.rendering.raycast import GroundHit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_atmosphere() -> AtmosphereConfig:
    """Return a default AtmosphereConfig built from DEFAULTS."""
    return _build_atmosphere_config(DEFAULTS["atmosphere"])


@pytest.fixture
def mock_backend() -> mock.MagicMock:
    """Return a mock RenderBackend with a known surface size."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (640, 360)
    return backend


@pytest.fixture
def mock_scene(default_atmosphere) -> mock.MagicMock:
    """Return a mock Scene with default atmosphere config and haze enabled."""
    scene = mock.MagicMock()
    scene.config.atmosphere = default_atmosphere
    return scene


@pytest.fixture
def mock_camera() -> mock.MagicMock:
    """Return a mock Camera with known matrix methods."""
    camera = mock.MagicMock()
    camera.get_view_matrix.return_value = np.eye(4)
    camera.get_projection_matrix.return_value = np.eye(4)
    camera.position = np.array([0.0, 0.0, 500.0])
    camera.far = 10000.0
    return camera


def _make_gbuffer(
    hit_x: np.ndarray,
    hit_y: np.ndarray,
    ground_mask: np.ndarray,
    ray_z: np.ndarray,
) -> GroundHit:
    """Build a GroundHit from per-pixel hit coords, mask, and ray elevation."""
    h, w = hit_x.shape
    ray_dirs = np.zeros((h, w, 3), dtype=np.float64)
    ray_dirs[:, :, 2] = ray_z
    return GroundHit(ray_dirs=ray_dirs, hit_x=hit_x, hit_y=hit_y, ground_mask=ground_mask)


def _capture_layer(backend: mock.MagicMock) -> list:
    """Wire a backend mock to capture layers passed to add_linear."""
    captured: list = []
    backend.add_linear.side_effect = lambda layer: captured.append(layer)
    return captured


# ---------------------------------------------------------------------------
# render_atmosphere — top-level behavior
# ---------------------------------------------------------------------------


def test_render_atmosphere_accumulates_linear(mock_scene, mock_camera, mock_backend):
    gbuffer = mock.MagicMock()

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_airlight"
    ):
        render_atmosphere(mock_scene, mock_camera, mock_backend, gbuffer)

    assert mock_backend.add_linear.call_count >= 1


def test_render_atmosphere_calls_horizon_glow(mock_scene, mock_camera, mock_backend):
    gbuffer = mock.MagicMock()

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ) as mock_glow, mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_airlight"
    ):
        render_atmosphere(mock_scene, mock_camera, mock_backend, gbuffer)

    mock_glow.assert_called_once()


def test_render_atmosphere_calls_airlight_when_enabled(mock_scene, mock_camera, mock_backend):
    mock_scene.config.atmosphere.haze_enabled = True
    gbuffer = mock.MagicMock()

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_airlight"
    ) as mock_airlight:
        render_atmosphere(mock_scene, mock_camera, mock_backend, gbuffer)

    mock_airlight.assert_called_once()


def test_render_atmosphere_skips_airlight_when_disabled(mock_scene, mock_camera, mock_backend):
    mock_scene.config.atmosphere.haze_enabled = False
    gbuffer = mock.MagicMock()

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_airlight"
    ) as mock_airlight:
        render_atmosphere(mock_scene, mock_camera, mock_backend, gbuffer)

    mock_airlight.assert_not_called()


# ---------------------------------------------------------------------------
# _render_horizon_glow — gradient behavior
# ---------------------------------------------------------------------------


def test_render_horizon_glow_adds_linear_layer(mock_backend, default_atmosphere):
    _render_horizon_glow(mock_backend, 640, 360, 180, default_atmosphere)

    mock_backend.add_linear.assert_called_once()
    layer_arg = mock_backend.add_linear.call_args[0][0]
    assert layer_arg.shape == (360, 640, 3)


def test_render_horizon_glow_horizon_row_has_nonzero_intensity(default_atmosphere):
    fake_backend = mock.MagicMock(spec=RenderBackend)
    fake_backend.get_surface_size.return_value = (10, 10)
    captured = _capture_layer(fake_backend)

    _render_horizon_glow(fake_backend, 10, 10, 5, default_atmosphere)

    assert captured[0][5, 5, :].sum() > 0


def test_render_horizon_glow_far_from_horizon_is_zero(default_atmosphere):
    fake_backend = mock.MagicMock(spec=RenderBackend)
    fake_backend.get_surface_size.return_value = (640, 360)
    captured = _capture_layer(fake_backend)
    default_atmosphere.horizon_glow_width_px = 10

    _render_horizon_glow(fake_backend, 640, 360, 180, default_atmosphere)

    assert captured[0][0, 320, :].sum() == 0


def test_render_horizon_glow_skips_when_width_is_zero(mock_backend, default_atmosphere):
    default_atmosphere.horizon_glow_width_px = 1  # // 2 == 0

    _render_horizon_glow(mock_backend, 640, 360, 180, default_atmosphere)

    mock_backend.add_linear.assert_not_called()


# ---------------------------------------------------------------------------
# _render_airlight — depth-based path radiance
# ---------------------------------------------------------------------------


def test_render_airlight_adds_linear_layer(mock_camera, default_atmosphere):
    fake_backend = mock.MagicMock(spec=RenderBackend)
    captured = _capture_layer(fake_backend)
    hit_x = np.full((4, 5), 100.0)
    hit_y = np.full((4, 5), 100.0)
    gbuffer = _make_gbuffer(hit_x, hit_y, np.ones((4, 5), dtype=bool), np.full((4, 5), -0.5))

    _render_airlight(fake_backend, gbuffer, mock_camera, default_atmosphere)

    fake_backend.add_linear.assert_called_once()
    assert captured[0].shape == (4, 5, 3)


def test_render_airlight_farther_ground_is_hazier(mock_camera, default_atmosphere):
    fake_backend = mock.MagicMock(spec=RenderBackend)
    captured = _capture_layer(fake_backend)
    # Column 0 is directly below the camera; column 1 is 20 km out.
    hit_x = np.array([[0.0, 20000.0]])
    hit_y = np.array([[0.0, 0.0]])
    gbuffer = _make_gbuffer(hit_x, hit_y, np.ones((1, 2), dtype=bool), np.full((1, 2), -0.5))

    _render_airlight(fake_backend, gbuffer, mock_camera, default_atmosphere)

    near = captured[0][0, 0, :].sum()
    far = captured[0][0, 1, :].sum()
    assert far > near


def test_render_airlight_sky_hazier_near_horizon(mock_camera, default_atmosphere):
    fake_backend = mock.MagicMock(spec=RenderBackend)
    captured = _capture_layer(fake_backend)
    # Two sky pixels (no ground hit): one near the horizon (ray_z≈0), one near zenith.
    hit_x = np.zeros((1, 2))
    hit_y = np.zeros((1, 2))
    ray_z = np.array([[0.05, 0.9]])
    gbuffer = _make_gbuffer(hit_x, hit_y, np.zeros((1, 2), dtype=bool), ray_z)

    _render_airlight(fake_backend, gbuffer, mock_camera, default_atmosphere)

    near_horizon = captured[0][0, 0, :].sum()
    near_zenith = captured[0][0, 1, :].sum()
    assert near_horizon > near_zenith


# ---------------------------------------------------------------------------
# _column_alpha_to_linear_layer
# ---------------------------------------------------------------------------


def test_column_alpha_to_linear_layer_scales_color_by_alpha():
    alpha = np.array([0.0, 1.0], dtype=np.float32)

    layer = _column_alpha_to_linear_layer(alpha, width=3, color_linear=(0.2, 0.4, 0.6))

    assert layer.shape == (2, 3, 3)
    np.testing.assert_allclose(layer[0], 0.0)
    np.testing.assert_allclose(layer[1, 0], [0.2, 0.4, 0.6], atol=1e-6)
