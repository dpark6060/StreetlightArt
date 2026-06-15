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
    _haze_alpha_column,
    _render_distance_haze,
    _render_horizon_glow,
    render_atmosphere,
)
from streets_at_night.rendering.backend import RenderBackend


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


# ---------------------------------------------------------------------------
# render_atmosphere — top-level behavior
# ---------------------------------------------------------------------------


def test_render_atmosphere_accumulates_linear(mock_scene, mock_camera, mock_backend):
    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ):
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    assert mock_backend.add_linear.call_count >= 1


def test_render_atmosphere_calls_horizon_glow(mock_scene, mock_camera, mock_backend):
    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ) as mock_glow, mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_distance_haze"
    ):
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    mock_glow.assert_called_once()


def test_render_atmosphere_calls_haze_when_enabled(mock_scene, mock_camera, mock_backend):
    mock_scene.config.atmosphere.haze_enabled = True

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_distance_haze"
    ) as mock_haze:
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    mock_haze.assert_called_once()


def test_render_atmosphere_skips_haze_when_disabled(mock_scene, mock_camera, mock_backend):
    mock_scene.config.atmosphere.haze_enabled = False

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_distance_haze"
    ) as mock_haze:
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    mock_haze.assert_not_called()


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
    captured = []
    fake_backend.add_linear.side_effect = lambda layer: captured.append(layer)

    _render_horizon_glow(fake_backend, 10, 10, 5, default_atmosphere)

    assert captured[0][5, 5, :].sum() > 0


def test_render_horizon_glow_far_from_horizon_is_zero(default_atmosphere):
    fake_backend = mock.MagicMock(spec=RenderBackend)
    fake_backend.get_surface_size.return_value = (640, 360)
    captured = []
    fake_backend.add_linear.side_effect = lambda layer: captured.append(layer)
    default_atmosphere.horizon_glow_width_px = 10

    _render_horizon_glow(fake_backend, 640, 360, 180, default_atmosphere)

    assert captured[0][0, 320, :].sum() == 0


def test_render_horizon_glow_skips_when_width_is_zero(mock_backend, default_atmosphere):
    default_atmosphere.horizon_glow_width_px = 1  # // 2 == 0

    _render_horizon_glow(mock_backend, 640, 360, 180, default_atmosphere)

    mock_backend.add_linear.assert_not_called()


# ---------------------------------------------------------------------------
# _render_distance_haze
# ---------------------------------------------------------------------------


def test_render_distance_haze_adds_linear_layer(mock_backend, default_atmosphere):
    _render_distance_haze(mock_backend, 640, 360, 180, default_atmosphere)

    mock_backend.add_linear.assert_called_once()
    layer_arg = mock_backend.add_linear.call_args[0][0]
    assert layer_arg.shape == (360, 640, 3)


# ---------------------------------------------------------------------------
# _haze_alpha_column — vectorized alpha calculation
# ---------------------------------------------------------------------------


def test_haze_alpha_column_at_horizon_row_equals_peak():
    alpha = _haze_alpha_column(
        height=200, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak=30.0
    )

    assert alpha[100] == pytest.approx(30.0)


def test_haze_alpha_column_at_ground_fade_bottom_is_zero():
    alpha = _haze_alpha_column(
        height=200, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak=30.0
    )

    assert alpha[133] == pytest.approx(0.0, abs=1e-5)


def test_haze_alpha_column_above_sky_fade_top_is_zero():
    alpha = _haze_alpha_column(
        height=200, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak=30.0
    )

    assert alpha[0] == pytest.approx(0.0, abs=1e-5)


def test_haze_alpha_column_sky_peaks_at_half():
    alpha = _haze_alpha_column(
        height=200, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak=30.0
    )

    # Just above the horizon (sky side), alpha ramps from peak/2 downward.
    assert alpha[99] == pytest.approx(15.0, abs=1.0)


# ---------------------------------------------------------------------------
# _column_alpha_to_linear_layer
# ---------------------------------------------------------------------------


def test_column_alpha_to_linear_layer_scales_color_by_alpha():
    alpha = np.array([0.0, 1.0], dtype=np.float32)

    layer = _column_alpha_to_linear_layer(alpha, width=3, color_linear=(0.2, 0.4, 0.6))

    assert layer.shape == (2, 3, 3)
    np.testing.assert_allclose(layer[0], 0.0)
    np.testing.assert_allclose(layer[1, 0], [0.2, 0.4, 0.6], atol=1e-6)
