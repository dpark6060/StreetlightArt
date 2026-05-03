# Radical
"""Unit tests for streets_at_night.rendering.atmosphere_pass module."""

from unittest import mock

import numpy as np
import pytest
from PIL import Image

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_atmosphere_config
from streets_at_night.config.schema import AtmosphereConfig
from streets_at_night.rendering.atmosphere_pass import (
    _haze_alpha_for_row,
    _parse_hex_color,
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
    """Return a mock RenderBackend with a known surface size and layer factory."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (640, 360)
    backend.create_layer.return_value = Image.new("RGBA", (640, 360), (0, 0, 0, 0))
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


def test_render_atmosphere_calls_composite_additive(mock_scene, mock_camera, mock_backend):
    # Arrange — patch compute_horizon_y so it returns a fixed value
    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ):
        # Act
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    # Assert — composite_additive called at least once (glow layer + haze layer)
    assert mock_backend.composite_additive.call_count >= 1


def test_render_atmosphere_calls_horizon_glow(mock_scene, mock_camera, mock_backend):
    # Arrange
    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ) as mock_glow, mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_distance_haze"
    ):
        # Act
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_glow.assert_called_once()


def test_render_atmosphere_calls_haze_when_enabled(mock_scene, mock_camera, mock_backend):
    # Arrange — haze_enabled defaults to True
    mock_scene.config.atmosphere.haze_enabled = True

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_distance_haze"
    ) as mock_haze:
        # Act
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_haze.assert_called_once()


def test_render_atmosphere_skips_haze_when_disabled(mock_scene, mock_camera, mock_backend):
    # Arrange
    mock_scene.config.atmosphere.haze_enabled = False

    with mock.patch(
        "streets_at_night.rendering.atmosphere_pass.compute_horizon_y",
        return_value=180,
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_horizon_glow"
    ), mock.patch(
        "streets_at_night.rendering.atmosphere_pass._render_distance_haze"
    ) as mock_haze:
        # Act
        render_atmosphere(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_haze.assert_not_called()


# ---------------------------------------------------------------------------
# _render_horizon_glow — gradient behavior
# ---------------------------------------------------------------------------


def test_render_horizon_glow_composites_additive(mock_backend, default_atmosphere):
    # Act
    _render_horizon_glow(mock_backend, 640, 360, 180, default_atmosphere)

    # Assert
    mock_backend.composite_additive.assert_called_once()
    layer_arg = mock_backend.composite_additive.call_args[0][0]
    assert isinstance(layer_arg, Image.Image)
    assert layer_arg.mode == "RGBA"


def test_render_horizon_glow_horizon_row_has_nonzero_alpha(default_atmosphere):
    # Arrange — build a real 10x10 layer to check pixel values
    fake_backend = mock.MagicMock(spec=RenderBackend)
    fake_backend.get_surface_size.return_value = (10, 10)
    captured_layer = []
    fake_backend.composite_additive.side_effect = lambda img: captured_layer.append(img)

    # Act
    _render_horizon_glow(fake_backend, 10, 10, 5, default_atmosphere)

    # Assert — pixel at horizon_y row has non-zero alpha
    arr = np.asarray(captured_layer[0])
    assert arr[5, 5, 3] > 0


def test_render_horizon_glow_far_from_horizon_has_zero_alpha(default_atmosphere):
    # Arrange — use large image so row 0 is far from horizon at center
    fake_backend = mock.MagicMock(spec=RenderBackend)
    fake_backend.get_surface_size.return_value = (640, 360)
    captured_layer = []
    fake_backend.composite_additive.side_effect = lambda img: captured_layer.append(img)

    # Use a narrow glow width so row 0 is outside the band
    default_atmosphere.horizon_glow_width_px = 10

    # Act — horizon at y=180; row 0 is 180 rows away, far beyond half_width=5
    _render_horizon_glow(fake_backend, 640, 360, 180, default_atmosphere)

    # Assert — alpha at row 0 should be 0
    arr = np.asarray(captured_layer[0])
    assert arr[0, 320, 3] == 0


# ---------------------------------------------------------------------------
# _render_distance_haze — haze disabled path
# ---------------------------------------------------------------------------


def test_render_distance_haze_composites_additive(mock_backend, default_atmosphere):
    # Act
    _render_distance_haze(mock_backend, 640, 360, 180, default_atmosphere)

    # Assert
    mock_backend.composite_additive.assert_called_once()


def test_render_distance_haze_layer_is_rgba(mock_backend, default_atmosphere):
    # Arrange
    captured_layer = []
    mock_backend.composite_additive.side_effect = lambda img: captured_layer.append(img)

    # Act
    _render_distance_haze(mock_backend, 640, 360, 180, default_atmosphere)

    # Assert
    assert captured_layer[0].mode == "RGBA"


# ---------------------------------------------------------------------------
# _haze_alpha_for_row — alpha calculation logic
# ---------------------------------------------------------------------------


def test_haze_alpha_at_horizon_row_equals_peak_alpha():
    # At exactly the horizon row, ground section starts: alpha = peak * (1 - 0) = peak
    result = _haze_alpha_for_row(
        row=100, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak_alpha=30
    )
    assert result == 30


def test_haze_alpha_at_ground_fade_bottom_is_zero():
    # At the fade bottom, t=1 so alpha = peak * (1 - 1) = 0
    result = _haze_alpha_for_row(
        row=133, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak_alpha=30
    )
    assert result == 0


def test_haze_alpha_above_sky_fade_top_is_zero():
    # Well above sky_fade_top, the fade reaches 0
    result = _haze_alpha_for_row(
        row=0, horizon_y=100, ground_fade_bottom=133, sky_fade_top=50, peak_alpha=30
    )
    assert result == 0


# ---------------------------------------------------------------------------
# _parse_hex_color — hex string parsing
# ---------------------------------------------------------------------------


def test_parse_hex_color_with_hash_prefix():
    # Act
    r, g, b = _parse_hex_color("#28180A")

    # Assert
    assert r == 0x28
    assert g == 0x18
    assert b == 0x0A


def test_parse_hex_color_without_hash_prefix():
    # Act
    r, g, b = _parse_hex_color("FF8040")

    # Assert
    assert r == 255
    assert g == 128
    assert b == 64
