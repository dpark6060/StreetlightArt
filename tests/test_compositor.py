# Radical
"""Unit tests for streets_at_night.rendering.compositor module."""

from unittest import mock

import numpy as np
import pytest
from PIL import Image

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.loader import _build_atmosphere_config, _build_stars_config
from streets_at_night.config.schema import AtmosphereConfig, StarsConfig
from streets_at_night.rendering.backend import RenderBackend
from streets_at_night.rendering.compositor import (
    _apply_scalar_to_image,
    _build_vignette_scalar,
    _render_stars,
    _render_vignette,
    render_composite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_atmosphere() -> AtmosphereConfig:
    """Return a default AtmosphereConfig built from DEFAULTS."""
    return _build_atmosphere_config(DEFAULTS["atmosphere"])


@pytest.fixture
def default_stars() -> StarsConfig:
    """Return a default StarsConfig with stars disabled."""
    return _build_stars_config(DEFAULTS["stars"])


@pytest.fixture
def enabled_stars() -> StarsConfig:
    """Return a StarsConfig with stars enabled and a small count for speed."""
    return StarsConfig(
        stars_enabled=True,
        count=10,
        brightness_min=0.5,
        brightness_max=0.8,
        color_variation=5,
        radius=1.0,
    )


@pytest.fixture
def white_image() -> Image.Image:
    """Return a small solid-white RGB image for vignette testing."""
    return Image.new("RGB", (100, 100), (255, 255, 255))


@pytest.fixture
def mock_backend(white_image) -> mock.MagicMock:
    """Return a mock RenderBackend returning a white image from get_image."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (100, 100)
    backend.get_image.return_value = white_image
    return backend


@pytest.fixture
def mock_scene(default_atmosphere, default_stars) -> mock.MagicMock:
    """Return a mock Scene with default atmosphere and disabled stars."""
    scene = mock.MagicMock()
    scene.config.atmosphere = default_atmosphere
    scene.config.stars = default_stars
    scene.config.seed = 42
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
# render_composite — top-level behavior
# ---------------------------------------------------------------------------


def test_render_composite_always_calls_vignette(mock_scene, mock_camera, mock_backend):
    # Arrange
    with mock.patch(
        "streets_at_night.rendering.compositor._render_vignette"
    ) as mock_vignette, mock.patch(
        "streets_at_night.rendering.compositor.compute_horizon_y",
        return_value=50,
    ):
        # Act
        render_composite(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_vignette.assert_called_once()


def test_render_composite_skips_stars_when_disabled(mock_scene, mock_camera, mock_backend):
    # Arrange — default_stars has stars_enabled=False
    with mock.patch(
        "streets_at_night.rendering.compositor._render_vignette"
    ), mock.patch(
        "streets_at_night.rendering.compositor._render_stars"
    ) as mock_stars:
        # Act
        render_composite(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_stars.assert_not_called()


def test_render_composite_calls_stars_when_enabled(mock_scene, mock_camera, mock_backend, enabled_stars):
    # Arrange
    mock_scene.config.stars = enabled_stars

    with mock.patch(
        "streets_at_night.rendering.compositor._render_vignette"
    ), mock.patch(
        "streets_at_night.rendering.compositor._render_stars"
    ) as mock_stars, mock.patch(
        "streets_at_night.rendering.compositor.compute_horizon_y",
        return_value=50,
    ):
        # Act
        render_composite(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_stars.assert_called_once()


def test_render_composite_skips_stars_when_stars_config_is_none(mock_scene, mock_camera, mock_backend):
    # Arrange — stars is None
    mock_scene.config.stars = None

    with mock.patch(
        "streets_at_night.rendering.compositor._render_vignette"
    ), mock.patch(
        "streets_at_night.rendering.compositor._render_stars"
    ) as mock_stars:
        # Act
        render_composite(mock_scene, mock_camera, mock_backend)

    # Assert
    mock_stars.assert_not_called()


# ---------------------------------------------------------------------------
# _render_vignette — darkens edges via paste_image
# ---------------------------------------------------------------------------


def test_render_vignette_calls_get_image_and_paste_image(mock_backend, default_atmosphere):
    # Act
    _render_vignette(mock_backend, 100, 100, default_atmosphere)

    # Assert
    mock_backend.get_image.assert_called_once()
    mock_backend.paste_image.assert_called_once()


def test_render_vignette_darkens_edges(default_atmosphere):
    # Arrange — real white image; apply real vignette
    width, height = 100, 100
    white_img = Image.new("RGB", (width, height), (255, 255, 255))

    real_backend = mock.MagicMock(spec=RenderBackend)
    real_backend.get_surface_size.return_value = (width, height)
    real_backend.get_image.return_value = white_img

    captured = []
    real_backend.paste_image.side_effect = lambda img: captured.append(img)

    # Act
    _render_vignette(real_backend, width, height, default_atmosphere)

    # Assert — corner pixel should be darker than center pixel
    result_arr = np.asarray(captured[0])
    center_val = int(result_arr[50, 50, 0])
    corner_val = int(result_arr[0, 0, 0])
    assert corner_val < center_val


def test_render_vignette_center_pixel_unchanged(default_atmosphere):
    # Arrange
    width, height = 100, 100
    white_img = Image.new("RGB", (width, height), (255, 255, 255))

    real_backend = mock.MagicMock(spec=RenderBackend)
    real_backend.get_surface_size.return_value = (width, height)
    real_backend.get_image.return_value = white_img

    captured = []
    real_backend.paste_image.side_effect = lambda img: captured.append(img)

    # Act — use vignette_radius=1.0 so darkening starts at the very edge only
    default_atmosphere.vignette_radius = 1.0
    _render_vignette(real_backend, width, height, default_atmosphere)

    # Assert — center pixel at (50, 50) should be at or near 255
    result_arr = np.asarray(captured[0])
    assert result_arr[50, 50, 0] >= 250


# ---------------------------------------------------------------------------
# _build_vignette_scalar — radial scalar field
# ---------------------------------------------------------------------------


def test_build_vignette_scalar_center_is_one():
    # Act
    scalar = _build_vignette_scalar(100, 100, strength=0.5, radius=0.7)

    # Assert — center pixel (50, 50) should be 1.0 (no darkening)
    assert abs(scalar[50, 50] - 1.0) < 0.01


def test_build_vignette_scalar_corners_are_darkened():
    # Act
    scalar = _build_vignette_scalar(100, 100, strength=0.5, radius=0.0)

    # Assert — corner scalar must be less than 1.0
    assert scalar[0, 0] < 1.0


def test_build_vignette_scalar_zero_strength_returns_all_ones():
    # Act
    scalar = _build_vignette_scalar(100, 100, strength=0.0, radius=0.7)

    # Assert — no darkening at any pixel
    assert np.all(scalar >= 0.999)


# ---------------------------------------------------------------------------
# _apply_scalar_to_image — channel multiplication
# ---------------------------------------------------------------------------


def test_apply_scalar_to_image_full_scalar_preserves_values():
    # Arrange — scalar of 1.0 everywhere should leave image unchanged
    image = Image.new("RGB", (10, 10), (200, 150, 100))
    scalar = np.ones((10, 10), dtype=np.float32)

    # Act
    result = _apply_scalar_to_image(image, scalar)

    # Assert
    arr = np.asarray(result)
    assert arr[5, 5, 0] == 200
    assert arr[5, 5, 1] == 150
    assert arr[5, 5, 2] == 100


def test_apply_scalar_to_image_zero_scalar_produces_black():
    # Arrange
    image = Image.new("RGB", (10, 10), (200, 150, 100))
    scalar = np.zeros((10, 10), dtype=np.float32)

    # Act
    result = _apply_scalar_to_image(image, scalar)

    # Assert
    arr = np.asarray(result)
    assert np.all(arr == 0)


# ---------------------------------------------------------------------------
# _render_stars — draws only above horizon
# ---------------------------------------------------------------------------


def test_render_stars_draws_only_above_horizon(enabled_stars):
    # Arrange
    fake_backend = mock.MagicMock(spec=RenderBackend)
    rng = np.random.default_rng(42)

    # Act — horizon at y=50, surface 100px tall
    _render_stars(fake_backend, 100, 100, horizon_y=50, config=enabled_stars, rng=rng)

    # Assert — all draw_point calls have y < 50
    for call in fake_backend.draw_point.call_args_list:
        center = call[0][0]
        assert center[1] < 50


def test_render_stars_skips_rendering_when_horizon_at_zero(enabled_stars):
    # Arrange — horizon_y=0 means no sky pixels available
    fake_backend = mock.MagicMock(spec=RenderBackend)
    rng = np.random.default_rng(42)

    # Act
    _render_stars(fake_backend, 640, 360, horizon_y=0, config=enabled_stars, rng=rng)

    # Assert — nothing drawn
    fake_backend.draw_point.assert_not_called()
    fake_backend.draw_circle.assert_not_called()


def test_render_stars_uses_draw_circle_for_large_radius():
    # Arrange
    stars = StarsConfig(
        stars_enabled=True,
        count=5,
        brightness_min=0.5,
        brightness_max=0.8,
        color_variation=0,
        radius=3.0,
    )
    fake_backend = mock.MagicMock(spec=RenderBackend)
    rng = np.random.default_rng(42)

    # Act
    _render_stars(fake_backend, 640, 360, horizon_y=200, config=stars, rng=rng)

    # Assert — draw_circle used, not draw_point
    assert fake_backend.draw_circle.call_count > 0
    fake_backend.draw_point.assert_not_called()
