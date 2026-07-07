# Radical
"""Unit tests for streets_at_night.rendering.compositor module."""

from unittest import mock

import numpy as np
import pytest
from PIL import Image

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import AtmosphereConfig, GradeConfig, StarsConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.rendering.backend import RenderBackend
from streets_at_night.rendering.compositor import (
    _add_stars,
    _apply_color_grade,
    _apply_vignette,
    _build_vignette_scalar,
    render_composite,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_atmosphere() -> AtmosphereConfig:
    """Return a default AtmosphereConfig built from DEFAULTS."""
    return dataclass_from_dict(AtmosphereConfig, DEFAULTS["atmosphere"])


@pytest.fixture
def default_stars() -> StarsConfig:
    """Return a default StarsConfig with stars disabled."""
    return dataclass_from_dict(StarsConfig, DEFAULTS["stars"])


@pytest.fixture
def default_grade() -> GradeConfig:
    """Return a default GradeConfig (neutral exposure, ACES)."""
    return GradeConfig(exposure=0.0, tonemap="aces")


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
def mock_backend() -> mock.MagicMock:
    """Return a mock RenderBackend resolving to a mid-gray linear frame."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (100, 100)
    backend.resolve_linear.return_value = np.full((100, 100, 3), 0.2, dtype=np.float32)
    return backend


@pytest.fixture
def mock_scene(default_atmosphere, default_stars, default_grade) -> mock.MagicMock:
    """Return a mock Scene with default atmosphere, grade, and disabled stars."""
    scene = mock.MagicMock()
    scene.config.atmosphere = default_atmosphere
    scene.config.stars = default_stars
    scene.config.grade = default_grade
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


def test_render_composite_applies_vignette_and_grade(mock_scene, mock_camera, mock_backend):
    with mock.patch(
        "streets_at_night.rendering.compositor._apply_vignette"
    ) as mock_vignette, mock.patch(
        "streets_at_night.rendering.compositor._apply_color_grade"
    ) as mock_grade:
        render_composite(mock_scene, mock_camera, mock_backend)

    mock_vignette.assert_called_once()
    mock_grade.assert_called_once()


def test_render_composite_skips_stars_when_disabled(mock_scene, mock_camera, mock_backend):
    with mock.patch(
        "streets_at_night.rendering.compositor._apply_vignette"
    ), mock.patch(
        "streets_at_night.rendering.compositor._apply_color_grade"
    ), mock.patch(
        "streets_at_night.rendering.compositor._add_stars"
    ) as mock_stars:
        render_composite(mock_scene, mock_camera, mock_backend)

    mock_stars.assert_not_called()


def test_render_composite_adds_stars_when_enabled(mock_scene, mock_camera, mock_backend, enabled_stars):
    mock_scene.config.stars = enabled_stars

    with mock.patch(
        "streets_at_night.rendering.compositor._apply_vignette"
    ), mock.patch(
        "streets_at_night.rendering.compositor._apply_color_grade"
    ), mock.patch(
        "streets_at_night.rendering.compositor._add_stars"
    ) as mock_stars, mock.patch(
        "streets_at_night.rendering.compositor.compute_horizon_y",
        return_value=50,
    ):
        render_composite(mock_scene, mock_camera, mock_backend)

    mock_stars.assert_called_once()


def test_render_composite_skips_stars_when_stars_config_is_none(mock_scene, mock_camera, mock_backend):
    mock_scene.config.stars = None

    with mock.patch(
        "streets_at_night.rendering.compositor._apply_vignette"
    ), mock.patch(
        "streets_at_night.rendering.compositor._apply_color_grade"
    ), mock.patch(
        "streets_at_night.rendering.compositor._add_stars"
    ) as mock_stars:
        render_composite(mock_scene, mock_camera, mock_backend)

    mock_stars.assert_not_called()


# ---------------------------------------------------------------------------
# _apply_vignette — multiply in linear
# ---------------------------------------------------------------------------


def test_apply_vignette_darkens_edges(default_atmosphere):
    linear = np.ones((100, 100, 3), dtype=np.float32)

    _apply_vignette(linear, 100, 100, default_atmosphere)

    assert linear[0, 0, 0] < linear[50, 50, 0]


def test_apply_vignette_center_unchanged_with_full_radius(default_atmosphere):
    linear = np.ones((100, 100, 3), dtype=np.float32)
    default_atmosphere.vignette_radius = 1.0

    _apply_vignette(linear, 100, 100, default_atmosphere)

    assert linear[50, 50, 0] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# _build_vignette_scalar — radial scalar field
# ---------------------------------------------------------------------------


def test_build_vignette_scalar_center_is_one():
    scalar = _build_vignette_scalar(100, 100, strength=0.5, radius=0.7)

    assert abs(scalar[50, 50] - 1.0) < 0.01


def test_build_vignette_scalar_corners_are_darkened():
    scalar = _build_vignette_scalar(100, 100, strength=0.5, radius=0.0)

    assert scalar[0, 0] < 1.0


def test_build_vignette_scalar_zero_strength_returns_all_ones():
    scalar = _build_vignette_scalar(100, 100, strength=0.0, radius=0.7)

    assert np.all(scalar >= 0.999)


# ---------------------------------------------------------------------------
# _apply_color_grade — tone-map and encode
# ---------------------------------------------------------------------------


def test_apply_color_grade_pastes_rgb_uint8_image(default_grade):
    linear = np.full((8, 8, 3), 0.2, dtype=np.float32)
    backend = mock.MagicMock(spec=RenderBackend)
    captured = []
    backend.paste_image.side_effect = lambda img: captured.append(img)

    _apply_color_grade(backend, linear, default_grade)

    assert captured[0].mode == "RGB"
    assert captured[0].size == (8, 8)


def test_apply_color_grade_rolls_off_bright_without_clip(default_grade):
    # A bright (>1.0) linear pixel must map below pure white, not hard-clip.
    linear = np.zeros((4, 4, 3), dtype=np.float32)
    linear[1, 1, :] = 1.2
    backend = mock.MagicMock(spec=RenderBackend)
    captured = []
    backend.paste_image.side_effect = lambda img: captured.append(img)

    _apply_color_grade(backend, linear, default_grade)

    arr = np.asarray(captured[0])
    assert 0 < arr[1, 1, 0] < 255
    assert arr[0, 0, 0] == 0


def test_apply_color_grade_exposure_brightens():
    linear = np.full((4, 4, 3), 0.1, dtype=np.float32)
    backend_low = mock.MagicMock(spec=RenderBackend)
    backend_high = mock.MagicMock(spec=RenderBackend)
    low, high = [], []
    backend_low.paste_image.side_effect = lambda img: low.append(img)
    backend_high.paste_image.side_effect = lambda img: high.append(img)

    _apply_color_grade(backend_low, linear, GradeConfig(exposure=0.0, tonemap="aces"))
    _apply_color_grade(backend_high, linear, GradeConfig(exposure=2.0, tonemap="aces"))

    assert int(np.asarray(high[0])[0, 0, 0]) > int(np.asarray(low[0])[0, 0, 0])


# ---------------------------------------------------------------------------
# _add_stars — adds emission only above horizon
# ---------------------------------------------------------------------------


def test_add_stars_only_above_horizon(enabled_stars):
    linear = np.zeros((100, 100, 3), dtype=np.float32)
    rng = np.random.default_rng(42)

    _add_stars(linear, 100, 100, horizon_y=50, config=enabled_stars, rng=rng)

    lit_rows = np.where(linear.sum(axis=2) > 0)[0]
    assert lit_rows.size > 0
    assert np.all(lit_rows < 50)


def test_add_stars_skips_when_horizon_at_zero(enabled_stars):
    linear = np.zeros((640, 360, 3), dtype=np.float32)
    rng = np.random.default_rng(42)

    _add_stars(linear, 360, 640, horizon_y=0, config=enabled_stars, rng=rng)

    assert np.all(linear == 0)
