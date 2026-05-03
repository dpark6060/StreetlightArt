# Radical
"""Unit tests for streets_at_night.rendering.illumination_pass module."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.models.light_source import LightSource
from streets_at_night.rendering.illumination_pass import (
    _build_illumination_map,
    _paint_illumination_blob,
    render_illumination,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_light(
    position: np.ndarray,
    brightness: float = 1.0,
    illumination_radius: float = 50.0,
    illumination_intensity: float = 1.0,
    illumination_falloff: float = 2.0,
    illumination_inner_fraction: float = 0.0,
) -> LightSource:
    """Return a LightSource with sensible defaults, overriding selected fields."""
    return LightSource(
        position=position,
        color=(255, 255, 255),
        brightness=brightness,
        size=0.5,
        glow_radius=10.0,
        illumination_radius=illumination_radius,
        illumination_intensity=illumination_intensity,
        illumination_falloff=illumination_falloff,
        illumination_inner_fraction=illumination_inner_fraction,
    )


def _make_blank_map(size: int = 200) -> np.ndarray:
    """Return a zero-filled (size, size, 3) float32 illumination map."""
    return np.zeros((size, size, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# _paint_illumination_blob — near vs far brightness
# ---------------------------------------------------------------------------


def test_paint_illumination_blob_near_light_brighter_than_far():
    # Arrange — two identical lights, one near the camera, one far away
    near_map = _make_blank_map()
    far_map = _make_blank_map()

    light_pos = np.array([200.0, 200.0, 5.0])
    light = _make_light(position=light_pos)

    near_camera = np.array([200.0, 200.0, 100.0])       # 95 m above light (near)
    far_camera = np.array([200.0, 200.0, 10100.0])      # 10 095 m above light (far)

    min_x, max_y, resolution = 0.0, 400.0, 1.0
    illumination_brightness = 60

    # Act
    _paint_illumination_blob(
        near_map, light, min_x, max_y, resolution,
        illumination_brightness, near_camera, falloff_distance=8000.0,
    )
    _paint_illumination_blob(
        far_map, light, min_x, max_y, resolution,
        illumination_brightness, far_camera, falloff_distance=8000.0,
    )

    # Assert — near blob has higher total intensity than far blob
    assert near_map.sum() > far_map.sum()


# ---------------------------------------------------------------------------
# _paint_illumination_blob — effectively no falloff at large falloff_distance
# ---------------------------------------------------------------------------


def test_paint_illumination_blob_zero_falloff_distance_clamps():
    # Arrange — camera_distance clamps to 1.0; a very large falloff_distance
    # means exp(-1 / 1e10) ≈ 1.0, so blob is painted at approximately full brightness.
    # This validates that the camera_distance clamp prevents a divide-by-zero when the
    # camera and light are co-located.
    illum_map = _make_blank_map()
    light = _make_light(position=np.array([100.0, 100.0, 0.0]))
    camera_pos = np.array([100.0, 100.0, 0.0])   # same position → distance clamped to 1.0 m

    min_x, max_y, resolution = 0.0, 200.0, 1.0
    illumination_brightness = 60
    falloff_distance = 1e10   # effectively no falloff

    # Act
    _paint_illumination_blob(
        illum_map, light, min_x, max_y, resolution,
        illumination_brightness, camera_pos, falloff_distance,
    )

    # Assert — map has non-zero values; the clamp prevented a divide-by-zero and
    # the nearly-zero attenuation left the blob at close to full brightness
    assert illum_map.sum() > 0.0


# ---------------------------------------------------------------------------
# _paint_illumination_blob — tiny radius skipped
# ---------------------------------------------------------------------------


def test_paint_illumination_blob_skips_tiny_radius():
    # Arrange — illumination_radius of 0.0 produces radius_px < 1.0
    illum_map = _make_blank_map()
    light = _make_light(position=np.array([100.0, 100.0, 0.0]), illumination_radius=0.0)
    camera_pos = np.array([0.0, 0.0, 1000.0])

    # Act
    _paint_illumination_blob(
        illum_map, light, 0.0, 200.0, 1.0,
        60, camera_pos, 8000.0,
    )

    # Assert — map remains all zeros; the early-return guard fired
    assert np.all(illum_map == 0.0)


# ---------------------------------------------------------------------------
# _build_illumination_map — multiple lights accumulate
# ---------------------------------------------------------------------------


def test_build_illumination_map_multiple_lights_accumulate():
    # Arrange — two identical lights at the same position
    light_pos = np.array([200.0, 200.0, 5.0])
    single_light = [_make_light(position=light_pos)]
    double_lights = [_make_light(position=light_pos), _make_light(position=light_pos)]

    bounds = (0.0, 0.0, 400.0, 400.0)
    resolution = 1.0
    illumination_brightness = 60
    camera_pos = np.array([0.0, 0.0, 1000.0])
    falloff_distance = 8000.0

    # Act
    single_map, _ = _build_illumination_map(
        single_light, bounds, resolution, illumination_brightness,
        camera_pos, falloff_distance,
    )
    double_map, _ = _build_illumination_map(
        double_lights, bounds, resolution, illumination_brightness,
        camera_pos, falloff_distance,
    )

    # Assert — two lights produce more total intensity than one
    # (maps are clipped to 255 so compare at the light centre pixel)
    cx = int((light_pos[0] - 0.0) / resolution)
    cy = int((400.0 - light_pos[1]) / resolution)

    single_val = single_map[cy, cx, 0]
    double_val = double_map[cy, cx, 0]
    assert double_val > single_val


# ---------------------------------------------------------------------------
# render_illumination — empty lights list
# ---------------------------------------------------------------------------


def test_render_illumination_returns_empty_when_no_lights():
    # Arrange — scene with an empty lights list
    scene = mock.MagicMock()
    scene.lights = []

    camera = mock.MagicMock()
    backend = mock.MagicMock()

    # Act
    result = render_illumination(scene, camera, backend)

    # Assert — returns empty dict and no backend methods called
    assert result == {}
    backend.composite_additive.assert_not_called()
    backend.get_surface_size.assert_not_called()
