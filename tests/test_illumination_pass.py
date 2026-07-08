# Radical
"""Unit tests for streets_at_night.rendering.illumination_pass module."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import Camera
from streets_at_night.models.illumination import BeamIllumination, ConeIllumination
from streets_at_night.models.light_source import LightSource
from streets_at_night.rendering.illumination_pass import (
    _auto_scale_illumination_resolution,
    _build_illumination_map,
    _compute_visible_bounds,
    _effective_pool_radius,
    _is_light_visible,
    _paint_illumination_blob,
    render_illumination,
)
from streets_at_night.rendering.raycast import GroundHit


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
    beam_angle_deg=None,
    illumination="cone",
) -> LightSource:
    """Return a LightSource with sensible defaults, overriding selected fields.

    ``illumination`` selects the ground-pool model: a ``beam_angle_deg`` makes
    it a BeamIllumination, otherwise a ConeIllumination from the radius/falloff/
    inner-fraction args. Pass ``illumination=None`` for a light with no pool.
    """
    if illumination is None:
        model = None
    elif beam_angle_deg is not None:
        model = BeamIllumination(beam_angle_deg)
    else:
        model = ConeIllumination(
            illumination_radius, illumination_falloff, illumination_inner_fraction
        )
    return LightSource(
        position=position,
        color=(255, 255, 255),
        brightness=brightness,
        size=0.5,
        glow_radius=10.0,
        illumination_intensity=illumination_intensity,
        illumination=model,
    )


def _make_blank_map(size: int = 200) -> np.ndarray:
    """Return a zero-filled (size, size, 3) float32 illumination map."""
    return np.zeros((size, size, 3), dtype=np.float32)


def _make_camera(position: np.ndarray, far: float = 5000.0) -> Camera:
    """Return a Camera with a fixed position/far, target/fov/near unused."""
    return Camera(
        position=position,
        target=np.array([0.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=1.0,
        near=1.0,
        far=far,
    )


def _make_gbuffer(
    hit_x: np.ndarray, hit_y: np.ndarray, ground_mask: np.ndarray
) -> GroundHit:
    """Return a GroundHit with the given per-pixel hit coords and mask."""
    ray_dirs = np.zeros(hit_x.shape + (3,), dtype=np.float64)
    hit_t = np.where(ground_mask, 1.0, -1.0)
    return GroundHit(
        ray_dirs=ray_dirs, hit_x=hit_x, hit_y=hit_y, hit_t=hit_t, ground_mask=ground_mask,
    )


def _make_grid_gbuffer(
    center: float, half_extent: float, size: int = 4
) -> GroundHit:
    """Return a GroundHit whose hits form a square grid centered on `center`."""
    coords = np.linspace(center - half_extent, center + half_extent, size)
    hit_x, hit_y = np.meshgrid(coords, coords)
    ground_mask = np.ones((size, size), dtype=bool)
    return _make_gbuffer(hit_x, hit_y, ground_mask)


def _make_atmosphere(
    illumination_brightness: int = 60,
    falloff_distance: float = 8000.0,
    min_brightness_threshold: float = 0.01,
) -> mock.MagicMock:
    """Return a MagicMock standing in for scene.config.atmosphere."""
    atmos = mock.MagicMock()
    atmos.illumination_brightness = illumination_brightness
    atmos.falloff_distance = falloff_distance
    atmos.min_brightness_threshold = min_brightness_threshold
    return atmos


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
# _paint_illumination_blob — physical height + beam-angle model
# ---------------------------------------------------------------------------


def test_paint_illumination_blob_physical_taller_pole_wider_pool():
    # Arrange — same beam angle, two pole heights; physical model derives radius
    short_map = _make_blank_map()
    tall_map = _make_blank_map()
    short = _make_light(position=np.array([100.0, 100.0, 10.0]), beam_angle_deg=120.0)
    tall = _make_light(position=np.array([100.0, 100.0, 20.0]), beam_angle_deg=120.0)
    camera_pos = np.array([100.0, 100.0, 5000.0])

    # Act
    _paint_illumination_blob(short_map, short, 0.0, 200.0, 1.0, 60, camera_pos, 1e10)
    _paint_illumination_blob(tall_map, tall, 0.0, 200.0, 1.0, 60, camera_pos, 1e10)

    # Assert — taller pole spreads light over more ground
    assert np.count_nonzero(tall_map[:, :, 0]) > np.count_nonzero(short_map[:, :, 0])


def test_paint_illumination_blob_physical_center_is_brightest():
    # Arrange
    illum_map = _make_blank_map()
    light = _make_light(position=np.array([100.0, 100.0, 10.0]), beam_angle_deg=120.0)
    camera_pos = np.array([100.0, 100.0, 5000.0])

    # Act
    _paint_illumination_blob(illum_map, light, 0.0, 200.0, 1.0, 60, camera_pos, 1e10)

    # Assert — directly under the lamp is the brightest point
    center = illum_map[100, 100, 0]
    assert center > 0.0
    assert center == illum_map[:, :, 0].max()


def test_paint_illumination_blob_physical_zero_beyond_beam_radius():
    # Arrange — h=10, beam 120° → r_max ≈ 17.3 m
    illum_map = _make_blank_map()
    light = _make_light(position=np.array([100.0, 100.0, 10.0]), beam_angle_deg=120.0)
    camera_pos = np.array([100.0, 100.0, 5000.0])

    # Act
    _paint_illumination_blob(illum_map, light, 0.0, 200.0, 1.0, 60, camera_pos, 1e10)

    # Assert — a point 30 m out is well beyond the beam and stays dark
    assert illum_map[100, 130, 0] == 0.0


def test_paint_illumination_blob_legacy_cone_used_when_no_beam_angle():
    # Arrange — no beam angle → authored illumination_radius drives the cone
    illum_map = _make_blank_map()
    light = _make_light(
        position=np.array([100.0, 100.0, 0.0]),
        illumination_radius=50.0,
        beam_angle_deg=None,
    )
    camera_pos = np.array([100.0, 100.0, 5000.0])

    # Act
    _paint_illumination_blob(illum_map, light, 0.0, 200.0, 1.0, 60, camera_pos, 1e10)

    # Assert — the legacy cone path painted the authored-radius pool
    assert illum_map.sum() > 0.0
    assert illum_map[100, 100, 0] > 0.0


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
    min_brightness_threshold = 0.01

    # Act
    single_map, _ = _build_illumination_map(
        single_light, bounds, resolution, illumination_brightness,
        camera_pos, falloff_distance, min_brightness_threshold,
    )
    double_map, _ = _build_illumination_map(
        double_lights, bounds, resolution, illumination_brightness,
        camera_pos, falloff_distance, min_brightness_threshold,
    )

    # Assert — two lights produce more total intensity than one
    # (maps are clipped to 255 so compare at the light centre pixel)
    cx = int((light_pos[0] - 0.0) / resolution)
    cy = int((400.0 - light_pos[1]) / resolution)

    single_val = single_map[cy, cx, 0]
    double_val = double_map[cy, cx, 0]
    assert double_val > single_val


def test_build_illumination_map_culls_light_outside_bounds():
    # Arrange — one light inside bounds, one far outside
    inside_light = _make_light(position=np.array([200.0, 200.0, 5.0]))
    outside_light = _make_light(position=np.array([50000.0, 50000.0, 5.0]))
    bounds = (0.0, 0.0, 400.0, 400.0)
    camera_pos = np.array([0.0, 0.0, 1000.0])

    # Act
    with mock.patch(
        "streets_at_night.rendering.illumination_pass._paint_illumination_blob"
    ) as mock_paint:
        _build_illumination_map(
            [inside_light, outside_light], bounds, 1.0, 60, camera_pos, 8000.0, 0.01,
        )

    # Assert — only the light inside bounds was painted
    assert mock_paint.call_count == 1
    painted_light = mock_paint.call_args.args[1]
    assert painted_light is inside_light


# ---------------------------------------------------------------------------
# _effective_pool_radius
# ---------------------------------------------------------------------------


def test_effective_pool_radius_physical_light_uses_beam_geometry():
    # Arrange — h=10, beam 120° → r = h * tan(60°) ≈ 17.32 m
    light = _make_light(position=np.array([0.0, 0.0, 10.0]), beam_angle_deg=120.0)

    # Act
    radius = _effective_pool_radius(light)

    # Assert
    assert radius == pytest.approx(17.32, abs=0.01)


def test_effective_pool_radius_legacy_light_uses_authored_radius():
    # Arrange — no beam angle → falls back to illumination_radius
    light = _make_light(position=np.array([0.0, 0.0, 0.0]), illumination_radius=42.0)

    # Act
    radius = _effective_pool_radius(light)

    # Assert
    assert radius == 42.0


def test_effective_pool_radius_beam_light_at_ground_level_raises():
    # Arrange — a BeamIllumination light at z=0 has an undefined physical pool
    light = _make_light(position=np.array([0.0, 0.0, 0.0]), beam_angle_deg=120.0)

    # Act & Assert — fail loud rather than silently produce a wrong pool
    with pytest.raises(ValueError, match="positive pole height"):
        _effective_pool_radius(light)


def test_build_illumination_map_none_illumination_paints_nothing():
    # Arrange — a light with no ground-pool model
    light = _make_light(position=np.array([100.0, 100.0, 5.0]), illumination=None)
    bounds = (0.0, 0.0, 200.0, 200.0)

    # Act
    illum_map, _ = _build_illumination_map(
        [light], bounds, resolution=1.0, illumination_brightness=200,
        camera_position=np.array([0.0, 0.0, 100.0]), falloff_distance=10000.0,
        min_brightness_threshold=0.0,
    )

    # Assert — nothing was painted
    assert illum_map.sum() == 0.0


# ---------------------------------------------------------------------------
# _is_light_visible
# ---------------------------------------------------------------------------


def test_is_light_visible_false_when_outside_bounds():
    # Arrange — light far to the east of the map bounds
    light = _make_light(position=np.array([10000.0, 0.0, 5.0]), illumination_radius=50.0)
    bounds = (0.0, 0.0, 400.0, 400.0)
    camera_pos = np.array([0.0, 0.0, 1000.0])

    # Act
    visible = _is_light_visible(light, bounds, camera_pos, 8000.0, 0.01)

    # Assert
    assert visible is False


def test_is_light_visible_false_when_below_brightness_threshold():
    # Arrange — light in bounds but very dim and far from the camera
    light = _make_light(
        position=np.array([200.0, 200.0, 5.0]), brightness=0.001, illumination_radius=50.0,
    )
    bounds = (0.0, 0.0, 400.0, 400.0)
    camera_pos = np.array([200.0, 200.0, 100000.0])

    # Act
    visible = _is_light_visible(light, bounds, camera_pos, falloff_distance=100.0, min_brightness_threshold=0.5)

    # Assert
    assert visible is False


def test_is_light_visible_true_when_in_bounds_and_bright():
    # Arrange — light in bounds, near camera, well above threshold
    light = _make_light(position=np.array([200.0, 200.0, 5.0]), illumination_radius=50.0)
    bounds = (0.0, 0.0, 400.0, 400.0)
    camera_pos = np.array([200.0, 200.0, 500.0])

    # Act
    visible = _is_light_visible(light, bounds, camera_pos, falloff_distance=8000.0, min_brightness_threshold=0.01)

    # Assert
    assert visible is True


# ---------------------------------------------------------------------------
# _compute_visible_bounds
# ---------------------------------------------------------------------------


def test_compute_visible_bounds_returns_none_for_empty_ground_mask():
    # Arrange — an all-sky frame, no ray hits the ground
    gbuffer = _make_gbuffer(
        hit_x=np.zeros((4, 4)), hit_y=np.zeros((4, 4)), ground_mask=np.zeros((4, 4), dtype=bool),
    )
    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]))

    # Act
    bounds = _compute_visible_bounds(gbuffer, camera, padding=200.0)

    # Assert
    assert bounds is None


def test_compute_visible_bounds_adds_padding_around_hits():
    # Arrange — hits confined to a small square
    gbuffer = _make_grid_gbuffer(center=0.0, half_extent=100.0, size=4)
    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]), far=5000.0)

    # Act
    min_x, min_y, max_x, max_y = _compute_visible_bounds(gbuffer, camera, padding=200.0)

    # Assert
    assert min_x == pytest.approx(-300.0)
    assert max_x == pytest.approx(300.0)
    assert min_y == pytest.approx(-300.0)
    assert max_y == pytest.approx(300.0)


def test_compute_visible_bounds_clamps_hits_to_camera_far():
    # Arrange — one hit reports a distance far beyond the camera's far plane
    hit_x = np.array([[50000.0]])
    hit_y = np.array([[0.0]])
    ground_mask = np.array([[True]])
    gbuffer = _make_gbuffer(hit_x, hit_y, ground_mask)
    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]), far=1000.0)

    # Act
    min_x, min_y, max_x, max_y = _compute_visible_bounds(gbuffer, camera, padding=0.0)

    # Assert — hit_x is clamped to camera.position[0] + far = 1000.0
    assert max_x == pytest.approx(1000.0)


def test_compute_visible_bounds_crops_to_lights_aabb():
    # Arrange — visible ground spans +/-1 km, but lights only exist within +/-100 m
    gbuffer = _make_grid_gbuffer(center=0.0, half_extent=1000.0, size=4)
    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]), far=50000.0)
    lights_aabb = (-100.0, -100.0, 100.0, 100.0)

    # Act
    min_x, min_y, max_x, max_y = _compute_visible_bounds(
        gbuffer, camera, padding=200.0, lights_aabb=lights_aabb
    )

    # Assert — bounds crop to the lights AABB expanded by padding, not the
    # much larger visible ground extent
    assert min_x == pytest.approx(-300.0)
    assert max_x == pytest.approx(300.0)
    assert min_y == pytest.approx(-300.0)
    assert max_y == pytest.approx(300.0)


def test_compute_visible_bounds_returns_none_when_no_light_overlap():
    # Arrange — visible ground near the origin, all lights far to the east
    gbuffer = _make_grid_gbuffer(center=0.0, half_extent=500.0, size=4)
    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]), far=50000.0)
    lights_aabb = (40000.0, 40000.0, 41000.0, 41000.0)

    # Act
    bounds = _compute_visible_bounds(
        gbuffer, camera, padding=200.0, lights_aabb=lights_aabb
    )

    # Assert — no overlap between visible ground and lights, so no map
    assert bounds is None


# ---------------------------------------------------------------------------
# _auto_scale_illumination_resolution
# ---------------------------------------------------------------------------


def test_auto_scale_illumination_resolution_unchanged_when_under_budget():
    # Arrange — a small map well within the pixel budget
    resolution = _auto_scale_illumination_resolution(0.0, 0.0, 1000.0, 1000.0, resolution=2.0)

    # Assert
    assert resolution == 2.0


def test_auto_scale_illumination_resolution_coarsens_when_over_budget(capsys):
    # Arrange — a huge map that would blow the pixel budget at 2.0 m/px
    resolution = _auto_scale_illumination_resolution(0.0, 0.0, 200000.0, 200000.0, resolution=2.0)

    # Assert — resolution was coarsened and the coarsening was reported
    assert resolution > 2.0
    captured = capsys.readouterr()
    assert "coarsening" in captured.out.lower()


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


# ---------------------------------------------------------------------------
# render_illumination — empty ground mask (all-sky frame)
# ---------------------------------------------------------------------------


def test_render_illumination_returns_empty_when_gbuffer_has_no_ground_hits():
    # Arrange — a light exists, but the frame never hits the ground
    scene = mock.MagicMock()
    scene.lights = [_make_light(position=np.array([0.0, 0.0, 5.0]))]
    scene.config.atmosphere = _make_atmosphere()

    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]))
    backend = mock.MagicMock()
    backend.get_surface_size.return_value = (4, 4)

    gbuffer = _make_gbuffer(
        hit_x=np.zeros((4, 4)), hit_y=np.zeros((4, 4)), ground_mask=np.zeros((4, 4), dtype=bool),
    )

    # Act
    result = render_illumination(scene, camera, backend, gbuffer=gbuffer)

    # Assert — no map is built or composited
    assert result == {}
    backend.add_linear.assert_not_called()


# ---------------------------------------------------------------------------
# render_illumination — bounds follow the visible G-buffer, not road extent
# ---------------------------------------------------------------------------


def test_render_illumination_bounds_follow_gbuffer_not_roads():
    # Arrange — road network spans +/-20 km, but the visible ground is +/-1 km
    scene = mock.MagicMock()
    scene.lights = [_make_light(position=np.array([0.0, 0.0, 5.0]), illumination_radius=50.0)]
    scene.config.atmosphere = _make_atmosphere()
    scene.road_segments = [
        mock.MagicMock(geometry=np.array([[-20000.0, -20000.0, 0.0], [20000.0, 20000.0, 0.0]]))
    ]
    # Lights span the whole road network, so the visible ground (not the
    # lights AABB) is what must crop the map here.
    scene.get_lights_aabb.return_value = (-20000.0, -20000.0, 20000.0, 20000.0)

    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]), far=50000.0)
    backend = mock.MagicMock()
    backend.get_surface_size.return_value = (4, 4)

    gbuffer = _make_grid_gbuffer(center=0.0, half_extent=1000.0, size=4)

    # Act
    result = render_illumination(scene, camera, backend, gbuffer=gbuffer)

    # Assert — map bounds track the gbuffer extent (~1 km), not the roads (~20 km)
    min_x, max_y, resolution = result["map_info"]
    assert min_x == pytest.approx(-1200.0)
    assert max_y == pytest.approx(1200.0)


# ---------------------------------------------------------------------------
# render_illumination — off-screen lights are culled before painting
# ---------------------------------------------------------------------------


def test_offscreen_light_is_culled():
    # Arrange — one light near the visible ground, one far off-screen
    near_light = _make_light(position=np.array([0.0, 0.0, 5.0]), illumination_radius=50.0)
    offscreen_light = _make_light(
        position=np.array([30000.0, 0.0, 5.0]), illumination_radius=50.0
    )
    scene = mock.MagicMock()
    scene.lights = [near_light, offscreen_light]
    scene.config.atmosphere = _make_atmosphere()
    scene.get_lights_aabb.return_value = (0.0, 0.0, 30000.0, 0.0)

    camera = _make_camera(position=np.array([0.0, 0.0, 500.0]), far=50000.0)
    backend = mock.MagicMock()
    backend.get_surface_size.return_value = (4, 4)

    gbuffer = _make_grid_gbuffer(center=0.0, half_extent=500.0, size=4)

    # Act
    with mock.patch(
        "streets_at_night.rendering.illumination_pass._paint_illumination_blob"
    ) as mock_paint:
        render_illumination(scene, camera, backend, gbuffer=gbuffer)

    # Assert — only the near light was painted
    assert mock_paint.call_count == 1
    painted_light = mock_paint.call_args.args[1]
    assert painted_light is near_light
