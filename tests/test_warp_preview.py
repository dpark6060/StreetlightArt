# Radical
"""Unit tests for warp_preview and flat_map modules."""

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from streets_at_night.camera.camera import create_camera
from streets_at_night.rendering.flat_map import (
    FlatMapMeta,
    generate_flat_map,
    get_flat_map,
    _compute_cache_key,
    _auto_scale_resolution,
)
from streets_at_night.rendering.warp_preview import (
    _build_ray_directions,
    _raycast_to_flat_map,
    _sample_flat_map,
    render_warp_preview,
)
from streets_at_night.models.road_segment import RoadSegment
from streets_at_night.models.road_types import RoadType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_camera() -> "create_camera.__class__":
    """Return a camera positioned above, looking down toward origin."""
    return create_camera(
        position=np.array([0.0, -2000.0, 1000.0]),
        target=np.array([0.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


@pytest.fixture
def flat_meta() -> FlatMapMeta:
    """Return a simple FlatMapMeta covering a 10km x 10km area at 10m/px."""
    return FlatMapMeta(
        min_x=-5000.0,
        min_y=-5000.0,
        max_x=5000.0,
        max_y=5000.0,
        resolution=10.0,
    )


@pytest.fixture
def minimal_scene():
    """Return a mock Scene with two road segments."""
    seg1 = RoadSegment(
        geometry=np.array([
            [-1000.0, -1000.0, 0.0],
            [1000.0,  1000.0, 0.0],
        ]),
        road_type=RoadType.RESIDENTIAL,
        width=4.0,
    )
    seg2 = RoadSegment(
        geometry=np.array([
            [0.0, -2000.0, 0.0],
            [0.0,  2000.0, 0.0],
        ]),
        road_type=RoadType.MAJOR_ROAD,
        width=8.0,
    )
    scene = mock.MagicMock()
    scene.road_segments = [seg1, seg2]
    return scene


# ---------------------------------------------------------------------------
# FlatMapMeta — world_to_pixel and pixel_to_world
# ---------------------------------------------------------------------------


def test_flatmapmeta_worldtopixel_origin_maps_to_correct_pixel(flat_meta):
    # Arrange — world origin at center of the 10km x 10km area
    # min_x=-5000, max_y=5000, resolution=10 → px=(0-(-5000))/10=500, py=(5000-0)/10=500

    # Act
    px, py = flat_meta.world_to_pixel(0.0, 0.0)

    # Assert
    assert px == pytest.approx(500.0)
    assert py == pytest.approx(500.0)


def test_flatmapmeta_worldtopixel_corner_maps_to_image_origin(flat_meta):
    # Arrange — (min_x, max_y) should map to pixel (0, 0)

    # Act
    px, py = flat_meta.world_to_pixel(-5000.0, 5000.0)

    # Assert
    assert px == pytest.approx(0.0)
    assert py == pytest.approx(0.0)


def test_flatmapmeta_pixeltoworld_roundtrip(flat_meta):
    # Arrange
    original_x, original_y = 1234.5, -678.9

    # Act
    px, py = flat_meta.world_to_pixel(original_x, original_y)
    recovered_x, recovered_y = flat_meta.pixel_to_world(px, py)

    # Assert
    assert recovered_x == pytest.approx(original_x, abs=1e-6)
    assert recovered_y == pytest.approx(original_y, abs=1e-6)


# ---------------------------------------------------------------------------
# _auto_scale_resolution
# ---------------------------------------------------------------------------


def test_auto_scale_resolution_returns_initial_when_small_enough():
    # Arrange — 5km x 5km at 10m/px = 500x500px, well within 8000 limit

    # Act
    result = _auto_scale_resolution(0.0, 0.0, 5000.0, 5000.0, 10.0)

    # Assert
    assert result == pytest.approx(10.0)


def test_auto_scale_resolution_increases_when_too_large():
    # Arrange — 200km x 200km at 10m/px = 20000x20000px, exceeds limit

    # Act
    result = _auto_scale_resolution(0.0, 0.0, 200_000.0, 200_000.0, 10.0)

    # Assert — resolution must be large enough that 200000/result <= 8000
    assert 200_000.0 / result <= 8000.0


# ---------------------------------------------------------------------------
# generate_flat_map
# ---------------------------------------------------------------------------


def test_generate_flat_map_returns_rgb_image(minimal_scene):
    # Act
    img = generate_flat_map(minimal_scene, resolution=50.0)

    # Assert
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_generate_flat_map_has_nonzero_dimensions(minimal_scene):
    # Act
    img = generate_flat_map(minimal_scene, resolution=50.0)

    # Assert
    assert img.size[0] > 0
    assert img.size[1] > 0


# ---------------------------------------------------------------------------
# get_flat_map — cache behavior
# ---------------------------------------------------------------------------


def test_get_flat_map_saves_cache_files(minimal_scene):
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Act
        img, meta = get_flat_map(minimal_scene, cache_dir, resolution=50.0)

        # Assert — cache files were written under the flat_map/ subdir
        flat_map_dir = cache_dir / "flat_map"
        png_files = list(flat_map_dir.glob("flat_map_*.png"))
        json_files = list(flat_map_dir.glob("flat_map_*.json"))
        assert len(png_files) == 1
        assert len(json_files) == 1


def test_get_flat_map_loads_from_cache_on_second_call(minimal_scene):
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        img1, meta1 = get_flat_map(minimal_scene, cache_dir, resolution=50.0)

        # Act — second call should load from cache
        with mock.patch(
            "streets_at_night.rendering.flat_map.generate_flat_map"
        ) as mock_gen:
            img2, meta2 = get_flat_map(minimal_scene, cache_dir, resolution=50.0)

        # Assert — generate was NOT called on the second hit
        mock_gen.assert_not_called()
        assert img2.size == img1.size


def test_get_flat_map_returns_meta_with_correct_resolution(minimal_scene):
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Act
        _, meta = get_flat_map(minimal_scene, cache_dir, resolution=50.0)

        # Assert
        assert meta.resolution == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _compute_cache_key
# ---------------------------------------------------------------------------


def test_compute_cache_key_returns_12_char_hex(flat_meta, minimal_scene):
    # Act
    key = _compute_cache_key(flat_meta, minimal_scene.road_segments)

    # Assert
    assert len(key) == 12
    assert all(c in "0123456789abcdef" for c in key)


def test_compute_cache_key_differs_for_different_segment_counts(flat_meta, minimal_scene):
    # Arrange
    fewer_segments = minimal_scene.road_segments[:1]

    # Act
    key1 = _compute_cache_key(flat_meta, minimal_scene.road_segments)
    key2 = _compute_cache_key(flat_meta, fewer_segments)

    # Assert
    assert key1 != key2


def test_compute_cache_key_differs_for_different_road_type_on_one_segment(
    flat_meta, minimal_scene
):
    # Arrange — two scenes identical except one segment's road_type
    changed_segments = [
        RoadSegment(
            geometry=minimal_scene.road_segments[0].geometry,
            road_type=RoadType.HIGHWAY,
            width=minimal_scene.road_segments[0].width,
        ),
        minimal_scene.road_segments[1],
    ]

    # Act
    key1 = _compute_cache_key(flat_meta, minimal_scene.road_segments)
    key2 = _compute_cache_key(flat_meta, changed_segments)

    # Assert
    assert key1 != key2


def test_compute_cache_key_same_scene_twice_returns_same_key(flat_meta, minimal_scene):
    # Act
    key1 = _compute_cache_key(flat_meta, minimal_scene.road_segments)
    key2 = _compute_cache_key(flat_meta, minimal_scene.road_segments)

    # Assert
    assert key1 == key2


# ---------------------------------------------------------------------------
# _build_ray_directions
# ---------------------------------------------------------------------------


def test_build_ray_directions_returns_correct_shape(simple_camera):
    # Arrange
    width, height = 320, 180
    inv_proj = np.linalg.inv(simple_camera.get_projection_matrix())
    inv_view = np.linalg.inv(simple_camera.get_view_matrix())

    # Act
    rays = _build_ray_directions(width, height, inv_proj, inv_view)

    # Assert
    assert rays.shape == (height, width, 3)


def test_build_ray_directions_are_normalized(simple_camera):
    # Arrange
    width, height = 64, 36
    inv_proj = np.linalg.inv(simple_camera.get_projection_matrix())
    inv_view = np.linalg.inv(simple_camera.get_view_matrix())

    # Act
    rays = _build_ray_directions(width, height, inv_proj, inv_view)

    # Assert — all ray directions should be unit length
    norms = np.linalg.norm(rays, axis=2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_build_ray_directions_center_points_toward_target(simple_camera):
    # Arrange — camera at (0, -2000, 1000) looking at (0, 0, 0)
    width, height = 64, 36
    inv_proj = np.linalg.inv(simple_camera.get_projection_matrix())
    inv_view = np.linalg.inv(simple_camera.get_view_matrix())

    # Act
    rays = _build_ray_directions(width, height, inv_proj, inv_view)
    center_ray = rays[height // 2, width // 2]

    # Assert — center ray should point roughly toward the target
    expected_dir = np.array([0.0, 2000.0, -1000.0])
    expected_dir /= np.linalg.norm(expected_dir)
    dot = np.dot(center_ray, expected_dir)
    assert dot > 0.95


# ---------------------------------------------------------------------------
# _raycast_to_flat_map
# ---------------------------------------------------------------------------


def test_raycast_to_flat_map_downward_camera_all_ground(flat_meta):
    # Arrange — camera nearly straight down, all rays should hit the ground
    camera = create_camera(
        position=np.array([0.0, 0.0, 500.0]),
        target=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )
    width, height = 64, 36
    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())
    rays = _build_ray_directions(width, height, inv_proj, inv_view)
    flat_map = Image.new("RGB", (1000, 1000), (100, 100, 100))

    # Act
    flat_px, flat_py, ground_mask = _raycast_to_flat_map(
        camera.position, rays, width, height, flat_meta, flat_map
    )

    # Assert — all pixels should hit ground (camera looking straight down)
    assert ground_mask.all()


def test_raycast_to_flat_map_sky_pixels_are_false(flat_meta):
    # Arrange — camera looking mostly at the horizon, some rays miss ground
    camera = create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 10000.0, 990.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )
    width, height = 64, 36
    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())
    rays = _build_ray_directions(width, height, inv_proj, inv_view)
    flat_map = Image.new("RGB", (1000, 1000), (100, 100, 100))

    # Act
    flat_px, flat_py, ground_mask = _raycast_to_flat_map(
        camera.position, rays, width, height, flat_meta, flat_map
    )

    # Assert — at least some pixels should be sky (top rows)
    assert not ground_mask.all()
    # Bottom row should still hit ground
    assert ground_mask[-1, :].any()


# ---------------------------------------------------------------------------
# _sample_flat_map
# ---------------------------------------------------------------------------


def test_sample_flat_map_sky_pixels_get_sky_color():
    # Arrange
    width, height = 4, 4
    flat_map = Image.new("RGB", (100, 100), (200, 200, 200))
    flat_px = np.zeros((height, width), dtype=np.int32)
    flat_py = np.zeros((height, width), dtype=np.int32)
    ground_mask = np.zeros((height, width), dtype=bool)

    # Act
    output = _sample_flat_map(flat_map, flat_px, flat_py, ground_mask, width, height)

    # Assert — all pixels should be sky color (5, 5, 15)
    arr = np.asarray(output)
    np.testing.assert_array_equal(arr[0, 0], [5, 5, 15])


def test_sample_flat_map_ground_pixels_get_map_color():
    # Arrange
    width, height = 4, 4
    flat_map = Image.new("RGB", (100, 100), (200, 150, 100))
    flat_px = np.full((height, width), 50, dtype=np.int32)
    flat_py = np.full((height, width), 50, dtype=np.int32)
    ground_mask = np.ones((height, width), dtype=bool)

    # Act
    output = _sample_flat_map(flat_map, flat_px, flat_py, ground_mask, width, height)

    # Assert — all pixels should be the flat map color
    arr = np.asarray(output)
    np.testing.assert_array_equal(arr[0, 0], [200, 150, 100])


# ---------------------------------------------------------------------------
# render_warp_preview — integration
# ---------------------------------------------------------------------------


def test_render_warp_preview_calls_blit(simple_camera, minimal_scene, flat_meta):
    # Arrange
    flat_map = Image.new("RGB", (1000, 1000), (100, 100, 100))
    mock_backend = mock.MagicMock()
    mock_backend.get_surface_size.return_value = (320, 180)

    # Act
    render_warp_preview(minimal_scene, simple_camera, mock_backend, flat_map, flat_meta)

    # Assert
    mock_backend.blit_pil_image.assert_called_once()
    blitted_img = mock_backend.blit_pil_image.call_args[0][0]
    assert isinstance(blitted_img, Image.Image)
    assert blitted_img.size == (320, 180)
