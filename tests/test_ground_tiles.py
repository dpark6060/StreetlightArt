# Radical
"""Unit tests for the ground_tiles module."""

import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon

from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import (
    AtmosphereConfig,
    BoundsConfig,
    CameraConfig,
    ColorPoolConfig,
    HouseConfig,
    LightConfig,
    QualityPreset,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    SceneConfig,
    ZoneColorsConfig,
    ZoneConfig,
)
from streets_at_night.models.road_segment import RoadSegment
from streets_at_night.models.road_types import RoadType
from streets_at_night.models.zones import DarkZone
from streets_at_night.scene.ground_tiles import (
    GroundTileGrid,
    create_tile_grid,
    generate_tile,
    get_tile,
    get_tile_cache_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zone_config() -> ZoneConfig:
    """Return a minimal ZoneConfig for testing."""
    return ZoneConfig(
        house=HouseConfig(
            exterior_colors=ColorPoolConfig(colors=[("#0D0A08", 1.0)]),
            yard_base_color="#060A04",
            yard_variation=2,
            min_lights=0,
            max_lights=5,
            light_weights=[5.0, 3.0, 1.5, 0.5, 0.3, 0.1],
            footprint_coverage=0.65,
            lot_width=15.0,
            lot_depth=25.0,
            setback=10.0,
        ),
        colors=ZoneColorsConfig(
            commercial="#0A0A0C",
            industrial="#080808",
            park_forest="#040804",
            water="#040408",
            road_surface="#0C0C0C",
            default_ground="#080706",
        ),
        tile_size_meters=256.0,
        tile_resolution_px=64,
    )


@pytest.fixture
def scene_config(
    zone_config: ZoneConfig,
    light_config: LightConfig,
    atmosphere_config: AtmosphereConfig,
) -> SceneConfig:
    """Return a minimal SceneConfig for testing."""
    _style = RoadStyleConfig(
        highway=RoadStyleEntry(color="#141210", width=4),
        major_road=RoadStyleEntry(color="#121010", width=3),
        collector=RoadStyleEntry(color="#100E0E", width=2),
        residential=RoadStyleEntry(color="#0E0C0C", width=1),
        service=RoadStyleEntry(color="#0C0A0A", width=1),
        minor=RoadStyleEntry(color="#0A0908", width=1),
    )
    return SceneConfig(
        camera=CameraConfig(
            lat=34.2349,
            lon=-117.3116,
            altitude_ft=5186.0,
            target_lat=34.1,
            target_lon=-117.3,
            target_altitude_ft=0.0,
            fov=60.0,
        ),
        bounds=BoundsConfig(north=34.3, south=34.0, east=-117.1, west=-117.5),
        quality_presets={"low": QualityPreset(width=640, height=360)},
        seed=42,
        zones=zone_config,
        lights=light_config,
        roads=RoadsConfig(export=_style, preview=_style),
        atmosphere=atmosphere_config,
    )


@pytest.fixture
def camera() -> object:
    """Return a basic camera for testing."""
    return create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 5000.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=100.0,
        far=60000.0,
    )


@pytest.fixture
def road_segments() -> list:
    """Return two road segments spanning a known world area."""
    return [
        RoadSegment(
            geometry=np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
            road_type=RoadType.HIGHWAY,
            width=12.0,
        ),
        RoadSegment(
            geometry=np.array([[500.0, 200.0, 0.0], [500.0, 800.0, 0.0]]),
            road_type=RoadType.RESIDENTIAL,
            width=4.0,
        ),
    ]


@pytest.fixture
def minimal_scene(road_segments, camera, scene_config):
    """Return a mock Scene with real road segments and config."""
    scene = mock.MagicMock()
    scene.road_segments = road_segments
    scene.config = scene_config
    scene.dark_zones = []
    scene.commercial_zones = []
    scene.residential_zones = []
    return scene


@pytest.fixture
def tile_grid(scene_config) -> GroundTileGrid:
    """Return a simple GroundTileGrid for a 256m tile size."""
    return GroundTileGrid(
        origin_x=-256.0,
        origin_y=-256.0,
        tile_size=256.0,
        tile_res=64,
        num_tiles_x=8,
        num_tiles_y=8,
    )


# ---------------------------------------------------------------------------
# create_tile_grid
# ---------------------------------------------------------------------------


def test_create_tile_grid_returns_valid_metadata(minimal_scene):
    # Act
    grid = create_tile_grid(minimal_scene)

    # Assert
    assert grid.num_tiles_x > 0
    assert grid.num_tiles_y > 0
    assert grid.tile_size > 0
    assert grid.tile_res > 0


def test_create_tile_grid_covers_road_bounds(minimal_scene):
    # Arrange — road segments span x=[0, 1000], y=[0, 800]
    grid = create_tile_grid(minimal_scene)

    # Act — compute how far the grid extends in each direction
    grid_max_x = grid.origin_x + grid.num_tiles_x * grid.tile_size
    grid_max_y = grid.origin_y + grid.num_tiles_y * grid.tile_size

    # Assert — grid must contain the road bounds plus padding
    assert grid.origin_x < 0.0
    assert grid.origin_y < 0.0
    assert grid_max_x > 1000.0
    assert grid_max_y > 800.0


def test_create_tile_grid_raises_when_no_segments(camera, scene_config):
    # Arrange
    scene = mock.MagicMock()
    scene.road_segments = []
    scene.config = scene_config

    # Act & Assert
    with pytest.raises(ValueError):
        create_tile_grid(scene)


# ---------------------------------------------------------------------------
# generate_tile
# ---------------------------------------------------------------------------


def test_generate_tile_returns_correct_size(minimal_scene, tile_grid):
    # Act
    img = generate_tile(0, 0, tile_grid, minimal_scene)

    # Assert
    assert isinstance(img, Image.Image)
    assert img.size == (tile_grid.tile_res, tile_grid.tile_res)


def test_generate_tile_default_color_when_no_zones(minimal_scene, tile_grid):
    # Arrange — scene has no zones; tile should be filled with default_ground color
    # default_ground is "#080706" → (8, 7, 6)
    expected_color = (8, 7, 6)

    # Act
    img = generate_tile(0, 0, tile_grid, minimal_scene)

    # Assert — sample center pixel; should match default ground color
    center_px = img.getpixel((tile_grid.tile_res // 2, tile_grid.tile_res // 2))
    assert center_px == expected_color


def test_generate_tile_with_dark_zone_calls_draw_dark_zones(minimal_scene, tile_grid):
    # Arrange — add a dark zone that overlaps tile (0, 0)
    tile_min_x, tile_min_y, tile_max_x, tile_max_y = tile_grid.get_tile_world_bounds(0, 0)
    cx = (tile_min_x + tile_max_x) / 2.0
    cy = (tile_min_y + tile_max_y) / 2.0
    half = tile_grid.tile_size * 0.1
    zone_polygon = Polygon([
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ])
    dark_zone = DarkZone(boundary=zone_polygon, zone_type="park", base_color=(4, 8, 4))
    minimal_scene.dark_zones = [dark_zone]

    with mock.patch(
        "streets_at_night.scene.ground_tiles._draw_dark_zones"
    ) as mock_draw_dark:
        # Act
        generate_tile(0, 0, tile_grid, minimal_scene)

        # Assert
        mock_draw_dark.assert_called_once()


# ---------------------------------------------------------------------------
# get_tile — caching behavior
# ---------------------------------------------------------------------------


def test_get_tile_caches_to_disk(minimal_scene, tile_grid, scene_config):
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tile_dir = get_tile_cache_dir(cache_dir, scene_config)

        # Act
        get_tile(0, 0, tile_grid, minimal_scene, tile_dir)

        # Assert — PNG file was written under the config-hashed tile cache directory
        tile_path = tile_dir / "tile_0_0.png"
        assert tile_path.exists()


def test_get_tile_loads_from_cache_on_second_call(minimal_scene, tile_grid, scene_config):
    # Arrange
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        tile_dir = get_tile_cache_dir(cache_dir, scene_config)
        img1 = get_tile(0, 0, tile_grid, minimal_scene, tile_dir)

        # Act — second call should load from disk, not regenerate
        with mock.patch(
            "streets_at_night.scene.ground_tiles.generate_tile"
        ) as mock_generate:
            img2 = get_tile(0, 0, tile_grid, minimal_scene, tile_dir)

            # Assert — generate_tile was not called on cache hit
            mock_generate.assert_not_called()

        assert img2.size == img1.size


# ---------------------------------------------------------------------------
# GroundTileGrid.get_tile_world_bounds
# ---------------------------------------------------------------------------


def test_get_tile_world_bounds_returns_correct_coordinates(tile_grid):
    # Arrange — tile (0, 0) should start at (origin_x, origin_y)
    # and end at (origin_x + tile_size, origin_y + tile_size)

    # Act
    min_x, min_y, max_x, max_y = tile_grid.get_tile_world_bounds(0, 0)

    # Assert
    assert min_x == pytest.approx(tile_grid.origin_x)
    assert min_y == pytest.approx(tile_grid.origin_y)
    assert max_x == pytest.approx(tile_grid.origin_x + tile_grid.tile_size)
    assert max_y == pytest.approx(tile_grid.origin_y + tile_grid.tile_size)


def test_get_tile_world_bounds_nonzero_indices_offset_correctly(tile_grid):
    # Arrange — tile (2, 3) should be offset by 2 and 3 tile widths
    tile_x, tile_y = 2, 3

    # Act
    min_x, min_y, max_x, max_y = tile_grid.get_tile_world_bounds(tile_x, tile_y)

    # Assert
    assert min_x == pytest.approx(tile_grid.origin_x + tile_x * tile_grid.tile_size)
    assert min_y == pytest.approx(tile_grid.origin_y + tile_y * tile_grid.tile_size)
    assert max_x == pytest.approx(min_x + tile_grid.tile_size)
    assert max_y == pytest.approx(min_y + tile_grid.tile_size)
