# Radical
"""Unit tests for the ground_pass render module."""

from unittest import mock

import numpy as np
import pytest
from shapely.geometry import Polygon

from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import (
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
from streets_at_night.models.zones import CommercialZone, DarkZone, ResidentialZone
from streets_at_night.rendering.backend import RenderBackend
from streets_at_night.rendering.ground_pass import render_ground, render_ground_export


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
def scene_config(zone_config: ZoneConfig, light_config: LightConfig) -> SceneConfig:
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
    )


@pytest.fixture
def camera():
    """Return a camera positioned directly above, looking nearly straight down.

    A square aspect ratio and wide FOV ensures ground plane polygons near the
    origin project their vertices within the screen NDC bounds.
    """
    return create_camera(
        position=np.array([0.0, 0.0, 500.0]),
        target=np.array([0.0, 50.0, 0.0]),
        fov_deg=90.0,
        aspect_ratio=1.0,
        near=1.0,
        far=5000.0,
    )


@pytest.fixture
def mock_backend():
    """Return a mock RenderBackend with spec."""
    backend = mock.MagicMock(spec=RenderBackend)
    backend.get_surface_size.return_value = (640, 360)
    return backend


@pytest.fixture
def empty_zone_scene(scene_config):
    """Return a mock scene with no zones."""
    scene = mock.MagicMock()
    scene.config = scene_config
    scene.dark_zones = []
    scene.commercial_zones = []
    scene.residential_zones = []
    return scene


def _make_ground_polygon() -> Polygon:
    """Return a polygon on the ground plane whose vertices project on screen.

    Sized to fit within the frustum of the test camera at (0, 0, 500) looking
    at (0, 50, 0) with 90-degree FOV and square aspect ratio.
    """
    return Polygon([
        (-200.0, -200.0),
        (200.0, -200.0),
        (200.0, 200.0),
        (-200.0, 200.0),
    ])


# ---------------------------------------------------------------------------
# render_ground — top-level behavior
# ---------------------------------------------------------------------------


def test_render_ground_calls_clear(empty_zone_scene, camera, mock_backend):
    # Arrange — sky color constant is (5, 5, 15)
    sky_color = (5, 5, 15)

    # Act
    render_ground(empty_zone_scene, camera, mock_backend)

    # Assert — clear is called once with the sky color
    mock_backend.clear.assert_called_once_with(sky_color)


def test_render_ground_draws_ground_polygon(empty_zone_scene, camera, mock_backend):
    # Arrange — backend should receive at least one draw_polygon call for the ground fill

    # Act
    render_ground(empty_zone_scene, camera, mock_backend)

    # Assert — draw_polygon is called at least once (ground fill)
    assert mock_backend.draw_polygon.call_count >= 1


def test_render_ground_with_empty_zones_does_not_raise(empty_zone_scene, camera, mock_backend):
    # Arrange — scene has no zones; should complete without error

    # Act & Assert — must not raise
    render_ground(empty_zone_scene, camera, mock_backend)


def test_render_ground_with_dark_zones_draws_additional_polygons(
    scene_config, camera, mock_backend
):
    # Arrange — scene with one dark zone that will project onto the screen
    dark_polygon = _make_ground_polygon()
    dark_zone = DarkZone(
        boundary=dark_polygon,
        zone_type="park",
        base_color=(4, 8, 4),
    )
    scene = mock.MagicMock()
    scene.config = scene_config
    scene.dark_zones = [dark_zone]
    scene.commercial_zones = []
    scene.residential_zones = []

    # Act
    render_ground(scene, camera, mock_backend)

    # Assert — at least two draw_polygon calls: ground fill + the dark zone
    assert mock_backend.draw_polygon.call_count >= 2


def test_render_ground_clears_before_drawing(empty_zone_scene, camera, mock_backend):
    # Arrange — track call order
    call_order = []
    mock_backend.clear.side_effect = lambda *a, **kw: call_order.append("clear")
    mock_backend.draw_polygon.side_effect = lambda *a, **kw: call_order.append("draw_polygon")

    # Act
    render_ground(empty_zone_scene, camera, mock_backend)

    # Assert — clear is the first operation
    assert call_order[0] == "clear"


def test_render_ground_draws_horizon_glow_lines(empty_zone_scene, camera, mock_backend):
    # Arrange — draw_line_rgba is used for the horizon glow band

    # Act
    render_ground(empty_zone_scene, camera, mock_backend)

    # Assert — horizon glow draws at least one semi-transparent line
    assert mock_backend.draw_line_rgba.call_count >= 1


def test_render_ground_with_commercial_zone_draws_additional_polygons(
    scene_config, camera, mock_backend
):
    # Arrange — scene with one commercial zone visible to the camera
    commercial_polygon = _make_ground_polygon()
    commercial_zone = CommercialZone(
        boundary=commercial_polygon,
        building_footprints=[],
        parking_areas=[],
    )
    scene = mock.MagicMock()
    scene.config = scene_config
    scene.dark_zones = []
    scene.commercial_zones = [commercial_zone]
    scene.residential_zones = []

    # Act
    render_ground(scene, camera, mock_backend)

    # Assert — ground fill + commercial zone polygon = at least 2 calls
    assert mock_backend.draw_polygon.call_count >= 2


# ---------------------------------------------------------------------------
# render_ground_export — tile-based export ground pass
# ---------------------------------------------------------------------------


@pytest.fixture
def export_scene(scene_config):
    """Return a mock scene with real road segments for tile grid creation."""
    scene = mock.MagicMock()
    scene.config = scene_config
    scene.dark_zones = []
    scene.commercial_zones = []
    scene.residential_zones = []
    scene.road_segments = [
        RoadSegment(
            geometry=np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0]]),
            road_type=RoadType.HIGHWAY,
            width=12.0,
        ),
    ]
    return scene


def _make_mock_tile_grid() -> mock.MagicMock:
    """Return a mock GroundTileGrid for export pass tests."""
    grid = mock.MagicMock()
    grid.num_tiles_x = 1
    grid.num_tiles_y = 1
    grid.world_to_tile_coords.return_value = (0, 0)
    grid.origin_x = -256.0
    grid.origin_y = -256.0
    grid.tile_size = 256.0
    grid.tile_res = 64
    return grid


def test_render_ground_export_calls_render_ground(
    export_scene, camera, mock_backend
):
    # Arrange — render_ground must run first to set sky/horizon/zone base
    from PIL import Image as PILImage
    mock_backend.get_image.return_value = PILImage.new("RGB", (640, 360), (8, 7, 6))
    mock_backend.get_surface_size.return_value = (640, 360)
    grid = _make_mock_tile_grid()
    mock_tile_image = PILImage.new("RGB", (64, 64), (10, 10, 10))

    with mock.patch(
        "streets_at_night.rendering.ground_pass.render_ground"
    ) as mock_render_ground, mock.patch(
        "streets_at_night.scene.ground_tiles.create_tile_grid", return_value=grid
    ), mock.patch(
        "streets_at_night.scene.ground_tiles.get_tile", return_value=mock_tile_image
    ):
        # Act
        render_ground_export(export_scene, camera, mock_backend)

    # Assert — render_ground was called once with the same arguments
    mock_render_ground.assert_called_once_with(export_scene, camera, mock_backend)


def test_render_ground_export_calls_paste_image(
    export_scene, camera, mock_backend
):
    # Arrange — verify composited result is pasted back to the backend
    from PIL import Image as PILImage
    mock_backend.get_image.return_value = PILImage.new("RGB", (640, 360), (8, 7, 6))
    mock_backend.get_surface_size.return_value = (640, 360)
    grid = _make_mock_tile_grid()
    mock_tile_image = PILImage.new("RGB", (64, 64), (10, 10, 10))

    with mock.patch(
        "streets_at_night.rendering.ground_pass.render_ground"
    ), mock.patch(
        "streets_at_night.scene.ground_tiles.create_tile_grid", return_value=grid
    ), mock.patch(
        "streets_at_night.scene.ground_tiles.get_tile", return_value=mock_tile_image
    ):
        # Act
        render_ground_export(export_scene, camera, mock_backend)

    # Assert — paste_image was called once with a PIL Image
    mock_backend.paste_image.assert_called_once()
    pasted_arg = mock_backend.paste_image.call_args[0][0]
    assert isinstance(pasted_arg, PILImage.Image)


def test_render_ground_export_creates_tile_grid(
    export_scene, camera, mock_backend
):
    # Arrange — verify the tile grid is created from the scene
    from PIL import Image as PILImage
    mock_backend.get_image.return_value = PILImage.new("RGB", (640, 360), (8, 7, 6))
    mock_backend.get_surface_size.return_value = (640, 360)
    grid = _make_mock_tile_grid()
    mock_tile_image = PILImage.new("RGB", (64, 64), (10, 10, 10))

    with mock.patch(
        "streets_at_night.rendering.ground_pass.render_ground"
    ), mock.patch(
        "streets_at_night.scene.ground_tiles.create_tile_grid", return_value=grid
    ) as mock_create_grid, mock.patch(
        "streets_at_night.scene.ground_tiles.get_tile", return_value=mock_tile_image
    ):
        # Act
        render_ground_export(export_scene, camera, mock_backend)

    # Assert — tile grid was built from the scene
    mock_create_grid.assert_called_once_with(export_scene)
