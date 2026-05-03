# Radical
"""Unit tests for the ground_pass render module."""

from unittest import mock

import numpy as np
import pytest
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
    StarsConfig,
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
def scene_config(zone_config: ZoneConfig, light_config: LightConfig, atmosphere_config: AtmosphereConfig, stars_config: StarsConfig) -> SceneConfig:
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
        atmosphere=atmosphere_config,
        stars=stars_config,
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
    scene.get_zones_in_rect.return_value = ([], [], [])
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
    scene.get_zones_in_rect.return_value = ([], [], [dark_zone])

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
    scene.get_zones_in_rect.return_value = ([], [commercial_zone], [])

    # Act
    render_ground(scene, camera, mock_backend)

    # Assert — ground fill + commercial zone polygon = at least 2 calls
    assert mock_backend.draw_polygon.call_count >= 2


# ---------------------------------------------------------------------------
# Spatial index integration — zone query via get_zones_in_rect
# ---------------------------------------------------------------------------


def test_render_zones_queries_spatial_index_with_camera_bounds(
    scene_config, camera, mock_backend
):
    # Arrange — verify that _render_zones calls get_zones_in_rect (spatial index path)
    # rather than reading scene.*_zones directly
    scene = mock.MagicMock()
    scene.config = scene_config
    scene.get_zones_in_rect.return_value = ([], [], [])

    # Act
    render_ground(scene, camera, mock_backend)

    # Assert — spatial index was queried once
    scene.get_zones_in_rect.assert_called_once()
    call_args = scene.get_zones_in_rect.call_args[0]
    assert len(call_args) == 4, "get_zones_in_rect must receive (min_x, min_y, max_x, max_y)"


def test_render_zones_uses_filtered_zone_lists(scene_config, camera, mock_backend):
    # Arrange — scene.dark_zones has two zones but get_zones_in_rect returns only one.
    # Verify that draw_polygon is called exactly once for the dark zone (one visible
    # polygon), proving the code consumes the filtered list, not scene.dark_zones.
    included_polygon = _make_ground_polygon()
    included_zone = DarkZone(
        boundary=included_polygon,
        zone_type="park",
        base_color=(4, 8, 4),
    )
    extra_zone = DarkZone(
        boundary=_make_ground_polygon(),
        zone_type="forest",
        base_color=(2, 6, 2),
    )
    scene = mock.MagicMock()
    scene.config = scene_config
    # scene.dark_zones has two zones, but the spatial index returns only one
    scene.dark_zones = [included_zone, extra_zone]
    scene.get_zones_in_rect.return_value = ([], [], [included_zone])

    # Act
    render_ground(scene, camera, mock_backend)

    # Assert — only one dark-zone polygon drawn (not two), confirming filtered list is used
    # The ground fill call uses _GROUND_COLOR; zone calls use zone.base_color.
    # Count calls where the color matches the included zone's base color.
    zone_draw_calls = [
        c for c in mock_backend.draw_polygon.call_args_list
        if c.args[1] == included_zone.base_color
    ]
    assert len(zone_draw_calls) == 1


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


# ---------------------------------------------------------------------------
# Near-plane clipping — zone polygon straddling near plane
# ---------------------------------------------------------------------------


def test_render_ground_with_zone_straddling_near_plane_draws_polygon(
    scene_config, mock_backend
):
    # Arrange — camera at (0, -1000, 500) looking at origin, near=10.
    # The zone polygon straddles the near plane: 2 vertices are in front and
    # 2 vertices are behind the near plane (the camera z-boundary at z=0 is
    # around y=-1250 for this camera, so y=-1500 vertices are behind it).
    # Without near-plane clipping all 4 raw vertices fail valid_mask and the
    # polygon is silently dropped. With clipping, the in-front portion is
    # passed through clip_polygon_to_near_plane before projection.
    # We verify that clip_polygon_to_near_plane is called for the zone.
    straddling_camera = create_camera(
        position=np.array([0.0, -1000.0, 500.0]),
        target=np.array([0.0, 0.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )
    straddling_polygon = Polygon([
        (-200.0, -500.0),
        (200.0, -500.0),
        (200.0, -1500.0),
        (-200.0, -1500.0),
    ])
    dark_zone = DarkZone(
        boundary=straddling_polygon,
        zone_type="park",
        base_color=(4, 8, 4),
    )
    scene = mock.MagicMock()
    scene.config = scene_config
    scene.get_zones_in_rect.return_value = ([], [], [dark_zone])

    with mock.patch(
        "streets_at_night.rendering.ground_pass.clip_polygon_to_near_plane",
        wraps=__import__(
            "streets_at_night.camera.projection",
            fromlist=["clip_polygon_to_near_plane"],
        ).clip_polygon_to_near_plane,
    ) as mock_clip:
        # Act
        render_ground(scene, straddling_camera, mock_backend)

    # Assert — near-plane clipping was invoked for the zone polygon
    mock_clip.assert_called_once()
