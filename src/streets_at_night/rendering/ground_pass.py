"""Ground render pass — fill sky and ground regions, render zones."""

import typing as t
from pathlib import Path

import numpy as np
from PIL import Image

from ..camera.camera import Camera
from ..camera.projection import clip_polygon_to_near_plane, project_points
from ..config.schema import SceneConfig
from ..models.zones import CommercialZone, DarkZone, ResidentialZone
from ..utils.colors import hex_to_rgb
from .backend import RenderBackend
from .raycast import GroundHit, build_ground_gbuffer
from .render_utils import compute_horizon_y

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


# Sky and ground background colors
_SKY_COLOR: t.Tuple[int, int, int] = (5, 5, 15)
_GROUND_COLOR: t.Tuple[int, int, int] = (8, 7, 6)

# Minimum number of projected vertices required to draw a zone polygon
_MIN_POLYGON_VERTICES: int = 3


def render_ground(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Fill sky and ground regions, then render zone polygons.

    Fills the whole surface with sky color, fills below the horizon with
    ground color, then renders zone polygons on top in layer order: dark
    zones, commercial zones, residential zones. Horizon glow is handled by
    the atmosphere pass which runs later in the pipeline.

    Args:
        scene (Scene): The world scene containing zones and config.
        camera (Camera): Configured 3D camera.
        backend (RenderBackend): Target render surface.
    """
    width, height = backend.get_surface_size()
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()

    backend.clear(_SKY_COLOR)

    horizon_y = compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
    _fill_ground(backend, width, height, horizon_y)
    _render_zones(scene, camera, view_matrix, proj_matrix, width, height, backend, camera.near)


def render_ground_export(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    cache_dir: Path = Path("cache"),
    gbuffer: t.Optional[GroundHit] = None,
) -> None:
    """Render ground using tiled textures for high-res export.

    Draws the sky/ground/horizon base first (via render_ground), then
    composites per-tile ground colors over the ground region using the shared
    per-pixel ground raycast. This provides per-house color variation that the
    flat zone-polygon pass cannot produce.

    The tile cache directory is derived from a hash of the scene's color
    config so that stale tiles are bypassed automatically when colors change.

    Args:
        scene (Scene): The world scene with zone and tile config.
        camera (Camera): Configured 3D camera.
        backend (RenderBackend): Target render surface (must support paste_image).
        cache_dir (Path): Root cache directory; tile subdirectory is chosen automatically.
        gbuffer (GroundHit | None): Precomputed ground raycast for this camera and
            size. When None, it is computed here. The pipeline shares one G-buffer
            between this pass and the illumination pass to avoid raycasting twice.
    """
    from ..scene.ground_tiles import create_tile_grid, get_tile_cache_dir

    render_ground(scene, camera, backend)

    width, height = backend.get_surface_size()
    if gbuffer is None:
        gbuffer = build_ground_gbuffer(camera, width, height)

    grid = create_tile_grid(scene)
    tile_cache_dir = get_tile_cache_dir(cache_dir, scene.config)

    current_image = backend.get_image()
    composited = _sample_tiles(
        current_image, gbuffer.hit_x, gbuffer.hit_y, gbuffer.ground_mask,
        grid, scene, tile_cache_dir, width, height,
    )
    backend.paste_image(composited)


def _sample_tiles(
    base_image: Image.Image,
    world_x: np.ndarray,
    world_y: np.ndarray,
    ground_mask: np.ndarray,
    grid: t.Any,
    scene: "Scene",
    tile_cache_dir: Path,
    width: int,
    height: int,
) -> Image.Image:
    """Sample tile colors for all ground pixels and composite onto the base image.

    For each pixel where ground_mask is True, maps the world position to a
    tile, samples the tile pixel color, and writes it into the output image.

    Args:
        base_image (Image.Image): The already-rendered sky/horizon/zone image.
        world_x (np.ndarray): (H, W) ENU x positions of ground hits.
        world_y (np.ndarray): (H, W) ENU y positions of ground hits.
        ground_mask (np.ndarray): (H, W) bool mask of valid ground pixels.
        grid: GroundTileGrid describing the tile layout.
        scene (Scene): Scene with zone data and config.
        tile_cache_dir (Path): Config-hashed tile cache directory.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        Image.Image: RGB image with tile colors composited over ground regions.
    """
    from ..scene.ground_tiles import get_tile

    output = np.asarray(base_image).copy()

    ground_ys, ground_xs = np.where(ground_mask)
    if ground_ys.size == 0:
        return Image.fromarray(output, "RGB")

    tile_cache: t.Dict[t.Tuple[int, int], np.ndarray] = {}

    for screen_y, screen_x in zip(ground_ys, ground_xs):
        wx = float(world_x[screen_y, screen_x])
        wy = float(world_y[screen_y, screen_x])

        tile_x, tile_y = grid.world_to_tile_coords(wx, wy)

        if tile_x < 0 or tile_x >= grid.num_tiles_x:
            continue
        if tile_y < 0 or tile_y >= grid.num_tiles_y:
            continue

        tile_key = (tile_x, tile_y)
        if tile_key not in tile_cache:
            tile_img = get_tile(tile_x, tile_y, grid, scene, tile_cache_dir)
            tile_cache[tile_key] = np.asarray(tile_img)

        tile_arr = tile_cache[tile_key]
        tile_px, tile_py = _world_to_tile_pixel(wx, wy, tile_x, tile_y, grid)
        output[screen_y, screen_x] = tile_arr[tile_py, tile_px]

    return Image.fromarray(output, "RGB")


def _world_to_tile_pixel(
    world_x: float,
    world_y: float,
    tile_x: int,
    tile_y: int,
    grid: t.Any,
) -> t.Tuple[int, int]:
    """Convert a world position to pixel coordinates within a specific tile.

    Args:
        world_x (float): ENU x coordinate in meters.
        world_y (float): ENU y coordinate in meters.
        tile_x (int): Tile column index.
        tile_y (int): Tile row index.
        grid: GroundTileGrid with origin, tile_size, and tile_res.

    Returns:
        tuple[int, int]: (px, py) pixel coordinates clamped to tile bounds.
    """
    tile_min_x = grid.origin_x + tile_x * grid.tile_size
    tile_max_y = grid.origin_y + (tile_y + 1) * grid.tile_size
    scale = grid.tile_res / grid.tile_size

    px = int((world_x - tile_min_x) * scale)
    py = int((tile_max_y - world_y) * scale)

    px = max(0, min(grid.tile_res - 1, px))
    py = max(0, min(grid.tile_res - 1, py))

    return px, py


def _camera_ground_rect(camera: Camera) -> t.Tuple[float, float, float, float]:
    """Compute a conservative bounding rectangle for the camera's ground footprint.

    Uses the camera position and far clip distance to produce a square bounding
    box that is guaranteed to contain every visible ground point. This is an
    over-approximation: it is a square of side 2*far centered on the camera XY
    position, regardless of look direction.

    Args:
        camera (Camera): The active camera.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in ENU meters.
    """
    cam_x = camera.position[0]
    cam_y = camera.position[1]
    half = camera.far
    return cam_x - half, cam_y - half, cam_x + half, cam_y + half


def _render_zones(
    scene: "Scene",
    camera: Camera,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
    near: float,
) -> None:
    """Render zone polygons on top of the ground fill in back-to-front order.

    Queries the spatial index for zones within the camera's visible footprint,
    then renders dark zones first (parks, water), then commercial zones, then
    residential zones. Each group is drawn on top of the previous, matching
    the intended layer order. Near-plane clipping is applied per polygon so
    that large zones straddling the near plane are not silently discarded.

    Args:
        scene (Scene): Scene with zone lists, spatial index, and config.
        camera (Camera): The active camera, used to compute the query rect.
        view_matrix (np.ndarray): (4, 4) view matrix from the active camera.
        proj_matrix (np.ndarray): (4, 4) projection matrix from the active camera.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
        near (float): Camera near clip plane distance in meters.
    """
    min_x, min_y, max_x, max_y = _camera_ground_rect(camera)
    residential, commercial, dark = scene.get_zones_in_rect(min_x, min_y, max_x, max_y)

    _render_dark_zones(dark, view_matrix, proj_matrix, width, height, backend, near)
    _render_commercial_zones(
        commercial, scene.config, view_matrix, proj_matrix, width, height, backend, near
    )
    _render_residential_zones(
        residential, scene.config, view_matrix, proj_matrix, width, height, backend, near
    )


def _render_dark_zones(
    dark_zones: t.List[DarkZone],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
    near: float,
) -> None:
    """Project and draw dark zone (park, forest, water) polygons.

    Uses each zone's own base_color for the fill.

    Args:
        dark_zones (list[DarkZone]): Pre-filtered dark zones to render.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
        near (float): Camera near clip plane distance in meters.
    """
    for zone in dark_zones:
        vertices = _project_polygon(zone.boundary, view_matrix, proj_matrix, width, height, near)
        if len(vertices) >= _MIN_POLYGON_VERTICES:
            backend.draw_polygon(vertices, zone.base_color)


def _render_commercial_zones(
    commercial_zones: t.List[CommercialZone],
    scene_config: SceneConfig,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
    near: float,
) -> None:
    """Project and draw commercial zone polygons.

    Uses the commercial color from scene config.

    Args:
        commercial_zones (list[CommercialZone]): Pre-filtered commercial zones to render.
        scene_config (SceneConfig): Scene configuration for zone colors.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
        near (float): Camera near clip plane distance in meters.
    """
    color = hex_to_rgb(scene_config.zones.colors.commercial)
    for zone in commercial_zones:
        vertices = _project_polygon(zone.boundary, view_matrix, proj_matrix, width, height, near)
        if len(vertices) >= _MIN_POLYGON_VERTICES:
            backend.draw_polygon(vertices, color)


def _render_residential_zones(
    residential_zones: t.List[ResidentialZone],
    scene_config: SceneConfig,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
    near: float,
) -> None:
    """Project and draw residential zone polygons using a flat yard color.

    In preview mode, each zone is filled with a single flat color derived from
    the configured yard base color. No per-house variation is applied here
    (too slow for real-time preview).

    Args:
        residential_zones (list[ResidentialZone]): Pre-filtered residential zones to render.
        scene_config (SceneConfig): Scene configuration for zone colors.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
        near (float): Camera near clip plane distance in meters.
    """
    color = hex_to_rgb(scene_config.zones.house.yard_base_color)
    for zone in residential_zones:
        vertices = _project_polygon(zone.boundary, view_matrix, proj_matrix, width, height, near)
        if len(vertices) >= _MIN_POLYGON_VERTICES:
            backend.draw_polygon(vertices, color)


def _project_polygon(
    boundary: t.Any,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    near: float,
) -> t.List[t.Tuple[float, float]]:
    """Project a Shapely Polygon's exterior coordinates to screen space.

    Clips the polygon to the camera near plane using Sutherland-Hodgman
    before projecting, so large zone polygons that straddle the near plane
    are not silently discarded. After projection, only vertices within the
    NDC bounds are kept.

    Shapely's exterior.coords includes a closing duplicate (last == first),
    which is stripped before clipping.

    Args:
        boundary: Shapely Polygon with ENU meter coordinates.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        near (float): Camera near clip plane distance in meters.

    Returns:
        list[tuple[float, float]]: Screen-space (x, y) tuples for valid points.
    """
    exterior_coords = np.array(boundary.exterior.coords)
    world_vertices = _coords_to_vertex_list(exterior_coords)

    clipped = clip_polygon_to_near_plane(world_vertices, view_matrix, near)
    if not clipped:
        return []

    points_3d = np.stack(clipped, axis=0)
    screen_coords, valid_mask = project_points(
        points_3d, view_matrix, proj_matrix, (width, height)
    )
    return [
        (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        for i in range(len(valid_mask))
        if valid_mask[i]
    ]


def _coords_to_vertex_list(coords: np.ndarray) -> t.List[np.ndarray]:
    """Convert exterior ring coordinates to a list of (3,) world-space arrays.

    Strips the closing duplicate that Shapely includes in exterior.coords
    (last vertex == first vertex). Each returned array has z=0 (ground plane).

    Args:
        coords (np.ndarray): Shape (N, 2) or (N, 3) coordinate array from
            a Shapely exterior ring.

    Returns:
        list[np.ndarray]: List of (3,) float arrays with z=0, no closing duplicate.
    """
    ring = coords[:-1] if len(coords) > 1 else coords
    vertices = []
    for row in ring:
        v = np.zeros(3, dtype=float)
        v[0] = row[0]
        v[1] = row[1]
        vertices.append(v)
    return vertices


def _fill_ground(
    backend: RenderBackend,
    width: int,
    height: int,
    horizon_y: int,
) -> None:
    """Fill everything below the horizon with the ground color.

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Image width.
        height (int): Image height.
        horizon_y (int): Screen y of the horizon.
    """
    if horizon_y >= height:
        return
    backend.draw_polygon(
        [(0, horizon_y), (width - 1, horizon_y), (width - 1, height - 1), (0, height - 1)],
        _GROUND_COLOR,
    )


