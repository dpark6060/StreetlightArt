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
from .raycast import GroundHit, build_ground_gbuffer, visible_ground_aabb
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
    tile and an in-tile pixel coordinate (vectorized over all ground pixels),
    then loads each unique visible tile once and writes its pixels into the
    output image with a single fancy-index assignment per tile.

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
    output = np.asarray(base_image).copy()

    ground_ys, ground_xs = np.where(ground_mask)
    if ground_ys.size == 0:
        return Image.fromarray(output, "RGB")

    wx = world_x[ground_ys, ground_xs]
    wy = world_y[ground_ys, ground_xs]
    tile_x, tile_y, tile_px, tile_py, in_grid = _compute_tile_pixel_coords(wx, wy, grid)

    flat_idx = ground_ys[in_grid] * width + ground_xs[in_grid]
    tile_cache: t.Dict[t.Tuple[int, int], np.ndarray] = {}
    _composite_tile_pixels(
        output.reshape(-1, 3),
        flat_idx,
        tile_x[in_grid],
        tile_y[in_grid],
        tile_px[in_grid],
        tile_py[in_grid],
        grid,
        scene,
        tile_cache_dir,
        tile_cache,
    )

    return Image.fromarray(output, "RGB")


def _compute_tile_pixel_coords(
    world_x: np.ndarray,
    world_y: np.ndarray,
    grid: t.Any,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized world-to-tile-pixel mapping for an array of ground pixels.

    Uses np.floor rather than the truncate-toward-zero behavior of int().
    The grid origin is padded one tile below the scene minimum (see
    ``_compute_padded_origin`` in ground_tiles.py), so world coordinates below
    origin occur only outside the tile grid, where ``in_grid`` already
    excludes the pixel. floor and truncation therefore always agree wherever
    ``in_grid`` is True, so this is a safe (and more correct) substitution.

    Args:
        world_x (np.ndarray): (N,) ENU x positions of ground hits.
        world_y (np.ndarray): (N,) ENU y positions of ground hits.
        grid: GroundTileGrid describing the tile layout.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (tile_x, tile_y, tile_px, tile_py, in_grid), each shape (N,). tile_x
        and tile_y are tile grid indices (int64). tile_px and tile_py are
        in-tile pixel coordinates clamped to [0, tile_res - 1] (int64).
        in_grid is a bool mask of pixels whose tile indices fall within the
        grid bounds.
    """
    scale = grid.tile_res / grid.tile_size

    tile_x = np.floor((world_x - grid.origin_x) / grid.tile_size).astype(np.int64)
    tile_y = np.floor((world_y - grid.origin_y) / grid.tile_size).astype(np.int64)

    in_grid = (
        (tile_x >= 0) & (tile_x < grid.num_tiles_x)
        & (tile_y >= 0) & (tile_y < grid.num_tiles_y)
    )

    tile_min_x = grid.origin_x + tile_x * grid.tile_size
    tile_max_y = grid.origin_y + (tile_y + 1) * grid.tile_size
    tile_px = np.clip(((world_x - tile_min_x) * scale).astype(np.int64), 0, grid.tile_res - 1)
    tile_py = np.clip(((tile_max_y - world_y) * scale).astype(np.int64), 0, grid.tile_res - 1)

    return tile_x, tile_y, tile_px, tile_py, in_grid


def _composite_tile_pixels(
    output: np.ndarray,
    flat_idx: np.ndarray,
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    tile_px: np.ndarray,
    tile_py: np.ndarray,
    grid: t.Any,
    scene: "Scene",
    tile_cache_dir: Path,
    tile_cache: t.Dict[t.Tuple[int, int], np.ndarray],
) -> None:
    """Load each unique visible tile once and write its pixels into output.

    Groups the in-grid ground pixels by tile id, loads each unique tile via
    ``get_tile`` (using ``tile_cache`` so a tile is loaded at most once),
    and assigns all pixels belonging to that tile with a single fancy-index
    write.

    Args:
        output (np.ndarray): (H * W, 3) flattened output image array,
            modified in place.
        flat_idx (np.ndarray): (N,) flattened output indices for the in-grid
            ground pixels.
        tile_x (np.ndarray): (N,) tile column index for each in-grid pixel.
        tile_y (np.ndarray): (N,) tile row index for each in-grid pixel.
        tile_px (np.ndarray): (N,) in-tile pixel x coordinate for each
            in-grid pixel.
        tile_py (np.ndarray): (N,) in-tile pixel y coordinate for each
            in-grid pixel.
        grid: GroundTileGrid describing the tile layout.
        scene (Scene): Scene with zone data and config.
        tile_cache_dir (Path): Config-hashed tile cache directory.
        tile_cache (dict[tuple[int, int], np.ndarray]): Cache of already
            loaded tile arrays, keyed by (tile_x, tile_y). Shared across
            calls so a tile already loaded elsewhere is not reloaded.
    """
    from ..scene.ground_tiles import get_tile

    if flat_idx.size == 0:
        return

    tile_id = tile_y * grid.num_tiles_x + tile_x
    unique_ids, inverse = np.unique(tile_id, return_inverse=True)

    for k, tile_id_value in enumerate(unique_ids):
        tx = int(tile_id_value % grid.num_tiles_x)
        ty = int(tile_id_value // grid.num_tiles_x)
        tile_key = (tx, ty)
        if tile_key not in tile_cache:
            tile_cache[tile_key] = np.asarray(get_tile(tx, ty, grid, scene, tile_cache_dir))
        tile_arr = tile_cache[tile_key]

        sel = inverse == k
        output[flat_idx[sel]] = tile_arr[tile_py[sel], tile_px[sel]]


def _camera_ground_rect(camera: Camera) -> t.Tuple[float, float, float, float]:
    """Compute a bounding rectangle for the camera's actual visible ground footprint.

    Delegates to ``visible_ground_aabb``, which intersects the view frustum's
    four corner rays with the z=0 ground plane. This is still conservative
    (an AABB around a wedge-shaped footprint) but far tighter than a square
    of side 2*far centered on the camera regardless of look direction.

    Args:
        camera (Camera): The active camera.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in ENU meters.
    """
    return visible_ground_aabb(camera)


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
    are not silently discarded. Every clipped vertex is in front of the
    camera by construction, so all projected vertices are returned —
    including ones that land off-screen. Both PIL and pygame clip polygon
    primitives to the surface themselves, so passing off-screen pixel
    coordinates is safe and preserves the polygon's true silhouette.

    Shapely's exterior.coords includes a closing duplicate (last == first),
    which is stripped before clipping.

    Note: this preview pass only projects the exterior ring — interior
    rings (holes, e.g. a lake carved out of a park) are not subtracted.
    The interactive preview is an approximation; only the export ground
    pass (ground_tiles.py) draws holes correctly.

    Args:
        boundary: Shapely Polygon with ENU meter coordinates.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        near (float): Camera near clip plane distance in meters.

    Returns:
        list[tuple[float, float]]: Screen-space (x, y) tuples for all
        near-clipped vertices.
    """
    exterior_coords = np.array(boundary.exterior.coords)
    world_vertices = _coords_to_vertex_list(exterior_coords)

    clipped = clip_polygon_to_near_plane(world_vertices, view_matrix, near)
    if not clipped:
        return []

    points_3d = np.stack(clipped, axis=0)
    screen_coords, _ = project_points(
        points_3d, view_matrix, proj_matrix, (width, height)
    )
    return [
        (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        for i in range(len(screen_coords))
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


