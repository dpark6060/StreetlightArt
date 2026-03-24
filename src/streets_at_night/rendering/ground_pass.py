"""Ground render pass — fill sky and ground regions, draw horizon glow, render zones."""

import typing as t
from pathlib import Path

import numpy as np
from PIL import Image

from ..camera.camera import Camera
from ..camera.projection import project_points
from ..utils.colors import hex_to_rgb
from .backend import RenderBackend

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


# Sky and ground background colors
_SKY_COLOR: t.Tuple[int, int, int] = (5, 5, 15)
_GROUND_COLOR: t.Tuple[int, int, int] = (8, 7, 6)

# Horizon glow band color (RGBA with alpha for blending)
_HORIZON_GLOW_COLOR: t.Tuple[int, int, int, int] = (40, 25, 10, 60)

# Number of semi-transparent lines for the horizon glow band
_HORIZON_GLOW_LINES: int = 10

# Minimum number of projected vertices required to draw a zone polygon
_MIN_POLYGON_VERTICES: int = 3


def render_ground(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Fill sky and ground regions, draw horizon glow, and render zone polygons.

    Fills the whole surface with sky color, fills below the horizon with
    ground color, draws a subtle glow band at the horizon, then renders
    zone polygons on top in layer order: dark zones, commercial zones,
    residential zones.

    Args:
        scene (Scene): The world scene containing zones and config.
        camera (Camera): Configured 3D camera.
        backend (RenderBackend): Target render surface.
    """
    width, height = backend.get_surface_size()
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()

    backend.clear(_SKY_COLOR)

    horizon_y = _compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
    _fill_ground(backend, width, height, horizon_y)
    _draw_horizon_glow(backend, width, horizon_y)
    _render_zones(scene, view_matrix, proj_matrix, width, height, backend)


def render_ground_export(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    cache_dir: Path = Path("cache"),
) -> None:
    """Render ground using tiled textures for high-res export.

    Draws the sky/ground/horizon base first (via render_ground), then
    composites per-tile ground colors over the ground region using a
    per-pixel raycast approach. This provides per-house color variation
    that the flat zone-polygon pass cannot produce.

    The tile cache directory is derived from a hash of the scene's color
    config so that stale tiles are bypassed automatically when colors change.

    Args:
        scene (Scene): The world scene with zone and tile config.
        camera (Camera): Configured 3D camera.
        backend (RenderBackend): Target render surface (must support paste_image).
        cache_dir (Path): Root cache directory; tile subdirectory is chosen automatically.
    """
    from ..scene.ground_tiles import create_tile_grid, get_tile_cache_dir

    render_ground(scene, camera, backend)

    width, height = backend.get_surface_size()
    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())

    grid = create_tile_grid(scene)
    tile_cache_dir = get_tile_cache_dir(cache_dir, scene.config)
    ray_dirs = _build_export_ray_directions(width, height, inv_proj, inv_view)
    world_x, world_y, ground_mask = _raycast_ground_plane(camera.position, ray_dirs)

    current_image = backend.get_image()
    composited = _sample_tiles(
        current_image, world_x, world_y, ground_mask, grid, scene, tile_cache_dir, width, height
    )
    backend.paste_image(composited)


def _build_export_ray_directions(
    width: int,
    height: int,
    inv_proj: np.ndarray,
    inv_view: np.ndarray,
) -> np.ndarray:
    """Build a world-space ray direction for every screen pixel.

    Args:
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        inv_proj (np.ndarray): (4, 4) inverse projection matrix.
        inv_view (np.ndarray): (4, 4) inverse view matrix.

    Returns:
        np.ndarray: (H, W, 3) normalized ray directions in world space.
    """
    sx, sy = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    ndc_x = 2.0 * sx / width - 1.0
    ndc_y = 1.0 - 2.0 * sy / height

    n_pixels = width * height
    clip = np.stack([
        ndc_x.ravel(),
        ndc_y.ravel(),
        np.full(n_pixels, -1.0),
        np.ones(n_pixels),
    ], axis=1)

    eye = (inv_proj @ clip.T).T
    eye_dir = np.column_stack([
        eye[:, 0],
        eye[:, 1],
        np.full(n_pixels, -1.0),
        np.zeros(n_pixels),
    ])
    world_dir = (inv_view @ eye_dir.T).T[:, :3]

    norms = np.linalg.norm(world_dir, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    world_dir /= norms

    return world_dir.reshape(height, width, 3)


def _raycast_ground_plane(
    cam_pos: np.ndarray,
    ray_dirs: np.ndarray,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect all rays with the z=0 ground plane.

    Args:
        cam_pos (np.ndarray): (3,) camera position in world ENU.
        ray_dirs (np.ndarray): (H, W, 3) normalized ray directions.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - world_x: (H, W) float array of ENU x hit positions.
            - world_y: (H, W) float array of ENU y hit positions.
            - ground_mask: (H, W) bool array, True where ray hits z=0 (t > 0).
    """
    dz = ray_dirs[:, :, 2]
    ray_hits = np.abs(dz) > 1e-10
    t_val = np.where(ray_hits, -cam_pos[2] / np.where(ray_hits, dz, 1.0), -1.0)
    ground_mask = ray_hits & (t_val > 0.0)

    world_x = cam_pos[0] + t_val * ray_dirs[:, :, 0]
    world_y = cam_pos[1] + t_val * ray_dirs[:, :, 1]

    return world_x, world_y, ground_mask


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


def _render_zones(
    scene: "Scene",
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
) -> None:
    """Render zone polygons on top of the ground fill in back-to-front order.

    Renders dark zones first (parks, water), then commercial zones, then
    residential zones. Each group is drawn on top of the previous, matching
    the intended layer order.

    Args:
        scene (Scene): Scene with zone lists and config.
        view_matrix (np.ndarray): (4, 4) view matrix from the active camera.
        proj_matrix (np.ndarray): (4, 4) projection matrix from the active camera.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
    """
    _render_dark_zones(scene, view_matrix, proj_matrix, width, height, backend)
    _render_commercial_zones(scene, view_matrix, proj_matrix, width, height, backend)
    _render_residential_zones(scene, view_matrix, proj_matrix, width, height, backend)


def _render_dark_zones(
    scene: "Scene",
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
) -> None:
    """Project and draw dark zone (park, forest, water) polygons.

    Uses each zone's own base_color for the fill.

    Args:
        scene (Scene): Scene with dark_zones list.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
    """
    for zone in scene.dark_zones:
        vertices = _project_polygon(zone.boundary, view_matrix, proj_matrix, width, height)
        if len(vertices) >= _MIN_POLYGON_VERTICES:
            backend.draw_polygon(vertices, zone.base_color)


def _render_commercial_zones(
    scene: "Scene",
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
) -> None:
    """Project and draw commercial zone polygons.

    Uses the commercial color from scene config.

    Args:
        scene (Scene): Scene with commercial_zones list and config.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
    """
    color = hex_to_rgb(scene.config.zones.colors.commercial)
    for zone in scene.commercial_zones:
        vertices = _project_polygon(zone.boundary, view_matrix, proj_matrix, width, height)
        if len(vertices) >= _MIN_POLYGON_VERTICES:
            backend.draw_polygon(vertices, color)


def _render_residential_zones(
    scene: "Scene",
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
) -> None:
    """Project and draw residential zone polygons using a flat yard color.

    In preview mode, each zone is filled with a single flat color derived from
    the configured yard base color. No per-house variation is applied here
    (too slow for real-time preview).

    Args:
        scene (Scene): Scene with residential_zones list and config.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        backend (RenderBackend): Target render surface.
    """
    color = hex_to_rgb(scene.config.zones.house.yard_base_color)
    for zone in scene.residential_zones:
        vertices = _project_polygon(zone.boundary, view_matrix, proj_matrix, width, height)
        if len(vertices) >= _MIN_POLYGON_VERTICES:
            backend.draw_polygon(vertices, color)


def _project_polygon(
    boundary: t.Any,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> t.List[t.Tuple[float, float]]:
    """Project a Shapely Polygon's exterior coordinates to screen space.

    Extracts (x, y) exterior ring coordinates, appends z=0, and projects
    all points via project_points. Returns only the screen-space positions
    for points that fall within the valid projection bounds.

    Args:
        boundary: Shapely Polygon with ENU meter coordinates.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.

    Returns:
        list[tuple[float, float]]: Screen-space (x, y) tuples for valid points.
    """
    exterior_coords = np.array(boundary.exterior.coords)
    points_3d = _coords_to_3d(exterior_coords)
    screen_coords, valid_mask = project_points(
        points_3d, view_matrix, proj_matrix, (width, height)
    )
    return [
        (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        for i in range(len(valid_mask))
        if valid_mask[i]
    ]


def _coords_to_3d(coords: np.ndarray) -> np.ndarray:
    """Convert 2D ENU exterior ring coordinates to 3D by appending z=0.

    Args:
        coords (np.ndarray): Shape (N, 2) or (N, 3) coordinate array.
            If already 3D, the z column is overwritten with zeros.

    Returns:
        np.ndarray: Shape (N, 3) with z=0 on the ground plane.
    """
    n = coords.shape[0]
    points_3d = np.zeros((n, 3), dtype=float)
    points_3d[:, 0] = coords[:, 0]
    points_3d[:, 1] = coords[:, 1]
    return points_3d


def _compute_horizon_y(
    camera: Camera,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> int:
    """Approximate the horizon screen y-coordinate.

    Projects a point far ahead on the ground plane (z=0) to find where the
    horizon falls in screen space. Falls back to the image quarter-point if
    the far point is behind the camera.

    Args:
        camera (Camera): The camera.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        int: Screen y-coordinate of the horizon line.
    """
    far_ahead = camera.position + np.array([0.0, camera.far * 0.9, 0.0])
    far_ahead[2] = 0.0
    coords, valid = project_points(
        far_ahead.reshape(1, 3), view_matrix, proj_matrix, (width, height)
    )
    if valid[0]:
        return int(np.clip(coords[0, 1], 0, height - 1))
    return height // 4


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


def _draw_horizon_glow(
    backend: RenderBackend,
    width: int,
    horizon_y: int,
) -> None:
    """Draw a subtle glow band centered on the horizon line.

    Draws semi-transparent lines above and below the horizon, fading in
    opacity toward the edges of the band. The backend must support RGBA
    colors for the glow to be visible (PillowBackend does via ImageDraw RGBA).

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Image width.
        horizon_y (int): Screen y of the horizon.
    """
    r, g, b, max_alpha = _HORIZON_GLOW_COLOR
    half = _HORIZON_GLOW_LINES // 2
    for offset in range(-half, half + 1):
        fade = 1.0 - abs(offset) / (half + 1)
        alpha = int(max_alpha * fade)
        y = horizon_y + offset
        backend.draw_line_rgba((0, y), (width - 1, y), (r, g, b, alpha), width=1)
