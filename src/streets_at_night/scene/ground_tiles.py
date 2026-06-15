"""Ground tile system — generate, cache, and load small ground color tiles.

Each tile is a PIL Image covering a fixed world area (tile_size_meters x tile_size_meters).
Zones are rasterized into tiles so the export renderer can sample ground colors at any
world coordinate. Tiles are cached to disk to avoid regeneration between runs.

Tile coordinate conventions:
- Tile (0, 0) covers the bottom-left corner of the world (min_x, min_y).
- Within a tile, pixel (0, 0) is the top-left corner (max_y of the tile area).
- World Y increases northward; image Y increases downward — consistent with flat_map.py.

Cache invalidation:
- The tile cache directory name includes a hash of color-relevant config fields.
- If zone colors or house colors change, a new directory is used and tiles regenerate.

Stages 5+ add light illumination on top of these base colors.
"""

import hashlib
import math
import typing as t
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..utils.colors import hex_to_rgb

if t.TYPE_CHECKING:
    from ..config.schema import SceneConfig
    from ..scene.scene import Scene
    from ..models.zones import CommercialZone, DarkZone, ResidentialZone


@dataclass
class GroundTileGrid:
    """Metadata describing the full ground tile grid for a scene.

    Attributes:
        origin_x: World x (ENU meters) of the (0, 0) tile's bottom-left corner.
        origin_y: World y (ENU meters) of the (0, 0) tile's bottom-left corner.
        tile_size: World area covered by each tile, in meters.
        tile_res: Pixel size of each tile (tile_res x tile_res).
        num_tiles_x: Number of tiles in the x (east) direction.
        num_tiles_y: Number of tiles in the y (north) direction.
    """

    origin_x: float
    origin_y: float
    tile_size: float
    tile_res: int
    num_tiles_x: int
    num_tiles_y: int

    def get_tile_world_bounds(
        self, tile_x: int, tile_y: int
    ) -> t.Tuple[float, float, float, float]:
        """Return the world bounds of a tile as (min_x, min_y, max_x, max_y).

        Args:
            tile_x (int): Tile column index (east direction).
            tile_y (int): Tile row index (north direction).

        Returns:
            Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in ENU meters.
        """
        min_x = self.origin_x + tile_x * self.tile_size
        min_y = self.origin_y + tile_y * self.tile_size
        max_x = min_x + self.tile_size
        max_y = min_y + self.tile_size
        return min_x, min_y, max_x, max_y

    def world_to_tile_coords(
        self, world_x: float, world_y: float
    ) -> t.Tuple[int, int]:
        """Return the tile indices that contain the given world coordinate.

        Args:
            world_x (float): ENU x coordinate in meters.
            world_y (float): ENU y coordinate in meters.

        Returns:
            Tuple[int, int]: (tile_x, tile_y) indices.
        """
        tile_x = int((world_x - self.origin_x) / self.tile_size)
        tile_y = int((world_y - self.origin_y) / self.tile_size)
        return tile_x, tile_y


def create_tile_grid(scene: "Scene") -> GroundTileGrid:
    """Compute a tile grid that covers all road segments in the scene.

    The grid is padded by one tile on each side so zones near the scene boundary
    are fully covered. Tile size and resolution come from scene config.

    Args:
        scene (Scene): Scene with road segments and configuration.

    Returns:
        GroundTileGrid: Tile grid metadata covering the scene.

    Raises:
        ValueError: If the scene has no road segments with geometry.
    """
    min_x, min_y, max_x, max_y = _compute_scene_bounds(scene)
    tile_size = scene.config.zones.tile_size_meters
    tile_res = scene.config.zones.tile_resolution_px

    origin_x, origin_y = _compute_padded_origin(min_x, min_y, tile_size)
    num_tiles_x, num_tiles_y = _compute_tile_counts(
        origin_x, origin_y, max_x, max_y, tile_size
    )

    return GroundTileGrid(
        origin_x=origin_x,
        origin_y=origin_y,
        tile_size=tile_size,
        tile_res=tile_res,
        num_tiles_x=num_tiles_x,
        num_tiles_y=num_tiles_y,
    )


def get_tile(
    tile_x: int,
    tile_y: int,
    grid: GroundTileGrid,
    scene: "Scene",
    tile_cache_dir: Path,
) -> Image.Image:
    """Return a ground tile, loading from cache or generating on demand.

    The caller is responsible for passing the correct tile cache directory.
    Use ``get_tile_cache_dir()`` to compute a config-hashed directory path
    so that stale tiles are automatically bypassed when config changes.

    Args:
        tile_x (int): Tile column index.
        tile_y (int): Tile row index.
        grid (GroundTileGrid): Tile grid metadata.
        scene (Scene): Scene with zone data and config.
        tile_cache_dir (Path): Directory for this config's tile cache files.

    Returns:
        Image.Image: RGB PIL image for the tile.
    """
    tile_cache_dir.mkdir(parents=True, exist_ok=True)
    tile_path = tile_cache_dir / f"tile_{tile_x}_{tile_y}.png"

    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")

    img = generate_tile(tile_x, tile_y, grid, scene)
    img.save(str(tile_path))
    return img


def generate_tile(
    tile_x: int,
    tile_y: int,
    grid: GroundTileGrid,
    scene: "Scene",
) -> Image.Image:
    """Generate a single ground tile by rasterizing zone colors.

    Fills the tile with the default ground color, then overlays zones in
    draw order: dark zones, commercial zones, residential zones (yards then
    house footprints).

    Args:
        tile_x (int): Tile column index.
        tile_y (int): Tile row index.
        grid (GroundTileGrid): Tile grid metadata.
        scene (Scene): Scene with zone data and configuration.

    Returns:
        Image.Image: RGB PIL image of size (tile_res, tile_res).
    """
    tile_bounds = grid.get_tile_world_bounds(tile_x, tile_y)
    default_color = hex_to_rgb(scene.config.zones.colors.default_ground)

    img = Image.new("RGB", (grid.tile_res, grid.tile_res), default_color)
    draw = ImageDraw.Draw(img)

    _draw_dark_zones(draw, scene.dark_zones, tile_bounds, grid)
    _draw_commercial_zones(draw, scene.commercial_zones, tile_bounds, grid, scene)
    _draw_residential_zones(draw, scene.residential_zones, tile_bounds, grid, scene)

    return img


# ------------------------------------------------------------------
# Zone drawing
# ------------------------------------------------------------------


def _draw_dark_zones(
    draw: ImageDraw.ImageDraw,
    zones: "t.List[DarkZone]",
    tile_bounds: t.Tuple[float, float, float, float],
    grid: GroundTileGrid,
) -> None:
    """Draw all dark zones that overlap this tile.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        zones (list[DarkZone]): All dark zones in the scene.
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        grid (GroundTileGrid): Tile grid metadata for coordinate conversion.
    """
    for zone in zones:
        if not _bounds_overlap(tile_bounds, zone.boundary.bounds):
            continue
        _draw_polygon_on_tile(draw, zone.boundary, tile_bounds, grid, zone.base_color)


def _draw_commercial_zones(
    draw: ImageDraw.ImageDraw,
    zones: "t.List[CommercialZone]",
    tile_bounds: t.Tuple[float, float, float, float],
    grid: GroundTileGrid,
    scene: "Scene",
) -> None:
    """Draw all commercial zones that overlap this tile.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        zones (list[CommercialZone]): All commercial zones in the scene.
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        grid (GroundTileGrid): Tile grid metadata for coordinate conversion.
        scene (Scene): Scene providing config colors.
    """
    color = hex_to_rgb(scene.config.zones.colors.commercial)
    for zone in zones:
        if not _bounds_overlap(tile_bounds, zone.boundary.bounds):
            continue
        _draw_polygon_on_tile(draw, zone.boundary, tile_bounds, grid, color)


def _draw_residential_zones(
    draw: ImageDraw.ImageDraw,
    zones: "t.List[ResidentialZone]",
    tile_bounds: t.Tuple[float, float, float, float],
    grid: GroundTileGrid,
    scene: "Scene",
) -> None:
    """Draw residential zones: yard fill first, then per-house footprints.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        zones (list[ResidentialZone]): All residential zones in the scene.
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        grid (GroundTileGrid): Tile grid metadata for coordinate conversion.
        scene (Scene): Scene providing config colors.
    """
    default_yard = hex_to_rgb(scene.config.zones.house.yard_base_color)

    for zone in zones:
        if not _bounds_overlap(tile_bounds, zone.boundary.bounds):
            continue
        yard_color = _get_zone_yard_color(zone, default_yard)
        _draw_polygon_on_tile(draw, zone.boundary, tile_bounds, grid, yard_color)
        _draw_house_footprints(draw, zone, tile_bounds, grid)


def _draw_house_footprints(
    draw: ImageDraw.ImageDraw,
    zone: "ResidentialZone",
    tile_bounds: t.Tuple[float, float, float, float],
    grid: GroundTileGrid,
) -> None:
    """Draw each house footprint within a residential zone.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        zone (ResidentialZone): Residential zone containing houses.
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        grid (GroundTileGrid): Tile grid metadata for coordinate conversion.
    """
    for house in zone.houses:
        if not _bounds_overlap(tile_bounds, house.footprint.bounds):
            continue
        _draw_polygon_on_tile(draw, house.footprint, tile_bounds, grid, house.base_color)


# ------------------------------------------------------------------
# Polygon rasterization
# ------------------------------------------------------------------


def _draw_polygon_on_tile(
    draw: ImageDraw.ImageDraw,
    polygon: t.Any,
    tile_bounds: t.Tuple[float, float, float, float],
    grid: GroundTileGrid,
    color: t.Tuple[int, int, int],
) -> None:
    """Rasterize a world-space polygon onto a tile using PIL.

    Converts the polygon's exterior ring vertices to tile-local pixel
    coordinates and calls PIL ImageDraw.polygon(). Requires at least 3
    vertices to form a valid polygon.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        polygon: Shapely Polygon in ENU meters.
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        grid (GroundTileGrid): Tile grid metadata (tile_size and tile_res).
        color (Tuple[int, int, int]): RGB fill color.
    """
    pixel_coords = _polygon_to_tile_pixels(polygon, tile_bounds, grid)
    if len(pixel_coords) < 3:
        return
    draw.polygon(pixel_coords, fill=color)


def _polygon_to_tile_pixels(
    polygon: t.Any,
    tile_bounds: t.Tuple[float, float, float, float],
    grid: GroundTileGrid,
) -> t.List[t.Tuple[float, float]]:
    """Convert a polygon's exterior vertices to tile-local pixel coordinates.

    World Y increases northward; image Y increases downward. So pixel_y is
    computed as (tile_max_y - world_y) * scale — matching the convention in
    flat_map.py.

    Args:
        polygon: Shapely Polygon in ENU meters.
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        grid (GroundTileGrid): Tile grid metadata.

    Returns:
        List[Tuple[float, float]]: Pixel (x, y) coordinates of exterior ring.
    """
    tile_min_x, tile_min_y, _tile_max_x, tile_max_y = tile_bounds
    scale = grid.tile_res / grid.tile_size

    coords = list(polygon.exterior.coords)
    pixel_coords = []
    for world_x, world_y in coords:
        px = (world_x - tile_min_x) * scale
        py = (tile_max_y - world_y) * scale
        pixel_coords.append((px, py))
    return pixel_coords


# ------------------------------------------------------------------
# Bounds and grid helpers
# ------------------------------------------------------------------


def _compute_scene_bounds(
    scene: "Scene",
) -> t.Tuple[float, float, float, float]:
    """Compute the bounding box of all road segment geometries in the scene.

    Args:
        scene (Scene): Scene with road segments.

    Returns:
        Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in ENU meters.

    Raises:
        ValueError: If the scene has no road segments with geometry.
    """
    valid_geoms = [s.geometry for s in scene.road_segments if s.geometry.shape[0] > 0]
    if not valid_geoms:
        raise ValueError("Cannot create tile grid: no road segments have geometry.")

    all_points = np.vstack(valid_geoms)
    return (
        float(all_points[:, 0].min()),
        float(all_points[:, 1].min()),
        float(all_points[:, 0].max()),
        float(all_points[:, 1].max()),
    )


def _compute_padded_origin(
    min_x: float, min_y: float, tile_size: float
) -> t.Tuple[float, float]:
    """Compute a grid origin padded one tile below and left of the scene min.

    Args:
        min_x (float): Scene minimum x in meters.
        min_y (float): Scene minimum y in meters.
        tile_size (float): Tile size in meters.

    Returns:
        Tuple[float, float]: (origin_x, origin_y) of tile (0, 0).
    """
    origin_x = min_x - tile_size
    origin_y = min_y - tile_size
    return origin_x, origin_y


def _compute_tile_counts(
    origin_x: float,
    origin_y: float,
    max_x: float,
    max_y: float,
    tile_size: float,
) -> t.Tuple[int, int]:
    """Compute the number of tiles needed to cover the scene plus one tile of padding.

    Args:
        origin_x (float): Grid origin x in meters.
        origin_y (float): Grid origin y in meters.
        max_x (float): Scene maximum x in meters.
        max_y (float): Scene maximum y in meters.
        tile_size (float): Tile size in meters.

    Returns:
        Tuple[int, int]: (num_tiles_x, num_tiles_y).
    """
    num_tiles_x = math.ceil((max_x - origin_x) / tile_size) + 1
    num_tiles_y = math.ceil((max_y - origin_y) / tile_size) + 1
    return num_tiles_x, num_tiles_y


def _bounds_overlap(
    tile_bounds: t.Tuple[float, float, float, float],
    shapely_bounds: t.Tuple[float, float, float, float],
) -> bool:
    """Return True if two axis-aligned bounding boxes overlap.

    Shapely's `.bounds` returns (min_x, min_y, max_x, max_y) which matches
    the tile_bounds format.

    Args:
        tile_bounds (Tuple): (min_x, min_y, max_x, max_y) of the tile.
        shapely_bounds (Tuple): (min_x, min_y, max_x, max_y) from shapely .bounds.

    Returns:
        bool: True if the two rectangles overlap (touching edges count).
    """
    t_min_x, t_min_y, t_max_x, t_max_y = tile_bounds
    s_min_x, s_min_y, s_max_x, s_max_y = shapely_bounds
    return (
        t_min_x <= s_max_x
        and t_max_x >= s_min_x
        and t_min_y <= s_max_y
        and t_max_y >= s_min_y
    )


# ------------------------------------------------------------------
# Color helpers
# ------------------------------------------------------------------


def _get_zone_yard_color(
    zone: "ResidentialZone",
    default_yard: t.Tuple[int, int, int],
) -> t.Tuple[int, int, int]:
    """Compute the representative yard color for a residential zone.

    Uses the average yard color of the zone's houses. Falls back to the
    default yard color if the zone has no houses.

    Args:
        zone (ResidentialZone): Residential zone with houses.
        default_yard (Tuple[int, int, int]): Fallback yard color (R, G, B).

    Returns:
        Tuple[int, int, int]: Representative (R, G, B) yard color.
    """
    if not zone.houses:
        return default_yard
    return _average_colors([h.yard_color for h in zone.houses])


def _average_colors(
    colors: t.List[t.Tuple[int, int, int]],
) -> t.Tuple[int, int, int]:
    """Compute the per-channel average of a list of RGB tuples.

    Args:
        colors (List[Tuple[int, int, int]]): List of (R, G, B) color tuples.

    Returns:
        Tuple[int, int, int]: Averaged (R, G, B) color.
    """
    count = len(colors)
    r = sum(c[0] for c in colors) // count
    g = sum(c[1] for c in colors) // count
    b = sum(c[2] for c in colors) // count
    return r, g, b


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------


def get_tile_cache_dir(cache_dir: Path, config: "SceneConfig") -> Path:
    """Return the config-hashed tile cache directory path.

    The directory name embeds an 8-character MD5 hash of the color-relevant
    config fields. When zone colors or house colors change, the hash changes
    and a new directory is used, so stale tiles are never served.

    The directory is not created here; ``get_tile()`` creates it on demand.

    Args:
        cache_dir (Path): Root cache directory for the project.
        config (SceneConfig): Full scene config; only color fields are hashed.

    Returns:
        Path: Path to the config-specific tile cache subdirectory.
    """
    key = _config_cache_key(config)
    return cache_dir / "ground_tiles" / key


def _config_cache_key(config: "SceneConfig") -> str:
    """Compute an 8-character hash of all color fields that affect tile rendering.

    Hashes ZoneColorsConfig fields, HouseConfig.exterior_colors, yard_base_color,
    and yard_variation. Any change to these values produces a different key.

    Args:
        config (SceneConfig): Full scene configuration.

    Returns:
        str: 8-character lowercase hex string (MD5 prefix).
    """
    colors = config.zones.colors
    house = config.zones.house

    parts = [
        colors.commercial,
        colors.industrial,
        colors.park_forest,
        colors.water,
        colors.road_surface,
        colors.default_ground,
        str(house.exterior_colors.colors),
        house.yard_base_color,
        str(house.yard_variation),
    ]
    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:8]
