"""Flat top-down road map — generate and cache a PIL image of the road network.

Used by the warp-preview path to avoid per-segment projection on every frame.
The image is generated once at startup and cached to disk so subsequent runs
skip generation entirely.
"""

import hashlib
import json
import typing as t
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..models.road_types import RoadType

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


# Background color: very dark blue (night ground)
_BACKGROUND_COLOR: t.Tuple[int, int, int] = (10, 12, 18)

# Road type → (RGB color, base line width in pixels at 1m/px)
# Widths in meters; these are scaled down by resolution at draw time.
_FLAT_MAP_ROAD_STYLE: t.Dict[RoadType, t.Tuple[t.Tuple[int, int, int], float]] = {
    RoadType.HIGHWAY:      ((255, 215,   0), 12.0),
    RoadType.MAJOR_ROAD:   ((255, 107,  53),  8.0),
    RoadType.COLLECTOR:    ((255, 255, 255),  6.0),
    RoadType.RESIDENTIAL:  ((136, 136, 136),  4.0),
    RoadType.SERVICE:      (( 85,  85,  85),  3.0),
    RoadType.MINOR:        ((100, 100, 120),  4.0),
}

# Maximum pixel dimension for either axis of the flat map
_MAX_FLAT_MAP_DIM: int = 8000

# Cache schema version — bump to invalidate cached flat maps when the drawing
# logic changes but the style table/census/resolution/bounds inputs to the
# cache key do not.
_FLAT_MAP_CACHE_VERSION: int = 1


@dataclass
class FlatMapMeta:
    """Metadata for the flat map — world bounds and resolution.

    Attributes:
        min_x: Minimum ENU x (east) coordinate in meters.
        min_y: Minimum ENU y (north) coordinate in meters.
        max_x: Maximum ENU x coordinate in meters.
        max_y: Maximum ENU y coordinate in meters.
        resolution: Meters per pixel.
    """

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    resolution: float

    def world_to_pixel(
        self,
        x: t.Union[float, np.ndarray],
        y: t.Union[float, np.ndarray],
    ) -> t.Tuple[t.Union[float, np.ndarray], t.Union[float, np.ndarray]]:
        """Convert ENU world coordinates to flat map pixel coordinates.

        Pure elementwise arithmetic, so ``x``/``y`` may be scalars or numpy
        arrays of any shape (the raycast paths pass full (H, W) arrays here to
        vectorize world-to-pixel conversion across an entire screen).

        Args:
            x (float | np.ndarray): ENU east coordinate(s) in meters.
            y (float | np.ndarray): ENU north coordinate(s) in meters.

        Returns:
            tuple[float | np.ndarray, float | np.ndarray]: (px, py) pixel coordinate(s).
        """
        px = (x - self.min_x) / self.resolution
        py = (self.max_y - y) / self.resolution
        return px, py

    def pixel_to_world(self, px: float, py: float) -> t.Tuple[float, float]:
        """Convert flat map pixel coordinates to ENU world coordinates.

        Args:
            px (float): Pixel x coordinate.
            py (float): Pixel y coordinate.

        Returns:
            tuple[float, float]: (world_x, world_y) in ENU meters.
        """
        x = px * self.resolution + self.min_x
        y = self.max_y - py * self.resolution
        return x, y


def get_flat_map(
    scene: "Scene",
    cache_dir: Path,
    resolution: float = 10.0,
) -> t.Tuple[Image.Image, FlatMapMeta]:
    """Return a flat top-down PIL image of the road network.

    Loads from cache if available, otherwise generates and saves to cache.

    Args:
        scene (Scene): Scene with road segments.
        cache_dir (Path): Directory to store cached flat map files.
        resolution (float): Meters per pixel. Smaller = higher detail but
            more RAM. Auto-increased if the map would exceed _MAX_FLAT_MAP_DIM
            in either dimension.

    Returns:
        tuple[Image.Image, FlatMapMeta]: The flat map image (RGB) and its
            world-coordinate metadata.
    """
    flat_map_dir = cache_dir / "flat_map"
    flat_map_dir.mkdir(parents=True, exist_ok=True)
    meta = _compute_meta(scene, resolution)
    cache_key = _compute_cache_key(meta, scene.road_segments)
    img_path = flat_map_dir / f"flat_map_{cache_key}.png"
    json_path = flat_map_dir / f"flat_map_{cache_key}.json"

    if img_path.exists() and json_path.exists():
        loaded_meta = _load_meta(json_path)
        img = Image.open(img_path).convert("RGB")
        return img, loaded_meta

    img = generate_flat_map(scene, resolution)
    img.save(str(img_path))
    _save_meta(meta, json_path)
    return img, meta


def save_flat_map_to_output(
    img: Image.Image,
    region_name: str,
    output_dir: Path,
) -> Path:
    """Save a copy of the flat map into the output directory.

    Gives the user a viewable artifact at a predictable path (the disk cache
    copy lives under an opaque hash name in the cache directory).

    Args:
        img (Image.Image): The generated flat map image.
        region_name (str): Config region name used in the filename; an empty
            string falls back to "flat_map.png".
        output_dir (Path): Directory to write into; created if missing.

    Returns:
        Path: Path of the written PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{region_name}" if region_name else ""
    out_path = output_dir / f"flat_map{suffix}.png"
    img.save(str(out_path))
    return out_path


def generate_flat_map(
    scene: "Scene",
    resolution: float = 10.0,
) -> Image.Image:
    """Render all road segments as a top-down PIL image.

    The image covers the full world bounds. One pixel = `resolution` meters.
    Origin (0, 0) of the image = (world_min_x, world_max_y) in ENU coordinates
    (image Y increases downward, world Y increases northward).

    Automatically increases resolution (coarser) if the computed image would
    exceed _MAX_FLAT_MAP_DIM pixels in either dimension.

    Args:
        scene (Scene): Scene with road segments.
        resolution (float): Initial meters per pixel.

    Returns:
        Image.Image: RGB image with roads drawn on a dark background.
    """
    meta = _compute_meta(scene, resolution)
    img_w = int((meta.max_x - meta.min_x) / meta.resolution)
    img_h = int((meta.max_y - meta.min_y) / meta.resolution)

    img = Image.new("RGB", (img_w, img_h), _BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    for segment in scene.road_segments:
        _draw_segment_on_flat_map(draw, segment, meta)

    return img


def _compute_meta(scene: "Scene", resolution: float) -> FlatMapMeta:
    """Compute FlatMapMeta from scene bounds, auto-scaling resolution if needed.

    Increases resolution (coarser pixels) until both image dimensions fit
    within _MAX_FLAT_MAP_DIM.

    Args:
        scene (Scene): Scene with road segments.
        resolution (float): Initial meters per pixel.

    Returns:
        FlatMapMeta: Metadata with adjusted resolution if necessary.
    """
    min_x, min_y, max_x, max_y = _compute_world_bounds(scene)
    adjusted = _auto_scale_resolution(min_x, min_y, max_x, max_y, resolution)
    return FlatMapMeta(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        resolution=adjusted,
    )


def _compute_world_bounds(
    scene: "Scene",
) -> t.Tuple[float, float, float, float]:
    """Compute overall world bounds from all road segment geometries.

    Args:
        scene (Scene): Scene with road segments.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in meters.

    Raises:
        ValueError: If no segments have geometry points.
    """
    valid_geoms = [s.geometry for s in scene.road_segments if s.geometry.shape[0] > 0]
    if not valid_geoms:
        raise ValueError("Cannot generate flat map: no road segments have geometry.")
    all_points = np.vstack(valid_geoms)
    return (
        float(all_points[:, 0].min()),
        float(all_points[:, 1].min()),
        float(all_points[:, 0].max()),
        float(all_points[:, 1].max()),
    )


def _auto_scale_resolution(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    resolution: float,
) -> float:
    """Increase resolution (m/px) until image fits within _MAX_FLAT_MAP_DIM.

    Args:
        min_x (float): World min x in meters.
        min_y (float): World min y in meters.
        max_x (float): World max x in meters.
        max_y (float): World max y in meters.
        resolution (float): Starting meters per pixel.

    Returns:
        float: Adjusted meters per pixel.
    """
    world_w = max_x - min_x
    world_h = max_y - min_y
    while True:
        img_w = int(world_w / resolution)
        img_h = int(world_h / resolution)
        if img_w <= _MAX_FLAT_MAP_DIM and img_h <= _MAX_FLAT_MAP_DIM:
            return resolution
        resolution *= 1.5


def _draw_segment_on_flat_map(
    draw: ImageDraw.ImageDraw,
    segment: t.Any,
    meta: FlatMapMeta,
) -> None:
    """Draw one road segment onto the flat map PIL draw context.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        segment: RoadSegment with .geometry and .road_type.
        meta (FlatMapMeta): World-to-pixel conversion metadata.
    """
    style = _FLAT_MAP_ROAD_STYLE.get(segment.road_type)
    if style is None:
        return
    color, width_m = style
    px_width = max(1, int(width_m / meta.resolution))

    points = segment.geometry
    if points.shape[0] < 2:
        return

    pixel_coords = [meta.world_to_pixel(float(pt[0]), float(pt[1])) for pt in points]
    for i in range(len(pixel_coords) - 1):
        draw.line([pixel_coords[i], pixel_coords[i + 1]], fill=color, width=px_width)


def _compute_cache_key(meta: FlatMapMeta, road_segments: t.Sequence[t.Any]) -> str:
    """Compute a short hex hash to identify a cached flat map.

    Hash inputs: resolution, world bounds (rounded to 1 meter to absorb
    floating-point noise between runs), the road style table (colors and
    widths), the background color, a census of road types present in
    the scene, and a cache schema version. This ensures a style or
    classification change invalidates the cache even when segment count and
    bounds are unchanged, and lets the version be bumped to invalidate the
    cache on drawing-logic changes alone.

    Args:
        meta (FlatMapMeta): Flat map metadata.
        road_segments (Sequence): Road segments in the scene.

    Returns:
        str: 12-character hex string.
    """
    style_repr = repr(
        sorted(
            (road_type.value, color, width)
            for road_type, (color, width) in _FLAT_MAP_ROAD_STYLE.items()
        )
    )
    road_type_census = sorted(
        Counter(segment.road_type.value for segment in road_segments).items()
    )
    key_data = (
        f"{meta.resolution:.4f}"
        f"_{len(road_segments)}"
        f"_{round(meta.min_x)}_{round(meta.min_y)}"
        f"_{round(meta.max_x)}_{round(meta.max_y)}"
        f"_{style_repr}"
        f"_{_BACKGROUND_COLOR}"
        f"_{road_type_census}"
        f"_{_FLAT_MAP_CACHE_VERSION}"
    )
    return hashlib.md5(key_data.encode()).hexdigest()[:12]


def _save_meta(meta: FlatMapMeta, path: Path) -> None:
    """Serialize FlatMapMeta to a JSON file.

    Args:
        meta (FlatMapMeta): Metadata to save.
        path (Path): Destination file path.
    """
    data = {
        "min_x": meta.min_x,
        "min_y": meta.min_y,
        "max_x": meta.max_x,
        "max_y": meta.max_y,
        "resolution": meta.resolution,
    }
    path.write_text(json.dumps(data))


def _load_meta(path: Path) -> FlatMapMeta:
    """Deserialize FlatMapMeta from a JSON file.

    Args:
        path (Path): Source file path.

    Returns:
        FlatMapMeta: Loaded metadata.
    """
    data = json.loads(path.read_text())
    return FlatMapMeta(
        min_x=data["min_x"],
        min_y=data["min_y"],
        max_x=data["max_x"],
        max_y=data["max_y"],
        resolution=data["resolution"],
    )
