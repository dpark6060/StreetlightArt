"""Loads a clipped strip of real OSM road data for devmode parameter tuning."""

import hashlib
import typing as t
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from ..camera.coordinates import create_transformer
from ..config.defaults import DEFAULTS
from ..config.loader import _build_devmode_camera_config
from ..config.schema import SceneConfig
from ..data.osm_loader import load_roads
from ..data.road_classifier import classify_roads
from ..placement.scene_builder import build_scene
from ..scene.scene import Scene
from .devmode_camera import build_devmode_camera

_DEFAULT_STRIP_HALF_WIDTH_DEG: float = 0.025  # ≈2.5 km per side


def build_real_scene_strip(
    config: SceneConfig,
    cache_dir: Path,
    strip_half_width_deg: float = _DEFAULT_STRIP_HALF_WIDTH_DEG,
) -> Scene:
    """Build a scene from a clipped strip of real OSM road data.

    Loads the full road cache (or downloads if absent), clips to a narrow
    east-west corridor centred on the camera target longitude, saves the
    clipped data to a separate debug cache file, then builds a Scene with the
    devmode ENU camera so controls and resolution match the synthetic mode.

    The coordinate origin is the midpoint between the real camera and target,
    which places real San Bernardino roads near ENU (0, 0) — exactly where
    the devmode camera is looking.

    Args:
        config (SceneConfig): Scene configuration with camera, bounds, and lights.
        cache_dir (Path): Directory for OSM cache files.
        strip_half_width_deg (float): Half-width of the longitude strip in
            degrees. Defaults to 0.025 (≈2.5 km per side).

    Returns:
        Scene: Fully constructed scene with real road geometry and devmode camera.
    """
    roads_gdf = _load_strip_roads(config, cache_dir, strip_half_width_deg)
    roads = classify_roads(roads_gdf)
    print(f"  Classified {len(roads)} road segments in strip.")

    origin_lat, origin_lon = _compute_origin(config)
    transformer = create_transformer(origin_lat, origin_lon)
    camera = _build_devmode_camera(config)

    print("Building real strip scene...")
    return build_scene(roads, transformer, camera, config)


def _load_strip_roads(
    config: SceneConfig,
    cache_dir: Path,
    strip_half_width_deg: float,
) -> gpd.GeoDataFrame:
    """Return clipped road data, using a strip-specific cache file.

    On first run, loads the full roads cache and clips to the strip; subsequent
    runs load the smaller strip cache directly.

    Args:
        config (SceneConfig): Scene configuration.
        cache_dir (Path): Directory for cache files.
        strip_half_width_deg (float): Half-width of the longitude strip.

    Returns:
        GeoDataFrame: Clipped road features in WGS84.
    """
    strip_path = _strip_cache_path(cache_dir, config, strip_half_width_deg)
    if strip_path.exists():
        print(f"Loading debug strip from {strip_path.name} ...")
        return gpd.read_file(strip_path)

    print("Clipping real roads to debug strip (one-time) ...")
    bbox = (config.bounds.north, config.bounds.south, config.bounds.east, config.bounds.west)
    full_roads = load_roads(bbox, cache_dir)
    clipped = _clip_to_strip(full_roads, config, strip_half_width_deg)
    clipped.to_file(strip_path, driver="GPKG")
    print(f"Saved debug strip: {strip_path.name} ({len(clipped)} segments)")
    return clipped


def _clip_to_strip(
    roads: gpd.GeoDataFrame,
    config: SceneConfig,
    strip_half_width_deg: float,
) -> gpd.GeoDataFrame:
    """Clip roads to a narrow longitude corridor centred on the camera target.

    Args:
        roads (GeoDataFrame): Full road network in WGS84.
        config (SceneConfig): Scene configuration.
        strip_half_width_deg (float): Half-width of the east-west strip.

    Returns:
        GeoDataFrame: Road features intersecting the strip.
    """
    center_lon = config.camera.target_lon
    clip_box = box(
        center_lon - strip_half_width_deg,
        config.bounds.south,
        center_lon + strip_half_width_deg,
        config.bounds.north,
    )
    return roads[roads.geometry.intersects(clip_box)].copy()


def _strip_cache_path(
    cache_dir: Path,
    config: SceneConfig,
    strip_half_width_deg: float,
) -> Path:
    """Return the cache path for the debug strip file.

    The filename encodes the strip half-width, the camera target longitude
    (the strip is centered on it), and a hash of the clip bounds — so moving
    the camera target or changing the bounds produces a different cache file
    instead of silently reusing a stale strip.

    Args:
        cache_dir (Path): Cache directory.
        config (SceneConfig): Scene configuration providing target_lon and bounds.
        strip_half_width_deg (float): Half-width encoded in the filename.

    Returns:
        Path: Path to the debug strip .gpkg file.
    """
    osm_dir = cache_dir / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    filename = _build_strip_cache_filename(config, strip_half_width_deg)
    return osm_dir / filename


def _build_strip_cache_filename(config: SceneConfig, strip_half_width_deg: float) -> str:
    """Build the debug strip cache filename from strip width, target, and bounds.

    Args:
        config (SceneConfig): Scene configuration providing target_lon and bounds.
        strip_half_width_deg (float): Half-width of the longitude strip.

    Returns:
        str: Filename (with .gpkg extension) uniquely identifying this strip.
    """
    width_tag = f"{strip_half_width_deg:.4f}".replace(".", "p")
    target_lon_tag = f"{round(config.camera.target_lon, 4)}".replace(".", "p").replace("-", "m")
    bounds = config.bounds
    bounds_tuple = (bounds.north, bounds.south, bounds.east, bounds.west)
    bounds_hash = hashlib.md5(str(bounds_tuple).encode()).hexdigest()[:8]
    return f"debug_strip_{width_tag}_{target_lon_tag}_{bounds_hash}.gpkg"


def _compute_origin(config: SceneConfig) -> t.Tuple[float, float]:
    """Compute coordinate origin as the midpoint between camera and target.

    Args:
        config (SceneConfig): Scene configuration.

    Returns:
        tuple[float, float]: (origin_lat, origin_lon) in WGS84 degrees.
    """
    origin_lat = (config.camera.lat + config.camera.target_lat) / 2.0
    origin_lon = (config.camera.lon + config.camera.target_lon) / 2.0
    return origin_lat, origin_lon


def _build_devmode_camera(config: SceneConfig) -> "Camera":
    """Build the devmode ENU camera from config (same as the synthetic scene).

    Args:
        config (SceneConfig): Scene configuration; devmode section may be None.

    Returns:
        Camera: Configured ENU-space devmode camera.
    """
    from ..camera.camera import Camera  # noqa: F401 — satisfies TYPE_CHECKING guard

    if config.devmode is not None:
        camera_cfg = config.devmode.camera
        width = config.devmode.render_width
        height = config.devmode.render_height
    else:
        camera_cfg = _build_devmode_camera_config(DEFAULTS["devmode"]["camera"])
        width = DEFAULTS["devmode"]["render_width"]
        height = DEFAULTS["devmode"]["render_height"]

    return build_devmode_camera(camera_cfg, width, height)
