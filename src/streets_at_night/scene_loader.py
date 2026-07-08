"""Scene loading and coordinate helpers shared by the CLI and UI.

Builds a fully populated :class:`~streets_at_night.scene.scene.Scene` from a
``SceneConfig`` and provides the small WGS84↔ENU conversion helpers the scene
build depends on. Extracted from ``main.py`` so the UI tabs can load scenes and
convert camera coordinates without importing the CLI module.

Import direction is one-way: ``main`` imports from ``scene_loader``; this module
must never import ``main``.
"""

import typing as t
from pathlib import Path

import numpy as np

from .camera.camera import create_camera
from .camera.coordinates import (
    CoordinateTransformer,
    create_transformer,
    feet_to_meters,
    wgs84_to_enu,
)
from .config.paths import compute_origin
from .config.schema import BoundsConfig, SceneConfig
from .data.osm_loader import load_buildings, load_land_use, load_roads
from .data.road_classifier import classify_roads
from .placement.scene_builder import build_scene
from .scene.scene import Scene
from .scene.scene_cache import (
    compute_scene_cache_key,
    load_scene_content,
    save_scene_content,
)


def _report(progress: t.Optional[t.Callable[[str], None]], msg: str) -> None:
    """Emit a progress message either to stdout or a supplied callback.

    Args:
        progress (Callable[[str], None] | None): Progress sink. When None the
            message is printed to stdout (preserving CLI output); otherwise the
            callback receives the message and nothing is printed.
        msg (str): The progress message.
    """
    if progress is None:
        print(msg)
        return
    progress(msg)


def load_scene(
    config: SceneConfig,
    quality: str,
    cache_dir: Path,
    progress: t.Optional[t.Callable[[str], None]] = None,
    use_scene_cache: bool = True,
) -> Scene:
    """Load OSM data and build a fully populated Scene.

    On a scene-cache hit, the pre-built content (segments, zones, buildings,
    lights, spatial indexes) is unpickled and rehydrated with a freshly built
    camera and the live config — skipping OSM loading and the whole scene
    build. The cache key covers everything the content depends on (bounds,
    origin, seed, lights/zones/buildings config, OSM file signatures), so a hit
    is content-identical to a fresh build; camera pose beyond lat/lon, quality,
    and all render-time config (atmosphere, grade, stars, heat-haze) are free to
    change without invalidating it.

    Args:
        config (SceneConfig): Loaded scene configuration.
        quality (str): Quality preset name for aspect ratio.
        cache_dir (Path): Directory for OSM data cache and scene cache files.
        progress (Callable[[str], None] | None): Optional progress sink. When
            None (the CLI default) messages print to stdout; otherwise each
            message is passed to the callback and nothing is printed.
        use_scene_cache (bool): When True (default), read/write the built-scene
            disk cache. Pass False to always rebuild (e.g. for benchmarking or
            when debugging the builders).

    Returns:
        Scene: Fully constructed scene with camera, spatial index, zones,
            and all light sources placed.
    """
    preset = config.quality_presets[quality]
    bbox = bounds_to_bbox(config.bounds)

    origin_lat, origin_lon = compute_origin(config)
    transformer = create_transformer(origin_lat, origin_lon)
    camera = _build_camera(config, preset, transformer)

    if use_scene_cache:
        key = compute_scene_cache_key(config, bbox, cache_dir)
        content = load_scene_content(cache_dir, key)
        if content is not None:
            _report(progress, "Loaded scene from cache.")
            return _scene_from_content(content, camera, config, transformer)

    _report(progress, f"Coordinate origin: {origin_lat:.4f}N, {origin_lon:.4f}W")

    _report(progress, "Loading road network...")
    roads_raw = load_roads(bbox, cache_dir)
    roads = classify_roads(roads_raw)
    _report(progress, f"  {len(roads)} road segments loaded.")

    _report(
        progress,
        f"Camera at ENU {camera.position}, looking at {camera.target}, "
        f"FOV {config.camera.fov}deg",
    )

    _report(progress, "Loading land use data...")
    land_use = load_land_use(bbox, cache_dir)
    _report(progress, f"  {len(land_use)} land use features loaded.")

    _report(progress, "Loading building footprints...")
    buildings = load_buildings(bbox, cache_dir)
    _report(progress, f"  {len(buildings)} building footprints loaded.")

    _report(progress, "Building scene and spatial index...")
    scene = build_scene(roads, transformer, camera, config, land_use, buildings)
    _report(progress, f"  {len(scene.road_segments)} segments indexed.")

    if use_scene_cache:
        save_scene_content(cache_dir, key, _content_from_scene(scene))
    return scene


def _build_camera(config: SceneConfig, preset, transformer: CoordinateTransformer):
    """Build the ENU Camera from config, quality preset, and transformer.

    Args:
        config (SceneConfig): Scene configuration with the camera section.
        preset: Active quality preset (supplies the aspect ratio).
        transformer (CoordinateTransformer): WGS84 -> ENU converter.

    Returns:
        Camera: Configured camera in ENU space.
    """
    return create_camera(
        position=camera_position_enu(config, transformer),
        target=target_position_enu(config, transformer),
        fov_deg=config.camera.fov,
        aspect_ratio=preset.width / preset.height,
        near=config.camera.near,
        far=config.camera.far,
    )


def _content_from_scene(scene: Scene) -> dict:
    """Extract the cacheable, camera-independent content from a built scene.

    Args:
        scene (Scene): A freshly built scene.

    Returns:
        dict: Picklable scene content (geometry, zones, buildings, lights,
            spatial indexes, and derived bounds) — no camera or config.
    """
    return {
        "road_segments": scene.road_segments,
        "residential_zones": scene.residential_zones,
        "commercial_zones": scene.commercial_zones,
        "dark_zones": scene.dark_zones,
        "buildings": scene.buildings,
        "lights": scene.lights,
        "car_pairs": scene.car_pairs,
        "road_index": scene._road_index,
        "zone_index": scene._zone_index,
        "building_index": scene._building_index,
        "light_index": scene._light_index,
        "index_bounds": scene._index_bounds,
        "lights_aabb": scene.get_lights_aabb(),
    }


def _scene_from_content(
    content: dict,
    camera,
    config: SceneConfig,
    transformer: CoordinateTransformer,
) -> Scene:
    """Rehydrate a Scene from cached content plus a live camera and config.

    Args:
        content (dict): Content produced by ``_content_from_scene``.
        camera: The freshly built camera for this render.
        config (SceneConfig): The live scene configuration.
        transformer (CoordinateTransformer): WGS84 -> ENU converter for this
            origin (deterministic from the origin baked into the cache key).

    Returns:
        Scene: A fully populated scene equivalent to a fresh build.
    """
    scene = Scene(
        content["road_segments"],
        camera,
        config,
        residential_zones=content["residential_zones"],
        commercial_zones=content["commercial_zones"],
        dark_zones=content["dark_zones"],
        transformer=transformer,
        buildings=content["buildings"],
    )
    scene.lights = content["lights"]
    scene.car_pairs = content["car_pairs"]
    scene._road_index = content["road_index"]
    scene._zone_index = content["zone_index"]
    scene._building_index = content["building_index"]
    scene._light_index = content["light_index"]
    scene._index_bounds = content["index_bounds"]
    scene._lights_aabb = content["lights_aabb"]
    return scene


def bounds_to_bbox(bounds: BoundsConfig) -> tuple:
    """Convert a BoundsConfig to a (north, south, east, west) tuple.

    Args:
        bounds (BoundsConfig): Geographic bounds from config.

    Returns:
        tuple: (north, south, east, west) floats.
    """
    return (bounds.north, bounds.south, bounds.east, bounds.west)


def camera_position_enu(
    config: SceneConfig,
    transformer: CoordinateTransformer,
) -> np.ndarray:
    """Convert the camera position from WGS84+ft to ENU meters.

    Args:
        config (SceneConfig): Scene configuration with camera section.
        transformer (CoordinateTransformer): Configured coordinate transformer.

    Returns:
        np.ndarray: (3,) ENU position in meters.
    """
    alt_m = feet_to_meters(config.camera.altitude_ft)
    e, n, u = wgs84_to_enu(transformer, config.camera.lat, config.camera.lon, alt_m)
    return np.array([e, n, u])


def target_position_enu(
    config: SceneConfig,
    transformer: CoordinateTransformer,
) -> np.ndarray:
    """Convert the camera target from WGS84+ft to ENU meters.

    Args:
        config (SceneConfig): Scene configuration with camera section.
        transformer (CoordinateTransformer): Configured coordinate transformer.

    Returns:
        np.ndarray: (3,) ENU position in meters.
    """
    alt_m = feet_to_meters(config.camera.target_altitude_ft)
    e, n, u = wgs84_to_enu(
        transformer, config.camera.target_lat, config.camera.target_lon, alt_m
    )
    return np.array([e, n, u])
