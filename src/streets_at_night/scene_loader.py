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
) -> Scene:
    """Load OSM data and build a fully populated Scene.

    Downloads (or loads cached) road, land use, and building data, classifies
    roads, creates the coordinate transformer and camera, then delegates to
    build_scene which populates road segments, zone lists, and all lights.

    Args:
        config (SceneConfig): Loaded scene configuration.
        quality (str): Quality preset name for aspect ratio.
        cache_dir (Path): Directory for OSM data cache files.
        progress (Callable[[str], None] | None): Optional progress sink. When
            None (the CLI default) messages print to stdout; otherwise each
            message is passed to the callback and nothing is printed.

    Returns:
        Scene: Fully constructed scene with camera, spatial index, zones,
            and all light sources placed.
    """
    preset = config.quality_presets[quality]
    bbox = bounds_to_bbox(config.bounds)

    _report(progress, "Loading road network...")
    roads_raw = load_roads(bbox, cache_dir)
    roads = classify_roads(roads_raw)
    _report(progress, f"  {len(roads)} road segments loaded.")

    origin_lat, origin_lon = compute_origin(config)
    _report(progress, f"Coordinate origin: {origin_lat:.4f}N, {origin_lon:.4f}W")
    transformer = create_transformer(origin_lat, origin_lon)

    cam_pos = camera_position_enu(config, transformer)
    target_pos = target_position_enu(config, transformer)
    aspect_ratio = preset.width / preset.height

    camera = create_camera(
        position=cam_pos,
        target=target_pos,
        fov_deg=config.camera.fov,
        aspect_ratio=aspect_ratio,
        near=config.camera.near,
        far=config.camera.far,
    )
    _report(
        progress,
        f"Camera at ENU {cam_pos}, looking at {target_pos}, "
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
