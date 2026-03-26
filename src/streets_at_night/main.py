"""CLI entry point for Streets at Night.

Subcommands:
    verify  — Download OSM data and produce a verification plot.
    preview — Launch interactive pygame preview.
    render  — Render a perspective road image from the configured camera.
"""

import argparse
import time
import tomllib
import tomli_w
import typing as t
from pathlib import Path

import numpy as np

from .camera.camera import create_camera
from .camera.coordinates import (
    CoordinateTransformer,
    create_transformer,
    enu_to_wgs84,
    feet_to_meters,
    meters_to_feet,
    wgs84_to_enu,
)
from .config.loader import load_config
from .config.schema import BoundsConfig, SceneConfig
from .data.osm_loader import load_buildings, load_land_use, load_roads
from .data.road_classifier import classify_roads
from .data.verify_plot import save_verification_plot
from .rendering.pillow_backend import PillowBackend
from .rendering.render_pipeline import get_export_passes, render_scene_frame
from .placement.scene_builder import build_scene
from .scene.scene import Scene


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        description="Streets at Night — aerial city night renderer"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scene_config.toml"),
        help="Path to scene config TOML file",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Produce OSM verification plot")
    verify_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/verify.png"),
        help="Output image path",
    )
    verify_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for OSM data cache files",
    )

    preview_parser = subparsers.add_parser("preview", help="Launch interactive pygame preview")
    preview_parser.add_argument(
        "--quality",
        default="low",
        choices=["low", "medium", "high"],
        help="Quality preset for screenshots (default: low)",
    )
    preview_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for OSM data cache files",
    )
    preview_parser.add_argument(
        "--no-lights",
        action="store_true",
        default=False,
        help="Disable light rendering (faster preview)",
    )

    render_parser = subparsers.add_parser(
        "render", help="Render a perspective road image"
    )
    render_parser.add_argument(
        "--quality",
        default="low",
        choices=["low", "medium", "high"],
        help="Quality preset (default: low)",
    )
    render_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/render.png"),
        help="Output image path (default: output/render.png)",
    )
    render_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for OSM data cache files",
    )
    render_parser.add_argument(
        "--no-lights",
        action="store_true",
        default=False,
        help="Disable illumination and light passes (faster render, no light effects)",
    )

    subparsers.add_parser(
        "devmode",
        help="Launch interactive devmode parameter tuning tool",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "verify":
        cmd_verify(config, args.output, args.cache_dir)
    elif args.command == "preview":
        cmd_preview(args, config)
    elif args.command == "render":
        cmd_render(config, args.quality, args.output, args.cache_dir, args.no_lights)
    elif args.command == "devmode":
        from .devmode.cmd_devmode import cmd_devmode
        cmd_devmode(args, config)


def cmd_verify(config: SceneConfig, output_path: Path, cache_dir: Path) -> None:
    """Load OSM data and produce a verification plot.

    Downloads (or loads cached) road, land use, and building data, classifies
    roads by type, and saves a matplotlib figure to output_path.

    Args:
        config (SceneConfig): Loaded scene configuration.
        output_path (Path): Where to save the output PNG.
        cache_dir (Path): Directory for OSM data cache files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bbox = _bounds_to_bbox(config.bounds)

    print("Loading road network...")
    roads_raw = load_roads(bbox, cache_dir)
    roads = classify_roads(roads_raw)

    print("Loading land use data...")
    land_use = load_land_use(bbox, cache_dir)

    print("Loading building footprints...")
    buildings = load_buildings(bbox, cache_dir)

    print("Generating verification plot...")
    save_verification_plot(config, roads, land_use, buildings, output_path)
    print(f"Saved to {output_path}")


def cmd_preview(args: argparse.Namespace, config: SceneConfig) -> None:
    """Launch interactive pygame preview.

    Loads road data, builds a Scene with spatial index, generates (or loads
    cached) a flat top-down road map, then runs a pygame event loop with
    camera controls and a HUD overlay. Each frame uses a perspective warp of
    the flat map instead of per-segment projection for high frame rates.

    Press P to save a screenshot, R to reset camera, C to save camera position
    to the TOML config, ESC to quit.

    Args:
        args (argparse.Namespace): Parsed CLI args (quality, cache_dir).
        config (SceneConfig): Loaded scene configuration.
    """
    import pygame
    from .rendering.pygame_backend import PygameBackend
    from .rendering.flat_map import get_flat_map
    from .rendering.render_pipeline import get_preview_passes_warp
    from .camera.controls import handle_input, render_hud, MouseDragState

    cache_dir = getattr(args, "cache_dir", Path("cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_scene(config, args.quality, cache_dir)

    print("Generating flat map for preview...")
    flat_map, flat_meta = get_flat_map(scene, cache_dir)
    print(f"Flat map ready: {flat_map.size[0]}x{flat_map.size[1]}px")

    warp_passes = get_preview_passes_warp(flat_map, flat_meta)

    pygame.init()
    preview_width, preview_height = 1280, 720
    screen = pygame.display.set_mode((preview_width, preview_height))
    pygame.display.set_caption("Streets At Night — Preview")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    original_position = scene.camera.position.copy()
    original_target = scene.camera.target.copy()
    original_fov = scene.camera.fov_deg

    backend = PygameBackend(screen)
    drag_state = MouseDragState()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()
        held_keys = pygame.key.get_pressed()

        result = handle_input(
            events, scene.camera, dt, held_keys, drag_state,
            original_position, original_target, original_fov,
        )

        if result.quit:
            running = False
        if result.reset:
            scene.camera.position = original_position.copy()
            scene.camera.target = original_target.copy()
            scene.camera.fov_deg = original_fov
            drag_state.clear()
        if result.screenshot:
            _save_screenshot(scene, args.quality, config)
        if result.save_camera:
            _save_camera_position(scene, args.config)

        for car_pair in scene.car_pairs:
            car_pair.resolve_facing(scene.camera.get_forward())

        stats = render_scene_frame(scene, backend, passes=warp_passes)
        fps = clock.get_fps()
        render_hud(screen, scene.camera, fps, stats.get("segments_drawn", 0), font)
        backend.present()

    pygame.quit()


def cmd_render(
    config: SceneConfig,
    quality: str,
    output_path: Path,
    cache_dir: Path,
    no_lights: bool = False,
) -> None:
    """Render a perspective road image from the configured camera.

    Loads and classifies road data, builds a fully populated Scene (roads,
    zones, and lights), resolves car headlight/taillight facing once for
    the fixed camera, and saves a PNG to output_path using the Pillow backend.

    Args:
        config (SceneConfig): Loaded scene configuration.
        quality (str): Quality preset name (e.g. "low", "medium", "high").
        output_path (Path): Where to save the rendered PNG.
        cache_dir (Path): Directory for OSM data cache files.
        no_lights (bool): When True, omit illumination and light passes.
            Defaults to False.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preset = config.quality_presets[quality]
    scene = _load_scene(config, quality, cache_dir)

    print(f"Rendering at {preset.width}×{preset.height}...")
    backend = PillowBackend(preset.width, preset.height)
    export_passes = get_export_passes(cache_dir, no_lights=no_lights)

    for car_pair in scene.car_pairs:
        car_pair.resolve_facing(scene.camera.get_forward())

    render_scene_frame(scene, backend, passes=export_passes)
    image = backend.get_image()
    image.save(str(output_path))
    print(f"Done. Output: {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_scene(
    config: SceneConfig,
    quality: str,
    cache_dir: Path,
) -> Scene:
    """Load OSM data and build a fully populated Scene.

    Downloads (or loads cached) road, land use, and building data, classifies
    roads, creates the coordinate transformer and camera, then delegates to
    build_scene which populates road segments, zone lists, and all lights.

    Args:
        config (SceneConfig): Loaded scene configuration.
        quality (str): Quality preset name for aspect ratio.
        cache_dir (Path): Directory for OSM data cache files.

    Returns:
        Scene: Fully constructed scene with camera, spatial index, zones,
            and all light sources placed.
    """
    preset = config.quality_presets[quality]
    bbox = _bounds_to_bbox(config.bounds)

    print("Loading road network...")
    roads_raw = load_roads(bbox, cache_dir)
    roads = classify_roads(roads_raw)
    print(f"  {len(roads)} road segments loaded.")

    origin_lat, origin_lon = _compute_origin(config)
    print(f"Coordinate origin: {origin_lat:.4f}N, {origin_lon:.4f}W")
    transformer = create_transformer(origin_lat, origin_lon)

    cam_pos = _camera_position_enu(config, transformer)
    target_pos = _target_position_enu(config, transformer)
    aspect_ratio = preset.width / preset.height

    camera = create_camera(
        position=cam_pos,
        target=target_pos,
        fov_deg=config.camera.fov,
        aspect_ratio=aspect_ratio,
        near=config.camera.near,
        far=config.camera.far,
    )
    print(
        f"Camera at ENU {cam_pos}, looking at {target_pos}, "
        f"FOV {config.camera.fov}deg"
    )

    print("Loading land use data...")
    land_use = load_land_use(bbox, cache_dir)
    print(f"  {len(land_use)} land use features loaded.")

    print("Loading building footprints...")
    buildings = load_buildings(bbox, cache_dir)
    print(f"  {len(buildings)} building footprints loaded.")

    print("Building scene and spatial index...")
    scene = build_scene(roads, transformer, camera, config, land_use, buildings)
    print(f"  {len(scene.road_segments)} segments indexed.")
    return scene


def _save_screenshot(scene: Scene, quality: str, config: SceneConfig) -> None:
    """Render a high-res screenshot via the Pillow backend and save it.

    Args:
        scene (Scene): The current world scene.
        quality (str): Quality preset name for resolution.
        config (SceneConfig): Loaded scene configuration.
    """
    preset = config.quality_presets.get(quality, config.quality_presets["low"])
    pillow_backend = PillowBackend(preset.width, preset.height)
    render_scene_frame(scene, pillow_backend)
    img = pillow_backend.get_image()
    screenshot_path = Path(f"output/screenshot_{int(time.time())}.png")
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(screenshot_path))
    print(f"Screenshot saved: {screenshot_path}")


def _save_camera_position(scene: Scene, toml_path: Path) -> None:
    """Save current camera position and target to the TOML config file.

    Converts camera ENU position and target back to WGS84 lat/lon and altitude
    in feet, then updates the [camera] section in the TOML file. All other
    sections are preserved.

    Prints an error if the scene has no transformer (camera cannot be converted).

    Args:
        scene (Scene): The current world scene (must have transformer set).
        toml_path (Path): Path to the TOML config file to update.
    """
    if scene.transformer is None:
        print("Cannot save camera: scene has no coordinate transformer.")
        return

    camera_data = _build_camera_toml_dict(scene)
    _write_camera_to_toml(toml_path, camera_data)
    print(f"Camera position saved to {toml_path}")


def _build_camera_toml_dict(scene: Scene) -> t.Dict[str, float]:
    """Convert the current camera ENU state to a TOML-ready dict.

    Args:
        scene (Scene): The current world scene with transformer set.

    Returns:
        dict[str, float]: Camera fields ready to write into the TOML [camera] section.
    """
    camera = scene.camera
    transformer = scene.transformer

    pos = camera.position
    lat, lon, alt_m = enu_to_wgs84(transformer, pos[0], pos[1], pos[2])
    alt_ft = meters_to_feet(alt_m)

    tgt = camera.target
    target_lat, target_lon, target_alt_m = enu_to_wgs84(
        transformer, tgt[0], tgt[1], tgt[2]
    )
    target_alt_ft = meters_to_feet(target_alt_m)

    return {
        "lat": lat,
        "lon": lon,
        "altitude_ft": alt_ft,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "target_altitude_ft": target_alt_ft,
        "fov": camera.fov_deg,
    }


def _write_camera_to_toml(toml_path: Path, camera_data: t.Dict[str, float]) -> None:
    """Read the existing TOML file, replace [camera] values, and write it back.

    Args:
        toml_path (Path): Path to the TOML config file.
        camera_data (dict[str, float]): Updated camera field values.
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    data["camera"].update(camera_data)

    with open(toml_path, "wb") as f:
        tomli_w.dump(data, f)


def _bounds_to_bbox(bounds: BoundsConfig) -> tuple:
    """Convert a BoundsConfig to a (north, south, east, west) tuple.

    Args:
        bounds (BoundsConfig): Geographic bounds from config.

    Returns:
        tuple: (north, south, east, west) floats.
    """
    return (bounds.north, bounds.south, bounds.east, bounds.west)


def _compute_origin(config: SceneConfig) -> t.Tuple[float, float]:
    """Compute the local coordinate origin as the midpoint between camera and target.

    Args:
        config (SceneConfig): Scene configuration with camera section.

    Returns:
        tuple[float, float]: (origin_lat, origin_lon) in WGS84 degrees.
    """
    origin_lat = (config.camera.lat + config.camera.target_lat) / 2.0
    origin_lon = (config.camera.lon + config.camera.target_lon) / 2.0
    return origin_lat, origin_lon


def _camera_position_enu(
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


def _target_position_enu(
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


if __name__ == "__main__":
    main()
