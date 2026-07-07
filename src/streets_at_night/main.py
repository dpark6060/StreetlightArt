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

import dataclasses
from dataclasses import dataclass

from PIL import Image

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
from .config.paths import compute_origin, resolve_cache_dir
from .config.schema import BoundsConfig, HeatHazeConfig, QualityPreset, SceneConfig
from .rendering.heat_haze import HeatHaze, compute_depth_map
from .data.osm_loader import load_buildings, load_land_use, load_roads
from .data.road_classifier import classify_roads
from .data.verify_plot import save_verification_plot
from .rendering.pillow_backend import PillowBackend
from .rendering.render_pipeline import (
    get_export_pass_names,
    get_export_passes,
    render_scene_frame,
    render_scene_frame_with_progress,
)
from .placement.scene_builder import build_scene
from .scene.scene import Scene

if t.TYPE_CHECKING:
    from .rendering.flat_map import FlatMapMeta
    from .rendering.raycast import GroundHit


@dataclass
class AnimationState:
    """Per-frame animation state for the interactive preview.

    base_frame holds the fully composited scene as a numpy array. It is
    rebuilt whenever the camera moves (base_frame_dirty=True) and reused
    otherwise — all per-frame work (heat haze) operates on a copy.
    """

    base_frame: np.ndarray
    base_frame_dirty: bool = False
    elapsed_t: float = 0.0


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
    render_parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override output width in pixels (overrides quality preset width)",
    )
    render_parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override output height in pixels (overrides quality preset height)",
    )
    render_parser.add_argument(
        "--format",
        choices=["png", "tiff"],
        default=None,
        help="Output format (default: auto-detect from output path extension)",
    )

    animate_parser = subparsers.add_parser(
        "animate",
        help="Animate heat haze over a pre-rendered image (live preview or offline MP4)",
    )
    animate_parser.add_argument(
        "--image",
        type=Path,
        default=Path("output/render.png"),
        help="Path to the base image to animate (default: output/render.png)",
    )
    animate_parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels. Pygame mode: display size (default 1280). "
             "Record mode: overrides the --quality preset width.",
    )
    animate_parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels. Pygame mode: display size (default 720). "
             "Record mode: overrides the --quality preset height.",
    )
    animate_parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Render an offline MP4 to this path instead of opening a pygame window.",
    )
    animate_parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="low",
        help="Quality preset for video resolution when --record is set (default: low). "
             "Ignored in pygame mode.",
    )
    animate_parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Video duration in seconds when --record is set "
             "(default: 2 cycles of the slowest haze layer).",
    )
    animate_parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for recorded video (default: 30).",
    )
    animate_parser.add_argument(
        "--seamless",
        action="store_true",
        default=False,
        help="Crossfade the end of the recorded video with the start so it loops seamlessly.",
    )
    animate_parser.add_argument(
        "--crossfade-seconds",
        type=float,
        default=0.5,
        help="Crossfade duration in seconds when --seamless is set (default: 0.5).",
    )

    devmode_parser = subparsers.add_parser(
        "devmode",
        help="Launch interactive devmode parameter tuning tool",
    )
    devmode_parser.add_argument(
        "--real-tiles",
        action="store_true",
        default=False,
        help="Load a strip of real OSM road data instead of the synthetic scene",
    )
    devmode_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Directory for OSM data cache files (used with --real-tiles)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "verify":
        cmd_verify(config, args.output, resolve_cache_dir(args.cache_dir, config))
    elif args.command == "preview":
        cmd_preview(args, config)
    elif args.command == "render":
        cmd_render(
            config,
            args.quality,
            args.output,
            resolve_cache_dir(args.cache_dir, config),
            args.no_lights,
            width_override=args.width,
            height_override=args.height,
            format_override=args.format,
        )
    elif args.command == "animate":
        cmd_animate(args, config)
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
    from .rendering.raycast import build_ground_gbuffer
    from .camera.controls import handle_input, render_hud, MouseDragState

    cache_dir = resolve_cache_dir(getattr(args, "cache_dir", Path("cache")), config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene = _load_scene(config, args.quality, cache_dir)

    print("Generating flat map for preview...")
    flat_map, flat_meta = get_flat_map(scene, cache_dir)
    flat_map_array = np.asarray(flat_map)
    print(f"Flat map ready: {flat_map.size[0]}x{flat_map.size[1]}px")

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

    print("Building initial base frame...")
    gbuffer = build_ground_gbuffer(scene.camera, preview_width, preview_height)
    anim = AnimationState(
        base_frame=_render_base_frame(
            scene, flat_map, flat_meta, flat_map_array,
            preview_width, preview_height, gbuffer,
        )
    )

    heat_haze_cfg = config.heat_haze
    haze = None
    if heat_haze_cfg is not None and heat_haze_cfg.enabled:
        haze = HeatHaze(preview_width, preview_height, heat_haze_cfg)
        haze.set_base_frame(anim.base_frame)
        haze.set_depth_map(compute_depth_map(
            scene.camera, preview_width, preview_height,
            heat_haze_cfg.max_distance_m, heat_haze_cfg.sky_distance_factor,
            gbuffer=gbuffer,
        ))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        events = pygame.event.get()
        held_keys = pygame.key.get_pressed()

        prev_pos = scene.camera.position.copy()
        prev_target = scene.camera.target.copy()
        prev_fov = scene.camera.fov_deg

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
            anim.base_frame_dirty = True
        if result.screenshot:
            _save_screenshot(scene, args.quality, config)
        if result.save_camera:
            _save_camera_position(scene, args.config)

        camera_moved = (
            not np.array_equal(scene.camera.position, prev_pos)
            or not np.array_equal(scene.camera.target, prev_target)
            or scene.camera.fov_deg != prev_fov
        )
        if camera_moved:
            anim.base_frame_dirty = True

        if anim.base_frame_dirty:
            gbuffer = build_ground_gbuffer(scene.camera, preview_width, preview_height)
            anim.base_frame = _render_base_frame(
                scene, flat_map, flat_meta, flat_map_array,
                preview_width, preview_height, gbuffer,
            )
            anim.base_frame_dirty = False
            if haze is not None:
                haze.set_base_frame(anim.base_frame)
                haze.set_depth_map(compute_depth_map(
                    scene.camera, preview_width, preview_height,
                    heat_haze_cfg.max_distance_m, heat_haze_cfg.sky_distance_factor,
                    gbuffer=gbuffer,
                ))

        anim.elapsed_t += dt

        for car_pair in scene.car_pairs:
            car_pair.resolve_facing(scene.camera.get_forward())

        if haze is not None:
            frame = haze.apply(anim.elapsed_t)
        else:
            frame = anim.base_frame

        backend.blit_array(frame)
        fps = clock.get_fps()
        render_hud(screen, scene.camera, fps, 0, font)
        backend.present()

    pygame.quit()


def cmd_animate(args: argparse.Namespace, config: SceneConfig) -> None:
    """Animate Kolmogorov heat haze distortion over a pre-rendered image.

    Two modes:
      * **Pygame mode** (default): runs a real-time interactive window.
      * **Record mode** (``--record FILE``): renders frames offline and writes
        an MP4 — used for high-resolution video that can't be played in real
        time.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        config (SceneConfig): Loaded scene configuration (camera + heat_haze).
    """
    image_path: Path = args.image
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Run `uv run streets-at-night render` first to produce a base image.")
        return

    heat_haze_cfg = config.heat_haze
    if heat_haze_cfg is None or not heat_haze_cfg.enabled:
        print("Heat haze is disabled in config. Set [heat_haze] enabled = true to see the effect.")
        return

    if args.record is not None:
        cmd_animate_record(args, config, heat_haze_cfg)
        return

    import pygame
    from .rendering.pygame_backend import PygameBackend

    display_width = args.width if args.width is not None else 1280
    display_height = args.height if args.height is not None else 720

    base_frame = _load_animate_base_frame(image_path, display_width, display_height)
    haze = _build_animate_haze(config, heat_haze_cfg, base_frame, display_width, display_height)

    pygame.init()
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("Streets At Night — Heat Haze")
    clock = pygame.time.Clock()
    backend = PygameBackend(screen)

    elapsed_t = 0.0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    _save_animate_frame(haze, elapsed_t)

        elapsed_t += dt
        frame = haze.apply(elapsed_t)
        backend.blit_array(frame)
        backend.present()

    pygame.quit()


def _load_animate_base_frame(
    image_path: Path, display_width: int, display_height: int,
) -> np.ndarray:
    """Load and resize the base image for ``cmd_animate``.

    Args:
        image_path (Path): Source image path.
        display_width (int): Target display width in pixels.
        display_height (int): Target display height in pixels.

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB frame at display size.
    """
    print(f"Loading base image: {image_path}")
    base_pil = Image.open(image_path).convert("RGB")
    if base_pil.size != (display_width, display_height):
        print(
            f"  Resizing {base_pil.size[0]}x{base_pil.size[1]} → "
            f"{display_width}x{display_height}"
        )
        base_pil = base_pil.resize((display_width, display_height), Image.LANCZOS)
    return np.asarray(base_pil).copy()


def _build_animate_haze(
    config: SceneConfig,
    heat_haze_cfg: HeatHazeConfig,
    base_frame: np.ndarray,
    display_width: int,
    display_height: int,
) -> HeatHaze:
    """Construct a HeatHaze instance for ``cmd_animate``.

    Builds a Camera from the scene config so the per-pixel depth map matches
    what the rendered image saw, computes that depth map at the display
    resolution, and uploads the base frame to the GPU.

    Args:
        config (SceneConfig): Full scene config (used for camera).
        heat_haze_cfg: Active HeatHazeConfig.
        base_frame (np.ndarray): (H, W, 3) base RGB frame.
        display_width (int): Target width.
        display_height (int): Target height.

    Returns:
        HeatHaze: Ready-to-use haze instance.
    """
    transformer = create_transformer(*compute_origin(config))
    camera = create_camera(
        position=_camera_position_enu(config, transformer),
        target=_target_position_enu(config, transformer),
        fov_deg=config.camera.fov,
        aspect_ratio=display_width / display_height,
        near=config.camera.near,
        far=config.camera.far,
    )
    depth = compute_depth_map(
        camera, display_width, display_height,
        heat_haze_cfg.max_distance_m, heat_haze_cfg.sky_distance_factor,
    )
    print("Building heat haze phase screens...")
    haze = HeatHaze(display_width, display_height, heat_haze_cfg)
    haze.set_base_frame(base_frame)
    haze.set_depth_map(depth)
    return haze


def cmd_animate_record(
    args: argparse.Namespace,
    config: SceneConfig,
    heat_haze_cfg: HeatHazeConfig,
) -> None:
    """Render an offline MP4 of heat-haze animation at high resolution.

    Used when ``--record FILE`` is supplied to ``animate``. Skips the pygame
    window entirely; renders frames sequentially and pipes them to ffmpeg via
    imageio. Memory-friendly: only the head buffer (for seamless crossfade)
    and one in-flight frame are held at a time.

    Args:
        args (argparse.Namespace): Parsed CLI args (image, record, quality,
            width, height, duration, fps, seamless, crossfade_seconds).
        config (SceneConfig): Loaded scene configuration.
        heat_haze_cfg: Active HeatHazeConfig (already verified non-null and enabled).
    """
    out_path: Path = args.record
    out_w, out_h = _resolve_record_dimensions(args, config)
    fps = max(1, int(args.fps))

    base_frame = _load_animate_base_frame(args.image, out_w, out_h)
    haze = _build_animate_haze(config, heat_haze_cfg, base_frame, out_w, out_h)

    # Duration is derived from the *built* haze so its loop period reflects the
    # effective (resolution-scaled) screens; computing it from raw config would
    # miss the loop point at any width != 1920 and break --seamless.
    duration = args.duration if args.duration is not None else _default_video_duration(haze)
    n_frames = max(1, int(round(duration * fps)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Recording {n_frames} frames @ {fps}fps ({duration:.1f}s) "
        f"at {out_w}x{out_h} → {out_path}"
    )

    if args.seamless:
        crossfade_frames = max(1, int(round(args.crossfade_seconds * fps)))
        crossfade_frames = min(crossfade_frames, n_frames // 2)
        _record_haze_seamless(haze, fps, n_frames, crossfade_frames, out_path)
    else:
        _record_haze(haze, fps, n_frames, out_path)

    print(f"Saved video: {out_path}")


def _resolve_record_dimensions(
    args: argparse.Namespace, config: SceneConfig,
) -> t.Tuple[int, int]:
    """Resolve output (width, height) for record mode.

    Explicit ``--width`` / ``--height`` override the ``--quality`` preset; if
    neither is supplied, the preset's dimensions are used.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        config (SceneConfig): Loaded scene configuration.

    Returns:
        tuple[int, int]: ``(width, height)`` in pixels.
    """
    preset = config.quality_presets[args.quality]
    width = args.width if args.width is not None else preset.width
    height = args.height if args.height is not None else preset.height
    return int(width), int(height)


def _default_video_duration(haze: HeatHaze) -> float:
    """Pick a default video duration covering one cyclic loop of the haze.

    Uses the haze's ``loop_period_s()`` — computed from the effective
    (resolution-scaled, clamped) phase screens and winds — so the default
    ``--duration`` lands exactly on the cycle boundary at any output width.
    Deriving it from raw config instead would miss the loop point by the
    resolution scale factor and make ``--seamless`` crossfade mid-cycle.

    Args:
        haze (HeatHaze): The built haze whose loop period sets the length.

    Returns:
        float: Default duration in seconds, no less than 4 s (10 s if the
        haze has no moving layers).
    """
    period = haze.loop_period_s()
    if period <= 0.0:
        return 10.0
    return max(4.0, period)


def _record_haze(haze: HeatHaze, fps: int, n_frames: int, out_path: Path) -> None:
    """Stream-render N frames of heat haze to an MP4 file.

    Args:
        haze (HeatHaze): Active haze pipeline (depth + base frame already set).
        fps (int): Frame rate.
        n_frames (int): Total frames to render.
        out_path (Path): Output MP4 path.
    """
    writer = _open_video_writer(out_path, fps)
    try:
        for i in range(n_frames):
            t = i / fps
            writer.append_data(haze.apply(t))
            _print_record_progress(i + 1, n_frames)
    finally:
        writer.close()


def _record_haze_seamless(
    haze: HeatHaze, fps: int, n_frames: int, crossfade_frames: int, out_path: Path,
) -> None:
    """Stream-render N frames with a crossfade tail blending back to frame 0.

    Pre-renders the first ``crossfade_frames`` into memory; while writing the
    final ``crossfade_frames`` of the output, blends each with its mirrored
    counterpart so that output[N-1] equals frame[0] and the loop is visually
    continuous.

    Args:
        haze (HeatHaze): Active haze pipeline.
        fps (int): Frame rate.
        n_frames (int): Total output frames.
        crossfade_frames (int): Length of the crossfade tail in frames.
        out_path (Path): Output MP4 path.
    """
    print(f"Pre-rendering {crossfade_frames} crossfade frames...")
    head = [haze.apply(i / fps) for i in range(crossfade_frames)]

    writer = _open_video_writer(out_path, fps)
    try:
        for i in range(n_frames):
            t = i / fps
            frame = haze.apply(t)
            if i >= n_frames - crossfade_frames:
                j = i - (n_frames - crossfade_frames)
                alpha = j / max(1, crossfade_frames - 1)
                mirror = head[crossfade_frames - 1 - j]
                frame = (
                    (1.0 - alpha) * frame.astype(np.float32)
                    + alpha * mirror.astype(np.float32)
                ).astype(np.uint8)
            writer.append_data(frame)
            _print_record_progress(i + 1, n_frames)
    finally:
        writer.close()


def _open_video_writer(out_path: Path, fps: int):
    """Open an imageio MP4 writer with H.264 + yuv420p settings.

    Returns:
        imageio writer: Object supporting ``append_data`` and ``close``.
    """
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-preset", "medium"],
    )


def _print_record_progress(done: int, total: int) -> None:
    """Print a compact progress line every ~5 % of total frames."""
    step = max(1, total // 20)
    if done % step == 0 or done == total:
        pct = 100.0 * done / total
        print(f"  {done}/{total} ({pct:.0f}%)")


def _save_animate_frame(haze: HeatHaze, elapsed_t: float) -> None:
    """Save the current heat haze frame to disk.

    Args:
        haze (HeatHaze): Active haze pipeline.
        elapsed_t (float): Current elapsed time for the haze effect.
    """
    frame = haze.apply(elapsed_t)
    out_path = Path(f"output/animate_{int(time.time())}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(str(out_path))
    print(f"Frame saved: {out_path}")


def cmd_render(
    config: SceneConfig,
    quality: str,
    output_path: Path,
    cache_dir: Path,
    no_lights: bool = False,
    width_override: t.Optional[int] = None,
    height_override: t.Optional[int] = None,
    format_override: t.Optional[str] = None,
) -> None:
    """Render a perspective road image from the configured camera.

    Loads and classifies road data, builds a fully populated Scene (roads,
    zones, and lights), resolves car headlight/taillight facing once for
    the fixed camera, then renders all passes with progress reporting.

    Args:
        config (SceneConfig): Loaded scene configuration.
        quality (str): Quality preset name (e.g. "low", "medium", "high").
        output_path (Path): Where to save the rendered image.
        cache_dir (Path): Directory for OSM data cache files.
        no_lights (bool): When True, omit illumination and light passes.
            Defaults to False.
        width_override (int | None): Override the preset width in pixels.
        height_override (int | None): Override the preset height in pixels.
        format_override (str | None): Force output format ("png" or "tiff").
            When None, format is auto-detected from the output path extension.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preset = config.quality_presets[quality]
    preset = _apply_resolution_overrides(preset, width_override, height_override)

    scene = _load_scene(config, quality, cache_dir)

    print(f"Rendering at {preset.width}×{preset.height}...")
    backend = PillowBackend(preset.width, preset.height)
    export_passes = get_export_passes(cache_dir, no_lights=no_lights, quality_preset=preset)
    pass_names = get_export_pass_names(no_lights=no_lights)

    for car_pair in scene.car_pairs:
        car_pair.resolve_facing(scene.camera.get_forward())

    render_scene_frame_with_progress(scene, backend, passes=export_passes, pass_names=pass_names)

    image = backend.get_image()
    output_format = _resolve_output_format(output_path, format_override)
    image.save(str(output_path), format=output_format)
    print(f"Done. Output: {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _render_base_frame(
    scene: Scene,
    flat_map: Image.Image,
    flat_meta: "FlatMapMeta",
    flat_map_array: np.ndarray,
    width: int,
    height: int,
    gbuffer: "GroundHit",
) -> np.ndarray:
    """Render one warp-preview frame and return the result as a numpy array.

    Uses a PillowBackend so the output can be converted to a (H, W, 3) uint8
    numpy array for per-frame manipulation. Calls render_warp_preview directly
    (rather than through the generic render-pass pipeline) so the caller's
    already-built GroundHit can be reused instead of raycasting again.

    Args:
        scene (Scene): The current world scene.
        flat_map (Image.Image): Pre-rendered top-down road map.
        flat_meta (FlatMapMeta): World bounds and resolution metadata.
        flat_map_array (np.ndarray): Pre-converted (H, W, 3) uint8 flat map array.
        width (int): Output width in pixels.
        height (int): Output height in pixels.
        gbuffer (GroundHit): Pre-computed ground raycast for the scene camera
            at this size, shared with the heat-haze depth computation.

    Returns:
        np.ndarray: (H, W, 3) uint8 array of the composited frame.
    """
    from .rendering.warp_preview import render_warp_preview

    pillow_backend = PillowBackend(width, height)
    render_warp_preview(
        scene, scene.camera, pillow_backend, flat_map, flat_meta,
        flat_map_array=flat_map_array, gbuffer=gbuffer,
    )
    return np.asarray(pillow_backend.get_image()).copy()


def _apply_resolution_overrides(
    preset: QualityPreset,
    width_override: t.Optional[int],
    height_override: t.Optional[int],
) -> QualityPreset:
    """Return a new preset with optional width and height overrides applied.

    Uses dataclasses.replace to avoid mutating the shared preset object.
    Only the provided dimensions are replaced; unspecified dimensions keep
    their preset values.

    Args:
        preset (QualityPreset): The base quality preset.
        width_override (int | None): New width in pixels, or None to keep preset.
        height_override (int | None): New height in pixels, or None to keep preset.

    Returns:
        QualityPreset: New preset with overrides applied.
    """
    kwargs: t.Dict[str, int] = {}
    if width_override is not None:
        kwargs["width"] = width_override
    if height_override is not None:
        kwargs["height"] = height_override
    if not kwargs:
        return preset
    return dataclasses.replace(preset, **kwargs)


def _resolve_output_format(output_path: Path, format_override: t.Optional[str]) -> str:
    """Determine the image format string for PIL Image.save.

    When format_override is provided, it takes precedence. Otherwise the format
    is inferred from the output path extension (.tiff/.tif → TIFF, otherwise PNG).

    Args:
        output_path (Path): The output file path.
        format_override (str | None): Explicit format name ("png" or "tiff"), or None.

    Returns:
        str: PIL format string, either "PNG" or "TIFF".
    """
    if format_override is not None:
        return "TIFF" if format_override == "tiff" else "PNG"
    suffix = output_path.suffix.lower()
    if suffix in (".tiff", ".tif"):
        return "TIFF"
    return "PNG"


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

    origin_lat, origin_lon = compute_origin(config)
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
