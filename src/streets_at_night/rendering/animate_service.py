"""Heat-haze animate service: base-frame loading, haze building, MP4 recording.

Extracted from ``main.py`` so both the CLI ``animate`` command and the UI
Animate tab can share the same logic. Every public helper takes an optional
``progress`` callback following the change-02 job pattern: when ``progress`` is
None the original ``print`` output is preserved byte-for-byte (so CLI behavior
is unchanged); otherwise progress is reported through the callback as
``(message, fraction)`` — loaders use ``fraction = -1.0`` (indeterminate),
recorders report ``done / total``.

This module must not import ``main`` (it is the other way around).
"""

import typing as t
from pathlib import Path

import numpy as np
from PIL import Image

from ..camera.camera import create_camera
from ..camera.coordinates import create_transformer
from ..config.paths import compute_origin
from ..config.schema import HeatHazeConfig, SceneConfig
from ..scene_loader import camera_position_enu, target_position_enu

if t.TYPE_CHECKING:
    from .heat_haze import HeatHaze

ProgressFn = t.Optional[t.Callable[[str, float], None]]


def _report(message: str, progress: ProgressFn) -> None:
    """Print the message (CLI) or forward it to the progress callback.

    Args:
        message (str): Human-readable progress message.
        progress (ProgressFn): Callback, or None to preserve CLI ``print``.
    """
    if progress is None:
        print(message)
    else:
        progress(message, -1.0)


def load_animate_base_frame(
    image_path: Path,
    display_width: int,
    display_height: int,
    progress: ProgressFn = None,
) -> np.ndarray:
    """Load and resize the base image for animation.

    Args:
        image_path (Path): Source image path.
        display_width (int): Target display width in pixels.
        display_height (int): Target display height in pixels.
        progress (ProgressFn): Optional progress callback (None => CLI prints).

    Returns:
        np.ndarray: (H, W, 3) uint8 RGB frame at display size.
    """
    _report(f"Loading base image: {image_path}", progress)
    base_pil = Image.open(image_path).convert("RGB")
    if base_pil.size != (display_width, display_height):
        _report(
            f"  Resizing {base_pil.size[0]}x{base_pil.size[1]} → "
            f"{display_width}x{display_height}",
            progress,
        )
        base_pil = base_pil.resize((display_width, display_height), Image.LANCZOS)
    return np.asarray(base_pil).copy()


def build_animate_haze(
    config: SceneConfig,
    heat_haze_cfg: HeatHazeConfig,
    base_frame: np.ndarray,
    display_width: int,
    display_height: int,
    progress: ProgressFn = None,
) -> "HeatHaze":
    """Construct a HeatHaze instance for animation.

    Builds a Camera from the scene config so the per-pixel depth map matches
    what the rendered image saw, computes that depth map at the display
    resolution, and uploads the base frame to the GPU.

    Args:
        config (SceneConfig): Full scene config (used for camera).
        heat_haze_cfg (HeatHazeConfig): Active heat haze config.
        base_frame (np.ndarray): (H, W, 3) base RGB frame.
        display_width (int): Target width.
        display_height (int): Target height.
        progress (ProgressFn): Optional progress callback (None => CLI prints).

    Returns:
        HeatHaze: Ready-to-use haze instance.
    """
    from .heat_haze import HeatHaze, compute_depth_map

    transformer = create_transformer(*compute_origin(config))
    camera = create_camera(
        position=camera_position_enu(config, transformer),
        target=target_position_enu(config, transformer),
        fov_deg=config.camera.fov,
        aspect_ratio=display_width / display_height,
        near=config.camera.near,
        far=config.camera.far,
    )
    depth = compute_depth_map(
        camera, display_width, display_height,
        heat_haze_cfg.max_distance_m, heat_haze_cfg.sky_distance_factor,
    )
    _report("Building heat haze phase screens...", progress)
    haze = HeatHaze(display_width, display_height, heat_haze_cfg)
    haze.set_base_frame(base_frame)
    haze.set_depth_map(depth)
    return haze


def default_video_duration(haze: "HeatHaze") -> float:
    """Pick a default video duration covering one cyclic loop of the haze.

    Uses the haze's ``loop_period_s()`` — computed from the effective
    (resolution-scaled, clamped) phase screens and winds — so the default
    duration lands exactly on the cycle boundary at any output width.

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


def record_haze(
    haze: "HeatHaze",
    fps: int,
    n_frames: int,
    out_path: Path,
    progress: ProgressFn = None,
) -> None:
    """Stream-render N frames of heat haze to an MP4 file.

    Args:
        haze (HeatHaze): Active haze pipeline (depth + base frame already set).
        fps (int): Frame rate.
        n_frames (int): Total frames to render.
        out_path (Path): Output MP4 path.
        progress (ProgressFn): Optional progress callback (None => CLI prints).
    """
    writer = open_video_writer(out_path, fps)
    try:
        for i in range(n_frames):
            t_sec = i / fps
            writer.append_data(haze.apply(t_sec))
            _report_record_progress(i + 1, n_frames, progress)
    finally:
        writer.close()


def record_haze_seamless(
    haze: "HeatHaze",
    fps: int,
    n_frames: int,
    crossfade_frames: int,
    out_path: Path,
    progress: ProgressFn = None,
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
        progress (ProgressFn): Optional progress callback (None => CLI prints).
    """
    _report(f"Pre-rendering {crossfade_frames} crossfade frames...", progress)
    head = [haze.apply(i / fps) for i in range(crossfade_frames)]

    writer = open_video_writer(out_path, fps)
    try:
        for i in range(n_frames):
            t_sec = i / fps
            frame = haze.apply(t_sec)
            if i >= n_frames - crossfade_frames:
                j = i - (n_frames - crossfade_frames)
                alpha = j / max(1, crossfade_frames - 1)
                mirror = head[crossfade_frames - 1 - j]
                frame = (
                    (1.0 - alpha) * frame.astype(np.float32)
                    + alpha * mirror.astype(np.float32)
                ).astype(np.uint8)
            writer.append_data(frame)
            _report_record_progress(i + 1, n_frames, progress)
    finally:
        writer.close()


def open_video_writer(out_path: Path, fps: int) -> t.Any:
    """Open an imageio MP4 writer with H.264 + yuv420p settings.

    Args:
        out_path (Path): Output MP4 path.
        fps (int): Frame rate.

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


def _report_record_progress(done: int, total: int, progress: ProgressFn) -> None:
    """Report record progress: CLI print cadence, or callback with the fraction.

    When ``progress`` is None the compact ~5%-cadence print is preserved so CLI
    output is unchanged; otherwise the fraction ``done / total`` is reported
    every frame so a progress bar advances smoothly.

    Args:
        done (int): Frames completed so far.
        total (int): Total frames to render.
        progress (ProgressFn): Optional progress callback.
    """
    if progress is None:
        step = max(1, total // 20)
        if done % step == 0 or done == total:
            pct = 100.0 * done / total
            print(f"  {done}/{total} ({pct:.0f}%)")
        return
    progress(f"{done}/{total} frames", done / total)
