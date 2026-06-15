"""Investigation 05+06 — Integrated heat haze prototype.

Combines:
  - Per-pixel depth map (from inv 01)
  - Multi-layer Kolmogorov phase screens (FFT-filtered noise, from inv 04)
  - Frozen-flow advection (sliding window, multi-direction)
  - Torch grid_sample MPS warping (from inv 03)

Modes:
  - "still"    save first/middle/last frames as PNG for review
  - "video"    save 240 frames as a 4-second 60fps MP4 (or write GIF if no ffmpeg)
  - "perf"     measure end-to-end fps over 200 frames

Usage:
    uv run python investigation/05_end_to_end.py [still|video|perf]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from streets_at_night.config.loader import load_config  # noqa: E402
from streets_at_night.camera.camera import create_camera  # noqa: E402
from streets_at_night.camera.coordinates import (  # noqa: E402
    create_transformer,
    feet_to_meters,
    wgs84_to_enu,
)
from streets_at_night.data.osm_loader import load_roads  # noqa: E402
from streets_at_night.data.road_classifier import classify_roads  # noqa: E402
from streets_at_night.placement.scene_builder import build_scene  # noqa: E402
from streets_at_night.rendering.flat_map import get_flat_map  # noqa: E402
from streets_at_night.rendering.warp_preview import _build_ray_directions  # noqa: E402
from streets_at_night.rendering.heat_haze import HazeField as _ProdHazeField  # noqa: E402
from streets_at_night.config.schema import HeatHazeLayerConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "investigation" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Phase screen generator (FFT-filtered Von Karman noise)
# ---------------------------------------------------------------------------


def kolmogorov_screen_fft(
    height: int, width: int, pixel_scale_m: float, r0: float, L0: float, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(height, d=pixel_scale_m)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_m)[None, :]
    k2 = (2 * np.pi * fx) ** 2 + (2 * np.pi * fy) ** 2
    psd = 0.023 * (r0 ** (-5.0 / 3.0)) * (k2 + (1.0 / L0) ** 2) ** (-11.0 / 6.0)
    psd[0, 0] = 0.0
    noise = rng.standard_normal((height, width)) + 1j * rng.standard_normal((height, width))
    field_F = noise * np.sqrt(psd) * np.sqrt(height * width) * pixel_scale_m
    return np.fft.ifft2(field_F).real.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth map (clones logic from investigation 01)
# ---------------------------------------------------------------------------


def compute_depth_map(
    camera, flat_meta, flat_size: tuple[int, int], max_haze_distance_m: float = 20000.0,
) -> tuple[np.ndarray, int, float]:
    """Return (depth_clamped_meters, horizon_row, max_haze_distance).

    The depth normalization is driven by the atmospheric fade distance — beyond
    `max_haze_distance_m` the existing atmospheric pass already fades the scene
    to haze, so heat shimmer should reach its maximum at that distance and
    saturate there. Trying to use a depth-array-derived "horizon" fails: with
    a slightly downward-tilted camera the topmost ground row contains rays
    whose t_val is in the millions of meters (asymptote at the actual horizon),
    so any percentile/median is dominated by that asymptote.
    """
    cam_pos = camera.position
    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())
    rays = _build_ray_directions(W, H, inv_proj, inv_view)
    dz = rays[:, :, 2]
    hits = np.abs(dz) > 1e-10
    t_val = np.where(hits, -cam_pos[2] / np.where(hits, dz, 1.0), -1.0)
    hits = hits & (t_val > 0.0)
    depth = np.where(hits, t_val, np.nan).astype(np.float32)

    row_hit_frac = hits.mean(axis=1)
    rows_majority = np.where(row_hit_frac >= 0.5)[0]
    horizon_row = int(rows_majority.min()) if rows_majority.size else int(np.argmax(row_hit_frac))

    clamped = np.where(np.isfinite(depth), depth, max_haze_distance_m).astype(np.float32)
    clamped = np.minimum(clamped, max_haze_distance_m)
    return clamped, horizon_row, float(max_haze_distance_m)


def build_distance_scale(depth: np.ndarray, horizon_max: float, exponent: float) -> np.ndarray:
    """Normalize depth to [0, 1] scale and apply exponent. Per-pixel float32."""
    norm = np.clip(depth / horizon_max, 0.0, 1.0)
    return (norm ** exponent).astype(np.float32)


# ---------------------------------------------------------------------------
# Frozen-flow displacement field — multi-layer
# ---------------------------------------------------------------------------


class HazeField:
    """Multi-layer atmospheric displacement field with frozen-flow animation."""

    def __init__(
        self,
        height: int,
        width_screen: int,
        out_width: int,
        layers: list[dict],
        amplitude_px: float,
        seed: int,
    ):
        self.h = height
        self.w_screen = width_screen
        self.w_out = out_width
        self.amplitude_px = amplitude_px
        self.layers = []
        for i, lc in enumerate(layers):
            scr_dx = kolmogorov_screen_fft(
                height, width_screen, lc["pixel_scale_m"], lc["r0"], lc["L0"], seed + 2 * i,
            )
            scr_dy = kolmogorov_screen_fft(
                height, width_screen, lc["pixel_scale_m"], lc["r0"], lc["L0"], seed + 2 * i + 1,
            )
            # Normalize each layer's std to its weight
            scr_dx = scr_dx / (scr_dx.std() + 1e-9) * lc["weight"]
            scr_dy = scr_dy / (scr_dy.std() + 1e-9) * lc["weight"]
            self.layers.append({
                "dx": scr_dx,
                "dy": scr_dy,
                "wind_x": lc["wind_x_pxs"],
                "wind_y": lc["wind_y_pxs"],
            })

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return per-pixel (dx, dy) at time t (seconds), summed across layers."""
        dx_total = np.zeros((self.h, self.w_out), dtype=np.float32)
        dy_total = np.zeros((self.h, self.w_out), dtype=np.float32)
        for lay in self.layers:
            x_off = (lay["wind_x"] * t) % self.w_screen
            y_off = (lay["wind_y"] * t)  # vertical advection, no wrap (small drift)
            dx_total += self._slide(lay["dx"], x_off, y_off)
            dy_total += self._slide(lay["dy"], x_off, y_off)
        # Renormalize total to amplitude_px scale
        # amplitude_px should be the std at distance_scale=1.0 (horizon)
        dx_total *= self.amplitude_px
        dy_total *= self.amplitude_px
        return dx_total, dy_total

    def _slide(self, screen: np.ndarray, x_off: float, y_off: float) -> np.ndarray:
        """Bilinear horizontal slide with vertical drift wrap."""
        H, W = screen.shape
        x_off = x_off % W
        i0 = int(np.floor(x_off))
        fx = x_off - i0
        cols = (np.arange(self.w_out) + i0) % W
        cols_p1 = (cols + 1) % W

        # Vertical: cyclic indices via wrap
        y_off_pix = y_off % H
        j0 = int(np.floor(y_off_pix))
        fy = y_off_pix - j0
        rows = (np.arange(H) + j0) % H
        rows_p1 = (rows + 1) % H

        a = screen[rows][:, cols]
        b = screen[rows][:, cols_p1]
        c = screen[rows_p1][:, cols]
        d = screen[rows_p1][:, cols_p1]
        return ((1 - fx) * (1 - fy) * a + fx * (1 - fy) * b
                + (1 - fx) * fy * c + fx * fy * d).astype(np.float32)


# ---------------------------------------------------------------------------
# Torch grid_sample warp (preloaded base image, dx/dy uploaded each frame)
# ---------------------------------------------------------------------------


class TorchWarper:
    def __init__(self, frame: np.ndarray, device: str):
        self.device = device
        self.h, self.w = frame.shape[:2]
        self.img_t = (
            torch.from_numpy(frame).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        )
        self.sx = torch.arange(self.w, device=device, dtype=torch.float32).unsqueeze(0)
        self.sy = torch.arange(self.h, device=device, dtype=torch.float32).unsqueeze(1)

    def warp(self, dx_np: np.ndarray, dy_np: np.ndarray) -> np.ndarray:
        dx = torch.from_numpy(np.ascontiguousarray(dx_np)).to(self.device)
        dy = torch.from_numpy(np.ascontiguousarray(dy_np)).to(self.device)
        src_x = (self.sx + dx).clamp(0, self.w - 1)
        src_y = (self.sy + dy).clamp(0, self.h - 1)
        gx = (src_x / (self.w - 1)) * 2.0 - 1.0
        gy = (src_y / (self.h - 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        out = F.grid_sample(self.img_t, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)
        return (out.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


# ---------------------------------------------------------------------------
# Pipeline: render base frame from scene, build depth map, animate
# ---------------------------------------------------------------------------


def load_scene_and_base_frame() -> tuple[np.ndarray, np.ndarray, int, float]:  # noqa: D401
    """Load scene, build flat map, render the warp-preview base frame, compute depth.

    Returns (base_frame, depth_clamped_m, horizon_row, horizon_max_m).
    """
    config = load_config(REPO_ROOT / "scene_config.toml")
    bbox = (config.bounds.north, config.bounds.south, config.bounds.east, config.bounds.west)
    cache_dir = REPO_ROOT / "cache"

    print("Loading roads...")
    roads = classify_roads(load_roads(bbox, cache_dir))

    origin_lat = (config.camera.lat + config.camera.target_lat) / 2.0
    origin_lon = (config.camera.lon + config.camera.target_lon) / 2.0
    transformer = create_transformer(origin_lat, origin_lon)
    cam_alt = feet_to_meters(config.camera.altitude_ft)
    tgt_alt = feet_to_meters(config.camera.target_altitude_ft)
    cam_e, cam_n, cam_u = wgs84_to_enu(transformer, config.camera.lat, config.camera.lon, cam_alt)
    tgt_e, tgt_n, tgt_u = wgs84_to_enu(
        transformer, config.camera.target_lat, config.camera.target_lon, tgt_alt
    )
    cam_pos = np.array([cam_e, cam_n, cam_u])
    tgt_pos = np.array([tgt_e, tgt_n, tgt_u])
    aspect = W / H
    camera = create_camera(
        position=cam_pos, target=tgt_pos, fov_deg=config.camera.fov,
        aspect_ratio=aspect, near=config.camera.near, far=config.camera.far,
    )
    print("Building scene + flat map (cached)...")
    scene = build_scene(roads, transformer, camera, config, None, None)
    flat_map, flat_meta = get_flat_map(scene, cache_dir)

    # Render base frame using the existing warp_preview machinery
    from streets_at_night.rendering.pillow_backend import PillowBackend
    from streets_at_night.rendering.render_pipeline import (
        render_scene_frame, get_preview_passes_warp,
    )
    print("Rendering base frame at 1280x720...")
    backend = PillowBackend(W, H)
    passes = get_preview_passes_warp(flat_map, flat_meta)
    render_scene_frame(scene, backend, passes=passes)
    base_frame = np.asarray(backend.get_image()).copy()

    depth, horizon_row, horizon_max = compute_depth_map(camera, flat_meta, flat_map.size)
    return base_frame, depth, horizon_row, horizon_max


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "still"
    print(f"Mode: {mode}, device: {DEVICE}")

    # Load scene
    base_frame, depth, horizon_row, horizon_max = load_scene_and_base_frame()
    Image.fromarray(base_frame).save(OUT_DIR / "base_frame.png")
    print(f"Base frame saved. Horizon at y={horizon_row}, max depth {horizon_max:.0f} m")

    # Build distance scale (per-pixel, in [0, 1]^exponent)
    distance_scale = build_distance_scale(depth, horizon_max, exponent=1.5)
    print(f"distance_scale: median {np.median(distance_scale):.3f}, "
          f"max {distance_scale.max():.3f}")

    # Configure 3 atmospheric layers.
    #
    # KEY INSIGHT: outer scale L0 must be << image extent in physical units,
    # otherwise the visible region sits in the PSD's flat low-k plateau and
    # the whole image bulges/contracts as one big eddy. With pixel_scale_m =
    # eddy outer-scale-in-pixels = L0 / pixel_scale_m, we want eddy sizes
    # roughly 5-50 image pixels (small-scale shimmer, not global motion).
    #
    # Each layer's eddy_max_px = L0 / pixel_scale_m:
    #   layer 1: 5.0/0.5 = 10 px  (medium eddies, slowest)
    #   layer 2: 2.0/0.2 = 10 px  (fine eddies, medium speed)
    #   layer 3: 0.6/0.1 = 6  px  (very fine, fast)
    # Bottom-to-top motion (rising heat): winds dominate the y axis.
    layers = [
        HeatHazeLayerConfig(r0=0.10, L0=2.5, pixel_scale_m=0.5, weight=0.6,
                            wind_x_pxs=4.0, wind_y_pxs=35.0),
        HeatHazeLayerConfig(r0=0.10, L0=1.0, pixel_scale_m=0.2, weight=0.8,
                            wind_x_pxs=-3.0, wind_y_pxs=70.0),
        HeatHazeLayerConfig(r0=0.10, L0=0.3, pixel_scale_m=0.1, weight=1.0,
                            wind_x_pxs=8.0, wind_y_pxs=120.0),
    ]

    print("Building 3-layer phase screens (this is one-time at startup)...")
    t0 = time.perf_counter()
    haze = _ProdHazeField(
        out_height=H, out_width=W, screen_height=4096, screen_width=2048,
        layers=layers, amplitude_px=0.5, seed=42,
    )
    t1 = time.perf_counter()
    print(f"Phase screen build time: {(t1 - t0) * 1000:.0f} ms")

    warper = TorchWarper(base_frame, DEVICE)

    if mode == "perf":
        n = 200
        if DEVICE == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        for f in range(n):
            t = f / 60.0
            dx, dy = haze.sample(t)
            dx *= distance_scale
            dy *= distance_scale
            _ = warper.warp(dx, dy)
        if DEVICE == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        per_frame = (t1 - t0) / n * 1000
        print(f"\nEnd-to-end {n} frames: {per_frame:.2f} ms/frame ({1000/per_frame:.1f} fps)")
        return

    if mode == "still":
        for label, t in [("t0.0", 0.0), ("t1.5", 1.5), ("t3.0", 3.0)]:
            dx, dy = haze.sample(t)
            dx *= distance_scale
            dy *= distance_scale
            print(f"  t={t}s  dx range [{dx.min():.2f}, {dx.max():.2f}], "
                  f"dy [{dy.min():.2f}, {dy.max():.2f}]")
            out = warper.warp(dx, dy)
            Image.fromarray(out).save(OUT_DIR / f"end_to_end_{label}.png")
        # Also save the baseline (current sine-based haze) for comparison
        from streets_at_night.config.schema import HeatHazeConfig
        from streets_at_night.rendering.warp_preview import apply_heat_haze
        cfg = HeatHazeConfig(
            enabled=True, amplitude=3.0, freq1=0.003, freq2=0.007,
            speed1=0.4, speed2=0.6, distance_exponent=1.5,
        )
        out_sine = apply_heat_haze(base_frame, t=1.5, config=cfg)
        Image.fromarray(out_sine).save(OUT_DIR / "end_to_end_sine_baseline.png")
        print(f"Saved stills + sine baseline to {OUT_DIR}/end_to_end_*.png")

    if mode == "video":
        # Make a short animation as a GIF (no ffmpeg dep)
        n = 90  # 3 seconds @ 30fps
        frames = []
        for f in range(n):
            t = f / 30.0
            dx, dy = haze.sample(t)
            dx *= distance_scale
            dy *= distance_scale
            out = warper.warp(dx, dy)
            # Downsample to keep GIF reasonable
            small = Image.fromarray(out).resize((640, 360), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE)
            frames.append(small)
        gif_path = OUT_DIR / "end_to_end_anim.gif"
        frames[0].save(
            gif_path, save_all=True, append_images=frames[1:],
            duration=int(1000 / 30), loop=0, optimize=True,
        )
        print(f"Saved {n}-frame animation: {gif_path}")


if __name__ == "__main__":
    main()
