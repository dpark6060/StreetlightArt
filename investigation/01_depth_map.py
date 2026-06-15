"""Investigation 01 — Extract a per-pixel depth map for the San Bernardino scene.

Mirrors the math in `rendering/warp_preview._raycast_to_flat_map` but ALSO retains
`t_val` (parametric ray distance, meters) as a depth map. Sky pixels are clamped
to the maximum ground-hit distance along the horizon row so they get a sensible
finite distance instead of infinity (per the investigation plan).

Outputs:
    investigation/out/depth_map.npy              raw float32 depth (H, W)
    investigation/out/depth_map_visualization.png pseudo-color visualization
    investigation/out/depth_stats.txt            min/max/median + horizon row info
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT_DIR = REPO_ROOT / "investigation" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PREVIEW_W, PREVIEW_H = 1280, 720


def raycast_depth(
    cam_pos: np.ndarray,
    ray_dirs: np.ndarray,
    width: int,
    height: int,
    flat_map_size: tuple[int, int],
    meta_min_x: float,
    meta_max_y: float,
    meta_resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raycast every pixel and return (flat_px, flat_py, ground_mask, depth_map_meters)."""
    dz = ray_dirs[:, :, 2]
    hits_ground = np.abs(dz) > 1e-10
    t_val = np.where(hits_ground, -cam_pos[2] / np.where(hits_ground, dz, 1.0), -1.0)
    hits_ground = hits_ground & (t_val > 0.0)

    world_x = cam_pos[0] + t_val * ray_dirs[:, :, 0]
    world_y = cam_pos[1] + t_val * ray_dirs[:, :, 1]

    flat_px_f = (world_x - meta_min_x) / meta_resolution
    flat_py_f = (meta_max_y - world_y) / meta_resolution

    fm_w, fm_h = flat_map_size
    in_bounds = (
        hits_ground
        & (flat_px_f >= 0)
        & (flat_px_f < fm_w)
        & (flat_py_f >= 0)
        & (flat_py_f < fm_h)
    )

    flat_px = np.clip(flat_px_f.astype(np.int32), 0, fm_w - 1)
    flat_py = np.clip(flat_py_f.astype(np.int32), 0, fm_h - 1)

    depth = t_val.astype(np.float32)
    depth = np.where(hits_ground, depth, np.float32(np.nan))

    return flat_px, flat_py, in_bounds, depth


def clamp_sky_to_horizon(depth: np.ndarray, ground_mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Replace NaN sky depths with the max ground depth along the horizon row.

    Returns (clamped_depth, horizon_row, horizon_max_depth).
    """
    H = depth.shape[0]
    horizon_row = -1
    for y in range(H - 1, -1, -1):
        if ground_mask[y, :].any():
            horizon_row_above = y
        else:
            break
    # The "horizon row" is the topmost row with any ground pixels.
    horizon_row = -1
    for y in range(H):
        if ground_mask[y, :].any():
            horizon_row = y
            break
    if horizon_row == -1:
        return depth, -1, 0.0
    row = depth[horizon_row]
    valid_row = row[np.isfinite(row)]
    horizon_max = float(np.nanmax(valid_row)) if valid_row.size else 0.0
    clamped = np.where(np.isfinite(depth), depth, horizon_max).astype(np.float32)
    return clamped, horizon_row, horizon_max


def main() -> None:
    config = load_config(REPO_ROOT / "scene_config.toml")

    bbox = (
        config.bounds.north,
        config.bounds.south,
        config.bounds.east,
        config.bounds.west,
    )
    cache_dir = REPO_ROOT / "cache"

    print("Loading roads...")
    roads_raw = load_roads(bbox, cache_dir)
    roads = classify_roads(roads_raw)

    origin_lat = (config.camera.lat + config.camera.target_lat) / 2.0
    origin_lon = (config.camera.lon + config.camera.target_lon) / 2.0
    transformer = create_transformer(origin_lat, origin_lon)

    cam_alt = feet_to_meters(config.camera.altitude_ft)
    tgt_alt = feet_to_meters(config.camera.target_altitude_ft)
    cam_e, cam_n, cam_u = wgs84_to_enu(transformer, config.camera.lat, config.camera.lon, cam_alt)
    tgt_e, tgt_n, tgt_u = wgs84_to_enu(
        transformer, config.camera.target_lat, config.camera.target_lon, tgt_alt
    )
    cam_pos = np.array([cam_e, cam_n, cam_u], dtype=np.float64)
    tgt_pos = np.array([tgt_e, tgt_n, tgt_u], dtype=np.float64)

    aspect = PREVIEW_W / PREVIEW_H
    camera = create_camera(
        position=cam_pos,
        target=tgt_pos,
        fov_deg=config.camera.fov,
        aspect_ratio=aspect,
        near=config.camera.near,
        far=config.camera.far,
    )
    print(f"Camera ENU pos: {cam_pos}, target: {tgt_pos}")
    print(f"Camera altitude (ENU z): {cam_pos[2]:.1f} m")

    print("Building scene + flat map (will be cached after first run)...")
    scene = build_scene(roads, transformer, camera, config, None, None)
    flat_map, flat_meta = get_flat_map(scene, cache_dir)
    print(f"Flat map: {flat_map.size}, resolution {flat_meta.resolution:.2f} m/px")

    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())

    print(f"Casting rays at {PREVIEW_W}x{PREVIEW_H}...")
    t0 = time.perf_counter()
    rays = _build_ray_directions(PREVIEW_W, PREVIEW_H, inv_proj, inv_view)
    flat_px, flat_py, ground_mask, depth = raycast_depth(
        cam_pos,
        rays,
        PREVIEW_W,
        PREVIEW_H,
        (flat_map.width, flat_map.height),
        flat_meta.min_x,
        flat_meta.max_y,
        flat_meta.resolution,
    )
    t1 = time.perf_counter()
    print(f"Raycast time: {(t1 - t0) * 1000:.1f} ms")

    clamped_depth, horizon_row, horizon_max = clamp_sky_to_horizon(depth, ground_mask)
    print(f"Topmost ground row (horizon): y={horizon_row}")
    print(f"Horizon row max depth: {horizon_max:.1f} m")

    finite = depth[np.isfinite(depth)]
    print("\n--- Ground-hit depth distribution (meters) ---")
    print(f"  count:  {finite.size} / {depth.size} ({100.0 * finite.size / depth.size:.1f}%)")
    print(f"  min:    {finite.min():.1f}")
    print(f"  median: {np.median(finite):.1f}")
    print(f"  mean:   {finite.mean():.1f}")
    print(f"  p90:    {np.percentile(finite, 90):.1f}")
    print(f"  p99:    {np.percentile(finite, 99):.1f}")
    print(f"  max:    {finite.max():.1f}")

    # Save raw .npy
    npy_path = OUT_DIR / "depth_map.npy"
    np.save(npy_path, clamped_depth)
    print(f"\nSaved depth map: {npy_path}")

    # Visualization — pseudo-color w/ horizon line
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].set_title("Depth (meters), sky clamped to horizon")
    im0 = axes[0].imshow(clamped_depth, cmap="viridis")
    axes[0].axhline(horizon_row, color="red", linewidth=0.5, alpha=0.7, label=f"horizon y={horizon_row}")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.colorbar(im0, ax=axes[0], label="meters")

    axes[1].set_title("Ground mask (white=ground hit)")
    axes[1].imshow(ground_mask, cmap="gray", vmin=0, vmax=1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "depth_map_visualization.png", dpi=120)
    print(f"Saved visualization: {OUT_DIR / 'depth_map_visualization.png'}")

    # Histogram of depth
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(finite, bins=80, color="#888")
    ax2.set_xlabel("depth (m)")
    ax2.set_ylabel("pixel count")
    ax2.set_title("Per-pixel ground-hit depth distribution")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "depth_histogram.png", dpi=120)
    print(f"Saved histogram: {OUT_DIR / 'depth_histogram.png'}")

    stats_path = OUT_DIR / "depth_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"size: {PREVIEW_W}x{PREVIEW_H}\n")
        f.write(f"camera_pos_enu: {cam_pos.tolist()}\n")
        f.write(f"camera_alt_m: {cam_pos[2]:.1f}\n")
        f.write(f"horizon_row_y: {horizon_row}\n")
        f.write(f"horizon_max_depth_m: {horizon_max:.1f}\n")
        f.write(f"ground_count: {finite.size}\n")
        f.write(f"sky_count: {depth.size - finite.size}\n")
        f.write(f"min_m: {finite.min():.1f}\n")
        f.write(f"median_m: {np.median(finite):.1f}\n")
        f.write(f"mean_m: {finite.mean():.1f}\n")
        f.write(f"p90_m: {np.percentile(finite, 90):.1f}\n")
        f.write(f"p99_m: {np.percentile(finite, 99):.1f}\n")
        f.write(f"max_m: {finite.max():.1f}\n")
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()
