"""Debug print: what does the per-row depth distribution look like?"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from streets_at_night.config.loader import load_config
from streets_at_night.camera.camera import create_camera
from streets_at_night.camera.coordinates import (
    create_transformer, feet_to_meters, wgs84_to_enu,
)
from streets_at_night.rendering.warp_preview import _build_ray_directions

W, H = 1280, 720
config = load_config(REPO_ROOT / "scene_config.toml")
origin_lat = (config.camera.lat + config.camera.target_lat) / 2.0
origin_lon = (config.camera.lon + config.camera.target_lon) / 2.0
transformer = create_transformer(origin_lat, origin_lon)
cam_alt = feet_to_meters(config.camera.altitude_ft)
tgt_alt = feet_to_meters(config.camera.target_altitude_ft)
cam_e, cam_n, cam_u = wgs84_to_enu(transformer, config.camera.lat, config.camera.lon, cam_alt)
tgt_e, tgt_n, tgt_u = wgs84_to_enu(transformer, config.camera.target_lat, config.camera.target_lon, tgt_alt)
cam_pos = np.array([cam_e, cam_n, cam_u])
tgt_pos = np.array([tgt_e, tgt_n, tgt_u])
camera = create_camera(
    position=cam_pos, target=tgt_pos, fov_deg=config.camera.fov,
    aspect_ratio=W / H, near=config.camera.near, far=config.camera.far,
)
print(f"Camera ENU: pos={cam_pos}, target={tgt_pos}")

inv_proj = np.linalg.inv(camera.get_projection_matrix())
inv_view = np.linalg.inv(camera.get_view_matrix())
rays = _build_ray_directions(W, H, inv_proj, inv_view)
dz = rays[:, :, 2]
hits = np.abs(dz) > 1e-10
t_val = np.where(hits, -cam_pos[2] / np.where(hits, dz, 1.0), -1.0)
hits = hits & (t_val > 0.0)
depth = np.where(hits, t_val, np.nan)

print("\nFine-grained transition rows:")
print("  row | hit_frac | dz_min   dz_max  | depth_min depth_med depth_max")
for y in range(245, 275):
    row_hits = hits[y]
    row_depth = depth[y]
    finite = row_depth[np.isfinite(row_depth)]
    dz_row = dz[y]
    print(f"  {y:3d} | {row_hits.mean():7.2%} "
          f"| {dz_row.min():7.4f} {dz_row.max():7.4f} | "
          f"{(finite.min() if finite.size else 0):10.0f} "
          f"{(np.median(finite) if finite.size else 0):10.0f} "
          f"{(finite.max() if finite.size else 0):10.0f}")

print("\nRow-by-row stats (every 30 rows):")
print("  row | hit_frac | dz_min   dz_max  | depth_min depth_med depth_max")
for y in range(0, H, 30):
    row_hits = hits[y]
    row_depth = depth[y]
    finite = row_depth[np.isfinite(row_depth)]
    dz_row = dz[y]
    print(f"  {y:3d} | {row_hits.mean():7.2%} "
          f"| {dz_row.min():7.4f} {dz_row.max():7.4f} | "
          f"{(finite.min() if finite.size else 0):8.0f} "
          f"{(np.median(finite) if finite.size else 0):8.0f} "
          f"{(finite.max() if finite.size else 0):8.0f}")
