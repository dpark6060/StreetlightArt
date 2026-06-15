"""Shared per-pixel ground raycast — the renderer's screen-space G-buffer.

Several passes need the same thing: for every screen pixel, the world (ENU)
position where its camera ray hits the z=0 ground plane. The ground export pass,
the illumination pass, and the warp/heat-haze paths all built this independently.
This module is the single implementation; ``build_ground_gbuffer`` is computed
once per frame and shared across passes that take the same camera and size.
"""

import typing as t
from dataclasses import dataclass

import numpy as np

if t.TYPE_CHECKING:
    from ..camera.camera import Camera


@dataclass
class GroundHit:
    """Per-pixel ground-plane intersection for a full screen.

    Attributes:
        ray_dirs: (H, W, 3) normalized world-space ray direction per pixel.
        hit_x: (H, W) ENU x of the z=0 intersection (undefined where not hit).
        hit_y: (H, W) ENU y of the z=0 intersection (undefined where not hit).
        ground_mask: (H, W) bool, True where the ray hits z=0 in front of camera.
    """

    ray_dirs: np.ndarray
    hit_x: np.ndarray
    hit_y: np.ndarray
    ground_mask: np.ndarray


def build_ray_directions(
    width: int,
    height: int,
    inv_proj: np.ndarray,
    inv_view: np.ndarray,
) -> np.ndarray:
    """Build a normalized world-space ray direction for every screen pixel.

    Constructs an NDC meshgrid for the full screen, transforms through the
    inverse projection matrix to view-space directions, then through the
    inverse view matrix to world-space directions, and normalizes.

    Args:
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        inv_proj (np.ndarray): (4, 4) inverse projection matrix.
        inv_view (np.ndarray): (4, 4) inverse view matrix.

    Returns:
        np.ndarray: (H, W, 3) normalized ray directions in world ENU space.
    """
    sx, sy = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    ndc_x = 2.0 * sx / width - 1.0
    ndc_y = 1.0 - 2.0 * sy / height

    n_pixels = width * height
    clip = np.stack([
        ndc_x.ravel(),
        ndc_y.ravel(),
        np.full(n_pixels, -1.0),
        np.ones(n_pixels),
    ], axis=1)

    eye = (inv_proj @ clip.T).T
    eye_dir = np.column_stack([
        eye[:, 0],
        eye[:, 1],
        np.full(n_pixels, -1.0),
        np.zeros(n_pixels),
    ])
    world_dir = (inv_view @ eye_dir.T).T[:, :3]

    norms = np.linalg.norm(world_dir, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    world_dir /= norms

    return world_dir.reshape(height, width, 3)


def build_ground_gbuffer(camera: "Camera", width: int, height: int) -> GroundHit:
    """Raycast every screen pixel to the z=0 ground plane.

    Builds the per-pixel ray directions, intersects each with z=0, and returns
    the hit positions plus a validity mask. Pixels whose ray is parallel to the
    ground or points away from it are masked out.

    Args:
        camera (Camera): Active camera (provides position + view + projection).
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.

    Returns:
        GroundHit: ray_dirs, hit_x, hit_y, and ground_mask for the screen.
    """
    cam_pos = camera.position
    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())
    ray_dirs = build_ray_directions(width, height, inv_proj, inv_view)

    dz = ray_dirs[:, :, 2]
    ray_hits = np.abs(dz) > 1e-10
    t_val = np.where(ray_hits, -cam_pos[2] / np.where(ray_hits, dz, 1.0), -1.0)
    ground_mask = ray_hits & (t_val > 0.0)

    hit_x = cam_pos[0] + t_val * ray_dirs[:, :, 0]
    hit_y = cam_pos[1] + t_val * ray_dirs[:, :, 1]

    return GroundHit(ray_dirs=ray_dirs, hit_x=hit_x, hit_y=hit_y, ground_mask=ground_mask)
