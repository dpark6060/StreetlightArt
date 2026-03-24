"""Per-pixel raycast preview renderer.

For each screen pixel, casts a ray through the camera, intersects with the
z=0 ground plane, and looks up the corresponding flat-map pixel. Pixels whose
rays miss the ground (sky region) are filled with the sky color.

This approach handles shallow camera angles correctly — no degenerate
perspective transform, no clamping artifacts, no black bands near the horizon.
"""

import typing as t

import numpy as np
from PIL import Image

from ..camera.camera import Camera
from .backend import RenderBackend
from .flat_map import FlatMapMeta

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


_SKY_COLOR: t.Tuple[int, int, int] = (5, 5, 15)


def render_warp_preview(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    flat_map: Image.Image,
    flat_meta: FlatMapMeta,
) -> None:
    """Render one preview frame via per-pixel raycasting against the flat map.

    For each screen pixel, unprojects a ray through the camera, intersects it
    with the z=0 ground plane, converts the hit point to a flat-map pixel, and
    samples the flat map. Sky pixels (no ground hit) get the sky color.

    Args:
        scene (Scene): The world scene (provides camera context).
        camera (Camera): Active camera.
        backend (RenderBackend): Target render surface.
        flat_map (Image.Image): Pre-rendered top-down road map.
        flat_meta (FlatMapMeta): World bounds and resolution for the flat map.
    """
    width, height = backend.get_surface_size()
    inv_proj = np.linalg.inv(camera.get_projection_matrix())
    inv_view = np.linalg.inv(camera.get_view_matrix())
    cam_pos = camera.position

    ray_dirs = _build_ray_directions(width, height, inv_proj, inv_view)
    flat_px, flat_py, ground_mask = _raycast_to_flat_map(
        cam_pos, ray_dirs, width, height, flat_meta, flat_map
    )
    output = _sample_flat_map(flat_map, flat_px, flat_py, ground_mask, width, height)
    backend.blit_pil_image(output)


def _build_ray_directions(
    width: int,
    height: int,
    inv_proj: np.ndarray,
    inv_view: np.ndarray,
) -> np.ndarray:
    """Build a world-space ray direction for every screen pixel.

    Args:
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        inv_proj (np.ndarray): (4, 4) inverse projection matrix.
        inv_view (np.ndarray): (4, 4) inverse view matrix.

    Returns:
        np.ndarray: (H, W, 3) normalized ray directions in world space.
    """
    sx, sy = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    ndc_x = 2.0 * sx / width - 1.0
    ndc_y = 1.0 - 2.0 * sy / height

    n_pixels = width * height
    ndc_x_flat = ndc_x.ravel()
    ndc_y_flat = ndc_y.ravel()

    clip = np.stack([
        ndc_x_flat,
        ndc_y_flat,
        np.full(n_pixels, -1.0),
        np.ones(n_pixels),
    ], axis=1)

    eye = (inv_proj @ clip.T).T
    eye_dir = np.column_stack([eye[:, 0], eye[:, 1], np.full(n_pixels, -1.0), np.zeros(n_pixels)])

    world_dir = (inv_view @ eye_dir.T).T[:, :3]

    norms = np.linalg.norm(world_dir, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    world_dir /= norms

    return world_dir.reshape(height, width, 3)


def _raycast_to_flat_map(
    cam_pos: np.ndarray,
    ray_dirs: np.ndarray,
    width: int,
    height: int,
    meta: FlatMapMeta,
    flat_map: Image.Image,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect all rays with z=0 and convert to flat map pixel coordinates.

    Args:
        cam_pos (np.ndarray): (3,) camera position in world ENU.
        ray_dirs (np.ndarray): (H, W, 3) ray directions.
        width (int): Screen width.
        height (int): Screen height.
        meta (FlatMapMeta): Flat map coordinate metadata.
        flat_map (Image.Image): Flat map image (for bounds).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - flat_px: (H, W) int array of flat map x pixel coordinates.
            - flat_py: (H, W) int array of flat map y pixel coordinates.
            - ground_mask: (H, W) bool array, True where ray hits ground
              within the flat map bounds.
    """
    dz = ray_dirs[:, :, 2]
    hits_ground = np.abs(dz) > 1e-10
    t_val = np.where(hits_ground, -cam_pos[2] / np.where(hits_ground, dz, 1.0), -1.0)
    hits_ground = hits_ground & (t_val > 0.0)

    world_x = cam_pos[0] + t_val * ray_dirs[:, :, 0]
    world_y = cam_pos[1] + t_val * ray_dirs[:, :, 1]

    flat_px_f = (world_x - meta.min_x) / meta.resolution
    flat_py_f = (meta.max_y - world_y) / meta.resolution

    in_bounds = (
        hits_ground
        & (flat_px_f >= 0) & (flat_px_f < flat_map.width)
        & (flat_py_f >= 0) & (flat_py_f < flat_map.height)
    )

    flat_px = np.clip(flat_px_f.astype(np.int32), 0, flat_map.width - 1)
    flat_py = np.clip(flat_py_f.astype(np.int32), 0, flat_map.height - 1)

    return flat_px, flat_py, in_bounds


def _sample_flat_map(
    flat_map: Image.Image,
    flat_px: np.ndarray,
    flat_py: np.ndarray,
    ground_mask: np.ndarray,
    width: int,
    height: int,
) -> Image.Image:
    """Sample the flat map at the computed pixel coords and compose the output.

    Args:
        flat_map (Image.Image): Source flat road map.
        flat_px (np.ndarray): (H, W) flat map x pixel indices.
        flat_py (np.ndarray): (H, W) flat map y pixel indices.
        ground_mask (np.ndarray): (H, W) bool, True where ground is visible.
        width (int): Output width.
        height (int): Output height.

    Returns:
        Image.Image: Composed RGB output image.
    """
    src = np.asarray(flat_map)
    output = np.full((height, width, 3), _SKY_COLOR, dtype=np.uint8)
    output[ground_mask] = src[flat_py[ground_mask], flat_px[ground_mask]]
    return Image.fromarray(output, "RGB")
