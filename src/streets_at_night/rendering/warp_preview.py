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
from .raycast import GroundHit, build_ground_gbuffer
# Deprecated: kept only because tests import build_ray_directions under this
# historical private name. Use raycast.build_ray_directions directly in new code.
from .raycast import build_ray_directions as _build_ray_directions

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


_SKY_COLOR: t.Tuple[int, int, int] = (5, 5, 15)


def render_warp_preview(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    flat_map: Image.Image,
    flat_meta: FlatMapMeta,
    flat_map_array: t.Optional[np.ndarray] = None,
    gbuffer: t.Optional[GroundHit] = None,
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
        flat_map_array (np.ndarray | None): Pre-converted (H, W, 3) uint8 array
            of ``flat_map``. When omitted, converted from ``flat_map`` on the
            spot; callers that render every frame should convert once and pass
            it in to avoid a full-image copy per frame.
        gbuffer (GroundHit | None): Pre-computed ground raycast for this camera
            and surface size. When omitted, built internally. Callers that
            also need the depth map for the same camera position (e.g. heat
            haze) should build it once and pass it to both.
    """
    width, height = backend.get_surface_size()
    if gbuffer is None:
        gbuffer = build_ground_gbuffer(camera, width, height)
    if flat_map_array is None:
        flat_map_array = np.asarray(flat_map)

    flat_px, flat_py, ground_mask = _raycast_to_flat_map(
        gbuffer, flat_meta, flat_map.width, flat_map.height
    )
    output = _sample_flat_map(flat_map_array, flat_px, flat_py, ground_mask, width, height)
    backend.blit_pil_image(output)


def _raycast_to_flat_map(
    gbuffer: GroundHit,
    meta: FlatMapMeta,
    map_width: int,
    map_height: int,
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a ground raycast's hit points to flat map pixel coordinates.

    Args:
        gbuffer (GroundHit): Ground-plane raycast result for the screen.
        meta (FlatMapMeta): Flat map coordinate metadata.
        map_width (int): Flat map image width in pixels.
        map_height (int): Flat map image height in pixels.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - flat_px: (H, W) int array of flat map x pixel coordinates.
            - flat_py: (H, W) int array of flat map y pixel coordinates.
            - ground_mask: (H, W) bool array, True where ray hits ground
              within the flat map bounds.
    """
    flat_px_f, flat_py_f = meta.world_to_pixel(gbuffer.hit_x, gbuffer.hit_y)

    in_bounds = (
        gbuffer.ground_mask
        & (flat_px_f >= 0) & (flat_px_f < map_width)
        & (flat_py_f >= 0) & (flat_py_f < map_height)
    )

    flat_px = np.clip(flat_px_f.astype(np.int32), 0, map_width - 1)
    flat_py = np.clip(flat_py_f.astype(np.int32), 0, map_height - 1)

    return flat_px, flat_py, in_bounds


def _sample_flat_map(
    flat_map_array: np.ndarray,
    flat_px: np.ndarray,
    flat_py: np.ndarray,
    ground_mask: np.ndarray,
    width: int,
    height: int,
) -> Image.Image:
    """Sample the flat map at the computed pixel coords and compose the output.

    Args:
        flat_map_array (np.ndarray): (H_map, W_map, 3) uint8 source flat road map.
        flat_px (np.ndarray): (H, W) flat map x pixel indices.
        flat_py (np.ndarray): (H, W) flat map y pixel indices.
        ground_mask (np.ndarray): (H, W) bool, True where ground is visible.
        width (int): Output width.
        height (int): Output height.

    Returns:
        Image.Image: Composed RGB output image.
    """
    output = np.full((height, width, 3), _SKY_COLOR, dtype=np.uint8)
    output[ground_mask] = flat_map_array[flat_py[ground_mask], flat_px[ground_mask]]
    return Image.fromarray(output, "RGB")


