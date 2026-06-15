"""Illumination render pass — world-space ground-plane light pooling for export.

Builds a float32 illumination map in world (ENU) space by painting cone-model
blobs for each light source. The map is then sampled via a per-pixel raycast from
camera through screen pixels down to the z=0 ground plane, and the result is
additively composited onto the main image.

Each blob uses a two-zone cone model: an inner flat zone at full brightness up to
inner_fraction * radius, followed by a fade zone where intensity falls from 1 to 0
using (1 - fade_t)^falloff. The falloff exponent controls the curve steepness in the
fade zone; the inner fraction controls the size of the flat core.

This approach is physically grounded: illumination represents light reflecting off
the road surface in world space, so perspective foreshortening and occlusion by the
horizon are handled automatically by the raycast — no special-casing needed.
"""

import typing as t

import numpy as np

from ..camera.camera import Camera
from .backend import RenderBackend
from .color_space import srgb255_to_linear
from .raycast import GroundHit, build_ground_gbuffer
from .render_utils import apply_brightness_falloff, camera_distance

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Base illumination map resolution in meters per pixel.
# Lower values = finer map = slower but more detailed. 2.0 m/px balances quality and performance.
_BASE_ILLUMINATION_RESOLUTION: float = 2.0

# Padding added around road bounds so lights near the edges illuminate fully.
_BOUNDS_PADDING_M: float = 200.0


def render_illumination(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    illumination_resolution_multiplier: float = 1.0,
    gbuffer: t.Optional[GroundHit] = None,
) -> t.Dict[str, t.Any]:
    """Render ground-plane illumination from light sources via world-space map.

    Builds a float32 linear illumination map by painting per-light cone-model
    blobs at their world positions. Each blob's brightness is attenuated by
    exponential distance falloff. Each screen pixel is raycast to the z=0 ground
    plane; the hit world coordinate is sampled from the map and added to the
    backend's linear light buffer.

    Returns an empty dict if there are no lights.

    Args:
        scene (Scene): World scene containing lights, road segments, and config.
            Uses scene.config.atmosphere.falloff_distance for distance attenuation.
        camera (Camera): Active camera providing position and matrices.
        backend (RenderBackend): Target render surface (must support add_linear).
        illumination_resolution_multiplier (float): Quality multiplier from the
            active QualityPreset. Higher values produce a finer illumination map.
            Defaults to 1.0.
        gbuffer (GroundHit | None): Precomputed ground raycast for this camera and
            size. When None, it is computed here. The pipeline shares one G-buffer
            between this pass and the ground export pass to avoid raycasting twice.

    Returns:
        dict[str, Any]: Stats dict (currently empty, reserved for future use).
    """
    if not scene.lights:
        return {}

    illumination_brightness = scene.config.atmosphere.illumination_brightness
    falloff_distance = scene.config.atmosphere.falloff_distance
    resolution = _BASE_ILLUMINATION_RESOLUTION / illumination_resolution_multiplier

    bounds = _compute_illumination_bounds(scene)
    illum_map, map_info = _build_illumination_map(
        scene.lights, bounds, resolution, illumination_brightness,
        camera.position, falloff_distance,
    )

    width, height = backend.get_surface_size()
    if gbuffer is None:
        gbuffer = build_ground_gbuffer(camera, width, height)

    min_x, max_y, res = map_info
    illum_img = _sample_illumination_map(
        illum_map, gbuffer.hit_x, gbuffer.hit_y, gbuffer.ground_mask, min_x, max_y, res
    )

    backend.add_linear(illum_img)

    return {}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_illumination_bounds(
    scene: "Scene",
) -> t.Tuple[float, float, float, float]:
    """Compute world ENU bounds from all road segment geometries with padding.

    Iterates all road segments to find the min/max x and y extent, then
    adds _BOUNDS_PADDING_M on every side so lights at the edges are fully
    captured in the illumination map.

    Args:
        scene (Scene): Scene with road_segments containing geometry arrays.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in meters.

    Raises:
        ValueError: If no road segments have geometry points.
    """
    valid_geoms = [s.geometry for s in scene.road_segments if s.geometry.shape[0] > 0]
    if not valid_geoms:
        raise ValueError("Cannot build illumination map: no road segments have geometry.")

    all_points = np.vstack(valid_geoms)
    min_x = float(all_points[:, 0].min()) - _BOUNDS_PADDING_M
    min_y = float(all_points[:, 1].min()) - _BOUNDS_PADDING_M
    max_x = float(all_points[:, 0].max()) + _BOUNDS_PADDING_M
    max_y = float(all_points[:, 1].max()) + _BOUNDS_PADDING_M

    return min_x, min_y, max_x, max_y


def _build_illumination_map(
    lights: t.List["LightSource"],
    bounds: t.Tuple[float, float, float, float],
    resolution: float,
    illumination_brightness: int,
    camera_position: np.ndarray,
    falloff_distance: float,
) -> t.Tuple[np.ndarray, t.Tuple[float, float, float]]:
    """Build a float32 RGB illumination map by painting cone-model blobs for each light.

    Initializes a zero-filled (H, W, 3) float32 array, paints each light's
    contribution via _paint_illumination_blob (with distance-based brightness
    falloff applied), then clips to [0, 255].

    Args:
        lights (list[LightSource]): All light sources to paint.
        bounds (tuple[float, float, float, float]): (min_x, min_y, max_x, max_y)
            in world ENU meters.
        resolution (float): Meters per pixel.
        illumination_brightness (int): Base brightness scalar from AtmosphereConfig.
        camera_position (np.ndarray): (3,) camera position in world ENU meters, used
            for per-light distance falloff.
        falloff_distance (float): Characteristic decay distance in meters for the
            exponential brightness attenuation.

    Returns:
        tuple[np.ndarray, tuple[float, float, float]]:
            - illum_map: (H, W, 3) float32 array of illumination RGB values.
            - map_info: (min_x, max_y, resolution) for coordinate conversion.
    """
    min_x, min_y, max_x, max_y = bounds

    width = max(1, int((max_x - min_x) / resolution))
    height = max(1, int((max_y - min_y) / resolution))

    illum_map = np.zeros((height, width, 3), dtype=np.float32)

    for light in lights:
        _paint_illumination_blob(
            illum_map, light, min_x, max_y, resolution, illumination_brightness,
            camera_position, falloff_distance,
        )

    map_info = (min_x, max_y, resolution)
    return illum_map, map_info


def _paint_illumination_blob(
    illum_map: np.ndarray,
    light: "LightSource",
    min_x: float,
    max_y: float,
    resolution: float,
    illumination_brightness: int,
    camera_position: np.ndarray,
    falloff_distance: float,
) -> None:
    """Paint a cone-model intensity blob for one light source into the illumination map.

    Intensity follows a two-zone cone model:

        t = clip(dist_m / illumination_radius, 0, 1)
        inner = illumination_inner_fraction
        fade_t = clip((t - inner) / max(1 - inner, 1e-6), 0, 1)
        falloff = (1 - fade_t) ^ illumination_falloff

    The inner zone [0, inner * radius] is at full brightness (fade_t = 0,
    falloff = 1.0). The fade zone [inner * radius, radius] ramps fade_t from
    0 to 1, driving falloff from 1 to 0. At exactly radius, falloff is zero,
    so there is no hard edge. Beyond radius, t is clipped to 1.0, keeping
    fade_t at 1.0 and intensity at zero.

    Light brightness is attenuated by exponential distance falloff from the
    camera before computing the peak, matching the formula used in the glow pass.

    Y-axis convention: top of image = max_y in world (same as FlatMapMeta).

    Args:
        illum_map (np.ndarray): (H, W, 3) float32 array to paint into (modified in-place).
        light (LightSource): Light source with position, color, brightness,
            illumination_radius, illumination_falloff, and illumination_inner_fraction.
        min_x (float): World x coordinate corresponding to pixel column 0.
        max_y (float): World y coordinate corresponding to pixel row 0.
        resolution (float): Meters per pixel.
        illumination_brightness (int): Base brightness scalar from AtmosphereConfig.
        camera_position (np.ndarray): (3,) camera position in world ENU meters.
        falloff_distance (float): Characteristic decay distance in meters for
            exponential brightness attenuation.
    """
    cx = (light.position[0] - min_x) / resolution
    cy = (max_y - light.position[1]) / resolution
    radius_px = light.illumination_radius / resolution

    if radius_px < 1.0:
        return

    map_h, map_w = illum_map.shape[:2]
    x0 = max(0, int(cx - radius_px))
    x1 = min(map_w, int(cx + radius_px + 1))
    y0 = max(0, int(cy - radius_px))
    y1 = min(map_h, int(cy + radius_px + 1))

    if x0 >= x1 or y0 >= y1:
        return

    grid_y, grid_x = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )

    dist_m = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2) * resolution

    t = np.clip(dist_m / light.illumination_radius, 0.0, 1.0)
    inner = light.illumination_inner_fraction
    denom = max(1.0 - inner, 1e-6)
    fade_t = np.clip((t - inner) / denom, 0.0, 1.0)
    fade_t = fade_t * fade_t * (3.0 - 2.0 * fade_t)
    falloff = (1.0 - fade_t) ** light.illumination_falloff

    dist = camera_distance(light.position, camera_position)
    effective_brightness = apply_brightness_falloff(light.brightness, dist, falloff_distance)
    # illumination_brightness is authored on a 0-255 scale; /255 maps it into
    # linear radiance so the painted values are linear-light, not sRGB codes.
    peak = (illumination_brightness / 255.0) * light.illumination_intensity * effective_brightness
    intensity = peak * falloff

    lin_r, lin_g, lin_b = srgb255_to_linear(light.color)
    illum_map[y0:y1, x0:x1, 0] += intensity * lin_r
    illum_map[y0:y1, x0:x1, 1] += intensity * lin_g
    illum_map[y0:y1, x0:x1, 2] += intensity * lin_b


def _sample_illumination_map(
    illum_map: np.ndarray,
    hit_x: np.ndarray,
    hit_y: np.ndarray,
    ground_mask: np.ndarray,
    min_x: float,
    max_y: float,
    resolution: float,
) -> np.ndarray:
    """Sample the illumination map at each ground-hit world position.

    Performs bilinear interpolation across the four surrounding map texels for
    each valid ground pixel. Pixels outside the ground mask remain zero (black).

    Args:
        illum_map (np.ndarray): (H_map, W_map, 3) float32 illumination map.
        hit_x (np.ndarray): (H, W) ENU x coordinates of ground intersections.
        hit_y (np.ndarray): (H, W) ENU y coordinates of ground intersections.
        ground_mask (np.ndarray): (H, W) bool mask of valid ground pixels.
        min_x (float): World x corresponding to column 0 of the illumination map.
        max_y (float): World y corresponding to row 0 of the illumination map.
        resolution (float): Meters per pixel in the illumination map.

    Returns:
        np.ndarray: (H, W, 3) float32 illumination values for the screen.
    """
    screen_h, screen_w = ground_mask.shape
    map_h, map_w = illum_map.shape[:2]

    result = np.zeros((screen_h, screen_w, 3), dtype=np.float32)

    # Fractional map coordinates for all pixels
    px_f = (hit_x - min_x) / resolution
    py_f = (max_y - hit_y) / resolution

    # Integer floor coordinates for the top-left texel of each 2x2 sample quad
    px0 = np.clip(np.floor(px_f).astype(int), 0, map_w - 1)
    py0 = np.clip(np.floor(py_f).astype(int), 0, map_h - 1)

    # Neighbour coordinates (clamped so we never go out of bounds)
    px1 = np.clip(px0 + 1, 0, map_w - 1)
    py1 = np.clip(py0 + 1, 0, map_h - 1)

    # Sub-texel fractional offsets, shape (H, W)
    fx = np.clip(px_f - px0, 0.0, 1.0).astype(np.float32)
    fy = np.clip(py_f - py0, 0.0, 1.0).astype(np.float32)

    m = ground_mask

    # Sample the four surrounding texels, shape (N, 3) where N = ground pixels
    c00 = illum_map[py0[m], px0[m]]   # top-left
    c10 = illum_map[py0[m], px1[m]]   # top-right
    c01 = illum_map[py1[m], px0[m]]   # bottom-left
    c11 = illum_map[py1[m], px1[m]]   # bottom-right

    # Expand fractional weights to (N, 1) for broadcast multiply over RGB
    fx_m = fx[m, np.newaxis]
    fy_m = fy[m, np.newaxis]

    result[m] = (
        c00 * (1.0 - fx_m) * (1.0 - fy_m)
        + c10 * fx_m * (1.0 - fy_m)
        + c01 * (1.0 - fx_m) * fy_m
        + c11 * fx_m * fy_m
    )

    return result
