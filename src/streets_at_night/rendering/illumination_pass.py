"""Illumination render pass — soft ground-light pools for export.

Uses a project-and-splat approach: each light's ground point (directly below
the light at z=0) is projected to screen space, a soft circle is drawn on an
illumination layer at the projected position with a radius derived from the
projected ground-circle edge, the layer is gaussian-blurred, then additively
composited onto the main image.

This avoids the per-pixel ground-plane raycast that would be prohibitively
slow in Python and produces perceptually correct pooled lighting.
"""

import math
import typing as t

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..camera.camera import Camera
from ..camera.projection import project_points
from .backend import RenderBackend
from .render_utils import camera_distance

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Gaussian blur radius applied to the illumination layer before compositing.
# Larger values produce wider, softer light pools.
_ILLUMINATION_BLUR_RADIUS: int = 20

# Minimum illumination radius in screen pixels before skipping a light.
_MIN_ILLUMINATION_RADIUS_PX: float = 3.0

# Number of concentric rings used to build each illumination splat.
_SPLAT_RING_COUNT: int = 6


def render_illumination(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Render ground illumination from nearby light sources (export only).

    Projects each light's ground point (z=0 directly below the light) and a
    sample point on the east edge of the ground circle to screen space.
    Draws a soft filled circle on an illumination layer, gaussian-blurs the
    layer, then additively composites it onto the main image.

    The pool center is the projected ground point, and the radius in pixels
    comes from the projected distance to the edge point — perspective
    foreshortening is handled automatically by the projection.

    The maximum splat alpha is derived from scene.config.atmosphere.illumination_brightness
    scaled by each light's illumination_intensity and brightness.

    The computed radius is clamped to at least scene.config.atmosphere.illumination_min_radius
    so distant lights whose projected circle would shrink below the gaussian blur radius still
    produce a visible pool.

    This pass should run after the ground and road passes but before the light
    core/glow pass so that illumination sits beneath individual light sources.

    Args:
        scene (Scene): The world scene containing all light sources and config.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface (must support
            create_layer and composite_additive).
    """
    if not scene.lights:
        return

    illumination_brightness = scene.config.atmosphere.illumination_brightness
    min_radius = scene.config.atmosphere.illumination_min_radius
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()

    ground_centers = _extract_ground_positions(scene.lights)
    ground_edges = _extract_ground_edge_positions(scene.lights)

    center_screen, center_valid = project_points(
        ground_centers, view_matrix, proj_matrix, (width, height)
    )
    edge_screen, _ = project_points(
        ground_edges, view_matrix, proj_matrix, (width, height)
    )

    illumination_layer = backend.create_layer()
    draw_ctx = _make_draw_context(illumination_layer)

    for i, light in enumerate(scene.lights):
        if not center_valid[i]:
            continue
        center = (float(center_screen[i, 0]), float(center_screen[i, 1]))
        radius_px = _compute_ground_radius_px(center_screen[i], edge_screen[i])
        radius_px = max(radius_px, min_radius)
        distance = camera_distance(light.position, camera.position)
        _draw_illumination_splat(draw_ctx, center, light, distance, radius_px, illumination_brightness)

    blurred = _blur_layer(illumination_layer)
    backend.composite_additive(blurred)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_ground_positions(lights: t.List["LightSource"]) -> np.ndarray:
    """Build a (N, 3) array of ground positions directly below each light.

    The ground position for each light is its (x, y) position with z=0.

    Args:
        lights (list[LightSource]): Light sources to extract positions from.

    Returns:
        np.ndarray: (N, 3) float array with z=0 for each light.
    """
    return np.array(
        [[light.position[0], light.position[1], 0.0] for light in lights],
        dtype=float,
    )


def _extract_ground_edge_positions(lights: t.List["LightSource"]) -> np.ndarray:
    """Build a (N, 3) array of east-edge points on each light's ground circle.

    The edge point is offset east by illumination_radius from the ground center,
    at z=0.  Projecting both the center and this edge point lets us derive the
    pixel radius via perspective projection directly.

    Args:
        lights (list[LightSource]): Light sources to extract edge positions from.

    Returns:
        np.ndarray: (N, 3) float array with x offset by illumination_radius, z=0.
    """
    return np.array(
        [
            [
                light.position[0] + light.illumination_radius,
                light.position[1],
                0.0,
            ]
            for light in lights
        ],
        dtype=float,
    )


def _compute_ground_radius_px(
    center: np.ndarray,
    edge: np.ndarray,
) -> float:
    """Compute screen-space radius of a ground circle from its projected points.

    Args:
        center (np.ndarray): (2,) projected screen coordinate of the ground center.
        edge (np.ndarray): (2,) projected screen coordinate of the east edge.

    Returns:
        float: Pixel distance between the projected center and edge.
    """
    return float(np.linalg.norm(edge - center))


def _make_draw_context(layer: Image.Image) -> ImageDraw.ImageDraw:
    """Create a PIL ImageDraw context for an RGBA layer.

    Args:
        layer (PIL.Image.Image): The RGBA layer to draw into.

    Returns:
        PIL.ImageDraw.ImageDraw: Draw context for the layer.
    """
    return ImageDraw.Draw(layer)


def _splat_ring_alpha(ring_index: int, max_alpha: int) -> int:
    """Compute alpha for one splat ring using Gaussian falloff.

    Args:
        ring_index (int): Ring index, 0 = innermost.
        max_alpha (int): Alpha at the center ring.

    Returns:
        int: Alpha in [0, 255].
    """
    sigma = _SPLAT_RING_COUNT / 2.0
    raw = max_alpha * math.exp(-(ring_index * ring_index) / (2.0 * sigma * sigma))
    return max(0, min(255, int(raw)))


def _draw_illumination_splat(
    draw_ctx: t.Any,
    center: t.Tuple[float, float],
    light: "LightSource",
    distance: float,
    radius_px: float,
    illumination_brightness: int,
) -> None:
    """Draw concentric filled circles to create a soft illumination splat.

    Rings are drawn outermost-first so that inner (brighter) rings paint over
    outer (dimmer) rings. Alpha is scaled by illumination_intensity and
    light.brightness so that lights with no ground contribution produce no splat.
    The base maximum alpha is controlled by illumination_brightness from
    the atmosphere config.

    Args:
        draw_ctx: PIL ImageDraw context for the illumination RGBA layer.
        center (tuple[float, float]): Screen-space center (x, y) of the ground pool.
        light (LightSource): The light source providing color and intensities.
        distance (float): Distance from camera to light in meters (used for intensity).
        radius_px (float): Pre-computed radius in screen pixels.
        illumination_brightness (int): Base max alpha (0–255) from AtmosphereConfig.
    """
    if radius_px < _MIN_ILLUMINATION_RADIUS_PX:
        return

    max_alpha = int(illumination_brightness * light.illumination_intensity * light.brightness)
    if max_alpha <= 0:
        return

    r, g, b = light.color
    cx, cy = center

    for ring_idx in range(_SPLAT_RING_COUNT - 1, -1, -1):
        ring_radius = radius_px * (ring_idx + 1) / _SPLAT_RING_COUNT
        alpha = _splat_ring_alpha(ring_idx, max_alpha)
        bbox = [cx - ring_radius, cy - ring_radius, cx + ring_radius, cy + ring_radius]
        draw_ctx.ellipse(bbox, fill=(r, g, b, alpha))


def _blur_layer(layer: Image.Image) -> Image.Image:
    """Apply gaussian blur to the illumination layer.

    Args:
        layer (PIL.Image.Image): RGBA illumination layer to blur.

    Returns:
        PIL.Image.Image: Blurred RGBA layer.
    """
    return layer.filter(ImageFilter.GaussianBlur(radius=_ILLUMINATION_BLUR_RADIUS))
