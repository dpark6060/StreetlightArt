"""Illumination render pass — soft ground-light pools for export.

Uses a project-and-splat approach: each light is projected to screen space,
a soft circle is drawn on an illumination layer at the projected position,
the layer is gaussian-blurred, then additively composited onto the main image.

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
from .render_utils import camera_distance, compute_focal_length, extract_positions

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Gaussian blur radius applied to the illumination layer before compositing.
# Larger values produce wider, softer light pools.
_ILLUMINATION_BLUR_RADIUS: int = 20

# Scale factor for the screen-space illumination radius relative to glow.
# Illumination pools are significantly wider than the glow halo.
_ILLUMINATION_RADIUS_SCALE: float = 4.0

# Minimum illumination radius in screen pixels before skipping a light.
_MIN_ILLUMINATION_RADIUS_PX: float = 3.0

# Number of concentric rings used to build each illumination splat.
_SPLAT_RING_COUNT: int = 6

# Maximum alpha at the center of each illumination splat.
_SPLAT_MAX_ALPHA: int = 60


def render_illumination(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Render ground illumination from nearby light sources (export only).

    Projects each visible light to screen space, draws a soft filled circle
    proportional to its illumination radius on a separate RGBA layer, gaussian-
    blurs the layer, then additively composites it onto the main image.

    This pass should run after the ground and road passes but before the light
    core/glow pass so that illumination sits beneath individual light sources.

    Args:
        scene (Scene): The world scene containing all light sources.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface (must support
            create_layer and composite_additive).
    """
    if not scene.lights:
        return

    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()

    positions = extract_positions(scene.lights)
    screen_coords, valid_mask = project_points(
        positions, view_matrix, proj_matrix, (width, height)
    )

    focal_length = compute_focal_length(camera, width)
    illumination_layer = backend.create_layer()
    draw_ctx = _make_draw_context(illumination_layer)

    for i, light in enumerate(scene.lights):
        if not valid_mask[i]:
            continue
        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        distance = camera_distance(light.position, camera.position)
        _draw_illumination_splat(draw_ctx, center, light, distance, focal_length)

    blurred = _blur_layer(illumination_layer)
    backend.composite_additive(blurred)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_draw_context(layer: Image.Image) -> ImageDraw.ImageDraw:
    """Create a PIL ImageDraw context for an RGBA layer.

    Args:
        layer (PIL.Image.Image): The RGBA layer to draw into.

    Returns:
        PIL.ImageDraw.ImageDraw: Draw context for the layer.
    """
    return ImageDraw.Draw(layer)


def _illumination_radius_px(
    light: "LightSource",
    distance: float,
    focal_length: float,
) -> float:
    """Compute screen-space illumination radius for a single light.

    Args:
        light (LightSource): The light source.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.

    Returns:
        float: Illumination radius in screen pixels.
    """
    return light.illumination_radius / distance * focal_length * _ILLUMINATION_RADIUS_SCALE


def _splat_ring_alpha(ring_index: int, max_alpha: int) -> int:
    """Compute alpha for one splat ring using inverse-square falloff.

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
    focal_length: float,
) -> None:
    """Draw concentric filled circles to create a soft illumination splat.

    Rings are drawn outermost-first so that inner (brighter) rings paint over
    outer (dimmer) rings. Alpha is scaled by illumination_intensity so that
    lights with no ground contribution produce no splat.

    Args:
        draw_ctx: PIL ImageDraw context for the illumination RGBA layer.
        center (tuple[float, float]): Screen-space center (x, y).
        light (LightSource): The light source providing color and radii.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.
    """
    radius_px = _illumination_radius_px(light, distance, focal_length)
    if radius_px < _MIN_ILLUMINATION_RADIUS_PX:
        return

    max_alpha = int(_SPLAT_MAX_ALPHA * light.illumination_intensity)
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
