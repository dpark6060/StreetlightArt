"""Light render passes — dots for preview, core + glow halos for export."""

import math
import typing as t

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from ..camera.camera import Camera
from ..camera.projection import project_points
from .backend import RenderBackend
from .render_utils import camera_distance, compute_focal_length, extract_positions

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Maximum alpha for the glow halo at its brightest center ring
_GLOW_MAX_ALPHA: int = 180

# Number of concentric rings drawn to approximate gaussian glow falloff
_GLOW_RING_COUNT: int = 8

# Minimum glow radius in screen pixels before we skip drawing the halo
_MIN_GLOW_RADIUS_PX: float = 2.0


def render_lights_preview(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> t.Dict[str, int]:
    """Render lights as colored dots for interactive preview.

    Projects every light in the scene to screen space. Draws a filled circle
    for each visible light. Circle radius scales with brightness but no glow
    halos are applied — this pass is intended for real-time pygame preview.

    Args:
        scene (Scene): The world scene containing all light sources.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface.

    Returns:
        dict[str, int]: Stats with key "lights_drawn" reporting visible count.
    """
    if not scene.lights:
        return {"lights_drawn": 0}

    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()

    positions = extract_positions(scene.lights)
    screen_coords, valid_mask = project_points(
        positions, view_matrix, proj_matrix, (width, height)
    )

    drawn = 0
    for i, light in enumerate(scene.lights):
        if not valid_mask[i]:
            continue
        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        radius = _preview_radius(light.brightness)
        backend.draw_circle(center, radius, light.color)
        drawn += 1

    return {"lights_drawn": drawn}


def render_lights_export(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> t.Dict[str, int]:
    """Render lights with bright cores and gaussian glow halos for high-res export.

    For each visible light:
      - Draws a bright core dot at the light's color and full brightness.
      - Draws concentric circles with decreasing alpha on a separate RGBA glow
        layer to approximate gaussian falloff.
    Composites the glow layer onto the main image using additive blending so
    overlapping glow regions accumulate naturally.

    Args:
        scene (Scene): The world scene containing all light sources.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface (must be PillowBackend or
            support create_layer / composite_additive).

    Returns:
        dict[str, int]: Stats with key "lights_drawn" reporting visible count.
    """
    if not scene.lights:
        return {"lights_drawn": 0}

    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()

    positions = extract_positions(scene.lights)
    screen_coords, valid_mask = project_points(
        positions, view_matrix, proj_matrix, (width, height)
    )

    focal_length = compute_focal_length(camera, width)
    glow_layer = backend.create_layer()
    glow_draw = ImageDraw.Draw(glow_layer)

    drawn = 0
    for i, light in enumerate(scene.lights):
        if not valid_mask[i]:
            continue

        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        distance = camera_distance(light.position, camera.position)
        core_radius = _core_radius(light, distance, focal_length)

        backend.draw_circle(center, core_radius, light.color)
        _draw_glow_halo(glow_draw, center, light, distance, focal_length)
        drawn += 1

    backend.composite_additive(glow_layer)
    return {"lights_drawn": drawn}


# ---------------------------------------------------------------------------
# Private helpers — preview
# ---------------------------------------------------------------------------


def _preview_radius(brightness: float) -> float:
    """Compute a screen-space dot radius for the preview pass.

    Args:
        brightness (float): Light brightness scalar (1.0 = normal).

    Returns:
        float: Radius in pixels, at least 1.
    """
    return max(1.0, brightness * 3.0)


# ---------------------------------------------------------------------------
# Private helpers — export
# ---------------------------------------------------------------------------


def _core_radius(light: "LightSource", distance: float, focal_length: float) -> float:
    """Compute the bright-core circle radius in screen pixels.

    Args:
        light (LightSource): The light source.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.

    Returns:
        float: Core radius in pixels, at least 1.
    """
    screen_size = light.size / distance * focal_length
    return max(1.0, screen_size * light.brightness)


def _glow_radius_px(light: "LightSource", distance: float, focal_length: float) -> float:
    """Compute the glow halo radius in screen pixels.

    Args:
        light (LightSource): The light source.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.

    Returns:
        float: Glow radius in pixels.
    """
    return light.glow_radius / distance * focal_length


def _ring_alpha(ring_index: int, total_rings: int, max_alpha: int) -> int:
    """Compute the alpha for one concentric glow ring using gaussian falloff.

    The sigma is set to half the total radius so the falloff is smooth and
    reaches near-zero by the outermost ring.

    Args:
        ring_index (int): Index of this ring, 0 = innermost.
        total_rings (int): Total number of rings.
        max_alpha (int): Alpha at the center (ring_index == 0).

    Returns:
        int: Clamped alpha in [0, 255].
    """
    sigma = total_rings / 2.0
    r = ring_index
    raw = max_alpha * math.exp(-(r * r) / (2.0 * sigma * sigma))
    return max(0, min(255, int(raw)))


def _draw_glow_halo(
    draw: ImageDraw.ImageDraw,
    center: t.Tuple[float, float],
    light: "LightSource",
    distance: float,
    focal_length: float,
) -> None:
    """Draw concentric filled circles on the glow layer to approximate a gaussian halo.

    Rings are drawn from outermost to innermost so inner rings (higher alpha)
    paint over the outer ones. Each ring decreases in radius and increases in alpha
    following a gaussian curve.

    Args:
        draw (ImageDraw.ImageDraw): Draw context for the RGBA glow layer.
        center (tuple[float, float]): Screen-space center (x, y).
        light (LightSource): The light source providing color and glow_radius.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.
    """
    radius_px = _glow_radius_px(light, distance, focal_length)
    if radius_px < _MIN_GLOW_RADIUS_PX:
        return

    r, g, b = light.color
    cx, cy = center

    for ring_idx in range(_GLOW_RING_COUNT - 1, -1, -1):
        ring_radius = radius_px * (ring_idx + 1) / _GLOW_RING_COUNT
        alpha = _ring_alpha(ring_idx, _GLOW_RING_COUNT, _GLOW_MAX_ALPHA)
        bbox = [cx - ring_radius, cy - ring_radius, cx + ring_radius, cy + ring_radius]
        draw.ellipse(bbox, fill=(r, g, b, alpha))
