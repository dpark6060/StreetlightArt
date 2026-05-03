"""Atmosphere render pass — horizon glow gradient and distance haze overlay.

Adds image-space atmospheric effects after the light pass. Both effects are
built as RGBA numpy arrays and composited additively, matching the pattern used
by the light glow pass in light_pass.py.

This pass runs after all geometry and light passes so it correctly overlays on
top of the fully composited scene.
"""

import typing as t

import numpy as np
from PIL import Image

from ..camera.camera import Camera
from .backend import RenderBackend
from .render_utils import compute_horizon_y

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def render_atmosphere(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Apply horizon glow and distance haze overlays to the current frame.

    Computes the horizon screen-y position, then renders a warm gradient band
    centered on the horizon (horizon glow) and optionally a distance haze
    overlay. Both effects are composited additively onto the backend.

    Args:
        scene (Scene): The world scene containing config.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface.
    """
    width, height = backend.get_surface_size()
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()

    horizon_y = compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
    config = scene.config.atmosphere

    _render_horizon_glow(backend, width, height, horizon_y, config)
    if config.haze_enabled:
        _render_distance_haze(backend, width, height, horizon_y, config)


def _render_horizon_glow(
    backend: RenderBackend,
    width: int,
    height: int,
    horizon_y: int,
    config: t.Any,
) -> None:
    """Build and composite a warm gradient band centered on the horizon.

    The glow is centered at horizon_y and extends horizon_glow_width_px // 2
    pixels above and below, with alpha peaking at horizon_glow_peak_alpha at
    the center row and fading to 0 at the band edges.

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        horizon_y (int): Screen y-coordinate of the horizon line.
        config: AtmosphereConfig with horizon_glow_color, horizon_glow_width_px,
            and horizon_glow_peak_alpha fields.
    """
    glow_color = _parse_hex_color(config.horizon_glow_color)
    half_width = config.horizon_glow_width_px // 2
    peak_alpha = config.horizon_glow_peak_alpha

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    for row in range(height):
        offset = abs(row - horizon_y)
        if offset > half_width:
            continue
        fade = 1.0 - offset / (half_width + 1)
        alpha = int(peak_alpha * fade)
        rgba[row, :, 0] = glow_color[0]
        rgba[row, :, 1] = glow_color[1]
        rgba[row, :, 2] = glow_color[2]
        rgba[row, :, 3] = alpha

    layer = Image.fromarray(rgba, "RGBA")
    backend.composite_additive(layer)


def _render_distance_haze(
    backend: RenderBackend,
    width: int,
    height: int,
    horizon_y: int,
    config: t.Any,
) -> None:
    """Build and composite a screen-space distance haze overlay.

    Applies a haze tint that is strongest at the horizon and fades to zero
    about one-third of the way down the image. The sky region above the
    horizon also receives a reduced haze tint. The haze color comes from
    AtmosphereConfig.haze_color (already parsed to an RGB tuple).

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        horizon_y (int): Screen y-coordinate of the horizon line.
        config: AtmosphereConfig with haze_color, haze_alpha fields.
    """
    haze_color = config.haze_color
    peak_alpha = config.haze_alpha

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    ground_fade_bottom = horizon_y + max(1, int((height - horizon_y) * 0.33))
    sky_fade_top = max(0, horizon_y - max(1, int(horizon_y * 0.5)))

    for row in range(height):
        alpha = _haze_alpha_for_row(
            row, horizon_y, ground_fade_bottom, sky_fade_top, peak_alpha
        )
        if alpha <= 0:
            continue
        rgba[row, :, 0] = haze_color[0]
        rgba[row, :, 1] = haze_color[1]
        rgba[row, :, 2] = haze_color[2]
        rgba[row, :, 3] = alpha

    layer = Image.fromarray(rgba, "RGBA")
    backend.composite_additive(layer)


def _haze_alpha_for_row(
    row: int,
    horizon_y: int,
    ground_fade_bottom: int,
    sky_fade_top: int,
    peak_alpha: int,
) -> int:
    """Compute the haze alpha for a single screen row.

    Ground rows: alpha decreases linearly from peak_alpha at horizon_y to 0 at
    ground_fade_bottom. Sky rows: alpha decreases linearly from half the peak
    at horizon_y to 0 at sky_fade_top. Rows beyond the fade boundaries get 0.

    Args:
        row (int): The screen row index.
        horizon_y (int): Screen y of the horizon.
        ground_fade_bottom (int): Row at which ground haze fades to zero.
        sky_fade_top (int): Row at which sky haze fades to zero.
        peak_alpha (int): Maximum alpha value at the horizon line.

    Returns:
        int: Alpha value for this row, in range [0, peak_alpha].
    """
    if row >= horizon_y:
        fade_range = ground_fade_bottom - horizon_y
        if fade_range <= 0:
            return 0
        t = (row - horizon_y) / fade_range
        return max(0, int(peak_alpha * (1.0 - t)))

    sky_fade_range = horizon_y - sky_fade_top
    if sky_fade_range <= 0:
        return 0
    t = (horizon_y - row) / sky_fade_range
    sky_peak = peak_alpha // 2
    return max(0, int(sky_peak * (1.0 - t)))


def _parse_hex_color(hex_color: str) -> t.Tuple[int, int, int]:
    """Parse a '#RRGGBB' hex color string to an (R, G, B) int tuple.

    Args:
        hex_color (str): Hex color string, with or without leading '#'.

    Returns:
        tuple[int, int, int]: RGB color values.
    """
    stripped = hex_color.lstrip("#")
    return (
        int(stripped[0:2], 16),
        int(stripped[2:4], 16),
        int(stripped[4:6], 16),
    )
