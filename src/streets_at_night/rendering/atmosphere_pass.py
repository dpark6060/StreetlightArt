"""Atmosphere render pass — horizon glow gradient and distance haze overlay.

Adds image-space atmospheric effects after the light pass. Both effects are
built as linear-radiance ``(H, W, 3)`` arrays and accumulated into the backend's
linear light buffer via ``add_linear`` (additive emission, unclamped), matching
the light and illumination passes.

This pass runs after all geometry and light passes so it correctly overlays on
top of the fully composited scene.
"""

import typing as t

import numpy as np

from ..camera.camera import Camera
from .backend import RenderBackend
from .color_space import hex_to_linear, srgb255_to_linear
from .render_utils import compute_horizon_y

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def render_atmosphere(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Apply horizon glow and distance haze overlays to the current frame.

    Computes the horizon screen-y position, then accumulates a warm gradient
    band centered on the horizon (horizon glow) and optionally a distance haze
    overlay. Both effects are added into the backend's linear light buffer.

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
    """Accumulate a warm gradient band centered on the horizon, in linear light.

    The glow is centered at horizon_y and extends horizon_glow_width_px // 2
    pixels above and below, with intensity peaking at horizon_glow_peak_alpha
    (on a 0-255 scale, mapped to linear) at the center row and fading to 0 at
    the band edges.

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        horizon_y (int): Screen y-coordinate of the horizon line.
        config: AtmosphereConfig with horizon_glow_color, horizon_glow_width_px,
            and horizon_glow_peak_alpha fields.
    """
    half_width = config.horizon_glow_width_px // 2
    if half_width <= 0:
        return

    glow_color = hex_to_linear(config.horizon_glow_color)
    peak = config.horizon_glow_peak_alpha / 255.0

    rows = np.arange(height, dtype=np.float32)
    offset = np.abs(rows - horizon_y)
    fade = np.where(offset < half_width, (1.0 - offset / half_width) ** 2, 0.0)
    alpha = (peak * fade).astype(np.float32)

    layer = _column_alpha_to_linear_layer(alpha, width, glow_color)
    backend.add_linear(layer)


def _render_distance_haze(
    backend: RenderBackend,
    width: int,
    height: int,
    horizon_y: int,
    config: t.Any,
) -> None:
    """Accumulate a screen-space distance haze overlay, in linear light.

    The haze is strongest at the horizon and fades to zero about one-third of
    the way down the image. The sky region above the horizon receives a reduced
    haze tint. The haze color comes from AtmosphereConfig.haze_color (an RGB
    0-255 tuple), linearized.

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        horizon_y (int): Screen y-coordinate of the horizon line.
        config: AtmosphereConfig with haze_color, haze_alpha fields.
    """
    haze_color = srgb255_to_linear(config.haze_color)
    peak = config.haze_alpha / 255.0

    ground_fade_bottom = horizon_y + max(1, int((height - horizon_y) * 0.33))
    sky_fade_top = max(0, horizon_y - max(1, int(horizon_y * 0.5)))

    alpha = _haze_alpha_column(
        height, horizon_y, ground_fade_bottom, sky_fade_top, peak
    )

    layer = _column_alpha_to_linear_layer(alpha, width, haze_color)
    backend.add_linear(layer)


def _haze_alpha_column(
    height: int,
    horizon_y: int,
    ground_fade_bottom: int,
    sky_fade_top: int,
    peak: float,
) -> np.ndarray:
    """Compute the per-row haze alpha for every screen row, vectorized.

    Ground rows: alpha decreases linearly from peak at horizon_y to 0 at
    ground_fade_bottom. Sky rows: alpha decreases linearly from half the peak
    at horizon_y to 0 at sky_fade_top. Rows beyond the fade boundaries get 0.

    Args:
        height (int): Number of screen rows.
        horizon_y (int): Screen y of the horizon.
        ground_fade_bottom (int): Row at which ground haze fades to zero.
        sky_fade_top (int): Row at which sky haze fades to zero.
        peak (float): Maximum linear intensity at the horizon line.

    Returns:
        np.ndarray: (H,) float32 alpha per row, each in [0, peak].
    """
    rows = np.arange(height, dtype=np.float32)

    ground_range = float(ground_fade_bottom - horizon_y)
    ground_t = (rows - horizon_y) / ground_range if ground_range > 0 else np.ones_like(rows)
    ground_alpha = np.clip(peak * (1.0 - ground_t), 0.0, None)

    sky_range = float(horizon_y - sky_fade_top)
    sky_t = (horizon_y - rows) / sky_range if sky_range > 0 else np.ones_like(rows)
    sky_alpha = np.clip((peak * 0.5) * (1.0 - sky_t), 0.0, None)

    alpha = np.where(rows >= horizon_y, ground_alpha, sky_alpha)
    return alpha.astype(np.float32)


def _column_alpha_to_linear_layer(
    alpha: np.ndarray,
    width: int,
    color_linear: t.Tuple[float, float, float],
) -> np.ndarray:
    """Expand a per-row alpha column into an (H, W, 3) linear color layer.

    Args:
        alpha (np.ndarray): (H,) per-row linear weight.
        width (int): Surface width in pixels.
        color_linear (tuple[float, float, float]): Linear RGB color in [0, 1].

    Returns:
        np.ndarray: (H, W, 3) float32 linear contribution.
    """
    height = alpha.shape[0]
    layer = np.empty((height, width, 3), dtype=np.float32)
    for c in range(3):
        layer[:, :, c] = color_linear[c] * alpha[:, None]
    return layer
