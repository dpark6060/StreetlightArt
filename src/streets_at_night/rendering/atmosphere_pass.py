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
from .raycast import GroundHit
from .render_utils import compute_horizon_y

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def render_atmosphere(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    gbuffer: GroundHit,
) -> None:
    """Apply horizon glow and depth-based airlight to the current frame.

    Computes the horizon screen-y position and accumulates a warm gradient band
    centered on the horizon (horizon glow). If haze is enabled, also accumulates
    the airlight veil — path radiance that grows with the line-of-sight distance
    to the ground, using the same extinction coefficient that dims the lights.
    Both effects are added into the backend's linear light buffer.

    Args:
        scene (Scene): The world scene containing config.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface.
        gbuffer (GroundHit): Shared per-pixel ground raycast for this camera and
            size, used to derive per-pixel line-of-sight distance for airlight.
    """
    width, height = backend.get_surface_size()
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()

    horizon_y = compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
    config = scene.config.atmosphere

    _render_horizon_glow(backend, width, height, horizon_y, config)
    if config.haze_enabled:
        _render_airlight(backend, gbuffer, camera, config)


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


def _render_airlight(
    backend: RenderBackend,
    gbuffer: GroundHit,
    camera: Camera,
    config: t.Any,
) -> None:
    """Accumulate the airlight veil from per-pixel line-of-sight distance.

    Airlight is the standard atmospheric path radiance: a pixel looking through
    distance ``d`` of air picks up ``airlight·(1 - exp(-d/k))`` of haze color,
    where ``k`` is the same extinction coefficient (``falloff_distance``) that
    dims the lights. Ground pixels use their true 3D distance to the z=0 hit;
    sky pixels (no hit) fade by elevation, saturating toward the horizon where
    the air path is longest and clearing toward the zenith.

    Args:
        backend (RenderBackend): Target render surface.
        gbuffer (GroundHit): Per-pixel ground raycast (hit positions, ray dirs,
            and ground mask).
        camera (Camera): Active camera, for the eye position.
        config: AtmosphereConfig with haze_color, haze_alpha, falloff_distance.
    """
    haze_color = srgb255_to_linear(config.haze_color)
    peak = config.haze_alpha / 255.0
    k = config.falloff_distance

    cam = camera.position
    dx = gbuffer.hit_x - cam[0]
    dy = gbuffer.hit_y - cam[1]
    ground_distance = np.sqrt(dx * dx + dy * dy + cam[2] * cam[2])
    ground_strength = 1.0 - np.exp(-ground_distance / k)

    elevation = np.clip(gbuffer.ray_dirs[:, :, 2], 0.0, 1.0)
    sky_strength = (1.0 - elevation) ** 2

    strength = np.where(gbuffer.ground_mask, ground_strength, sky_strength).astype(np.float32)

    layer = np.empty((*strength.shape, 3), dtype=np.float32)
    for c in range(3):
        layer[:, :, c] = haze_color[c] * peak * strength
    backend.add_linear(layer)


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
