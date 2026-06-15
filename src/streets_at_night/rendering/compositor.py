"""Compositor render pass — final grade: stars, vignette, tone-map, sRGB encode.

This is the final pass in the export pipeline. It resolves the backend's
combined linear frame (linearized substrate + accumulated light), applies
post-composition effects in linear space, then tone-maps and encodes to sRGB:

  - Stars: faint emissive points added into sky pixels above the horizon.
  - Vignette: radial edge darkening (multiply in linear).
  - Color grade: exposure -> ACES tone map -> linear->sRGB encode -> uint8.

The graded uint8 frame is pasted back into the backend so get_image() returns
the final image.
"""

import typing as t

import numpy as np
from PIL import Image

from ..camera.camera import Camera
from .backend import RenderBackend
from .color_space import linear_to_srgb, srgb255_to_linear, tonemap_aces
from .render_utils import compute_horizon_y

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def render_composite(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Resolve, grade, and encode the final frame.

    Resolves the combined linear frame, adds stars and applies the vignette in
    linear space, then tone-maps and encodes to sRGB. Stars are added only when
    stars_enabled is True in the scene config.

    Args:
        scene (Scene): The world scene containing config and seed.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface.
    """
    width, height = backend.get_surface_size()
    config = scene.config

    linear = backend.resolve_linear()

    if config.stars is not None and config.stars.stars_enabled:
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        horizon_y = compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
        rng = np.random.default_rng(scene.config.seed)
        _add_stars(linear, width, height, horizon_y, config.stars, rng)

    _apply_vignette(linear, width, height, config.atmosphere)
    _apply_color_grade(backend, linear, config.grade)


def _apply_vignette(
    linear: np.ndarray,
    width: int,
    height: int,
    config: t.Any,
) -> None:
    """Darken the frame edges with a radial vignette, multiplied in linear.

    Args:
        linear (np.ndarray): (H, W, 3) float32 linear frame, modified in place.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        config: AtmosphereConfig with vignette_strength and vignette_radius.
    """
    scalar = _build_vignette_scalar(
        width, height, config.vignette_strength, config.vignette_radius
    )
    linear *= scalar[:, :, np.newaxis]


def _build_vignette_scalar(
    width: int,
    height: int,
    strength: float,
    radius: float,
) -> np.ndarray:
    """Build a (H, W) float32 scalar field for vignette darkening.

    The scalar is 1.0 at the image center, decreasing toward the edges.
    Darkening begins at radius * half_diagonal from the center and reaches
    (1.0 - strength) at the corners.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        strength (float): Darkening factor at the edges (0=no effect, 1=black).
        radius (float): Fractional radius at which darkening starts (0–1).

    Returns:
        np.ndarray: (H, W) float32 scalar field in range [1-strength, 1.0].
    """
    cx = width / 2.0
    cy = height / 2.0
    half_diag = np.sqrt(cx ** 2 + cy ** 2)

    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)

    inner = radius * half_diag
    outer = half_diag
    fade_range = max(outer - inner, 1.0)

    t = np.clip((dist - inner) / fade_range, 0.0, 1.0)
    scalar = 1.0 - strength * t
    return scalar.astype(np.float32)


def _add_stars(
    linear: np.ndarray,
    width: int,
    height: int,
    horizon_y: int,
    config: t.Any,
    rng: np.random.Generator,
) -> None:
    """Add faint emissive stars into the sky region above the horizon.

    Stars are placed uniformly at random in the sky area (y < horizon_y), each
    near-white with small random color variation and brightness drawn from the
    configured range. Star colors are linearized before being added so they
    accumulate as linear emission like every other light.

    Args:
        linear (np.ndarray): (H, W, 3) float32 linear frame, modified in place.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        horizon_y (int): Screen y-coordinate of the horizon. Stars go above this.
        config: StarsConfig with count, brightness_min, brightness_max,
            color_variation, and radius fields.
        rng (np.random.Generator): Seeded random generator for reproducibility.
    """
    if horizon_y <= 0:
        return

    xs = rng.integers(0, width, size=config.count)
    ys = rng.integers(0, horizon_y, size=config.count)
    brightnesses = rng.uniform(config.brightness_min, config.brightness_max, size=config.count)
    variations = rng.integers(
        -config.color_variation, config.color_variation + 1, size=(config.count, 3)
    )

    base = (brightnesses[:, np.newaxis] * 255.0).astype(np.int32)
    rgb = np.clip(base + variations, 0, 255)

    for i in range(config.count):
        color = srgb255_to_linear((int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))
        _add_star_point(linear, int(xs[i]), int(ys[i]), config.radius, color)


def _add_star_point(
    linear: np.ndarray,
    x: int,
    y: int,
    radius: float,
    color_linear: t.Tuple[float, float, float],
) -> None:
    """Add a single star's linear emission at (x, y).

    Stars with radius <= 1 light a single pixel; larger radii fill a small disc.

    Args:
        linear (np.ndarray): (H, W, 3) float32 frame, modified in place.
        x (int): Screen x.
        y (int): Screen y.
        radius (float): Star radius in pixels.
        color_linear (tuple[float, float, float]): Linear RGB color.
    """
    if radius <= 1.0:
        linear[y, x, :] += color_linear
        return

    h, w, _ = linear.shape
    r = int(radius)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
    for c in range(3):
        linear[y0:y1, x0:x1, c] += color_linear[c] * mask


def _apply_color_grade(
    backend: RenderBackend,
    linear: np.ndarray,
    config: t.Any,
) -> None:
    """Tone-map and encode the linear frame to sRGB, writing the final image.

    Applies exposure (in stops) before the tone-map curve, runs the configured
    operator (ACES filmic by default), encodes linear->sRGB, and pastes the
    resulting uint8 frame back into the backend.

    Args:
        backend (RenderBackend): Target render surface (receives the final paste).
        linear (np.ndarray): (H, W, 3) float32 linear frame.
        config: GradeConfig with exposure (stops) and tonemap (operator name).
    """
    scaled = linear * (2.0 ** config.exposure)

    if config.tonemap == "aces":
        mapped = tonemap_aces(scaled)
    else:
        mapped = np.clip(scaled, 0.0, 1.0)

    srgb = linear_to_srgb(mapped)
    uint8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    backend.paste_image(Image.fromarray(uint8, "RGB"))
