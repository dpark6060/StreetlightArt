"""Compositor render pass — vignette, optional stars, and color grade stub.

This is the final pass in the export pipeline. It applies post-composition
effects that depend on the fully rendered scene:
  - Vignette: radial edge darkening via numpy multiply on the full image.
  - Stars: faint point lights drawn into sky pixels above the horizon.
  - Color grade: no-op stub for future LUT/curve adjustments.
"""

import typing as t

import numpy as np
from PIL import Image

from ..camera.camera import Camera
from .backend import RenderBackend
from .render_utils import compute_horizon_y

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def render_composite(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> None:
    """Apply vignette, optional stars, and color grade to the current frame.

    Always applies vignette. Stars are drawn only when stars_enabled is True
    in the scene config. Color grading is a no-op placeholder.

    Args:
        scene (Scene): The world scene containing config and seed.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface.
    """
    width, height = backend.get_surface_size()
    config = scene.config

    _render_vignette(backend, width, height, config.atmosphere)

    if config.stars is not None and config.stars.stars_enabled:
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        horizon_y = compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
        rng = np.random.default_rng(scene.config.seed)
        _render_stars(backend, width, height, horizon_y, config.stars, rng)

    _apply_color_grade(backend)


def _render_vignette(
    backend: RenderBackend,
    width: int,
    height: int,
    config: t.Any,
) -> None:
    """Darken image edges with a radial vignette applied via numpy multiply.

    Builds a scalar field where the center is 1.0 (no change) and the edges
    are darkened by vignette_strength. The scalar field is applied by getting
    the current image, multiplying RGB channels, and pasting the result back.

    Args:
        backend (RenderBackend): Target render surface (must support get_image
            and paste_image).
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        config: AtmosphereConfig with vignette_strength and vignette_radius.
    """
    scalar_field = _build_vignette_scalar(
        width, height, config.vignette_strength, config.vignette_radius
    )
    image = backend.get_image()
    darkened = _apply_scalar_to_image(image, scalar_field)
    backend.paste_image(darkened)


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


def _apply_scalar_to_image(image: Image.Image, scalar: np.ndarray) -> Image.Image:
    """Multiply the RGB channels of a PIL image by a scalar field.

    Args:
        image (PIL.Image.Image): Source RGB image.
        scalar (np.ndarray): (H, W) float32 scalar field, values in [0, 1].

    Returns:
        PIL.Image.Image: New RGB image with channels scaled and clamped to uint8.
    """
    arr = np.asarray(image, dtype=np.float32)
    darkened = arr * scalar[:, :, np.newaxis]
    clamped = np.clip(darkened, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(clamped, "RGB")


def _render_stars(
    backend: RenderBackend,
    width: int,
    height: int,
    horizon_y: int,
    config: t.Any,
    rng: np.random.Generator,
) -> None:
    """Draw faint point stars in the sky region above the horizon.

    Stars are placed uniformly at random in the sky area (y < horizon_y).
    Each star is near-white with small random color variation and brightness
    drawn from the configured range. Stars with radius <= 1 are drawn as
    single pixels; larger radii use draw_circle.

    Args:
        backend (RenderBackend): Target render surface.
        width (int): Surface width in pixels.
        height (int): Surface height in pixels.
        horizon_y (int): Screen y-coordinate of the horizon. Stars go above this.
        config: StarsConfig with count, brightness_min, brightness_max,
            color_variation, and radius fields.
        rng (np.random.Generator): Seeded random generator for reproducibility.
    """
    if horizon_y <= 0:
        return

    sky_height = horizon_y
    xs = rng.integers(0, width, size=config.count)
    ys = rng.integers(0, sky_height, size=config.count)
    brightnesses = rng.uniform(config.brightness_min, config.brightness_max, size=config.count)
    variations = rng.integers(-config.color_variation, config.color_variation + 1, size=(config.count, 3))

    for i in range(config.count):
        base_val = int(brightnesses[i] * 255)
        r = int(np.clip(base_val + variations[i, 0], 0, 255))
        g = int(np.clip(base_val + variations[i, 1], 0, 255))
        b = int(np.clip(base_val + variations[i, 2], 0, 255))
        color = (r, g, b)
        center = (float(xs[i]), float(ys[i]))

        if config.radius <= 1.0:
            backend.draw_point(center, color)
        else:
            backend.draw_circle(center, config.radius, color)


def _apply_color_grade(backend: RenderBackend) -> None:
    """Apply global color grading to the final image.

    Currently a no-op placeholder. Future implementations may apply LUT
    transformations or curve adjustments here.

    Args:
        backend (RenderBackend): Target render surface.
    """
