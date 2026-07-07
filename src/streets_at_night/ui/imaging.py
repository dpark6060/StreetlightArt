"""Image helpers shared by UI tabs: PIL <-> pygame and viewport letterboxing.

Copied from ``devmode.cmd_devmode`` so tabs can share them from a neutral spot
(change 09 deletes the originals). The private helpers there are exposed here
under public names.
"""

from dataclasses import dataclass

import pygame
from PIL import Image


@dataclass
class ViewportFit:
    """Scaled size and centering offset for fitting a render into a viewport.

    Attributes:
        width: Scaled width in pixels.
        height: Scaled height in pixels.
        x: Horizontal offset in pixels for centering within the viewport.
        y: Vertical offset in pixels for centering within the viewport.
    """

    width: int
    height: int
    x: int
    y: int


def compute_viewport_fit(
    render_width: int,
    render_height: int,
    viewport_width: int,
    viewport_height: int,
) -> ViewportFit:
    """Compute the scaled size and letterbox offset to fit a render into a viewport.

    Preserves the render's aspect ratio: scales uniformly to the largest size
    that fits entirely within the viewport, then centers it.

    Args:
        render_width (int): Width of the rendered image in pixels.
        render_height (int): Height of the rendered image in pixels.
        viewport_width (int): Width of the target viewport in pixels.
        viewport_height (int): Height of the target viewport in pixels.

    Returns:
        ViewportFit: Scaled width/height and top-left offset within the viewport.
    """
    scale = min(viewport_width / render_width, viewport_height / render_height)
    width = max(1, round(render_width * scale))
    height = max(1, round(render_height * scale))
    x = (viewport_width - width) // 2
    y = (viewport_height - height) // 2
    return ViewportFit(width=width, height=height, x=x, y=y)


def fit_image_to_viewport(
    image: Image.Image,
    viewport_width: int,
    viewport_height: int,
) -> Image.Image:
    """Scale a rendered image to fit a fixed-size viewport, preserving aspect ratio.

    Resizes the image to the largest size that fits within the viewport
    (via compute_viewport_fit) and pastes it onto a black canvas the size of
    the viewport, so the result is always exactly viewport_width x
    viewport_height regardless of the source image's aspect ratio.

    Args:
        image (PIL.Image.Image): The rendered image at its native resolution.
        viewport_width (int): Width of the display viewport in pixels.
        viewport_height (int): Height of the display viewport in pixels.

    Returns:
        PIL.Image.Image: Letterboxed image sized viewport_width x viewport_height.
    """
    fit = compute_viewport_fit(image.width, image.height, viewport_width, viewport_height)
    scaled = image.resize((fit.width, fit.height))
    canvas = Image.new("RGB", (viewport_width, viewport_height), (0, 0, 0))
    canvas.paste(scaled, (fit.x, fit.y))
    return canvas


def pil_to_pygame_surface(image: Image.Image) -> pygame.Surface:
    """Convert a PIL Image to a pygame Surface.

    Args:
        image (PIL.Image.Image): The image to convert.

    Returns:
        pygame.Surface: Equivalent pygame surface in RGB mode.
    """
    rgb_image = image.convert("RGB")
    raw = rgb_image.tobytes()
    return pygame.image.frombytes(raw, rgb_image.size, "RGB")
