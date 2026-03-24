"""Pillow-based render backend for static export images."""

import typing as t

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .backend import RenderBackend


class PillowBackend(RenderBackend):
    """Pillow-based render backend for static export images."""

    def __init__(self, width: int, height: int) -> None:
        """Initialise the backend with a blank RGB surface.

        Args:
            width (int): Surface width in pixels.
            height (int): Surface height in pixels.
        """
        self._width = width
        self._height = height
        self._image = Image.new("RGB", (width, height), (0, 0, 0))
        self._draw = ImageDraw.Draw(self._image, "RGBA")

    def get_surface_size(self) -> t.Tuple[int, int]:
        """Return (width, height) of the render surface.

        Returns:
            tuple[int, int]: (width, height) in pixels.
        """
        return (self._width, self._height)

    def clear(self, color: t.Tuple[int, int, int]) -> None:
        """Fill the entire surface with a solid color.

        Args:
            color (tuple[int, int, int]): RGB fill color.
        """
        self._draw.rectangle(
            [(0, 0), (self._width - 1, self._height - 1)],
            fill=color,
        )

    def draw_line(
        self,
        p1: t.Tuple[float, float],
        p2: t.Tuple[float, float],
        color: t.Tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a line segment.

        Args:
            p1 (tuple[float, float]): Start point (x, y) in screen pixels.
            p2 (tuple[float, float]): End point (x, y) in screen pixels.
            color (tuple[int, int, int]): RGB line color.
            width (int): Line width in pixels.
        """
        self._draw.line([p1, p2], fill=color, width=width)

    def draw_polygon(
        self,
        vertices: t.List[t.Tuple[float, float]],
        fill_color: t.Tuple[int, int, int],
    ) -> None:
        """Draw a filled polygon.

        Args:
            vertices (list[tuple[float, float]]): Screen-space vertex list.
            fill_color (tuple[int, int, int]): RGB fill color.
        """
        self._draw.polygon(vertices, fill=fill_color)

    def draw_circle(
        self,
        center: t.Tuple[float, float],
        radius: float,
        color: t.Tuple[int, int, int],
    ) -> None:
        """Draw a filled circle using an ellipse bounding box.

        Args:
            center (tuple[float, float]): Center (x, y) in screen pixels.
            radius (float): Circle radius in pixels.
            color (tuple[int, int, int]): RGB fill color.
        """
        cx, cy = center
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        self._draw.ellipse(bbox, fill=color)

    def draw_line_rgba(
        self,
        p1: t.Tuple[float, float],
        p2: t.Tuple[float, float],
        color: t.Tuple[int, int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a line with alpha blending via PIL's RGBA draw context.

        Args:
            p1 (tuple[float, float]): Start point (x, y) in screen pixels.
            p2 (tuple[float, float]): End point (x, y) in screen pixels.
            color (tuple[int, int, int, int]): RGBA color.
            width (int): Line width in pixels.
        """
        self._draw.line([p1, p2], fill=color, width=width)

    def get_image(self) -> Image.Image:
        """Return a copy of the current surface as a PIL Image.

        Returns:
            PIL.Image.Image: Copy of the rendered surface.
        """
        return self._image.copy()

    def paste_image(self, img: Image.Image) -> None:
        """Paste a PIL Image over the current internal surface at (0, 0).

        Used by the export ground pass to composite the tile-sampled ground
        layer onto the image after the sky/horizon base has been drawn.

        Args:
            img (PIL.Image.Image): RGB image to paste over the surface.
        """
        self._image.paste(img, (0, 0))
        self._draw = ImageDraw.Draw(self._image, "RGBA")

    def create_layer(self) -> Image.Image:
        """Create a new transparent RGBA image matching the surface size.

        Returns:
            PIL.Image.Image: New RGBA image, fully transparent, same size as surface.
        """
        return Image.new("RGBA", (self._width, self._height), (0, 0, 0, 0))

    def composite_additive(self, layer: Image.Image) -> None:
        """Additively blend an RGBA layer onto the internal RGB surface.

        Converts the RGBA layer to RGB (alpha pre-multiplied), then uses
        PIL's ImageChops.add to clamp-add each channel. The result replaces
        the internal surface and the draw context is refreshed.

        Args:
            layer (PIL.Image.Image): RGBA glow layer to blend onto the surface.
        """
        layer_rgb = _rgba_to_additive_rgb(layer)
        blended = ImageChops.add(self._image, layer_rgb)
        self._image.paste(blended, (0, 0))
        self._draw = ImageDraw.Draw(self._image, "RGBA")

    def present(self) -> None:
        """No-op for the Pillow backend (no display to flip)."""


def _rgba_to_additive_rgb(layer: Image.Image) -> Image.Image:
    """Convert an RGBA layer to RGB by pre-multiplying alpha for additive blending.

    Each RGB channel is scaled by alpha/255 so that transparent pixels contribute
    nothing and opaque pixels contribute their full color value.

    Args:
        layer (PIL.Image.Image): RGBA image to convert.

    Returns:
        PIL.Image.Image: RGB image ready for additive blending.
    """
    arr = np.asarray(layer, dtype=np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    rgb = (arr[:, :, :3] * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")
