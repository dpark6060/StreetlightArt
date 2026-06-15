"""Pillow-based render backend for static export images.

The backend holds two surfaces that together form a linear-light HDR pipeline:

  - ``_image`` — uint8 sRGB PIL image, the opaque substrate. Ground and roads
    draw here through the normal PIL primitives (keeping anti-aliasing).
  - ``_light`` — float32 ``(H, W, 3)`` linear-radiance buffer, unbounded, where
    all additive light (illumination, glow, cores, atmosphere) accumulates with
    no clamp.

``resolve_linear()`` combines the linearized substrate with the light buffer;
the compositor tone-maps that result and pastes the final uint8 frame back into
``_image`` so ``get_image()`` returns the graded image.
"""

import typing as t

import numpy as np
from PIL import Image, ImageDraw

from .backend import RenderBackend
from .color_space import srgb_to_linear


class PillowBackend(RenderBackend):
    """Pillow-based render backend for static export images."""

    def __init__(self, width: int, height: int) -> None:
        """Initialise the backend with a blank RGB surface and light buffer.

        Args:
            width (int): Surface width in pixels.
            height (int): Surface height in pixels.
        """
        self._width = width
        self._height = height
        self._image = Image.new("RGB", (width, height), (0, 0, 0))
        self._draw = ImageDraw.Draw(self._image, "RGBA")
        self._light = np.zeros((height, width, 3), dtype=np.float32)

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

    def draw_point(
        self,
        center: t.Tuple[float, float],
        color: t.Tuple[int, int, int],
    ) -> None:
        """Draw a single pixel using ImageDraw.point.

        Args:
            center (tuple[float, float]): Screen coordinate (x, y).
            color (tuple[int, int, int]): RGB fill color.
        """
        self._draw.point([(int(center[0]), int(center[1]))], fill=color)

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

    def blit_pil_image(self, img: Image.Image) -> None:
        """Paste a PIL Image onto the surface at (0, 0).

        Used by the warp preview pass to write the raycast result into
        the surface. Delegates to paste_image.

        Args:
            img (PIL.Image.Image): RGB image to blit onto the surface.
        """
        self.paste_image(img)

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
        """Accumulate an sRGB RGBA layer into the linear light buffer.

        The layer's RGB is decoded sRGB->linear, weighted by its alpha (treated
        as a linear coverage), and added to ``_light`` with no clamp. This is a
        convenience wrapper around ``add_linear`` for passes that build PIL RGBA
        layers; passes that already hold a float array should call ``add_linear``
        directly to avoid the 8-bit quantization round-trip.

        Args:
            layer (PIL.Image.Image): RGBA layer (sRGB, 0-255) to add.
        """
        arr = np.asarray(layer, dtype=np.float32)
        linear_rgb = srgb_to_linear(arr[:, :, :3] / 255.0)
        alpha = arr[:, :, 3:4] / 255.0
        self.add_linear(linear_rgb * alpha)

    def add_linear(self, layer: np.ndarray) -> None:
        """Add a linear-radiance ``(H, W, 3)`` float array to the light buffer.

        The canonical accumulation entry point for the light passes. Values are
        unbounded — bright overlaps build past 1.0 and are rolled off later by
        the tone map, never hard-clipped here.

        Args:
            layer (np.ndarray): (H, W, 3) float linear-radiance contribution.
        """
        self._light += np.asarray(layer, dtype=np.float32)

    def resolve_linear(self) -> np.ndarray:
        """Return the combined linear frame: linearized substrate plus light.

        Returns:
            np.ndarray: (H, W, 3) float32 linear radiance, unbounded.
        """
        base = srgb_to_linear(np.asarray(self._image, dtype=np.float32) / 255.0)
        return base + self._light

    def present(self) -> None:
        """No-op for the Pillow backend (no display to flip)."""
