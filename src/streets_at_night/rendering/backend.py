"""Abstract rendering backend interface.

Concrete implementations:
    PillowBackend — static export images via PIL.
    PygameBackend — interactive preview (Stage 3+).
"""

import typing as t
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class RenderBackend(ABC):
    """Abstract rendering backend. Implementations: PillowBackend (export), PygameBackend (preview)."""

    @abstractmethod
    def get_surface_size(self) -> t.Tuple[int, int]:
        """Return (width, height) of the render surface.

        Returns:
            tuple[int, int]: (width, height) in pixels.
        """

    @abstractmethod
    def clear(self, color: t.Tuple[int, int, int]) -> None:
        """Fill the surface with a solid color.

        Args:
            color (tuple[int, int, int]): RGB fill color.
        """

    @abstractmethod
    def draw_line(
        self,
        p1: t.Tuple[float, float],
        p2: t.Tuple[float, float],
        color: t.Tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a line segment.

        Args:
            p1 (tuple[float, float]): Start point in screen pixels (x, y).
            p2 (tuple[float, float]): End point in screen pixels (x, y).
            color (tuple[int, int, int]): RGB line color.
            width (int): Line width in pixels.
        """

    @abstractmethod
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

    @abstractmethod
    def draw_circle(
        self,
        center: t.Tuple[float, float],
        radius: float,
        color: t.Tuple[int, int, int],
    ) -> None:
        """Draw a filled circle.

        Args:
            center (tuple[float, float]): Center in screen pixels (x, y).
            radius (float): Circle radius in pixels.
            color (tuple[int, int, int]): RGB fill color.
        """

    @abstractmethod
    def get_image(self) -> Image.Image:
        """Return the current surface state as a PIL Image.

        Returns:
            PIL.Image.Image: Copy of the current render surface.
        """

    def draw_line_rgba(
        self,
        p1: t.Tuple[float, float],
        p2: t.Tuple[float, float],
        color: t.Tuple[int, int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a line with an RGBA color (alpha blended if supported).

        Default implementation discards alpha and delegates to draw_line.
        Override in backends that support alpha blending.

        Args:
            p1 (tuple[float, float]): Start point in screen pixels (x, y).
            p2 (tuple[float, float]): End point in screen pixels (x, y).
            color (tuple[int, int, int, int]): RGBA color.
            width (int): Line width in pixels.
        """
        r, g, b, _ = color
        self.draw_line(p1, p2, (r, g, b), width)

    def blit_pil_image(self, img: Image.Image) -> None:
        """Blit a PIL Image onto the render surface at (0, 0).

        Default implementation is a no-op. Override in backends that support
        direct image blitting (e.g. PygameBackend).

        Args:
            img (PIL.Image.Image): Image to blit. Must match surface size.
        """

    def paste_image(self, img: Image.Image) -> None:
        """Paste a PIL Image over the current surface at (0, 0).

        Default implementation is a no-op. Override in backends that need to
        composite a fully-rendered PIL image onto their internal surface
        (e.g. PillowBackend for tile compositing in the export path).

        Args:
            img (PIL.Image.Image): RGB image to paste. Must match surface size.
        """

    def create_layer(self) -> Image.Image:
        """Create a new transparent RGBA image matching the surface size.

        Returns a blank RGBA layer that callers can draw into and then
        composite back onto the surface via composite_additive. Default
        implementation returns a transparent image of the correct size.

        Returns:
            PIL.Image.Image: New RGBA image, fully transparent, same size as surface.
        """
        width, height = self.get_surface_size()
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))

    def composite_additive(self, layer: Image.Image) -> None:
        """Additively blend an RGBA layer onto the main surface.

        Default implementation is a no-op. Override in backends that support
        additive compositing (e.g. PillowBackend for glow layer blending).

        For additive blending, each output channel is:
            min(255, base_channel + layer_channel * layer_alpha / 255)

        Args:
            layer (PIL.Image.Image): RGBA layer to blend additively onto the surface.
        """

    @abstractmethod
    def present(self) -> None:
        """Display the frame (pygame: flip; Pillow: no-op)."""
