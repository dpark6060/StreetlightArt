"""Pygame-based render backend for interactive preview."""

import typing as t

import numpy as np
import pygame
from PIL import Image

from .backend import RenderBackend


class PygameBackend(RenderBackend):
    """Pygame-based render backend for interactive preview.

    Draws directly onto a pygame Surface using pygame.draw primitives.
    Out-of-bounds coordinates are clipped or skipped so that off-screen
    projected values do not crash pygame or distort on-screen geometry.
    """

    def __init__(self, surface: pygame.Surface) -> None:
        """Initialise the backend with an existing pygame surface.

        Args:
            surface (pygame.Surface): The pygame surface to draw onto.
        """
        self._surface = surface

    def get_surface_size(self) -> t.Tuple[int, int]:
        """Return (width, height) of the render surface.

        Returns:
            tuple[int, int]: (width, height) in pixels.
        """
        return self._surface.get_size()

    def clear(self, color: t.Tuple[int, int, int]) -> None:
        """Fill the surface with a solid color.

        Args:
            color (tuple[int, int, int]): RGB fill color.
        """
        self._surface.fill(color)

    def draw_line(
        self,
        p1: t.Tuple[float, float],
        p2: t.Tuple[float, float],
        color: t.Tuple[int, int, int],
        width: int = 1,
    ) -> None:
        """Draw a line segment, clipped to the surface bounds.

        The segment is clipped to the surface rect before drawing so
        off-screen endpoints do not distort the on-screen slope and pygame
        does not crash on out-of-range projected values. Segments fully
        outside the surface are skipped.

        Args:
            p1 (tuple[float, float]): Start point (x, y) in screen pixels.
            p2 (tuple[float, float]): End point (x, y) in screen pixels.
            color (tuple[int, int, int]): RGB line color.
            width (int): Line width in pixels.
        """
        w, h = self._surface.get_size()
        clipped = pygame.Rect(0, 0, w, h).clipline(
            (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
        )
        if not clipped:
            return
        ip1, ip2 = clipped
        pygame.draw.line(self._surface, color, ip1, ip2, max(1, int(width)))

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
        int_verts = [(int(x), int(y)) for x, y in vertices]
        pygame.draw.polygon(self._surface, fill_color, int_verts)

    def draw_circle(
        self,
        center: t.Tuple[float, float],
        radius: float,
        color: t.Tuple[int, int, int],
    ) -> None:
        """Draw a filled circle.

        Args:
            center (tuple[float, float]): Center (x, y) in screen pixels.
            radius (float): Circle radius in pixels.
            color (tuple[int, int, int]): RGB fill color.
        """
        pygame.draw.circle(
            self._surface,
            color,
            (int(center[0]), int(center[1])),
            max(1, int(radius)),
        )

    def draw_point(
        self,
        center: t.Tuple[float, float],
        color: t.Tuple[int, int, int],
    ) -> None:
        """Draw a single pixel using surface.set_at.

        Out-of-bounds coordinates are silently skipped since
        Surface.set_at raises IndexError outside the surface.

        Args:
            center (tuple[float, float]): Screen coordinate (x, y).
            color (tuple[int, int, int]): RGB fill color.
        """
        w, h = self._surface.get_size()
        x, y = int(center[0]), int(center[1])
        if not (0 <= x < w and 0 <= y < h):
            return
        self._surface.set_at((x, y), color)

    def get_image(self) -> Image.Image:
        """Convert the pygame surface to a PIL Image.

        Returns:
            PIL.Image.Image: Current surface state as an RGB PIL image.
        """
        w, h = self._surface.get_size()
        raw = pygame.image.tobytes(self._surface, "RGB")
        return Image.frombytes("RGB", (w, h), raw)

    def blit_pil_image(self, img: Image.Image) -> None:
        """Blit a PIL Image onto the pygame surface at (0, 0).

        Converts the PIL image to a pygame Surface and blits it at the origin.
        The image mode must be RGB or compatible with pygame.image.frombuffer.

        Args:
            img (PIL.Image.Image): Image to blit. Should match surface size.
        """
        surf = pygame.image.frombytes(img.tobytes(), img.size, img.mode)
        self._surface.blit(surf, (0, 0))

    def blit_array(self, frame: np.ndarray) -> None:
        """Blit an (H, W, 3) uint8 RGB numpy array onto the surface at (0, 0).

        One copy (``frame.tobytes()``) versus the two involved in going
        through ``blit_pil_image`` (PIL Image wrap + ``pygame.image.frombytes``
        re-encode). Preferred over ``blit_pil_image`` for callers that already
        hold a raw numpy frame, e.g. the preview and animate loops.

        Args:
            frame (np.ndarray): (H, W, 3) uint8 RGB array matching the surface size.
        """
        h, w, _ = frame.shape
        surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
        self._surface.blit(surf, (0, 0))

    def present(self) -> None:
        """Flip the pygame display to show the rendered frame."""
        pygame.display.flip()
