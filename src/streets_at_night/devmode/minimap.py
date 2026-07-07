"""2D top-down minimap: world<->screen mapping, hit testing, and drawing.

Kept pygame-light so the coordinate math is unit-testable. ``MinimapLayout``
holds the screen rect and world (ENU) extent; ``Minimap`` maps between the two
with a uniform, aspect-preserving scale (world +y/north maps to screen up) and
draws the road lines plus the draggable camera/target dots. Nothing here touches
pygame_gui — only plain ``pygame.Rect`` and surface drawing.
"""

import typing as t
from dataclasses import dataclass

import pygame

# Dot pick radius and draw radius, and the segment count above which the road
# overlay is subsampled to stay cheap.
_HIT_RADIUS_PX = 10
_DOT_RADIUS_PX = 6
_MAX_SEGMENTS = 2000

_CAMERA_COLOR = (80, 200, 255)
_TARGET_COLOR = (255, 160, 60)
_ROAD_COLOR = (120, 120, 120)
_LINK_COLOR = (200, 200, 200)
_OUTLINE_COLOR = (150, 150, 150)
_BACKGROUND_RGBA = (0, 0, 0, 160)


@dataclass
class MinimapLayout:
    """Screen rect and world extent for coordinate mapping.

    Attributes:
        rect: Screen-space minimap box.
        world_min: (x_min, y_min) ENU meters.
        world_max: (x_max, y_max) ENU meters.
    """

    rect: pygame.Rect
    world_min: t.Tuple[float, float]
    world_max: t.Tuple[float, float]


class Minimap:
    """Maps ENU world coordinates to a screen rect and draws the overlay."""

    def __init__(self, layout: MinimapLayout) -> None:
        """Initialise the minimap and precompute the uniform scale + offsets.

        Args:
            layout (MinimapLayout): Screen rect and world extent to map between.
        """
        self.layout = layout
        rect = layout.rect
        world_w = max(1e-6, layout.world_max[0] - layout.world_min[0])
        world_h = max(1e-6, layout.world_max[1] - layout.world_min[1])
        # Uniform scale: fit the larger world dimension, center the smaller axis.
        self._scale = min(rect.width / world_w, rect.height / world_h)
        content_w = world_w * self._scale
        content_h = world_h * self._scale
        self._off_x = rect.x + (rect.width - content_w) / 2.0
        self._off_y = rect.y + (rect.height - content_h) / 2.0

    def world_to_screen(self, x: float, y: float) -> t.Tuple[int, int]:
        """Map an ENU world point to a screen pixel (world north = screen up).

        Args:
            x (float): World east coordinate (ENU x).
            y (float): World north coordinate (ENU y).

        Returns:
            tuple[int, int]: Screen pixel (px, py).
        """
        px = self._off_x + (x - self.layout.world_min[0]) * self._scale
        py = self._off_y + (self.layout.world_max[1] - y) * self._scale
        return int(round(px)), int(round(py))

    def screen_to_world(self, px: int, py: int) -> t.Tuple[float, float]:
        """Map a screen pixel back to an ENU world point, clamped to the extent.

        Exact inverse of ``world_to_screen`` (before clamping), with the y axis
        flipped back. The result is clamped to the world extent so out-of-rect
        drags stay inside the scene bounds.

        Args:
            px (int): Screen x pixel.
            py (int): Screen y pixel.

        Returns:
            tuple[float, float]: World (x, y) in ENU meters.
        """
        x = self.layout.world_min[0] + (px - self._off_x) / self._scale
        y = self.layout.world_max[1] - (py - self._off_y) / self._scale
        x = _clamp(x, self.layout.world_min[0], self.layout.world_max[0])
        y = _clamp(y, self.layout.world_min[1], self.layout.world_max[1])
        return x, y

    def hit_test(
        self,
        pos: t.Tuple[int, int],
        camera_xy: t.Tuple[float, float],
        target_xy: t.Tuple[float, float],
    ) -> t.Optional[str]:
        """Return which dot a screen position hits, camera winning ties.

        Args:
            pos (tuple[int, int]): Screen pixel to test.
            camera_xy (tuple[float, float]): Camera world (x, y).
            target_xy (tuple[float, float]): Target world (x, y).

        Returns:
            str | None: "camera", "target", or None if neither is within
                the pick radius.
        """
        cam = self.world_to_screen(*camera_xy)
        tgt = self.world_to_screen(*target_xy)
        dc = _distance(pos, cam)
        dt = _distance(pos, tgt)
        if dc <= _HIT_RADIUS_PX and dc <= dt:
            return "camera"
        if dt <= _HIT_RADIUS_PX:
            return "target"
        if dc <= _HIT_RADIUS_PX:
            return "camera"
        return None

    def draw(
        self,
        surface: pygame.Surface,
        camera_xy: t.Tuple[float, float],
        target_xy: t.Tuple[float, float],
        road_segments: t.Sequence[t.Any],
    ) -> None:
        """Draw the minimap background, roads, link line, and both dots.

        Args:
            surface (pygame.Surface): The surface to draw onto.
            camera_xy (tuple[float, float]): Camera world (x, y).
            target_xy (tuple[float, float]): Target world (x, y).
            road_segments (Sequence): Scene road segments; each ``.geometry`` is
                an (N, 3) ENU array (columns 0/1 used).
        """
        rect = self.layout.rect
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        panel.fill(_BACKGROUND_RGBA)
        surface.blit(panel, (rect.x, rect.y))
        pygame.draw.rect(surface, _OUTLINE_COLOR, rect, width=1)

        self._draw_roads(surface, road_segments)

        cam = self.world_to_screen(*camera_xy)
        tgt = self.world_to_screen(*target_xy)
        pygame.draw.line(surface, _LINK_COLOR, cam, tgt, 1)
        pygame.draw.circle(surface, _CAMERA_COLOR, cam, _DOT_RADIUS_PX)
        pygame.draw.circle(surface, _TARGET_COLOR, tgt, _DOT_RADIUS_PX)

    def _draw_roads(
        self, surface: pygame.Surface, road_segments: t.Sequence[t.Any]
    ) -> None:
        """Draw each road segment as a 1 px polyline, subsampling large scenes.

        Args:
            surface (pygame.Surface): The surface to draw onto.
            road_segments (Sequence): Scene road segments.
        """
        clip = surface.get_clip()
        surface.set_clip(self.layout.rect)
        step = max(1, len(road_segments) // _MAX_SEGMENTS)
        for segment in road_segments[::step]:
            geometry = segment.geometry
            if geometry is None or len(geometry) < 2:
                continue
            points = [self.world_to_screen(pt[0], pt[1]) for pt in geometry]
            pygame.draw.lines(surface, _ROAD_COLOR, False, points, 1)
        surface.set_clip(clip)


def _clamp(value: float, low: float, high: float) -> float:
    """Clamp value to the inclusive [low, high] range."""
    return max(low, min(high, value))


def _distance(a: t.Tuple[float, float], b: t.Tuple[float, float]) -> float:
    """Return the Euclidean distance between two 2D points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
