# Radical
"""Unit tests for PygameBackend draw_point and draw_line correctness."""

import numpy as np
import pygame
import pytest

from streets_at_night.rendering.pygame_backend import PygameBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def surface() -> pygame.Surface:
    """Return a plain headless pygame Surface for drawing tests."""
    return pygame.Surface((100, 100))


@pytest.fixture
def backend(surface: pygame.Surface) -> PygameBackend:
    """Return a PygameBackend wrapping the test surface."""
    return PygameBackend(surface)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_pygame_backend_can_be_instantiated_with_surface(
    surface: pygame.Surface,
) -> None:
    """PygameBackend reports the wrapped surface's size."""
    backend = PygameBackend(surface)

    assert backend.get_surface_size() == (100, 100)


# ---------------------------------------------------------------------------
# draw_point
# ---------------------------------------------------------------------------


def test_draw_point_out_of_bounds_does_not_raise(
    backend: PygameBackend, surface: pygame.Surface
) -> None:
    """draw_point with off-screen coordinates does not raise and leaves the surface unchanged."""
    before = pygame.image.tobytes(surface, "RGB")

    backend.draw_point((-5, -5), (255, 255, 255))
    backend.draw_point((110, 110), (255, 255, 255))

    after = pygame.image.tobytes(surface, "RGB")
    assert before == after


def test_draw_point_in_bounds_sets_pixel(
    backend: PygameBackend, surface: pygame.Surface
) -> None:
    """draw_point with an in-bounds coordinate sets that pixel to the given color."""
    backend.draw_point((10, 20), (255, 0, 0))

    assert surface.get_at((10, 20)) == (255, 0, 0, 255)


# ---------------------------------------------------------------------------
# draw_line
# ---------------------------------------------------------------------------


def test_draw_line_offscreen_endpoint_preserves_slope(
    backend: PygameBackend, surface: pygame.Surface
) -> None:
    """draw_line clips a diagonal line to the surface instead of distorting its slope."""
    backend.draw_line((10, 10), (1000, 1000), (255, 255, 255))

    lit_pixels = [
        (x, y)
        for x in range(100)
        for y in range(100)
        if surface.get_at((x, y))[:3] == (255, 255, 255)
    ]

    assert lit_pixels
    assert all(x == y for x, y in lit_pixels)


def test_draw_line_fully_offscreen_draws_nothing(
    backend: PygameBackend, surface: pygame.Surface
) -> None:
    """draw_line with both endpoints entirely outside the surface draws nothing."""
    before = pygame.image.tobytes(surface, "RGB")

    backend.draw_line((200, 200), (300, 300), (255, 255, 255))

    after = pygame.image.tobytes(surface, "RGB")
    assert before == after


# ---------------------------------------------------------------------------
# blit_array
# ---------------------------------------------------------------------------


def test_blit_array_sets_surface_pixels_from_frame(
    backend: PygameBackend, surface: pygame.Surface
) -> None:
    """blit_array copies an (H, W, 3) uint8 array onto the surface at (0, 0)."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20, 10] = (255, 0, 0)

    backend.blit_array(frame)

    assert surface.get_at((10, 20)) == (255, 0, 0, 255)


def test_blit_array_matches_blit_pil_image_output(
    backend: PygameBackend, surface: pygame.Surface
) -> None:
    """blit_array and blit_pil_image produce identical surface contents for the same frame."""
    from PIL import Image

    frame = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)

    backend.blit_array(frame)
    via_array = pygame.image.tobytes(surface, "RGB")

    backend.blit_pil_image(Image.fromarray(frame))
    via_pil = pygame.image.tobytes(surface, "RGB")

    assert via_array == via_pil
