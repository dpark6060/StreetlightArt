# Radical
"""Unit tests for streets_at_night.ui.imaging: viewport fit + surface round-trips."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402
from PIL import Image  # noqa: E402

from streets_at_night.ui.imaging import (  # noqa: E402
    ViewportFit,
    array_to_pygame_surface,
    compute_viewport_fit,
    fit_image_to_viewport,
    pil_to_pygame_surface,
)


def test_compute_viewport_fit_wide_into_tall_letterboxes_vertically():
    # Act — a 200x100 render into a 100x100 viewport is width-limited.
    fit = compute_viewport_fit(200, 100, 100, 100)

    # Assert
    assert fit == ViewportFit(width=100, height=50, x=0, y=25)


def test_compute_viewport_fit_tall_into_wide_letterboxes_horizontally():
    # Act — a 100x200 render into a 100x100 viewport is height-limited.
    fit = compute_viewport_fit(100, 200, 100, 100)

    # Assert
    assert fit == ViewportFit(width=50, height=100, x=25, y=0)


def test_compute_viewport_fit_exact_fit_no_offset():
    # Act
    fit = compute_viewport_fit(100, 100, 100, 100)

    # Assert
    assert fit == ViewportFit(width=100, height=100, x=0, y=0)


def test_compute_viewport_fit_floors_scaled_dims_at_one():
    # Act — a very wide render into a 1px-tall viewport still yields >=1px height.
    fit = compute_viewport_fit(1000, 10, 5, 1)

    # Assert
    assert fit.width >= 1
    assert fit.height >= 1


def test_fit_image_to_viewport_output_size_equals_viewport():
    # Arrange — a 200x100 image into a 120x120 viewport is letterboxed to viewport size.
    image = Image.new("RGB", (200, 100), (10, 20, 30))

    # Act
    fitted = fit_image_to_viewport(image, 120, 120)

    # Assert
    assert fitted.size == (120, 120)


def test_pil_to_pygame_surface_round_trips_pixel_values(headless_pygame):
    # Arrange
    image = Image.new("RGB", (3, 2), (12, 34, 56))
    image.putpixel((1, 0), (200, 100, 50))

    # Act
    surface = pil_to_pygame_surface(image)

    # Assert
    assert surface.get_size() == (3, 2)
    assert surface.get_at((0, 0))[:3] == (12, 34, 56)
    assert surface.get_at((1, 0))[:3] == (200, 100, 50)


def test_array_to_pygame_surface_round_trips_pixel_values(headless_pygame):
    # Arrange — an (H, W, 3) uint8 RGB frame with two distinct pixels.
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (12, 34, 56)
    frame[1, 2] = (200, 100, 50)

    # Act
    surface = array_to_pygame_surface(frame)

    # Assert — pygame surfaces are (width, height); frame is (height, width).
    assert surface.get_size() == (3, 2)
    assert surface.get_at((0, 0))[:3] == (12, 34, 56)
    assert surface.get_at((2, 1))[:3] == (200, 100, 50)
