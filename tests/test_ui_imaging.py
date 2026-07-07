# Radical
"""Unit tests for streets_at_night.ui.imaging viewport-fit math."""

import pytest

from streets_at_night.ui.imaging import ViewportFit, compute_viewport_fit


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
