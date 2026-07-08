# Radical
"""Unit tests for streets_at_night.models.building."""

import numpy as np

from streets_at_night.models.building import Building


def test_building_init_stores_all_fields():
    # Arrange
    footprint = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

    # Act
    building = Building(footprint=footprint, height=6.5, building_kind="retail", window_seed=1)

    # Assert
    assert np.array_equal(building.footprint, footprint)
    assert building.height == 6.5
    assert building.building_kind == "retail"
    assert building.window_seed == 1


def test_building_centroid_is_footprint_mean():
    # Arrange
    footprint = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

    # Act
    building = Building(footprint=footprint, height=6.5, building_kind="retail", window_seed=1)

    # Assert — centroid is the mean of the footprint vertices
    assert np.array_equal(building.centroid, np.array([5.0, 5.0]))
