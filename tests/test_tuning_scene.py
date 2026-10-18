# Radical
"""Tests for streets_at_night.gpu.tuning_scene."""

import numpy as np
import pytest

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import SceneConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.gpu.tuning_scene import (
    CAR_SIZE_M,
    PERSON_SIZE_M,
    POLE_COUNT,
    build_tuning_scene,
)
from streets_at_night.models.illumination import BeamIllumination, DirectionalConeIllumination


@pytest.fixture
def config() -> SceneConfig:
    """Return a SceneConfig built from project defaults."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


@pytest.fixture
def highway_profile(config):
    """Return the first highway fixture profile config."""
    return config.lights.streetlights.highway[0].profile


def _pole_lights(scene):
    return [l for l in scene.lights if isinstance(l.illumination, BeamIllumination)]


def test_build_tuning_scene_places_pole_count_streetlights(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=False)

    # Assert
    assert len(_pole_lights(scene)) == POLE_COUNT


def test_build_tuning_scene_pole_spacing_within_profile_range(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=False)

    # Assert — every gap equals one sampled spacing, inside the authored range
    xs = sorted(float(l.position[0]) for l in _pole_lights(scene))
    gaps = np.diff(xs)
    np.testing.assert_allclose(gaps, gaps[0])
    assert highway_profile.spacing_min <= gaps[0] <= highway_profile.spacing_max


def test_build_tuning_scene_pole_height_within_profile_range(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=False)

    # Assert
    for light in _pole_lights(scene):
        assert highway_profile.height_min <= light.position[2] <= highway_profile.height_max


def test_build_tuning_scene_parking_fixture_uses_parking_beam(config):
    # Arrange
    commercial = config.lights.commercial

    # Act
    scene = build_tuning_scene(config, commercial, road_lights=True, building_lights=False)

    # Assert
    poles = _pole_lights(scene)
    assert len(poles) == POLE_COUNT
    assert all(l.illumination.beam_angle_deg == commercial.parking_beam_angle_deg for l in poles)


def test_build_tuning_scene_same_config_builds_same_scene(config, highway_profile):
    # Act
    a = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=True)
    b = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=True)

    # Assert — fixed seed, so slider drags never reshuffle the scene
    np.testing.assert_allclose(
        [l.position for l in a.lights], [l.position for l in b.lights]
    )


def test_build_tuning_scene_road_lights_off_drops_pole_and_car_lights(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=False, building_lights=False)

    # Assert
    assert scene.lights == []


def test_build_tuning_scene_includes_one_headlight_cone(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=False)

    # Assert
    cones = [l for l in scene.lights if isinstance(l.illumination, DirectionalConeIllumination)]
    assert len(cones) == 1
    assert cones[0].illumination.direction == pytest.approx((1.0, 0.0))


def test_build_tuning_scene_building_lights_off_has_no_windows(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=False)

    # Assert
    assert scene.windows == []


def test_build_tuning_scene_building_lights_on_adds_facade_lights(config, highway_profile):
    # Act
    off = build_tuning_scene(config, highway_profile, road_lights=False, building_lights=False)
    on = build_tuning_scene(config, highway_profile, road_lights=False, building_lights=True)

    # Assert — at least the sign plus one storefront light
    assert len(on.lights) - len(off.lights) >= 2


def test_build_tuning_scene_car_and_person_have_reference_sizes(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=True, building_lights=False)

    # Assert
    car, person = scene.boxes[0], scene.boxes[1]
    assert car.max_xy[0] - car.min_xy[0] == pytest.approx(CAR_SIZE_M[0])
    assert car.height == pytest.approx(CAR_SIZE_M[2])
    assert person.height == pytest.approx(PERSON_SIZE_M[2])


def test_build_tuning_scene_one_pole_box_per_pole(config, highway_profile):
    # Act
    scene = build_tuning_scene(config, highway_profile, road_lights=False, building_lights=False)

    # Assert — poles stay standing even with road lights off
    assert len(scene.boxes) == 2 + POLE_COUNT
