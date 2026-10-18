# Radical
"""Tests for streets_at_night.models.car.CarLightPair."""

import math
import random

import numpy as np
import pytest

from streets_at_night.config.schema import CarConfig, ColorPoolConfig
from streets_at_night.models.car import CarLightPair
from streets_at_night.models.illumination import DirectionalConeIllumination
from streets_at_night.models.light_source import LightSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> random.Random:
    """Return a seeded RNG for reproducible tests."""
    return random.Random(42)


@pytest.fixture
def road_position() -> np.ndarray:
    """Return a center road position in ENU meters."""
    return np.array([500.0, 300.0, 0.0])


@pytest.fixture
def car_config() -> CarConfig:
    """Return a CarConfig built from DEFAULTS values."""
    return CarConfig(
        headlight_colors=ColorPoolConfig(colors=[
            ("#FFFFFF", 0.5),
            ("#FFF4E0", 0.3),
            ("#FFEE80", 0.15),
            ("#CCE0FF", 0.05),
        ]),
        taillight_colors=ColorPoolConfig(colors=[
            ("#FF2020", 0.6),
            ("#CC1010", 0.25),
            ("#FF3010", 0.15),
        ]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        headlight_illumination_inner_fraction=0.3,
        headlight_beam_length=25.0,
        headlight_beam_angle_deg=40.0,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
    )


@pytest.fixture
def car_pair(road_position, rng, car_config) -> CarLightPair:
    """Return a CarLightPair traveling East (bearing=0)."""
    return CarLightPair(
        position=road_position,
        travel_bearing=0.0,
        config=car_config,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# __init__ — two point lights plus the illumination-only cone carrier
# ---------------------------------------------------------------------------


def test_get_lights_returns_three_lights(car_pair):
    # Act
    lights = car_pair.get_lights()

    # Assert — two point lights plus the headlight-cone carrier
    assert len(lights) == 3


def test_get_lights_cone_carrier_emits_no_core(car_pair):
    # Act
    lights = car_pair.get_lights()

    # Assert — the third light contributes ground pool only, no core/glow
    assert lights[2].emits_core is False


def test_cone_carrier_direction_matches_travel_bearing(road_position, rng, car_config):
    # Arrange — bearing 0.0 = travelling due East → unit heading (1, 0)
    pair = CarLightPair(
        position=road_position, travel_bearing=0.0, config=car_config, rng=rng
    )

    # Act
    cone = pair.get_lights()[2]

    # Assert
    assert cone.illumination.direction[0] == pytest.approx(1.0)
    assert cone.illumination.direction[1] == pytest.approx(0.0)


def test_cone_carrier_uses_headlight_color_and_beam_config(road_position, rng, car_config):
    # Arrange — single-color headlight pool so the sampled color is known
    car_config.headlight_colors = ColorPoolConfig(colors=[("#AABBCC", 1.0)])
    pair = CarLightPair(
        position=road_position, travel_bearing=0.0, config=car_config, rng=rng
    )

    # Act
    cone = pair.get_lights()[2]

    # Assert — carrier color is the headlight color; cone shape comes from config
    assert cone.color == (0xAA, 0xBB, 0xCC)
    assert cone.illumination.length == car_config.headlight_beam_length
    assert cone.illumination.beam_angle_deg == car_config.headlight_beam_angle_deg
    assert cone.illumination.falloff == car_config.headlight_illumination_falloff
    assert cone.illumination.inner_fraction == car_config.headlight_illumination_inner_fraction


def test_get_lights_are_light_source_instances(car_pair):
    # Act
    lights = car_pair.get_lights()

    # Assert
    assert all(isinstance(light, LightSource) for light in lights)


def test_travel_bearing_stored(road_position, rng, car_config):
    # Arrange
    bearing = math.pi / 4

    # Act
    pair = CarLightPair(
        position=road_position, travel_bearing=bearing, config=car_config, rng=rng
    )

    # Assert
    assert pair.travel_bearing == bearing


# ---------------------------------------------------------------------------
# __init__ — light positions are offset perpendicular to bearing
# ---------------------------------------------------------------------------


def test_light_positions_differ_from_center(car_pair, road_position):
    # Act
    lights = car_pair.get_lights()

    # Assert — at least one light must be offset from the road center
    positions = [light.position for light in lights]
    same_as_center = [
        np.allclose(p[:2], road_position[:2]) for p in positions
    ]
    assert not all(same_as_center)


def test_light_positions_symmetric_about_center(car_pair, road_position):
    # Act
    lights = car_pair.get_lights()

    # Assert — midpoint of the two lights equals the road center position
    midpoint = (lights[0].position + lights[1].position) / 2.0
    np.testing.assert_allclose(midpoint[:2], road_position[:2], atol=1e-9)


def test_light_separation_within_range(car_config):
    # Arrange — run many samples to verify separation is within config range
    for seed in range(30):
        rng = random.Random(seed)
        pos = np.array([0.0, 0.0, 0.0])
        pair = CarLightPair(position=pos, travel_bearing=0.0, config=car_config, rng=rng)
        lights = pair.get_lights()
        separation = np.linalg.norm(lights[0].position - lights[1].position)

        # Assert
        assert (
            car_config.separation_min <= separation <= car_config.separation_max
        ), f"seed={seed} separation={separation}"


# ---------------------------------------------------------------------------
# resolve_facing — headlights when traveling toward camera
# ---------------------------------------------------------------------------


def test_resolve_facing_toward_camera_sets_bright_glow(car_pair):
    # Arrange — car travels East (bearing=0.0), camera looks West → travel
    # opposes camera forward → dot < 0 → car approaches viewer → headlights
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — the two point lights (headlights) have large glow radius
    assert all(light.glow_radius == 8.0 for light in lights[:2])


def test_resolve_facing_toward_camera_sets_brightness_range(car_pair, car_config):
    # Arrange — car approaches viewer, camera looks West
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — headlight brightness within config range
    assert all(
        car_config.headlight_brightness_min <= light.brightness <= car_config.headlight_brightness_max
        for light in lights
    )


# ---------------------------------------------------------------------------
# resolve_facing — taillights when traveling away from camera
# ---------------------------------------------------------------------------


def test_resolve_facing_away_from_camera_sets_dim_glow(car_pair):
    # Arrange — car travels East (bearing=0.0), camera also looks East →
    # travel aligns with camera forward → dot > 0 → car moves away → taillights
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — the two point lights (taillights) have small glow radius
    assert all(light.glow_radius == 3.0 for light in lights[:2])


def test_resolve_facing_away_from_camera_sets_brightness_range(car_pair, car_config):
    # Arrange — car moves away, camera looks East
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — the two point lights get taillight brightness (the cone carrier
    # keeps its headlight brightness, so scope to the point lights)
    assert all(
        car_config.taillight_brightness_min <= light.brightness <= car_config.taillight_brightness_max
        for light in lights[:2]
    )


# ---------------------------------------------------------------------------
# resolve_facing — cone carrier is camera-independent
# ---------------------------------------------------------------------------


def test_resolve_facing_away_keeps_headlight_cone(car_pair, car_config):
    # Arrange — car moving away shows taillights, but the cone is world-space
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    cone = car_pair.get_lights()[2]

    # Assert — carrier still holds a forward cone with headlight color
    assert isinstance(cone.illumination, DirectionalConeIllumination)
    assert cone.illumination.length == car_config.headlight_beam_length


def test_resolve_facing_toward_camera_clears_point_light_pools(car_pair):
    # Arrange — car approaching shows headlights; point lights no longer pool
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert — the cone carrier owns the ground pool now, not the point lights
    assert lights[0].illumination is None
    assert lights[1].illumination is None


# ---------------------------------------------------------------------------
# resolve_facing — both lights get the same color
# ---------------------------------------------------------------------------


def test_resolve_facing_headlights_both_same_color(car_pair):
    # Arrange — car approaches viewer → headlights
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert
    assert lights[0].color == lights[1].color


def test_resolve_facing_taillights_both_same_color(car_pair):
    # Arrange — car moves away from viewer → taillights
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    car_pair.resolve_facing(camera_forward)
    lights = car_pair.get_lights()

    # Assert
    assert lights[0].color == lights[1].color


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_light_positions(car_config):
    # Arrange
    pos = np.array([0.0, 0.0, 0.0])
    pair_a = CarLightPair(
        position=pos, travel_bearing=0.5, config=car_config, rng=random.Random(99)
    )
    pair_b = CarLightPair(
        position=pos, travel_bearing=0.5, config=car_config, rng=random.Random(99)
    )

    # Act
    lights_a = pair_a.get_lights()
    lights_b = pair_b.get_lights()

    # Assert
    for la, lb in zip(lights_a, lights_b):
        np.testing.assert_array_equal(la.position, lb.position)


# ---------------------------------------------------------------------------
# Config passthrough
# ---------------------------------------------------------------------------


def test_headlight_color_comes_from_config():
    # Arrange — single-entry pool so we know exactly which color will be picked
    single_color_config = CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#AABBCC", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#FF0000", 1.0)]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
        headlight_illumination_inner_fraction=0.3,
    )
    rng = random.Random(0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=single_color_config,
        rng=rng,
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — headlight color matches the only color in the pool
    expected = single_color_config.headlight_colors.colors[0]
    expected_hex = expected[0]
    r = int(expected_hex[1:3], 16)
    g = int(expected_hex[3:5], 16)
    b = int(expected_hex[5:7], 16)
    assert lights[0].color == (r, g, b)


def test_taillight_color_comes_from_config():
    # Arrange — single-entry pool so we know exactly which color will be picked
    single_color_config = CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#FFFFFF", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#DD1122", 1.0)]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
        headlight_illumination_inner_fraction=0.3,
    )
    rng = random.Random(0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=single_color_config,
        rng=rng,
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — taillight color matches the only color in the pool
    expected = single_color_config.taillight_colors.colors[0]
    expected_hex = expected[0]
    r = int(expected_hex[1:3], 16)
    g = int(expected_hex[3:5], 16)
    b = int(expected_hex[5:7], 16)
    assert lights[0].color == (r, g, b)


def test_separation_respects_config_range():
    # Arrange — tight separation range so any sample must fall within it
    tight_config = CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#FFFFFF", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#FF0000", 1.0)]),
        separation_min=3.0,
        separation_max=3.1,
        headlight_brightness_min=0.7,
        headlight_brightness_max=0.9,
        taillight_brightness_min=0.3,
        taillight_brightness_max=0.5,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=8.0,
        headlight_illumination_intensity=0.6,
        headlight_illumination_falloff=1.5,
        taillight_glow_radius=3.0,
        taillight_illumination_radius=2.0,
        taillight_illumination_intensity=0.1,
        taillight_illumination_falloff=2.0,
        taillight_illumination_inner_fraction=0.0,
        headlight_illumination_inner_fraction=0.3,
    )
    for seed in range(20):
        rng = random.Random(seed)
        pair = CarLightPair(
            position=np.array([0.0, 0.0, 0.0]),
            travel_bearing=0.0,
            config=tight_config,
            rng=rng,
        )
        lights = pair.get_lights()
        separation = np.linalg.norm(lights[0].position - lights[1].position)

        # Assert
        assert (
            tight_config.separation_min <= separation <= tight_config.separation_max
        ), f"seed={seed} separation={separation}"


# ---------------------------------------------------------------------------
# Config-driven glow/illumination properties
# ---------------------------------------------------------------------------


def _make_config_with_light_overrides(
    headlight_glow_radius: float = 8.0,
    headlight_illumination_intensity: float = 0.6,
    headlight_illumination_falloff: float = 1.5,
    headlight_illumination_inner_fraction: float = 0.3,
    headlight_beam_length: float = 25.0,
    headlight_beam_angle_deg: float = 40.0,
    taillight_glow_radius: float = 3.0,
    taillight_illumination_radius: float = 2.0,
    taillight_illumination_intensity: float = 0.1,
    taillight_illumination_falloff: float = 2.0,
    taillight_illumination_inner_fraction: float = 0.0,
    headlight_brightness_min: float = 0.7,
    headlight_brightness_max: float = 0.9,
    taillight_brightness_min: float = 0.3,
    taillight_brightness_max: float = 0.5,
) -> CarConfig:
    """Build a CarConfig with controllable glow/illumination and brightness fields."""
    return CarConfig(
        headlight_colors=ColorPoolConfig(colors=[("#FFFFFF", 1.0)]),
        taillight_colors=ColorPoolConfig(colors=[("#FF0000", 1.0)]),
        separation_min=1.6,
        separation_max=2.4,
        headlight_brightness_min=headlight_brightness_min,
        headlight_brightness_max=headlight_brightness_max,
        taillight_brightness_min=taillight_brightness_min,
        taillight_brightness_max=taillight_brightness_max,
        density_highway_min=25.0,
        density_highway_max=50.0,
        density_major_min=40.0,
        density_major_max=80.0,
        density_collector_min=80.0,
        density_collector_max=150.0,
        density_residential_min=150.0,
        density_residential_max=400.0,
        density_service_min=300.0,
        density_service_max=800.0,
        headlight_glow_radius=headlight_glow_radius,
        headlight_illumination_intensity=headlight_illumination_intensity,
        headlight_illumination_falloff=headlight_illumination_falloff,
        headlight_illumination_inner_fraction=headlight_illumination_inner_fraction,
        headlight_beam_length=headlight_beam_length,
        headlight_beam_angle_deg=headlight_beam_angle_deg,
        taillight_glow_radius=taillight_glow_radius,
        taillight_illumination_radius=taillight_illumination_radius,
        taillight_illumination_intensity=taillight_illumination_intensity,
        taillight_illumination_falloff=taillight_illumination_falloff,
        taillight_illumination_inner_fraction=taillight_illumination_inner_fraction,
    )


def test_headlight_glow_radius_comes_from_config():
    # Arrange — use non-default value to confirm it is read from config
    config = _make_config_with_light_overrides(headlight_glow_radius=20.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — only the two point lights carry the headlight glow
    assert all(light.glow_radius == 20.0 for light in lights[:2])


def test_headlight_beam_length_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(headlight_beam_length=40.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    cone = pair.get_lights()[2]

    # Assert
    assert cone.illumination.length == 40.0


def test_headlight_beam_angle_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(headlight_beam_angle_deg=55.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    cone = pair.get_lights()[2]

    # Assert
    assert cone.illumination.beam_angle_deg == 55.0


def test_headlight_illumination_intensity_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(headlight_illumination_intensity=0.9)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    cone = pair.get_lights()[2]

    # Assert — the cone carrier's pool strength comes from headlight config
    assert cone.illumination_intensity == 0.9


def test_taillight_glow_radius_comes_from_config():
    # Arrange — use non-default value to confirm it is read from config
    config = _make_config_with_light_overrides(taillight_glow_radius=10.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — only the two point lights carry the taillight glow
    assert all(light.glow_radius == 10.0 for light in lights[:2])


def test_taillight_illumination_radius_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(taillight_illumination_radius=7.0)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — the two point lights keep their small radial taillight pools
    assert all(light.illumination.radius == 7.0 for light in lights[:2])


def test_taillight_illumination_intensity_comes_from_config():
    # Arrange
    config = _make_config_with_light_overrides(taillight_illumination_intensity=0.05)
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — only the two point lights carry the taillight pool strength
    assert all(light.illumination_intensity == 0.05 for light in lights[:2])


# ---------------------------------------------------------------------------
# Brightness config threading
# ---------------------------------------------------------------------------


def test_headlight_brightness_comes_from_config():
    # Arrange — tight range so any sampled value must fall within it
    config = _make_config_with_light_overrides(
        headlight_brightness_min=0.85,
        headlight_brightness_max=0.86,
    )
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([-1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — the two point lights get headlight brightness in range
    assert all(0.85 <= light.brightness <= 0.86 for light in lights[:2])


def test_taillight_brightness_comes_from_config():
    # Arrange — tight range so any sampled value must fall within it
    config = _make_config_with_light_overrides(
        taillight_brightness_min=0.35,
        taillight_brightness_max=0.36,
    )
    pair = CarLightPair(
        position=np.array([0.0, 0.0, 0.0]),
        travel_bearing=0.0,
        config=config,
        rng=random.Random(0),
    )
    camera_forward = np.array([1.0, 0.0, 0.0])

    # Act
    pair.resolve_facing(camera_forward)
    lights = pair.get_lights()

    # Assert — the two point lights get taillight brightness in range
    assert all(0.35 <= light.brightness <= 0.36 for light in lights[:2])
