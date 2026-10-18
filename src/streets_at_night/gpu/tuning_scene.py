"""Human-scale light-tuning scene: one road, one building, a car, a person, and poles.

Layout (ENU meters, origin at the first pole; the road runs east-west):

    building  ................  north side, set back from the road
    ===== road, y in [0, 10] =====      car in the eastbound (south) lane
    pole0 person    pole1    pole2    pole3      south curb, y = -0.3

Every light is built with the same factory functions the city scene uses, so a
value tuned here means the same thing in the 2D render.
"""

import random
import typing as t
from dataclasses import dataclass

import numpy as np

from ..config.schema import CommercialLightConfig, FixtureProfileConfig, SceneConfig
from ..models.building import Building
from ..models.car import CarLightPair
from ..models.commercial import (
    create_parking_lot_light,
    create_sign_light,
    create_storefront_light,
)
from ..models.light_source import LightSource
from ..models.road_types import RoadType
from ..models.streetlight import StreetLight, config_to_fixture_profile
from ..rendering.building_pass import LitWindow, get_lit_windows
from ..utils.colors import hex_to_rgb

# Road: one east-west strip, two 5 m lanes, south edge on y = 0.
ROAD_X_RANGE: t.Tuple[float, float] = (-150.0, 250.0)
ROAD_Y_RANGE: t.Tuple[float, float] = (0.0, 10.0)

# Poles stand just off the south edge; the first one is next to the person.
POLE_COUNT: int = 4
POLE_Y: float = -0.3
_POLE_WIDTH_M: float = 0.2
_POLE_COLOR: t.Tuple[int, int, int] = (60, 60, 62)

# Average sedan: 4.7 m long (plan-04), parked facing east in the south lane.
CAR_SIZE_M: t.Tuple[float, float, float] = (4.7, 1.8, 1.45)
CAR_CENTER: t.Tuple[float, float] = (3.0, 2.5)
_CAR_LIGHT_HEIGHT_M: float = 0.7
_CAR_COLOR: t.Tuple[int, int, int] = (40, 44, 52)

# Average adult standing next to the first pole.
PERSON_SIZE_M: t.Tuple[float, float, float] = (0.45, 0.3, 1.75)
PERSON_CENTER: t.Tuple[float, float] = (1.2, -1.2)
_PERSON_COLOR: t.Tuple[int, int, int] = (110, 90, 75)

# Small two-floor shop north of the road; its south wall faces the street.
_BUILDING_FOOTPRINT: np.ndarray = np.array(
    [[-18.0, 16.0], [-10.0, 16.0], [-10.0, 22.0], [-18.0, 22.0]]
)
_BUILDING_HEIGHT_M: float = 6.0
_BUILDING_KIND: str = "retail"
_BUILDING_WINDOW_SEED: int = 7
_FALLBACK_WALL_COLOR: str = "#0B0A09"
_FALLBACK_ROOF_COLOR: str = "#080807"

# Building-mounted lights sit just off the street-facing wall.
_FACADE_OFFSET_M: float = 0.3
_SIGN_HEIGHT_M: float = 5.0
_STOREFRONT_HEIGHT_M: float = 3.0

PoleFixture = t.Union[FixtureProfileConfig, CommercialLightConfig]


@dataclass
class Box:
    """An axis-aligned box resting on the ground (walls + roof, no floor).

    Attributes:
        min_xy: (x, y) southwest corner in meters.
        max_xy: (x, y) northeast corner in meters.
        height: Box height in meters.
        wall_color: sRGB wall color.
        roof_color: sRGB roof color.
    """

    min_xy: t.Tuple[float, float]
    max_xy: t.Tuple[float, float]
    height: float
    wall_color: t.Tuple[int, int, int]
    roof_color: t.Tuple[int, int, int]


@dataclass
class TuningScene:
    """Everything the GPU renderer needs for one build of the tuning scene.

    Attributes:
        ground_color: sRGB ground color.
        road_color: sRGB road surface color.
        boxes: Car, person, and pole boxes.
        building: The shop, rendered with config wall/roof colors.
        lights: Every light source, already resolved (car facing is fixed).
        windows: Lit windows on the building (empty when building lights are off).
    """

    ground_color: t.Tuple[int, int, int]
    road_color: t.Tuple[int, int, int]
    boxes: t.List[Box]
    building: Building
    lights: t.List[LightSource]
    windows: t.List[LitWindow]


def build_tuning_scene(
    config: SceneConfig,
    pole_fixture: PoleFixture,
    road_lights: bool,
    building_lights: bool,
) -> TuningScene:
    """Build the tuning scene for the selected pole fixture.

    Uses a fixed seed so dragging a slider never reshuffles the random draws
    (pole height, spacing, colors) underneath it.

    Args:
        config (SceneConfig): Working config; lights are built from it.
        pole_fixture (FixtureProfileConfig | CommercialLightConfig): A
            streetlight pool profile, or the commercial config to place
            parking lot lights on the poles.
        road_lights (bool): Include pole lights and car lights.
        building_lights (bool): Include windows, the sign, and storefronts.

    Returns:
        TuningScene: The built scene.
    """
    rng = random.Random(config.seed)
    pole_lights, pole_boxes = _build_poles(config, pole_fixture, rng)
    building = Building(
        footprint=_BUILDING_FOOTPRINT.copy(),
        height=_BUILDING_HEIGHT_M,
        building_kind=_BUILDING_KIND,
        window_seed=_BUILDING_WINDOW_SEED,
    )

    lights: t.List[LightSource] = []
    if road_lights:
        lights.extend(pole_lights)
        lights.extend(_build_car_lights(config, rng))
    windows: t.List[LitWindow] = []
    if building_lights:
        lights.extend(_build_facade_lights(config.lights.commercial, rng))
        if config.buildings is not None:
            windows = get_lit_windows(building, config.buildings)

    colors = config.zones.colors
    return TuningScene(
        ground_color=hex_to_rgb(colors.default_ground),
        road_color=hex_to_rgb(colors.road_surface),
        boxes=[_car_box(), _person_box(), *pole_boxes],
        building=building,
        lights=lights,
        windows=windows,
    )


def building_colors(config: SceneConfig) -> t.Tuple[t.Tuple[int, int, int], t.Tuple[int, int, int], float]:
    """Return the building wall color, roof color, and face shade strength.

    Args:
        config (SceneConfig): Working config.

    Returns:
        tuple: (wall sRGB, roof sRGB, face_shade_strength).
    """
    if config.buildings is None:
        return hex_to_rgb(_FALLBACK_WALL_COLOR), hex_to_rgb(_FALLBACK_ROOF_COLOR), 0.2
    cfg = config.buildings
    return hex_to_rgb(cfg.wall_color), hex_to_rgb(cfg.roof_color), cfg.face_shade_strength


def _build_poles(
    config: SceneConfig,
    pole_fixture: PoleFixture,
    rng: random.Random,
) -> t.Tuple[t.List[LightSource], t.List[Box]]:
    """Place POLE_COUNT poles east from x = 0 at the fixture's own spacing.

    Args:
        config (SceneConfig): Working config.
        pole_fixture (FixtureProfileConfig | CommercialLightConfig): What to mount.
        rng (random.Random): Scene RNG.

    Returns:
        tuple[list[LightSource], list[Box]]: Pole lights and pole boxes.
    """
    lights: t.List[LightSource] = []
    if isinstance(pole_fixture, FixtureProfileConfig):
        # One profile per segment, like the real placer: every pole on a road
        # segment shares height, brightness, and spacing.
        profile = config_to_fixture_profile(
            pole_fixture, rng, config.lights.streetlights.color_temp_shift_k
        )
        for i in range(POLE_COUNT):
            position = np.array([i * profile.spacing, POLE_Y, profile.height])
            lights.append(
                StreetLight.create_from_profile(profile, position, RoadType.RESIDENTIAL)
            )
    else:
        spacing = rng.uniform(
            pole_fixture.parking_grid_spacing_min, pole_fixture.parking_grid_spacing_max
        )
        for i in range(POLE_COUNT):
            height = rng.uniform(pole_fixture.parking_height_min, pole_fixture.parking_height_max)
            position = np.array([i * spacing, POLE_Y, height])
            lights.append(create_parking_lot_light(position, pole_fixture, rng))

    boxes = [_pole_box(light.position) for light in lights]
    return lights, boxes


def _build_car_lights(config: SceneConfig, rng: random.Random) -> t.List[LightSource]:
    """Build the parked car's headlights (with cone) and taillights.

    Two CarLightPairs: one at the front bumper resolved as headlights (it
    carries the headlight ground cone) and one at the rear resolved as
    taillights, so both are visible at once regardless of camera angle.

    Args:
        config (SceneConfig): Working config.
        rng (random.Random): Scene RNG.

    Returns:
        list[LightSource]: Headlight points + cone carrier, then taillight points.
    """
    half_length = CAR_SIZE_M[0] / 2.0
    cx, cy = CAR_CENTER
    cars = config.lights.cars

    front = CarLightPair(np.array([cx + half_length, cy, _CAR_LIGHT_HEIGHT_M]), 0.0, cars, rng)
    front.resolve_facing(np.array([-1.0, 0.0, 0.0]))
    rear = CarLightPair(np.array([cx - half_length, cy, _CAR_LIGHT_HEIGHT_M]), 0.0, cars, rng)
    rear.resolve_facing(np.array([1.0, 0.0, 0.0]))
    # get_lights() is (left point, right point, cone carrier); the rear pair's
    # cone would be a second headlight pool, so only its two points are kept.
    return [*front.get_lights(), *rear.get_lights()[:2]]


def _build_facade_lights(
    commercial: CommercialLightConfig, rng: random.Random
) -> t.List[LightSource]:
    """Mount one sign and a row of storefront lights on the street-facing wall.

    Args:
        commercial (CommercialLightConfig): Commercial light config.
        rng (random.Random): Scene RNG.

    Returns:
        list[LightSource]: The sign light followed by storefront lights.
    """
    (x0, y_wall), (x1, _) = _BUILDING_FOOTPRINT[0], _BUILDING_FOOTPRINT[1]
    y = y_wall - _FACADE_OFFSET_M
    lights = [create_sign_light(np.array([(x0 + x1) / 2.0, y, _SIGN_HEIGHT_M]), commercial, rng)]

    spacing = rng.uniform(commercial.storefront_spacing_min, commercial.storefront_spacing_max)
    x = x0 + spacing / 2.0
    while x < x1:
        lights.append(create_storefront_light(np.array([x, y, _STOREFRONT_HEIGHT_M]), commercial, rng))
        x += spacing
    return lights


def _car_box() -> Box:
    """Return the parked car's box."""
    return _centered_box(CAR_CENTER, CAR_SIZE_M, _CAR_COLOR)


def _person_box() -> Box:
    """Return the standing person's box."""
    return _centered_box(PERSON_CENTER, PERSON_SIZE_M, _PERSON_COLOR)


def _pole_box(light_position: np.ndarray) -> Box:
    """Return a pole box from the ground up to the light's height.

    Args:
        light_position (np.ndarray): (3,) light position at the pole top.

    Returns:
        Box: The pole.
    """
    size = (_POLE_WIDTH_M, _POLE_WIDTH_M, float(light_position[2]))
    return _centered_box((float(light_position[0]), float(light_position[1])), size, _POLE_COLOR)


def _centered_box(
    center: t.Tuple[float, float],
    size: t.Tuple[float, float, float],
    color: t.Tuple[int, int, int],
) -> Box:
    """Build a box centered on (x, y) with the given (length, width, height).

    Args:
        center (tuple[float, float]): Ground-plane center in meters.
        size (tuple[float, float, float]): (x extent, y extent, height) in meters.
        color (tuple[int, int, int]): sRGB color for walls and roof.

    Returns:
        Box: The box.
    """
    half_x, half_y = size[0] / 2.0, size[1] / 2.0
    return Box(
        min_xy=(center[0] - half_x, center[1] - half_y),
        max_xy=(center[0] + half_x, center[1] + half_y),
        height=size[2],
        wall_color=color,
        roof_color=color,
    )
