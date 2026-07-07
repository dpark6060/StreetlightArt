"""Builds a synthetic ENU-space test scene for devmode parameter tuning.

The scene contains 10 rows of content stacked along the Y axis. Each row
extends from X=0 to X=10000m. Rows 0-4 are road types with streetlights
and cars. Row 5 is a car-only row. Rows 6-9 are direct light source rows
(house lights, sign lights, parking lights, storefronts).
"""

import random
import typing as t

import numpy as np

from ..config.defaults import DEFAULTS
from ..config.loader import build_devmode_camera_config
from ..config.schema import SceneConfig
from ..models.commercial import create_parking_lot_light, create_sign_light, create_storefront_light
from ..models.house_light import create_house_light
from ..models.light_source import LightSource
from ..models.road_segment import RoadSegment, get_default_width
from ..models.road_types import RoadType
from ..models.streetlight import FixtureProfile, config_to_fixture_profile
from ..placement.car_placer import place_cars
from ..placement.streetlight_placer import place_streetlights
from ..scene.scene import Scene
from .devmode_camera import build_devmode_camera

if t.TYPE_CHECKING:
    from ..camera.camera import Camera


# Row layout: (label, y_offset_in_enu_meters)
ROW_LABELS: t.List[t.Tuple[str, float]] = [
    ("Highway",         900.0),
    ("Major Road",      700.0),
    ("Collector",       500.0),
    ("Residential",     300.0),
    ("Service",         100.0),
    ("Cars",           -100.0),
    ("House Lights",   -300.0),
    ("Sign Lights",    -500.0),
    ("Parking Lights", -700.0),
    ("Storefront",     -900.0),
]

# Road type for each of rows 0-4 (index matches ROW_LABELS)
_ROW_ROAD_TYPES: t.List[RoadType] = [
    RoadType.HIGHWAY,
    RoadType.MAJOR_ROAD,
    RoadType.COLLECTOR,
    RoadType.RESIDENTIAL,
    RoadType.SERVICE,
]

# X extent for all rows
_ROW_X_START = 0.0
_ROW_X_END = 10000.0

# Spacing between intermediate geometry points (meters)
_SEGMENT_STEP: float = 200.0

# Spacing between light instances in the direct-light rows (meters)
_DIRECT_LIGHT_SPACING = 80.0


def build_synthetic_scene(config: SceneConfig) -> Scene:
    """Build a synthetic ENU-space scene for devmode parameter tuning.

    Creates 10 rows of visual content. Rows 0-4 are road segments with
    streetlights and cars. Row 5 is a car-only row. Rows 6-9 are direct
    light source rows placed at evenly-spaced X positions.

    Args:
        config (SceneConfig): Scene configuration. Uses config.devmode for
            camera parameters if present; falls back to built-in defaults.

    Returns:
        Scene: Fully constructed synthetic scene with lights indexed.
    """
    camera = _build_camera(config)
    road_segments = _build_road_segments()
    scene = _create_scene(road_segments, camera, config)
    scene.build_spatial_index()

    rng = random.Random(config.seed)
    _populate_road_lights(scene, config, rng, road_segments)
    _populate_direct_light_rows(scene, config, rng)

    return scene


def _build_camera(config: SceneConfig) -> "Camera":
    """Construct the devmode camera from config or built-in defaults.

    Args:
        config (SceneConfig): Scene config; devmode section may be None.

    Returns:
        Camera: Configured ENU-space camera.
    """
    if config.devmode is not None:
        camera_cfg = config.devmode.camera
        width = config.devmode.render_width
        height = config.devmode.render_height
    else:
        camera_cfg = build_devmode_camera_config(DEFAULTS["devmode"]["camera"])
        width = DEFAULTS["devmode"]["render_width"]
        height = DEFAULTS["devmode"]["render_height"]

    return build_devmode_camera(camera_cfg, width, height)


_DIRECT_LIGHT_ROW_ROAD_WIDTH = 20.0


def _make_row_geometry(y: float) -> np.ndarray:
    """Create a multi-point road geometry with intermediate vertices.

    The road renderer requires both endpoints of each sub-segment to project
    on-screen. A 2-point 10km line fails this check at typical devmode
    distances. Adding intermediate points every _SEGMENT_STEP meters ensures
    sub-segments within the camera frustum have on-screen endpoints.

    Args:
        y (float): Y coordinate (ENU meters) for this row.

    Returns:
        np.ndarray: (N, 3) geometry array from _ROW_X_START to _ROW_X_END.
    """
    x_values = np.arange(_ROW_X_START, _ROW_X_END + _SEGMENT_STEP, _SEGMENT_STEP)
    if x_values[-1] != _ROW_X_END:
        x_values = np.append(x_values, _ROW_X_END)
    n = len(x_values)
    geom = np.zeros((n, 3))
    geom[:, 0] = x_values
    geom[:, 1] = y
    return geom


def _build_road_segments() -> t.List[RoadSegment]:
    """Create ten road segments for rows 0-9.

    Rows 0-4 correspond to the five road types with streetlights. Row 5 is
    the car-only row using RoadType.HIGHWAY geometry but with
    fixture_profile=None to prevent streetlight placement. Rows 6-9 are wide
    surface segments (RoadType.RESIDENTIAL, fixture_profile=None) placed under
    the direct light source rows so the illumination pass has road geometry to
    illuminate.

    Each segment is built with _make_row_geometry(), which inserts intermediate
    vertices every _SEGMENT_STEP meters so the road renderer can project
    individual sub-segments that fall within the camera frustum.

    Returns:
        list[RoadSegment]: Ten road segments, one per row.
    """
    segments = []

    for road_type, (_, y_offset) in zip(_ROW_ROAD_TYPES, ROW_LABELS[:5]):
        geom = _make_row_geometry(y_offset)
        width = get_default_width(road_type)
        segments.append(RoadSegment(geometry=geom, road_type=road_type, width=width))

    car_row_y = ROW_LABELS[5][1]
    car_segment = RoadSegment(
        geometry=_make_row_geometry(car_row_y),
        road_type=RoadType.HIGHWAY,
        width=get_default_width(RoadType.HIGHWAY),
        fixture_profile=None,
    )
    segments.append(car_segment)

    for _, y_offset in ROW_LABELS[6:10]:
        segments.append(RoadSegment(
            geometry=_make_row_geometry(y_offset),
            road_type=RoadType.RESIDENTIAL,
            width=_DIRECT_LIGHT_ROW_ROAD_WIDTH,
            fixture_profile=None,
        ))

    return segments


def _create_scene(
    road_segments: t.List[RoadSegment],
    camera: "Camera",
    config: SceneConfig,
) -> Scene:
    """Instantiate a Scene with the given segments and camera.

    Args:
        road_segments (list[RoadSegment]): All ten road segments.
        camera (Camera): The devmode camera.
        config (SceneConfig): Scene configuration.

    Returns:
        Scene: Uninitialised scene (no spatial index, no lights yet).
    """
    return Scene(
        road_segments=road_segments,
        camera=camera,
        config=config,
        residential_zones=[],
        commercial_zones=[],
        dark_zones=[],
        transformer=None,
    )


def _populate_road_lights(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
    road_segments: t.List[RoadSegment],
) -> None:
    """Place streetlights and cars on road segments and add to scene.

    Streetlights are placed on segments 0-4 (road type rows) only. After
    placement, any segment whose pool pick was None (no-lights) is forced to
    use the first real profile for its road type, ensuring every devmode row
    always has visible streetlights for parameter tuning.

    The car placer runs on segments 0-5 (road type rows and car-only row),
    excluding the direct light surface segments in rows 6-9.

    Args:
        scene (Scene): The scene to populate.
        config (SceneConfig): Scene configuration.
        rng (random.Random): Master RNG seeded from config.seed.
        road_segments (list[RoadSegment]): All ten road segments.
    """
    streetlight_rng = random.Random(rng.randint(0, 2**32))
    car_rng = random.Random(rng.randint(0, 2**32))

    streetlights = place_streetlights(road_segments[:5], config.lights, streetlight_rng)
    forced_lights = _force_lights_on_dark_segments(road_segments[:5], config, streetlight_rng)
    car_lights, car_pairs = place_cars(road_segments[:6], config.lights, car_rng)

    scene.add_lights(streetlights)
    scene.add_lights(forced_lights)
    scene.add_lights(car_lights)
    scene.add_car_pairs(car_pairs)


def _force_lights_on_dark_segments(
    road_segments: t.List[RoadSegment],
    config: SceneConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place streetlights on any segment that received a no-lights pool pick.

    After place_streetlights runs, segments whose pool pick was None have
    fixture_profile=None. For the devmode synthetic scene, every road-type row
    must be lit so the parameter panel is useful. This function assigns the
    first real FixtureProfile from the road type's pool to any dark segment
    and re-runs light placement for it.

    A forced pool containing only the non-None entries (equal weights) is
    used so the pick is guaranteed to return a real profile.

    Args:
        road_segments (list[RoadSegment]): The five road-type segments.
        config (SceneConfig): Scene configuration with pool entries.
        rng (random.Random): RNG to use for profile construction and placement.

    Returns:
        list[LightSource]: Lights for all previously-dark segments, or empty
            if all segments already had profiles.
    """
    from ..placement.streetlight_placer import place_lights_on_segment

    lights: t.List[LightSource] = []
    for segment in road_segments:
        if segment.fixture_profile is not None:
            continue
        profile = _pick_first_real_profile(segment.road_type, config, rng)
        if profile is None:
            continue
        segment.fixture_profile = profile
        lights.extend(place_lights_on_segment(segment, rng))
    return lights


def _pick_first_real_profile(
    road_type: RoadType,
    config: SceneConfig,
    rng: random.Random,
) -> t.Optional[FixtureProfile]:
    """Build and return a FixtureProfile from the first non-None pool entry.

    Reads the pool entries for the road type and creates a FixtureProfile from
    the first entry that has a non-None profile config. Returns None if all
    entries are no-lights entries.

    Args:
        road_type (RoadType): The road classification.
        config (SceneConfig): Scene configuration.
        rng (random.Random): RNG for sampling the profile's range fields.

    Returns:
        FixtureProfile or None: Profile from the first real pool entry, or None.
    """
    rt_attr = {
        RoadType.HIGHWAY: "highway",
        RoadType.MAJOR_ROAD: "major_road",
        RoadType.COLLECTOR: "collector",
        RoadType.RESIDENTIAL: "residential",
        RoadType.SERVICE: "service",
        RoadType.MINOR: "residential",
    }[road_type]
    shift = config.lights.streetlights.color_temp_shift_k
    entries = getattr(config.lights.streetlights, rt_attr)
    for entry in entries:
        if entry.profile is not None:
            return config_to_fixture_profile(entry.profile, rng, shift)
    return None


def _populate_direct_light_rows(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
) -> None:
    """Place direct light sources for rows 6-9 and add them to the scene.

    Each row gets lights at evenly-spaced X positions from 0 to 10000m.
    Row 6: house lights. Row 7: sign lights. Row 8: parking lights.
    Row 9: storefront lights.

    Args:
        scene (Scene): The scene to populate.
        config (SceneConfig): Scene configuration providing light params.
        rng (random.Random): Master RNG; child RNGs are derived from it.
    """
    house_rng = random.Random(rng.randint(0, 2**32))
    sign_rng = random.Random(rng.randint(0, 2**32))
    parking_rng = random.Random(rng.randint(0, 2**32))
    storefront_rng = random.Random(rng.randint(0, 2**32))

    x_positions = _get_row_x_positions()

    house_y = ROW_LABELS[6][1]
    sign_y = ROW_LABELS[7][1]
    parking_y = ROW_LABELS[8][1]
    storefront_y = ROW_LABELS[9][1]

    lights: t.List[LightSource] = []
    lights.extend(_create_house_light_row(x_positions, house_y, config, house_rng))
    lights.extend(_create_sign_light_row(x_positions, sign_y, config, sign_rng))
    lights.extend(_create_parking_light_row(x_positions, parking_y, config, parking_rng))
    lights.extend(_create_storefront_light_row(x_positions, storefront_y, config, storefront_rng))

    scene.add_lights(lights)


def _get_row_x_positions() -> t.List[float]:
    """Return evenly-spaced X positions for direct light rows.

    Returns:
        list[float]: X coordinates at _DIRECT_LIGHT_SPACING intervals.
    """
    positions = []
    x = _DIRECT_LIGHT_SPACING / 2.0
    while x < _ROW_X_END:
        positions.append(x)
        x += _DIRECT_LIGHT_SPACING
    return positions


def _create_house_light_row(
    x_positions: t.List[float],
    y: float,
    config: SceneConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Create house light instances along a row.

    Args:
        x_positions (list[float]): X coordinates for light placement.
        y (float): Y coordinate (ENU meters) for this row.
        config (SceneConfig): Scene configuration.
        rng (random.Random): RNG for this row.

    Returns:
        list[LightSource]: House lights for this row.
    """
    return [
        create_house_light(np.array([x, y, 3.0]), config.lights.house_lights, rng)
        for x in x_positions
    ]


def _create_sign_light_row(
    x_positions: t.List[float],
    y: float,
    config: SceneConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Create commercial sign light instances along a row.

    Args:
        x_positions (list[float]): X coordinates for light placement.
        y (float): Y coordinate (ENU meters) for this row.
        config (SceneConfig): Scene configuration.
        rng (random.Random): RNG for this row.

    Returns:
        list[LightSource]: Sign lights for this row.
    """
    return [
        create_sign_light(np.array([x, y, 5.0]), config.lights.commercial, rng)
        for x in x_positions
    ]


def _create_parking_light_row(
    x_positions: t.List[float],
    y: float,
    config: SceneConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Create parking lot light instances along a row.

    Args:
        x_positions (list[float]): X coordinates for light placement.
        y (float): Y coordinate (ENU meters) for this row.
        config (SceneConfig): Scene configuration.
        rng (random.Random): RNG for this row.

    Returns:
        list[LightSource]: Parking lot lights for this row.
    """
    return [
        create_parking_lot_light(np.array([x, y, 8.0]), config.lights.commercial, rng)
        for x in x_positions
    ]


def _create_storefront_light_row(
    x_positions: t.List[float],
    y: float,
    config: SceneConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Create storefront light instances along a row.

    Args:
        x_positions (list[float]): X coordinates for light placement.
        y (float): Y coordinate (ENU meters) for this row.
        config (SceneConfig): Scene configuration.
        rng (random.Random): RNG for this row.

    Returns:
        list[LightSource]: Storefront lights for this row.
    """
    return [
        create_storefront_light(np.array([x, y, 4.0]), config.lights.commercial, rng)
        for x in x_positions
    ]
