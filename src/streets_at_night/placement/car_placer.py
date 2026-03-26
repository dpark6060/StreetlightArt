"""Place cars (CarLightPair instances) along road segments."""

import math
import random
import typing as t

import numpy as np

from ..config.schema import CarConfig, LightConfig
from ..models.car import CarLightPair
from ..models.light_source import LightSource
from ..models.road_segment import RoadSegment
from ..models.road_types import RoadType
from .geometry_utils import perpendicular_2d as _perpendicular_2d


# Lane offset in meters — cars placed offset from road centerline
_LANE_OFFSET = 1.5


# Maps RoadType → (density_min, density_max) field names on CarConfig
_DENSITY_FIELD_MAP: t.Dict[RoadType, t.Tuple[str, str]] = {
    RoadType.HIGHWAY:     ("density_highway_min",     "density_highway_max"),
    RoadType.MAJOR_ROAD:  ("density_major_min",       "density_major_max"),
    RoadType.COLLECTOR:   ("density_collector_min",   "density_collector_max"),
    RoadType.RESIDENTIAL: ("density_residential_min", "density_residential_max"),
    RoadType.SERVICE:     ("density_service_min",     "density_service_max"),
    RoadType.MINOR:       ("density_residential_min", "density_residential_max"),
}


def place_cars(
    road_segments: t.List[RoadSegment],
    config: LightConfig,
    rng: random.Random,
) -> t.Tuple[t.List[LightSource], t.List[CarLightPair]]:
    """Place cars along all road segments and return lights and car pairs.

    Walks each segment at random intervals determined by road-type density
    from config. Each car is placed offset from the road centerline with a
    travel bearing derived from the local road tangent.

    Returns both the flat list of LightSources (for the scene) and the
    CarLightPair objects (needed for render-time facing resolution).

    Args:
        road_segments (List[RoadSegment]): All road segments in the scene.
        config (LightConfig): Light configuration providing car density ranges.
        rng (random.Random): Seeded RNG for reproducible car placement.

    Returns:
        Tuple[List[LightSource], List[CarLightPair]]:
            - Flat list of all car LightSources.
            - All CarLightPair objects for resolve_facing at render time.
    """
    all_lights: t.List[LightSource] = []
    all_pairs: t.List[CarLightPair] = []

    for segment in road_segments:
        pairs = _place_cars_on_segment(segment, config.cars, rng)
        for pair in pairs:
            all_lights.extend(pair.get_lights())
        all_pairs.extend(pairs)

    return all_lights, all_pairs


# ---------------------------------------------------------------------------
# Per-segment placement
# ---------------------------------------------------------------------------


def _place_cars_on_segment(
    segment: RoadSegment,
    car_config: CarConfig,
    rng: random.Random,
) -> t.List[CarLightPair]:
    """Place cars along a single road segment.

    Args:
        segment (RoadSegment): The road segment to populate.
        car_config (CarConfig): Car density and light configuration.
        rng (random.Random): Seeded RNG for spacing and direction decisions.

    Returns:
        List[CarLightPair]: All car pairs placed on this segment.
    """
    density_min, density_max = _get_density_range(segment.road_type, car_config)
    sample_points, sample_tangents = _walk_with_random_spacing(
        segment.geometry, density_min, density_max, rng
    )

    pairs = []
    for point, tangent in zip(sample_points, sample_tangents):
        pair = _create_car_at_point(
            point, tangent, segment.road_type, car_config, rng
        )
        pairs.append(pair)
    return pairs


def _get_density_range(
    road_type: RoadType,
    car_config: CarConfig,
) -> t.Tuple[float, float]:
    """Return the (min, max) spacing interval for cars on this road type.

    Args:
        road_type (RoadType): The road classification.
        car_config (CarConfig): Config object holding per-road-type density fields.

    Returns:
        Tuple[float, float]: (density_min, density_max) in meters between cars.
    """
    min_field, max_field = _DENSITY_FIELD_MAP[road_type]
    return getattr(car_config, min_field), getattr(car_config, max_field)


def _create_car_at_point(
    point: np.ndarray,
    tangent: np.ndarray,
    road_type: RoadType,
    car_config: CarConfig,
    rng: random.Random,
) -> CarLightPair:
    """Create a single CarLightPair at a sampled road point.

    Offsets the car from the centerline into a lane and assigns a bearing.
    Highway roads keep all cars on one bearing per lane side. Other roads
    assign direction randomly 50/50.

    Args:
        point (np.ndarray): 3-element ENU centerline position.
        tangent (np.ndarray): 3-element tangent unit vector at this point.
        road_type (RoadType): Used to determine highway vs bidirectional logic.
        car_config (CarConfig): Passed through to CarLightPair.
        rng (random.Random): RNG for direction and offset decisions.

    Returns:
        CarLightPair: Newly created car light pair.
    """
    bearing = _bearing_from_tangent(tangent)
    lane_sign, travel_bearing = _assign_lane_and_bearing(
        bearing, road_type, rng
    )
    position = _offset_to_lane(point, tangent, lane_sign)
    return CarLightPair(
        position=position,
        travel_bearing=travel_bearing,
        config=car_config,
        rng=rng,
    )


def _bearing_from_tangent(tangent: np.ndarray) -> float:
    """Compute the road bearing from a tangent vector.

    Uses atan2(x, y) convention: 0 = North, pi/2 = East.

    Args:
        tangent (np.ndarray): 3-element tangent vector.

    Returns:
        float: Road bearing in radians.
    """
    return math.atan2(tangent[0], tangent[1])


def _assign_lane_and_bearing(
    road_bearing: float,
    road_type: RoadType,
    rng: random.Random,
) -> t.Tuple[int, float]:
    """Determine lane side and travel bearing for a car.

    Highways use directional lanes: the lane side determines bearing. Other
    roads randomly choose direction and then pick a matching lane side.

    Args:
        road_bearing (float): The forward direction of the road in radians.
        road_type (RoadType): Road classification.
        rng (random.Random): RNG for random direction selection.

    Returns:
        Tuple[int, float]:
            - lane_sign: +1 for left lane, -1 for right lane.
            - travel_bearing: Actual travel direction in radians.
    """
    if road_type == RoadType.HIGHWAY:
        return _highway_lane_and_bearing(road_bearing, rng)
    return _bidirectional_lane_and_bearing(road_bearing, rng)


def _highway_lane_and_bearing(
    road_bearing: float,
    rng: random.Random,
) -> t.Tuple[int, float]:
    """Assign lane and bearing for a highway — same direction per lane side.

    Left lane (+1) travels in the road bearing direction. Right lane (-1)
    travels in the opposite direction.

    Args:
        road_bearing (float): Road bearing in radians.
        rng (random.Random): RNG for lane selection.

    Returns:
        Tuple[int, float]: (lane_sign, travel_bearing).
    """
    lane_sign = rng.choice([1, -1])
    travel_bearing = road_bearing if lane_sign == 1 else road_bearing + math.pi
    return lane_sign, travel_bearing


def _bidirectional_lane_and_bearing(
    road_bearing: float,
    rng: random.Random,
) -> t.Tuple[int, float]:
    """Assign lane and bearing for a bidirectional road — 50/50 direction.

    The car's travel direction is chosen randomly, then the lane sign is set
    to keep cars on the correct side of the road for that direction.

    Args:
        road_bearing (float): Road bearing in radians.
        rng (random.Random): RNG for direction selection.

    Returns:
        Tuple[int, float]: (lane_sign, travel_bearing).
    """
    going_forward = rng.random() < 0.5
    if going_forward:
        return 1, road_bearing
    return -1, road_bearing + math.pi


def _offset_to_lane(
    center: np.ndarray,
    tangent: np.ndarray,
    lane_sign: int,
) -> np.ndarray:
    """Offset a road centerline point laterally into a lane.

    Args:
        center (np.ndarray): 3-element ENU centerline position.
        tangent (np.ndarray): 3-element tangent unit vector at this point.
        lane_sign (int): +1 for left lane, -1 for right lane.

    Returns:
        np.ndarray: 3-element ENU position in the lane.
    """
    perp = _perpendicular_2d(tangent)
    position = center.copy()
    position[:2] += perp[:2] * _LANE_OFFSET * lane_sign
    return position


# ---------------------------------------------------------------------------
# Random-interval walk
# ---------------------------------------------------------------------------


def _walk_with_random_spacing(
    geometry: np.ndarray,
    spacing_min: float,
    spacing_max: float,
    rng: random.Random,
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]:
    """Walk a polyline at random intervals and collect positions and tangents.

    Advances along the polyline with a new random spacing drawn each step,
    stopping when the end of the geometry is reached.

    Args:
        geometry (np.ndarray): (N, 3) array of ENU points.
        spacing_min (float): Minimum inter-car spacing in meters.
        spacing_max (float): Maximum inter-car spacing in meters.
        rng (random.Random): RNG for spacing draws.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            Parallel lists of positions and tangent unit vectors.
    """
    deltas = np.diff(geometry, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cum_lengths[-1]

    positions = []
    tangents = []

    dist = rng.uniform(spacing_min, spacing_max)
    while dist < total_length:
        position, tangent = _sample_at_distance(
            geometry, cum_lengths, segment_lengths, dist
        )
        positions.append(position)
        tangents.append(tangent)
        dist += rng.uniform(spacing_min, spacing_max)

    return positions, tangents


def _sample_at_distance(
    geometry: np.ndarray,
    cum_lengths: np.ndarray,
    segment_lengths: np.ndarray,
    dist: float,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Interpolate a position and tangent at a given arc distance.

    Args:
        geometry (np.ndarray): (N, 3) array of ENU points.
        cum_lengths (np.ndarray): Cumulative arc lengths, shape (N,).
        segment_lengths (np.ndarray): Per-segment lengths, shape (N-1,).
        dist (float): Target arc distance in meters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated position and unit tangent.
    """
    seg_idx = int(np.searchsorted(cum_lengths, dist, side="right")) - 1
    seg_idx = max(0, min(seg_idx, len(cum_lengths) - 2))

    seg_len = segment_lengths[seg_idx]
    start = geometry[seg_idx]
    end = geometry[seg_idx + 1]
    direction = end - start

    if seg_len > 1e-9:
        t_frac = (dist - cum_lengths[seg_idx]) / seg_len
        position = start + direction * np.clip(t_frac, 0.0, 1.0)
        tangent = direction / seg_len
    else:
        position = start.copy()
        tangent = np.array([1.0, 0.0, 0.0])

    return position, tangent
