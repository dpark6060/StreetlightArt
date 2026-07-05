"""Place streetlights along road segments using their fixture profiles."""

import random
import typing as t

import numpy as np

from ..config.schema import LightConfig
from ..models.light_source import LightSource
from ..models.road_segment import RoadSegment
from ..models.road_types import RoadType
from ..models.streetlight import StreetLight, create_fixture_pool
from .geometry_utils import perpendicular_2d as _perpendicular_2d


# Lane offset in meters — lights placed on road edge (half road width outward)
_BOTH_SIDES = "both"
_ONE_SIDE = "one"
_ALTERNATING = "alternating"


def place_streetlights(
    road_segments: t.List[RoadSegment],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place streetlights along all road segments and return all LightSources.

    Creates a FixturePool per road type, assigns one FixtureProfile to each
    road segment, then walks the segment geometry at fixed intervals placing
    lights on the road edge(s) according to the profile's side rule.

    Segments whose profile is None receive no lights.

    Args:
        road_segments (List[RoadSegment]): All road segments in the scene.
        config (LightConfig): Light configuration (passed to pool factory).
        rng (random.Random): Seeded RNG for reproducible fixture selection.

    Returns:
        List[LightSource]: All streetlight LightSources for the scene.
    """
    pools = _build_pools_per_road_type(config, rng)
    lights: t.List[LightSource] = []

    for segment in road_segments:
        profile = pools[segment.road_type].pick()
        segment.fixture_profile = profile

        if profile is None:
            continue

        segment_lights = place_lights_on_segment(segment, rng)
        lights.extend(segment_lights)

    return lights


# ---------------------------------------------------------------------------
# Pool creation
# ---------------------------------------------------------------------------


def _build_pools_per_road_type(
    config: LightConfig,
    rng: random.Random,
) -> t.Dict[RoadType, t.Any]:
    """Build one FixturePool per RoadType.

    Args:
        config (LightConfig): Passed through to create_fixture_pool.
        rng (random.Random): Shared RNG for pool construction.

    Returns:
        Dict[RoadType, FixturePool]: One pool per road type.
    """
    return {
        road_type: create_fixture_pool(road_type, config.streetlights, rng)
        for road_type in RoadType
    }


# ---------------------------------------------------------------------------
# Per-segment placement
# ---------------------------------------------------------------------------


def place_lights_on_segment(
    segment: RoadSegment,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place all lights on a single road segment.

    Walks the segment at the profile spacing interval and creates LightSource
    instances at each point based on the profile's side rule.

    Args:
        segment (RoadSegment): The road segment with a non-None fixture_profile.
        rng (random.Random): RNG used to pick a consistent side for "one" mode.

    Returns:
        List[LightSource]: All lights created for this segment.
    """
    profile = segment.fixture_profile
    sample_points, sample_tangents = _sample_geometry(
        segment.geometry, profile.spacing
    )

    side_choice = rng.choice([-1, 1])
    lights: t.List[LightSource] = []

    for i, (point, tangent) in enumerate(zip(sample_points, sample_tangents)):
        sides = _get_sides_for_step(profile.side, i, side_choice)
        for side_sign in sides:
            position = _compute_light_position(
                point, tangent, side_sign, segment.width, profile.height
            )
            light = StreetLight.create_from_profile(
                profile, position, segment.road_type
            )
            lights.append(light)

    return lights


def _get_sides_for_step(
    side_rule: str,
    step_index: int,
    side_choice: int,
) -> t.List[int]:
    """Return the list of side signs (+1 / -1) to place lights on for this step.

    Args:
        side_rule (str): One of "both", "one", or "alternating".
        step_index (int): Index of the current spacing step along the segment.
        side_choice (int): Per-segment coin flip (+1 or -1) for "one" mode.

    Returns:
        List[int]: Side signs to place a light on. +1 = left, -1 = right.
    """
    if side_rule == _BOTH_SIDES:
        return [1, -1]
    if side_rule == _ONE_SIDE:
        return [side_choice]
    # alternating: odd steps go left (+1), even steps go right (-1)
    return [1 if step_index % 2 == 0 else -1]


def _compute_light_position(
    point: np.ndarray,
    tangent: np.ndarray,
    side_sign: int,
    road_width: float,
    pole_height: float,
) -> np.ndarray:
    """Compute the 3D ENU position of a light on the road edge.

    Offsets perpendicular to the road tangent by half the road width,
    then sets z to the pole height.

    Args:
        point (np.ndarray): 3-element centerline point in ENU meters.
        tangent (np.ndarray): 3-element unit tangent vector at this point.
        side_sign (int): +1 for left side, -1 for right side.
        road_width (float): Road width in meters.
        pole_height (float): Streetlight pole height in meters (becomes z).

    Returns:
        np.ndarray: 3-element ENU position for the light.
    """
    perp = _perpendicular_2d(tangent)
    offset = perp * (road_width / 2.0) * side_sign
    position = point.copy()
    position[:2] += offset[:2]
    position[2] = pole_height
    return position


# ---------------------------------------------------------------------------
# Geometry sampling
# ---------------------------------------------------------------------------


def _sample_geometry(
    geometry: np.ndarray,
    spacing: float,
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]:
    """Sample positions and tangents along a polyline at fixed intervals.

    Computes cumulative arc length, then samples at multiples of spacing.
    Each sample yields an interpolated position and the local tangent direction.

    Args:
        geometry (np.ndarray): (N, 3) array of ENU points along the road.
        spacing (float): Distance between samples in meters.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            Parallel lists of sampled positions and tangent unit vectors.
    """
    cum_lengths, segment_lengths = _compute_arc_lengths(geometry)
    total_length = cum_lengths[-1]

    if total_length < spacing:
        return [], []

    sample_distances = _get_sample_distances(total_length, spacing)
    return _interpolate_samples(geometry, cum_lengths, segment_lengths, sample_distances)


def _compute_arc_lengths(
    geometry: np.ndarray,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative arc lengths and per-segment lengths for a polyline.

    Args:
        geometry (np.ndarray): (N, 3) array of ENU points.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - cumulative arc lengths starting at 0, shape (N,)
            - per-segment lengths, shape (N-1,)
    """
    deltas = np.diff(geometry, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return cum_lengths, segment_lengths


def _get_sample_distances(total_length: float, spacing: float) -> t.List[float]:
    """Generate evenly spaced sample distances along the polyline.

    Starts at spacing/2 so lights appear centered within each interval.

    Args:
        total_length (float): Total arc length of the polyline in meters.
        spacing (float): Desired distance between samples in meters.

    Returns:
        List[float]: Sample distances in meters from the polyline start.
    """
    start = spacing / 2.0
    distances = []
    d = start
    while d < total_length:
        distances.append(d)
        d += spacing
    return distances


def _interpolate_samples(
    geometry: np.ndarray,
    cum_lengths: np.ndarray,
    segment_lengths: np.ndarray,
    sample_distances: t.List[float],
) -> t.Tuple[t.List[np.ndarray], t.List[np.ndarray]]:
    """Interpolate positions and tangents at each sample distance.

    Args:
        geometry (np.ndarray): (N, 3) array of ENU points.
        cum_lengths (np.ndarray): Cumulative arc lengths, shape (N,).
        segment_lengths (np.ndarray): Per-segment lengths, shape (N-1,).
        sample_distances (List[float]): Target distances for sampling.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            Parallel lists of interpolated positions and tangent unit vectors.
    """
    positions = []
    tangents = []

    for dist in sample_distances:
        seg_idx = _find_segment_index(cum_lengths, dist)
        position, tangent = _interpolate_on_segment(
            geometry, cum_lengths, segment_lengths, seg_idx, dist
        )
        positions.append(position)
        tangents.append(tangent)

    return positions, tangents


def _find_segment_index(cum_lengths: np.ndarray, dist: float) -> int:
    """Find the polyline segment index that contains the given arc distance.

    Args:
        cum_lengths (np.ndarray): Cumulative arc lengths, shape (N,).
        dist (float): Target arc distance.

    Returns:
        int: Index of the segment (0-based), clamped to valid range.
    """
    idx = int(np.searchsorted(cum_lengths, dist, side="right")) - 1
    return max(0, min(idx, len(cum_lengths) - 2))


def _interpolate_on_segment(
    geometry: np.ndarray,
    cum_lengths: np.ndarray,
    segment_lengths: np.ndarray,
    seg_idx: int,
    dist: float,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a position and tangent within a single segment.

    Args:
        geometry (np.ndarray): (N, 3) array of ENU points.
        cum_lengths (np.ndarray): Cumulative arc lengths, shape (N,).
        segment_lengths (np.ndarray): Per-segment lengths, shape (N-1,).
        seg_idx (int): Index of the segment to interpolate within.
        dist (float): Target arc distance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Interpolated position and unit tangent.
    """
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
