"""Commercial zone light placement — signs, storefronts, and parking lots."""

import random
import typing as t

import numpy as np
from shapely.geometry import Point, Polygon

from ..config.schema import LightConfig
from ..models.commercial import (
    create_parking_lot_light,
    create_sign_light,
    create_storefront_light,
)
from ..models.light_source import LightSource
from ..models.zones import CommercialZone


def place_commercial_lights(
    commercial_zones: t.List[CommercialZone],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place sign, storefront, and parking lot lights across all commercial zones.

    For each zone, sign lights are placed at midpoints of building footprint
    edges, storefront lights are placed along each building's longest edge, and
    parking lot lights are placed in a grid within each parking area.

    Args:
        commercial_zones (List[CommercialZone]): Zones to populate with lights.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG for reproducible placement.

    Returns:
        List[LightSource]: All generated commercial light sources.
    """
    lights: t.List[LightSource] = []
    for zone in commercial_zones:
        lights.extend(_place_zone_lights(zone, config, rng))
    return lights


def _place_zone_lights(
    zone: CommercialZone,
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place all light types for a single commercial zone.

    Args:
        zone (CommercialZone): The zone to process.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Lights generated for this zone.
    """
    lights: t.List[LightSource] = []
    lights.extend(_place_building_lights(zone.building_footprints, config, rng))
    lights.extend(_place_parking_lights(zone.parking_areas, config, rng))
    return lights


def _place_building_lights(
    footprints: t.List[Polygon],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place sign and storefront lights across all building footprints.

    Args:
        footprints (List[Polygon]): Building footprint polygons in ENU metres.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Sign and storefront lights.
    """
    lights: t.List[LightSource] = []
    for footprint in footprints:
        lights.extend(_place_footprint_lights(footprint, config, rng))
    return lights


def _place_footprint_lights(
    footprint: Polygon,
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place sign and storefront lights for a single building footprint.

    Sign lights are placed at midpoints of all qualifying edges. Storefront
    lights are placed along the building's longest edge.

    Args:
        footprint (Polygon): A building footprint polygon in ENU metres.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Lights for this footprint.
    """
    lights: t.List[LightSource] = []
    edges = _get_exterior_edges(footprint)
    lights.extend(_place_sign_lights(edges, config, rng))
    lights.extend(_place_storefront_lights_on_longest_edge(edges, config, rng))
    return lights


def _place_parking_lights(
    parking_areas: t.List[Polygon],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place parking lot pole lights in a grid within each parking polygon.

    Args:
        parking_areas (List[Polygon]): Parking area polygons in ENU metres.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Parking lot lights.
    """
    lights: t.List[LightSource] = []
    for area in parking_areas:
        lights.extend(_place_parking_grid(area, config, rng))
    return lights


def _get_exterior_edges(
    footprint: Polygon,
) -> t.List[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]]:
    """Extract consecutive coordinate pairs from a polygon's exterior ring.

    Args:
        footprint (Polygon): Building footprint polygon.

    Returns:
        List[Tuple[Tuple[float, float], Tuple[float, float]]]: List of
            (start, end) coordinate pairs for each exterior edge.
    """
    coords = list(footprint.exterior.coords)
    return [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]


def _get_edge_length(
    start: t.Tuple[float, float],
    end: t.Tuple[float, float],
) -> float:
    """Compute the Euclidean length of an edge in metres.

    Args:
        start (Tuple[float, float]): Edge start (x, y) in ENU metres.
        end (Tuple[float, float]): Edge end (x, y) in ENU metres.

    Returns:
        float: Edge length in metres.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return (dx * dx + dy * dy) ** 0.5


def _get_edge_midpoint(
    start: t.Tuple[float, float],
    end: t.Tuple[float, float],
) -> np.ndarray:
    """Compute the midpoint of an edge as a 3D ENU position at z=0.

    Args:
        start (Tuple[float, float]): Edge start (x, y) in ENU metres.
        end (Tuple[float, float]): Edge end (x, y) in ENU metres.

    Returns:
        np.ndarray: 3-element (x, y, z) midpoint array.
    """
    mx = (start[0] + end[0]) / 2.0
    my = (start[1] + end[1]) / 2.0
    return np.array([mx, my, 0.0])


def _place_sign_lights(
    edges: t.List[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]],
    config: LightConfig,
    rng: random.Random,
    min_length_m: float = 5.0,
) -> t.List[LightSource]:
    """Place a sign light at the midpoint of each sufficiently long edge.

    Only edges longer than min_length_m receive a sign light.

    Args:
        edges (List[Tuple]): Exterior edge coordinate pairs.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.
        min_length_m (float): Minimum edge length in metres to place a sign.
            Defaults to 5.0.

    Returns:
        List[LightSource]: One sign light per qualifying edge.
    """
    lights: t.List[LightSource] = []
    for start, end in edges:
        if _get_edge_length(start, end) <= min_length_m:
            continue
        position = _get_edge_midpoint(start, end)
        lights.append(create_sign_light(position, config.commercial, rng))
    return lights


def _place_storefront_lights_on_longest_edge(
    edges: t.List[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place storefront lights at regular intervals along the longest edge.

    Spacing is drawn randomly from the configured storefront spacing range.

    Args:
        edges (List[Tuple]): Exterior edge coordinate pairs.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Storefront lights along the longest edge.
    """
    if not edges:
        return []

    longest = max(edges, key=lambda e: _get_edge_length(e[0], e[1]))
    return _place_lights_along_edge(longest, config, rng)


def _place_lights_along_edge(
    edge: t.Tuple[t.Tuple[float, float], t.Tuple[float, float]],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place storefront lights at evenly-spaced intervals along an edge.

    Spacing is drawn once per edge from the commercial storefront spacing range.

    Args:
        edge (Tuple): (start, end) coordinate pair for the edge.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Storefront lights placed along this edge.
    """
    start, end = edge
    length = _get_edge_length(start, end)
    spacing = rng.uniform(
        config.commercial.storefront_spacing_min,
        config.commercial.storefront_spacing_max,
    )

    if length < spacing:
        return []

    dx = (end[0] - start[0]) / length
    dy = (end[1] - start[1]) / length

    lights: t.List[LightSource] = []
    dist = spacing / 2.0
    while dist < length:
        position = np.array([start[0] + dx * dist, start[1] + dy * dist, 0.0])
        lights.append(create_storefront_light(position, config.commercial, rng))
        dist += spacing
    return lights


def _place_parking_grid(
    polygon: Polygon,
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Place parking lot lights on a regular grid within a polygon.

    Grid spacing is drawn once from the configured range. Only grid points
    that fall inside the polygon are kept.

    Args:
        polygon (Polygon): Parking area polygon in ENU metres.
        config (LightConfig): Top-level light configuration.
        rng (random.Random): Seeded RNG.

    Returns:
        List[LightSource]: Parking lot lights inside the polygon.
    """
    spacing = rng.uniform(
        config.commercial.parking_grid_spacing_min,
        config.commercial.parking_grid_spacing_max,
    )
    grid_points = _compute_grid_points(polygon, spacing)
    lights: t.List[LightSource] = []
    for x, y in grid_points:
        position = np.array([x, y, 0.0])
        lights.append(create_parking_lot_light(position, config.commercial, rng))
    return lights


def _compute_grid_points(
    polygon: Polygon,
    spacing: float,
) -> t.List[t.Tuple[float, float]]:
    """Compute grid points inside a polygon at a given spacing.

    Generates candidate x and y coordinates using numpy, then filters to only
    those that fall within the polygon boundary.

    Args:
        polygon (Polygon): Bounding polygon in ENU metres.
        spacing (float): Grid cell size in metres.

    Returns:
        List[Tuple[float, float]]: (x, y) positions that lie inside the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx + spacing / 2.0, maxx, spacing)
    ys = np.arange(miny + spacing / 2.0, maxy, spacing)
    points: t.List[t.Tuple[float, float]] = []
    for x in xs:
        for y in ys:
            if polygon.contains(Point(x, y)):
                points.append((float(x), float(y)))
    return points
