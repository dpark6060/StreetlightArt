"""Build Building objects from OSM footprint data and residential house lots.

Converts the buildings GeoDataFrame to ENU `Building` objects, resolving
each footprint's height from OSM tags with graceful fallback to per-type
defaults (see `_resolve_building_height`). Residential zone house footprints
that OSM doesn't cover are also wrapped into `Building` objects, deduped
against OSM footprints via an `shapely.STRtree` so suburbs built from
synthetic lots (see `data.zone_builder`) aren't left empty.

A new module rather than an extension of `zone_builder` — that module is
already large and buildings are a distinct concern (mass geometry, not
zone/lot classification).
"""

import math
import random
import re
import typing as t

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.strtree import STRtree

from ..camera.coordinates import CoordinateTransformer, feet_to_meters, wgs84_to_enu_batch
from ..config.schema import BuildingConfig
from ..models.building import Building
from ..models.zones import ResidentialZone

_DEFAULT_HEIGHT_M: float = 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_buildings(
    buildings_gdf: t.Optional[gpd.GeoDataFrame],
    residential_zones: t.List[ResidentialZone],
    transformer: CoordinateTransformer,
    config: BuildingConfig,
    rng: random.Random,
) -> t.List[Building]:
    """Build Building objects from OSM footprints and residential house lots.

    Converts every OSM building footprint to ENU and resolves its height per
    the config's height-resolution rules, then wraps any residential house
    footprint not already covered by an OSM footprint into a Building too,
    so suburbs built from synthetic lots aren't left empty.

    Args:
        buildings_gdf (GeoDataFrame or None): OSM building footprints
            (WGS84). None or empty produces house-derived buildings only.
        residential_zones (list[ResidentialZone]): Zones whose houses are
            merged in where OSM coverage is missing.
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        config (BuildingConfig): Building generation parameters.
        rng (random.Random): Seeded RNG for height jitter and window seeds.

    Returns:
        list[Building]: All buildings in ENU coordinates. Empty if
            `config.enabled` is False.
    """
    if not config.enabled:
        return []

    osm_buildings = _build_osm_buildings(buildings_gdf, transformer, config, rng)
    house_buildings = _build_house_buildings(residential_zones, osm_buildings, config, rng)
    return osm_buildings + house_buildings


# ---------------------------------------------------------------------------
# OSM footprint conversion
# ---------------------------------------------------------------------------


def _build_osm_buildings(
    buildings_gdf: t.Optional[gpd.GeoDataFrame],
    transformer: CoordinateTransformer,
    config: BuildingConfig,
    rng: random.Random,
) -> t.List[Building]:
    """Convert an OSM buildings GeoDataFrame to a list of Building objects.

    Args:
        buildings_gdf (GeoDataFrame or None): OSM building footprints (WGS84).
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        config (BuildingConfig): Building generation parameters.
        rng (random.Random): Seeded RNG for height jitter and window seeds.

    Returns:
        list[Building]: One Building per valid footprint polygon.
    """
    if buildings_gdf is None or buildings_gdf.empty:
        return []

    records = _iter_building_records(buildings_gdf)
    polygons, kinds, height_tags, levels_tags = _flatten_building_records(records)
    footprints = _batch_convert_footprints(polygons, transformer)

    buildings = []
    for footprint, kind, height_tag, levels_tag in zip(
        footprints, kinds, height_tags, levels_tags
    ):
        if footprint is None:
            continue
        height = _resolve_building_height(height_tag, levels_tag, kind, config, rng)
        buildings.append(
            Building(
                footprint=footprint,
                height=height,
                building_kind=kind,
                window_seed=rng.randint(0, 2**32 - 1),
            )
        )
    return buildings


def _iter_building_records(buildings_gdf: gpd.GeoDataFrame) -> t.List[dict]:
    """Yield building rows as dicts, filling missing height/levels columns with None.

    A `.gpkg` cache written before the height/levels migration lacks the
    "height" and "building:levels" columns entirely; this treats that case
    identically to the column being present with a NaN/None value, so both
    a cache hit and a cache miss produce the same fallback behavior.

    Args:
        buildings_gdf (GeoDataFrame): OSM building footprints (WGS84).

    Returns:
        list[dict]: One dict per row with "geometry", "building", "height",
            and "building:levels" keys.
    """
    working = buildings_gdf.copy()
    for col in ("building", "height", "building:levels"):
        if col not in working.columns:
            working[col] = None
    return working[["geometry", "building", "height", "building:levels"]].to_dict("records")


def _flatten_building_records(
    records: t.List[dict],
) -> t.Tuple[t.List[Polygon], t.List[str], t.List[t.Any], t.List[t.Any]]:
    """Flatten building rows into individual polygons with per-polygon tags.

    A MultiPolygon row's tags apply to every constituent polygon it expands to.

    Args:
        records (list[dict]): Building rows from `_iter_building_records`.

    Returns:
        tuple: Parallel lists (polygons, building_kinds, height_tags,
            levels_tags), one entry per individual polygon.
    """
    polygons: t.List[Polygon] = []
    kinds: t.List[str] = []
    height_tags: t.List[t.Any] = []
    levels_tags: t.List[t.Any] = []

    for record in records:
        geom = record.get("geometry")
        if geom is None:
            continue
        kind = _tag_to_str(record.get("building")) or "yes"
        for polygon in _iter_polygons(geom):
            polygons.append(polygon)
            kinds.append(kind)
            height_tags.append(record.get("height"))
            levels_tags.append(record.get("building:levels"))

    return polygons, kinds, height_tags, levels_tags


def _iter_polygons(geom: t.Any) -> t.Iterator[Polygon]:
    """Yield individual Polygons from a Polygon or MultiPolygon.

    Args:
        geom: A shapely Polygon, MultiPolygon, or any geometry.

    Yields:
        Polygon: Each individual non-empty polygon from the geometry.
    """
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            yield geom
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if isinstance(part, Polygon) and not part.is_empty:
                yield part


def _batch_convert_footprints(
    polygons: t.List[Polygon],
    transformer: CoordinateTransformer,
) -> t.List[t.Optional[np.ndarray]]:
    """Convert many WGS84 polygon exterior rings to ENU footprints in one batch.

    Every polygon's exterior ring (minus its closing duplicate vertex) is
    concatenated into a single coordinate array and converted with one
    `wgs84_to_enu_batch` call, rather than one pyproj transform per building.

    Args:
        polygons (list[Polygon]): WGS84 polygons (exterior ring only; holes
            are not part of the Building footprint representation).
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        list[np.ndarray or None]: One entry per input polygon, in order.
            None for polygons whose exterior has fewer than 3 vertices after
            dropping the closing duplicate.
    """
    valid_indices = []
    rings = []
    lengths = []
    for idx, polygon in enumerate(polygons):
        coords = np.array(polygon.exterior.coords)
        ring = coords[:-1] if len(coords) > 0 else coords
        if len(ring) < 3:
            continue
        valid_indices.append(idx)
        rings.append(ring)
        lengths.append(len(ring))

    results: t.List[t.Optional[np.ndarray]] = [None] * len(polygons)
    if not rings:
        return results

    all_coords = np.concatenate(rings, axis=0)
    enu_all = wgs84_to_enu_batch(transformer, all_coords[:, 1], all_coords[:, 0], alt_m=0.0)
    split_points = np.cumsum(lengths)[:-1]
    enu_rings = np.split(enu_all, split_points)

    for idx, enu_ring in zip(valid_indices, enu_rings):
        results[idx] = _ensure_ccw(enu_ring[:, :2])
    return results


def _ensure_ccw(ring: np.ndarray) -> np.ndarray:
    """Return the ring reversed if it winds clockwise, unchanged if already CCW.

    Uses the shoelace formula's signed area: positive for CCW, negative for CW.

    Args:
        ring (np.ndarray): (N, 2) polygon exterior ring, no closing duplicate.

    Returns:
        np.ndarray: The same ring, guaranteed CCW.
    """
    x = ring[:, 0]
    y = ring[:, 1]
    # Shoelace signed area without np.roll: the interior terms plus the
    # explicit wrap term. Only the sign is used, so summation order is
    # irrelevant, and this avoids two full-array roll allocations per ring.
    signed_area = (
        np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1])
    )
    if signed_area < 0:
        return ring[::-1].copy()
    return ring


# ---------------------------------------------------------------------------
# Height resolution
# ---------------------------------------------------------------------------


def _tag_to_str(value: t.Any) -> t.Optional[str]:
    """Normalize an OSM tag cell to a stripped string, or None if absent.

    Handles the three ways a missing tag surfaces from a GeoDataFrame:
    Python None, float NaN (pandas' missing-value sentinel for object
    columns is still occasionally a float NaN), and an empty string.

    Args:
        value: Raw cell value from a building record.

    Returns:
        str or None: Stripped string value, or None if missing/empty.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_leading_float(text: str) -> t.Optional[float]:
    """Parse the leading numeric portion of a string as a float.

    Args:
        text (str): Text that may start with a number (e.g. "12", "12.5 m").

    Returns:
        float or None: The parsed leading number, or None if the string
            doesn't start with a parseable number.
    """
    match = re.match(r"[-+]?\d*\.?\d+", text.strip())
    if match is None:
        return None
    return float(match.group())


def _parse_height_tag(value: t.Any) -> t.Optional[float]:
    """Parse an OSM `height` tag value to meters.

    Accepts plain numbers ("12"), numbers with a unit suffix ("12 m"), and
    feet expressed with a trailing apostrophe ("39'"). Any value that
    doesn't start with a parseable number returns None so the caller can
    fall through to the next height-resolution rule.

    Args:
        value: Raw `height` tag cell value.

    Returns:
        float or None: Height in meters, or None if unparseable/missing.
    """
    text = _tag_to_str(value)
    if text is None:
        return None
    if text.endswith("'"):
        feet = _parse_leading_float(text[:-1])
        return feet_to_meters(feet) if feet is not None else None
    return _parse_leading_float(text)


def _parse_levels_tag(value: t.Any) -> t.Optional[float]:
    """Parse an OSM `building:levels` tag value to a floor count.

    Args:
        value: Raw `building:levels` tag cell value.

    Returns:
        float or None: Parsed floor count, or None if unparseable/missing.
    """
    text = _tag_to_str(value)
    if text is None:
        return None
    return _parse_leading_float(text)


def _get_type_default_height(building_kind: str, height_defaults: t.Dict[str, float]) -> float:
    """Look up the per-type default height, falling back to a generic default.

    Args:
        building_kind (str): Raw OSM `building` tag value.
        height_defaults (dict[str, float]): Config-provided type → height
            table (see `[buildings.height_defaults]`).

    Returns:
        float: The type's default height, `height_defaults["default"]`, or
            `_DEFAULT_HEIGHT_M` if neither is present.
    """
    if building_kind in height_defaults:
        return height_defaults[building_kind]
    return height_defaults.get("default", _DEFAULT_HEIGHT_M)


def _resolve_building_height(
    height_tag: t.Any,
    levels_tag: t.Any,
    building_kind: str,
    config: BuildingConfig,
    rng: random.Random,
) -> float:
    """Resolve a building's height in meters, first match wins.

    Resolution order: parsed `height` tag, then `building:levels` times
    `config.floor_height_m`, then the per-type default from
    `config.height_defaults`. Only the type-default fallback is jittered by
    `config.height_jitter` — a parsed `height` tag or a `building:levels`
    value is measured data and is returned as-is. Jitter exists so
    identical type defaults don't produce a perfectly flat skyline; it must
    not corrupt real height data.

    Args:
        height_tag: Raw `height` tag cell value.
        levels_tag: Raw `building:levels` tag cell value.
        building_kind (str): Raw OSM `building` tag value.
        config (BuildingConfig): Building generation parameters.
        rng (random.Random): Seeded RNG for jitter.

    Returns:
        float: Resolved height in meters, jittered only when derived from
            a type default.
    """
    height = _parse_height_tag(height_tag)
    if height is not None:
        return height

    levels = _parse_levels_tag(levels_tag)
    if levels is not None:
        return levels * config.floor_height_m

    default_height = _get_type_default_height(building_kind, config.height_defaults)
    return _apply_height_jitter(default_height, config.height_jitter, rng)


def _apply_height_jitter(height: float, jitter: float, rng: random.Random) -> float:
    """Apply symmetric proportional jitter to a resolved height.

    Args:
        height (float): Base height in meters before jitter.
        jitter (float): Fractional jitter amount (e.g. 0.12 for ±12%).
        rng (random.Random): Seeded RNG.

    Returns:
        float: Jittered height in meters.
    """
    return height * rng.uniform(1.0 - jitter, 1.0 + jitter)


# ---------------------------------------------------------------------------
# House-footprint merge
# ---------------------------------------------------------------------------


def _build_house_buildings(
    residential_zones: t.List[ResidentialZone],
    osm_buildings: t.List[Building],
    config: BuildingConfig,
    rng: random.Random,
) -> t.List[Building]:
    """Wrap house footprints not already covered by an OSM building into Buildings.

    Args:
        residential_zones (list[ResidentialZone]): Zones with House objects.
        osm_buildings (list[Building]): Already-built OSM buildings, used to
            dedupe against so a house isn't double-drawn on top of a real
            OSM footprint.
        config (BuildingConfig): Building generation parameters.
        rng (random.Random): Seeded RNG for height jitter and window seeds.

    Returns:
        list[Building]: One Building per house footprint with no OSM overlap.
    """
    house_polygons = _collect_house_footprints(residential_zones)
    if not house_polygons:
        return []

    osm_polygons = [Polygon(building.footprint) for building in osm_buildings]
    unmatched = _filter_houses_without_osm_match(house_polygons, osm_polygons)

    house_height_default = _get_type_default_height("house", config.height_defaults)
    return [
        Building(
            footprint=_polygon_to_ccw_footprint(polygon),
            height=_apply_height_jitter(house_height_default, config.height_jitter, rng),
            building_kind="house",
            window_seed=rng.randint(0, 2**32 - 1),
        )
        for polygon in unmatched
    ]


def _collect_house_footprints(residential_zones: t.List[ResidentialZone]) -> t.List[Polygon]:
    """Flatten every house footprint across all residential zones.

    Args:
        residential_zones (list[ResidentialZone]): Zones with House objects.

    Returns:
        list[Polygon]: House footprint polygons in ENU meters.
    """
    return [house.footprint for zone in residential_zones for house in zone.houses]


def _filter_houses_without_osm_match(
    house_polygons: t.List[Polygon],
    osm_polygons: t.List[Polygon],
) -> t.List[Polygon]:
    """Drop house footprints that intersect an OSM building footprint.

    Uses an STRtree over the OSM footprints for O(log n) candidate lookup
    per house, rather than an O(n^2) all-pairs intersection test.

    Args:
        house_polygons (list[Polygon]): Candidate house footprints.
        osm_polygons (list[Polygon]): OSM building footprints already built.

    Returns:
        list[Polygon]: House footprints that don't intersect any OSM footprint.
    """
    if not osm_polygons:
        return house_polygons
    tree = STRtree(osm_polygons)
    return [
        house
        for house in house_polygons
        if len(tree.query(house, predicate="intersects")) == 0
    ]


def _polygon_to_ccw_footprint(polygon: Polygon) -> np.ndarray:
    """Convert a shapely Polygon (already in ENU meters) to a CCW footprint array.

    Args:
        polygon (Polygon): House footprint polygon in ENU meters.

    Returns:
        np.ndarray: (N, 2) exterior ring, CCW, no closing duplicate vertex.
    """
    coords = np.array(polygon.exterior.coords)[:-1]
    return _ensure_ccw(coords)
