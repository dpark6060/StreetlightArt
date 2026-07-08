"""Build zone objects from OSM land use and building data.

Classifies land use polygons by OSM tag, converts geometries from WGS84 to
ENU metres, and constructs ResidentialZone, CommercialZone, and DarkZone
objects.  Residential zones include Houses derived from OSM building
footprints; where footprints are absent a simple grid subdivision is used.
"""

import logging
import random
import typing as t

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..camera.coordinates import CoordinateTransformer, wgs84_to_enu_batch
from ..config.schema import HouseConfig, ZoneColorsConfig, ZoneConfig
from ..models.color_pool import ColorPool
from ..models.house import House
from ..models.zones import CommercialZone, DarkZone, ResidentialZone
from ..utils.colors import hex_to_rgb

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSM tag classification — single source of truth
# ---------------------------------------------------------------------------
#
# Maps an OSM tag value (from any of the landuse/leisure/natural/waterway
# columns) to the zone bucket it belongs to. verify_plot.py imports this
# mapping so the verification plot can never show a land-use category that
# the zone builder silently drops (and vice versa).

_TAG_TO_ZONE_TYPE: t.Dict[str, str] = {
    "residential": "residential",
    "commercial": "commercial",
    "retail": "commercial",
    "industrial": "industrial",
    "park": "park",
    "garden": "park",
    "nature_reserve": "park",
    "wood": "park",
    "grassland": "park",
    "forest": "park",
    "grass": "park",
    "recreation_ground": "park",
    "meadow": "park",
    "water": "water",
    "riverbank": "water",
}

_TAG_COLUMNS: t.Tuple[str, ...] = ("landuse", "leisure", "natural", "waterway")

# Minimum house density (houses per square kilometre) below which synthetic
# lots are generated to fill out a residential zone. Prevents one
# boundary-clipping building from suppressing synthetic lots for an entire
# under-built zone.
_MIN_HOUSES_PER_KM2: float = 50.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_zones(
    land_use_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    transformer: CoordinateTransformer,
    zone_config: ZoneConfig,
    rng: random.Random,
) -> t.Tuple[t.List[ResidentialZone], t.List[CommercialZone], t.List[DarkZone]]:
    """Build zone objects from OSM land use and building data.

    Classifies land use polygons by OSM tags, converts all geometries to ENU
    metres, and constructs the three zone types used by the scene and ground
    renderer.

    Args:
        land_use_gdf (GeoDataFrame): Land use polygons from OSM (WGS84).
        buildings_gdf (GeoDataFrame): Building footprints from OSM (WGS84).
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        zone_config (ZoneConfig): Zone generation parameters.
        rng (random.Random): Seeded RNG for reproducible house generation.

    Returns:
        Tuple[List[ResidentialZone], List[CommercialZone], List[DarkZone]]:
            Populated zone lists; any list may be empty if no matching polygons
            were found.
    """
    if land_use_gdf.empty:
        return [], [], []

    classified = _classify_land_use(land_use_gdf)

    residential = _build_residential_zones(
        classified["residential"], buildings_gdf, transformer, zone_config.house, rng
    )
    commercial = _build_commercial_zones(
        classified["commercial"], buildings_gdf, transformer
    )
    dark = _build_dark_zones(classified, transformer, zone_config.colors)

    return residential, commercial, dark


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_land_use(gdf: gpd.GeoDataFrame) -> t.Dict[str, t.List[t.Any]]:
    """Classify GeoDataFrame rows into zone-type buckets by OSM tags.

    Args:
        gdf (GeoDataFrame): Land use polygons with OSM tag columns.

    Returns:
        Dict[str, List]: Keys are "residential", "commercial", "industrial",
            "park", "water". Values are lists of GeoDataFrame row objects.
    """
    buckets: t.Dict[str, t.List[t.Any]] = {
        "residential": [],
        "commercial": [],
        "industrial": [],
        "park": [],
        "water": [],
    }

    for row in gdf.itertuples():
        bucket = _get_row_bucket(row)
        if bucket is not None:
            buckets[bucket].append(row)

    return buckets


def _get_row_bucket(row: t.Any) -> t.Optional[str]:
    """Determine the zone type bucket for a single GeoDataFrame row.

    Checks the landuse, leisure, natural, and waterway tag columns in that
    order against the shared `_TAG_TO_ZONE_TYPE` mapping, returning the
    bucket for the first column with a recognized value.

    Args:
        row: A namedtuple row from GeoDataFrame.itertuples().

    Returns:
        str or None: Bucket name, or None if the row doesn't match any zone.
    """
    for column in _TAG_COLUMNS:
        value = _safe_tag(row, column)
        if value in _TAG_TO_ZONE_TYPE:
            return _TAG_TO_ZONE_TYPE[value]
    return None


def _safe_tag(row: t.Any, attr: str) -> str:
    """Safely read an OSM tag attribute from a row, returning empty string if absent.

    OSM merged-feature tags can carry list values (e.g. `landuse=['residential']`);
    the first element is used, mirroring the road classifier's normalization.

    Args:
        row: A namedtuple row from GeoDataFrame.itertuples().
        attr (str): Attribute name to read.

    Returns:
        str: Tag value, or empty string if missing or NaN.
    """
    value = getattr(row, attr, None)
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


# ---------------------------------------------------------------------------
# Residential zone construction
# ---------------------------------------------------------------------------


def _build_residential_zones(
    rows: t.List[t.Any],
    buildings_gdf: gpd.GeoDataFrame,
    transformer: CoordinateTransformer,
    house_config: HouseConfig,
    rng: random.Random,
) -> t.List[ResidentialZone]:
    """Build ResidentialZone objects from classified OSM rows.

    Args:
        rows (List): GeoDataFrame rows with residential land use geometry.
        buildings_gdf (GeoDataFrame): All building footprints (WGS84).
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        house_config (HouseConfig): House generation parameters.
        rng (random.Random): Seeded RNG passed through to House construction.

    Returns:
        List[ResidentialZone]: One zone per input polygon (MultiPolygons are
            split into individual Polygons).
    """
    zones = []
    for row in rows:
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        for polygon in _iter_polygons(geom):
            zone = _build_one_residential_zone(
                polygon, buildings_gdf, transformer, house_config, rng
            )
            zones.extend(_as_list(zone))
    return zones


def _build_one_residential_zone(
    wgs84_polygon: Polygon,
    buildings_gdf: gpd.GeoDataFrame,
    transformer: CoordinateTransformer,
    house_config: HouseConfig,
    rng: random.Random,
) -> t.Optional[ResidentialZone]:
    """Build a single ResidentialZone from one WGS84 polygon.

    A single ColorPool is created per zone so that all houses in the zone share
    the same pool instance, avoiding repeated color-list parsing.

    Args:
        wgs84_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        buildings_gdf (GeoDataFrame): All building footprints (WGS84).
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        house_config (HouseConfig): House generation parameters.
        rng (random.Random): Seeded RNG.

    Returns:
        ResidentialZone or None: None if ENU conversion fails.
    """
    enu_boundary = _polygon_wgs84_to_enu(wgs84_polygon, transformer)
    if enu_boundary is None:
        return None

    color_pool = ColorPool(house_config.exterior_colors.colors, rng)
    building_rows = _get_buildings_within(wgs84_polygon, buildings_gdf)
    enu_footprints = _convert_building_footprints(building_rows, transformer)

    houses = _build_houses_from_footprints(enu_footprints, house_config, rng, color_pool)

    if _needs_synthetic_lots(houses, enu_boundary):
        synthetic_footprints = _build_synthetic_lots(enu_boundary, house_config, enu_footprints)
        houses.extend(
            _build_houses_from_footprints(synthetic_footprints, house_config, rng, color_pool)
        )

    return ResidentialZone(boundary=enu_boundary, houses=houses)


def _needs_synthetic_lots(houses: t.List[House], zone_boundary: Polygon) -> bool:
    """Determine whether a zone's house density is too low and needs synthetic lots.

    A single boundary-clipping building intersecting a zone should not, by
    itself, suppress synthetic lot generation for the whole zone — so this
    checks house density against a minimum threshold rather than checking
    for zero houses.

    Args:
        houses (List[House]): Houses already built from real OSM footprints.
        zone_boundary (Polygon): Zone boundary polygon in ENU metres.

    Returns:
        bool: True if houses per square kilometre is below
            `_MIN_HOUSES_PER_KM2` (or the zone has zero area).
    """
    zone_area_km2 = zone_boundary.area / 1_000_000.0
    if zone_area_km2 <= 0:
        return not houses
    return (len(houses) / zone_area_km2) < _MIN_HOUSES_PER_KM2


def _build_houses_from_footprints(
    footprints: t.List[Polygon],
    house_config: HouseConfig,
    rng: random.Random,
    color_pool: ColorPool,
) -> t.List[House]:
    """Create House objects from a list of ENU footprint polygons.

    All houses receive the same shared color_pool to avoid re-parsing the
    color list for every House instantiation.

    Args:
        footprints (List[Polygon]): House footprint polygons in ENU metres.
        house_config (HouseConfig): House generation parameters.
        rng (random.Random): Seeded RNG.
        color_pool (ColorPool): Shared color pool for exterior color selection.

    Returns:
        List[House]: One House per footprint.
    """
    return [
        House(footprint=fp, config=house_config, rng=rng, color_pool=color_pool)
        for fp in footprints
    ]


def _build_synthetic_lots(
    zone_boundary: Polygon,
    house_config: HouseConfig,
    existing_footprints: t.Optional[t.List[Polygon]] = None,
) -> t.List[Polygon]:
    """Subdivide a zone boundary into synthetic rectangular house footprints.

    Uses a simple grid of lots based on lot_width and lot_depth from config.
    Only lots whose centroid falls inside the zone boundary are kept, and
    lots that intersect an existing OSM building footprint are dropped so
    synthetic lots only fill unoccupied space.

    Args:
        zone_boundary (Polygon): Zone boundary polygon in ENU metres.
        house_config (HouseConfig): Lot dimensions and footprint coverage ratio.
        existing_footprints (List[Polygon], optional): Real OSM footprints
            already placed in this zone. Synthetic lots overlapping any of
            these are skipped.

    Returns:
        List[Polygon]: Rectangular house footprint polygons in ENU metres.
    """
    min_x, min_y, max_x, max_y = zone_boundary.bounds
    lot_w = house_config.lot_width
    lot_d = house_config.lot_depth
    coverage = house_config.footprint_coverage

    candidate_lots = _grid_lots(min_x, min_y, max_x, max_y, lot_w, lot_d, coverage)
    contained_lots = _filter_lots_within_boundary(candidate_lots, zone_boundary)
    return _filter_lots_avoiding_footprints(contained_lots, existing_footprints or [])


def _filter_lots_within_boundary(
    lots: t.List[Polygon],
    zone_boundary: Polygon,
) -> t.List[Polygon]:
    """Keep only lots whose centroid falls inside the zone boundary.

    Uses a single vectorized `shapely.contains_xy` call over all lot
    centroids instead of one `Polygon.contains()` call per lot.

    Args:
        lots (List[Polygon]): Candidate lot footprint polygons.
        zone_boundary (Polygon): Zone boundary polygon in ENU metres.

    Returns:
        List[Polygon]: Lots whose centroid is inside the zone boundary.
    """
    if not lots:
        return []
    centroids = np.array([(lot.centroid.x, lot.centroid.y) for lot in lots])
    inside = shapely.contains_xy(zone_boundary, centroids[:, 0], centroids[:, 1])
    return [lot for lot, keep in zip(lots, inside) if keep]


def _filter_lots_avoiding_footprints(
    lots: t.List[Polygon],
    existing_footprints: t.List[Polygon],
) -> t.List[Polygon]:
    """Drop lots that intersect an existing building footprint.

    Uses an STRtree over the existing footprints so a boundary-clipping
    building only excludes the lots it actually overlaps, rather than
    suppressing synthetic lot generation for the whole zone.

    Args:
        lots (List[Polygon]): Candidate lot footprint polygons.
        existing_footprints (List[Polygon]): Real OSM footprints in this zone.

    Returns:
        List[Polygon]: Lots that do not intersect any existing footprint.
    """
    if not existing_footprints:
        return lots
    tree = STRtree(existing_footprints)
    return [lot for lot in lots if len(tree.query(lot, predicate="intersects")) == 0]


def _grid_lots(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    lot_w: float,
    lot_d: float,
    coverage: float,
) -> t.List[Polygon]:
    """Generate a grid of lot footprint polygons within a bounding box.

    Args:
        min_x (float): Left bound in ENU metres.
        min_y (float): Bottom bound in ENU metres.
        max_x (float): Right bound in ENU metres.
        max_y (float): Top bound in ENU metres.
        lot_w (float): Lot width in metres.
        lot_d (float): Lot depth in metres.
        coverage (float): Footprint coverage fraction (0–1).

    Returns:
        List[Polygon]: All footprint polygons on the grid (unfiltered).
    """
    # Generate lot origins with the same nested accumulation (x += lot_w,
    # y += lot_d) as before, so the exact floating-point sequence — and thus
    # which boundary lots are included — is preserved. Only the polygon
    # construction is vectorized: building N shapely polygons in one call
    # instead of one Polygon() per lot (the dominant cost here).
    origins = []
    x = min_x
    while x + lot_w <= max_x:
        y = min_y
        while y + lot_d <= max_y:
            origins.append((x, y))
            y += lot_d
        x += lot_w

    if not origins:
        return []
    return _make_lot_footprints(np.asarray(origins, dtype=float), lot_w, lot_d, coverage)


def _make_lot_footprints(
    origins: np.ndarray,
    lot_w: float,
    lot_d: float,
    coverage: float,
) -> t.List[Polygon]:
    """Build coverage-inset house footprints for every lot origin at once.

    Each footprint is inset from its lot edges by the coverage ratio and
    centered within the lot, matching the previous per-lot construction — the
    vertex order is identical, so the resulting polygons (and their centroids)
    are the same. All polygons are created with a single ``shapely.polygons``
    call rather than one ``Polygon()`` per lot.

    Args:
        origins (np.ndarray): (N, 2) lot lower-left corners in ENU metres.
        lot_w (float): Lot width in metres.
        lot_d (float): Lot depth in metres.
        coverage (float): Fraction of lot covered by the footprint (0–1).

    Returns:
        List[Polygon]: House footprint polygons in ENU metres.
    """
    fp_w = lot_w * coverage
    fp_d = lot_d * coverage
    margin_x = (lot_w - fp_w) / 2.0
    margin_y = (lot_d - fp_d) / 2.0

    x0 = origins[:, 0] + margin_x
    y0 = origins[:, 1] + margin_y
    x1 = x0 + fp_w
    y1 = y0 + fp_d

    # Ring order (x0,y0)->(x1,y0)->(x1,y1)->(x0,y1) matches the old per-lot
    # Polygon so geometry and centroids are unchanged.
    rings = np.stack(
        [
            np.column_stack([x0, y0]),
            np.column_stack([x1, y0]),
            np.column_stack([x1, y1]),
            np.column_stack([x0, y1]),
        ],
        axis=1,
    )
    return list(shapely.polygons(rings))


# ---------------------------------------------------------------------------
# Commercial zone construction
# ---------------------------------------------------------------------------


def _build_commercial_zones(
    rows: t.List[t.Any],
    buildings_gdf: gpd.GeoDataFrame,
    transformer: CoordinateTransformer,
) -> t.List[CommercialZone]:
    """Build CommercialZone objects from classified OSM rows.

    Args:
        rows (List): GeoDataFrame rows with commercial land use geometry.
        buildings_gdf (GeoDataFrame): All building footprints (WGS84).
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[CommercialZone]: One zone per input polygon.
    """
    zones = []
    for row in rows:
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        for polygon in _iter_polygons(geom):
            zone = _build_one_commercial_zone(polygon, buildings_gdf, transformer)
            zones.extend(_as_list(zone))
    return zones


def _build_one_commercial_zone(
    wgs84_polygon: Polygon,
    buildings_gdf: gpd.GeoDataFrame,
    transformer: CoordinateTransformer,
) -> t.Optional[CommercialZone]:
    """Build a single CommercialZone from one WGS84 polygon.

    Args:
        wgs84_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        buildings_gdf (GeoDataFrame): All building footprints (WGS84).
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        CommercialZone or None: None if ENU conversion fails.
    """
    enu_boundary = _polygon_wgs84_to_enu(wgs84_polygon, transformer)
    if enu_boundary is None:
        return None

    building_rows = _get_buildings_within(wgs84_polygon, buildings_gdf)
    enu_footprints = _convert_building_footprints(building_rows, transformer)
    parking_areas = _build_parking_areas(enu_boundary, enu_footprints)

    return CommercialZone(
        boundary=enu_boundary,
        building_footprints=enu_footprints,
        parking_areas=parking_areas,
    )


def _build_parking_areas(
    enu_boundary: Polygon,
    enu_footprints: t.List[Polygon],
) -> t.List[Polygon]:
    """Compute parking areas as zone boundary minus all building footprints.

    Building footprints are unioned into a single geometry before a single
    difference is taken against the zone boundary, rather than looping
    `difference()` once per footprint. Inputs are repaired with
    `shapely.make_valid` up front; if the difference still fails, the zone
    is logged and the boundary is returned with no parking excluded (an
    explicit fallback, not a silently-corrupted result).

    Args:
        enu_boundary (Polygon): Zone boundary in ENU metres.
        enu_footprints (List[Polygon]): Building footprints in ENU metres.

    Returns:
        List[Polygon]: Remaining open-area polygons after subtracting buildings.
    """
    if not enu_footprints:
        return [enu_boundary]

    valid_boundary = shapely.make_valid(enu_boundary)
    valid_footprints = [shapely.make_valid(fp) for fp in enu_footprints]

    try:
        remaining = valid_boundary.difference(unary_union(valid_footprints))
    except Exception:
        log.warning(
            "Parking-area difference failed for zone at centroid %s; "
            "falling back to boundary with no parking excluded.",
            enu_boundary.centroid,
        )
        return [enu_boundary]

    return list(_iter_polygons(remaining))


# ---------------------------------------------------------------------------
# Dark zone construction
# ---------------------------------------------------------------------------


def _build_dark_zones(
    classified: t.Dict[str, t.List[t.Any]],
    transformer: CoordinateTransformer,
    zone_colors: ZoneColorsConfig,
) -> t.List[DarkZone]:
    """Build DarkZone objects for parks, water, and industrial areas.

    Args:
        classified (Dict): Classified rows per bucket from _classify_land_use.
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        zone_colors (ZoneColorsConfig): Hex color strings per zone type.

    Returns:
        List[DarkZone]: All dark zones combined.
    """
    dark_zones: t.List[DarkZone] = []

    park_color = hex_to_rgb(zone_colors.park_forest)
    water_color = hex_to_rgb(zone_colors.water)
    industrial_color = hex_to_rgb(zone_colors.industrial)

    dark_zones.extend(
        _build_dark_zones_for_type(classified["park"], transformer, "park", park_color)
    )
    dark_zones.extend(
        _build_dark_zones_for_type(classified["water"], transformer, "water", water_color)
    )
    dark_zones.extend(
        _build_dark_zones_for_type(
            classified["industrial"], transformer, "industrial", industrial_color
        )
    )
    return dark_zones


def _build_dark_zones_for_type(
    rows: t.List[t.Any],
    transformer: CoordinateTransformer,
    zone_type: str,
    base_color: t.Tuple[int, int, int],
) -> t.List[DarkZone]:
    """Build DarkZone objects for one zone type from a list of rows.

    Args:
        rows (List): GeoDataFrame rows for this zone type.
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        zone_type (str): Semantic type string (e.g. "park", "water").
        base_color (Tuple[int, int, int]): (R, G, B) ambient color.

    Returns:
        List[DarkZone]: One DarkZone per valid polygon.
    """
    dark_zones = []
    for row in rows:
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        for polygon in _iter_polygons(geom):
            dark_zones.extend(
                _as_list(_dark_zone_from_polygon(polygon, transformer, zone_type, base_color))
            )
    return dark_zones


def _dark_zone_from_polygon(
    wgs84_polygon: Polygon,
    transformer: CoordinateTransformer,
    zone_type: str,
    base_color: t.Tuple[int, int, int],
) -> t.Optional[DarkZone]:
    """Build a single DarkZone from one WGS84 polygon.

    Args:
        wgs84_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        zone_type (str): Semantic type string (e.g. "park", "water").
        base_color (Tuple[int, int, int]): (R, G, B) ambient color.

    Returns:
        DarkZone or None: None if ENU conversion fails.
    """
    enu_boundary = _polygon_wgs84_to_enu(wgs84_polygon, transformer)
    if enu_boundary is None:
        return None
    return DarkZone(boundary=enu_boundary, zone_type=zone_type, base_color=base_color)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


_T = t.TypeVar("_T")


def _as_list(item: t.Optional[_T]) -> t.List[_T]:
    """Wrap an optional item in a list, returning empty list for None.

    Args:
        item: Value to wrap, or None.

    Returns:
        List: Single-element list, or empty list if item is None.
    """
    if item is None:
        return []
    return [item]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _polygon_wgs84_to_enu(
    polygon: Polygon,
    transformer: CoordinateTransformer,
) -> t.Optional[Polygon]:
    """Convert a shapely Polygon from WGS84 lon/lat to ENU metres.

    Interior rings (holes) are converted along with the exterior — a lake
    inside a park or a courtyard inside a residential block is preserved
    rather than painted over solid.

    Args:
        polygon (Polygon): Input polygon with WGS84 (lon, lat) coordinates.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        Polygon or None: ENU polygon, or None if conversion fails or the
            exterior ring produces fewer than 3 vertices.
    """
    try:
        shell = _ring_wgs84_to_enu(polygon.exterior.coords, transformer)
        if shell is None:
            return None
        holes = _convert_interior_rings(polygon.interiors, transformer)
        return Polygon(shell, holes=holes)
    except Exception:
        return None


def _ring_wgs84_to_enu(
    ring_coords: t.Any,
    transformer: CoordinateTransformer,
) -> t.Optional[t.List[t.Tuple[float, float]]]:
    """Convert a single WGS84 ring's coordinates to ENU (x, y) tuples.

    Shapely stores coordinates as (x=lon, y=lat). The transformer expects
    (lat, lon), so coordinates are swapped before conversion.

    Args:
        ring_coords: Iterable of (lon, lat) coordinate pairs from a Shapely
            LinearRing's `.coords`.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[Tuple[float, float]] or None: ENU (x, y) coordinates, or None
            if the ring has fewer than 3 vertices.
    """
    coords = np.array(ring_coords)
    if len(coords) < 3:
        return None
    lons = coords[:, 0]
    lats = coords[:, 1]
    enu_coords = wgs84_to_enu_batch(transformer, lats, lons, alt_m=0.0)
    return [(pt[0], pt[1]) for pt in enu_coords]


def _convert_interior_rings(
    interiors: t.Any,
    transformer: CoordinateTransformer,
) -> t.List[t.List[t.Tuple[float, float]]]:
    """Convert all interior rings (holes) of a polygon to ENU coordinates.

    Args:
        interiors: Iterable of Shapely LinearRings (a Polygon's `.interiors`).
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[List[Tuple[float, float]]]: One ENU coordinate list per hole
            that converts successfully; degenerate rings are dropped.
    """
    holes = []
    for interior in interiors:
        ring = _ring_wgs84_to_enu(interior.coords, transformer)
        if ring is not None:
            holes.append(ring)
    return holes


def _iter_polygons(geom: t.Any) -> t.Iterator[Polygon]:
    """Yield individual Polygons from a Polygon or MultiPolygon.

    Args:
        geom: A shapely Polygon, MultiPolygon, or any geometry.

    Yields:
        Polygon: Each individual polygon from the geometry.
    """
    if isinstance(geom, Polygon):
        if not geom.is_empty:
            yield geom
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            if isinstance(part, Polygon) and not part.is_empty:
                yield part


def _get_buildings_within(
    zone_polygon: Polygon,
    buildings_gdf: gpd.GeoDataFrame,
) -> t.List[t.Any]:
    """Find building rows uniquely assigned to the given zone polygon.

    Uses `_get_candidate_buildings` to find geometry that intersects the
    zone (via the spatial index), then assigns each candidate to at most
    one zone by testing whether its representative point falls inside this
    zone's polygon. Without this second step, a building straddling the
    shared edge of two adjacent zones would be duplicated into both.

    Args:
        zone_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        buildings_gdf (GeoDataFrame): Building footprints in WGS84.

    Returns:
        List: GeoDataFrame rows uniquely assigned to this zone.
    """
    if buildings_gdf.empty:
        return []

    candidates = _get_candidate_buildings(zone_polygon, buildings_gdf)
    return [row for row in candidates if _building_belongs_to_zone(row, zone_polygon)]


def _building_belongs_to_zone(row: t.Any, zone_polygon: Polygon) -> bool:
    """Determine whether a building row's representative point falls inside a zone.

    A representative point (rather than the centroid) is guaranteed by
    Shapely to lie within the geometry itself, which keeps this assignment
    stable for concave or irregular footprints.

    Args:
        row: A GeoDataFrame row with a `geometry` attribute.
        zone_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.

    Returns:
        bool: True if the building's representative point is inside the zone.
    """
    geom = getattr(row, "geometry", None)
    if geom is None:
        return False
    return zone_polygon.contains(geom.representative_point())


def _get_candidate_buildings(
    zone_polygon: Polygon,
    buildings_gdf: gpd.GeoDataFrame,
) -> t.List[t.Any]:
    """Find building rows whose geometry intersects the given zone polygon.

    Uses the GeoDataFrame spatial index (sindex) when available for O(log n)
    candidate selection, then confirms with a shapely intersects check.
    Falls back to iterating all rows if the spatial index is unavailable.
    These are intersection candidates only — a straddling building may be a
    candidate for more than one zone; `_get_buildings_within` narrows that
    down to a single zone.

    Args:
        zone_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        buildings_gdf (GeoDataFrame): Building footprints in WGS84.

    Returns:
        List: GeoDataFrame rows whose geometry intersects the zone boundary.
    """
    try:
        candidate_indices = buildings_gdf.sindex.query(zone_polygon, predicate="intersects")
        candidates = buildings_gdf.iloc[candidate_indices]
        return list(candidates.itertuples())
    except Exception:
        return _get_candidate_buildings_fallback(zone_polygon, buildings_gdf)


def _get_candidate_buildings_fallback(
    zone_polygon: Polygon,
    buildings_gdf: gpd.GeoDataFrame,
) -> t.List[t.Any]:
    """Find intersecting building rows by iterating all rows with a bbox pre-filter.

    Used as a fallback when the spatial index is unavailable.

    Args:
        zone_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        buildings_gdf (GeoDataFrame): Building footprints in WGS84.

    Returns:
        List: GeoDataFrame rows whose geometry intersects the zone boundary.
    """
    min_x, min_y, max_x, max_y = zone_polygon.bounds
    candidate_rows = []
    for row in buildings_gdf.itertuples():
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        bx0, by0, bx1, by1 = geom.bounds
        if bx1 < min_x or bx0 > max_x or by1 < min_y or by0 > max_y:
            continue
        if zone_polygon.intersects(geom):
            candidate_rows.append(row)
    return candidate_rows


def _convert_building_footprints(
    building_rows: t.List[t.Any],
    transformer: CoordinateTransformer,
) -> t.List[Polygon]:
    """Convert building row geometries to ENU Polygon objects.

    Handles Polygon and MultiPolygon geometries. All polygons are converted
    in a single batched pyproj transform (see `_polygons_wgs84_to_enu_batch`)
    rather than one transform call per building.

    Args:
        building_rows (List): GeoDataFrame rows with WGS84 geometry.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[Polygon]: Valid ENU footprint polygons.
    """
    polygons = _collect_building_polygons(building_rows)
    return _polygons_wgs84_to_enu_batch(polygons, transformer)


def _collect_building_polygons(building_rows: t.List[t.Any]) -> t.List[Polygon]:
    """Flatten building row geometries into individual WGS84 Polygons.

    Args:
        building_rows (List): GeoDataFrame rows with WGS84 geometry.

    Returns:
        List[Polygon]: Individual polygons (Polygon rows and MultiPolygon parts).
    """
    polygons = []
    for row in building_rows:
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        polygons.extend(_iter_polygons(geom))
    return polygons


def _polygons_wgs84_to_enu_batch(
    polygons: t.List[Polygon],
    transformer: CoordinateTransformer,
) -> t.List[Polygon]:
    """Convert many WGS84 polygons (with holes) to ENU in one batched transform.

    Every exterior and interior ring across all input polygons is
    concatenated into a single coordinate array and converted with one
    `wgs84_to_enu_batch` call, replacing one pyproj transform per polygon.
    Falls back to converting polygons one at a time if the batch fails for
    any reason, matching the per-polygon robustness of `_polygon_wgs84_to_enu`.

    Args:
        polygons (List[Polygon]): WGS84 polygons to convert.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[Polygon]: ENU polygons; polygons whose exterior has fewer than
            3 vertices are dropped.
    """
    if not polygons:
        return []
    try:
        return _batch_convert_polygons(polygons, transformer)
    except Exception:
        return _convert_polygons_individually(polygons, transformer)


def _convert_polygons_individually(
    polygons: t.List[Polygon],
    transformer: CoordinateTransformer,
) -> t.List[Polygon]:
    """Convert each polygon with its own `_polygon_wgs84_to_enu` call.

    Fallback path used when the batched conversion fails.

    Args:
        polygons (List[Polygon]): WGS84 polygons to convert.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[Polygon]: ENU polygons that converted successfully.
    """
    results = []
    for polygon in polygons:
        results.extend(_as_list(_polygon_wgs84_to_enu(polygon, transformer)))
    return results


def _batch_convert_polygons(
    polygons: t.List[Polygon],
    transformer: CoordinateTransformer,
) -> t.List[Polygon]:
    """Convert all polygons' rings in a single batched pyproj transform.

    Args:
        polygons (List[Polygon]): WGS84 polygons to convert.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[Polygon]: ENU polygons; polygons whose exterior has fewer than
            3 vertices are dropped.
    """
    rings, ring_owners = _flatten_polygon_rings(polygons)
    if not rings:
        return []

    lengths = [len(ring) for ring in rings]
    all_coords = np.concatenate(rings, axis=0)
    enu_all = wgs84_to_enu_batch(transformer, all_coords[:, 1], all_coords[:, 0], alt_m=0.0)
    enu_rings = _split_by_lengths(enu_all, lengths)

    return _rebuild_polygons(len(polygons), ring_owners, enu_rings)


def _flatten_polygon_rings(
    polygons: t.List[Polygon],
) -> t.Tuple[t.List[np.ndarray], t.List[t.Tuple[int, bool]]]:
    """Flatten every polygon's exterior and interior rings into one list.

    Args:
        polygons (List[Polygon]): WGS84 polygons.

    Returns:
        Tuple[List[np.ndarray], List[Tuple[int, bool]]]: Parallel lists of
            (N, 2) ring coordinate arrays and (polygon_index, is_shell) owners.
    """
    rings = []
    owners = []
    for poly_idx, polygon in enumerate(polygons):
        rings.append(np.array(polygon.exterior.coords))
        owners.append((poly_idx, True))
        for interior in polygon.interiors:
            rings.append(np.array(interior.coords))
            owners.append((poly_idx, False))
    return rings, owners


def _split_by_lengths(array: np.ndarray, lengths: t.List[int]) -> t.List[np.ndarray]:
    """Split a stacked array back into per-ring chunks using known lengths.

    Args:
        array (np.ndarray): Concatenated (N, 3) ENU coordinate array.
        lengths (List[int]): Number of rows contributed by each ring, in order.

    Returns:
        List[np.ndarray]: One chunk per input ring, in the same order.
    """
    split_points = np.cumsum(lengths)[:-1]
    return np.split(array, split_points)


def _rebuild_polygons(
    num_polygons: int,
    ring_owners: t.List[t.Tuple[int, bool]],
    enu_rings: t.List[np.ndarray],
) -> t.List[Polygon]:
    """Reassemble per-polygon Polygon objects from converted ENU rings.

    Args:
        num_polygons (int): Number of source polygons (for pre-sizing holes).
        ring_owners (List[Tuple[int, bool]]): (polygon_index, is_shell) per ring,
            in the same order as `enu_rings`.
        enu_rings (List[np.ndarray]): Converted ENU coordinates per ring.

    Returns:
        List[Polygon]: ENU polygons; polygons whose shell has fewer than 3
            vertices are dropped.
    """
    shells: t.Dict[int, t.List[t.Tuple[float, float]]] = {}
    holes: t.Dict[int, t.List[t.List[t.Tuple[float, float]]]] = {
        i: [] for i in range(num_polygons)
    }

    for (poly_idx, is_shell), enu_ring in zip(ring_owners, enu_rings):
        xy_coords = [(pt[0], pt[1]) for pt in enu_ring]
        if is_shell:
            shells[poly_idx] = xy_coords
        elif len(xy_coords) >= 3:
            holes[poly_idx].append(xy_coords)

    return [
        Polygon(shell, holes=holes[poly_idx])
        for poly_idx, shell in shells.items()
        if len(shell) >= 3
    ]
