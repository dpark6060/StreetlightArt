"""Build zone objects from OSM land use and building data.

Classifies land use polygons by OSM tag, converts geometries from WGS84 to
ENU metres, and constructs ResidentialZone, CommercialZone, and DarkZone
objects.  Residential zones include Houses derived from OSM building
footprints; where footprints are absent a simple grid subdivision is used.
"""

import random
import typing as t

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..camera.coordinates import CoordinateTransformer, wgs84_to_enu_batch
from ..config.schema import HouseConfig, ZoneColorsConfig, ZoneConfig
from ..models.color_pool import ColorPool
from ..models.house import House
from ..models.zones import CommercialZone, DarkZone, ResidentialZone
from ..utils.colors import hex_to_rgb


# ---------------------------------------------------------------------------
# OSM tag classification sets
# ---------------------------------------------------------------------------

_RESIDENTIAL_LANDUSE: t.FrozenSet[str] = frozenset({"residential"})
_COMMERCIAL_LANDUSE: t.FrozenSet[str] = frozenset({"commercial", "retail"})
_INDUSTRIAL_LANDUSE: t.FrozenSet[str] = frozenset({"industrial"})
_PARK_LEISURE: t.FrozenSet[str] = frozenset({"park", "garden", "nature_reserve"})
_PARK_NATURAL: t.FrozenSet[str] = frozenset({"wood", "grassland"})
_PARK_LANDUSE: t.FrozenSet[str] = frozenset({"forest"})
_WATER_NATURAL: t.FrozenSet[str] = frozenset({"water"})
_WATER_WATERWAY: t.FrozenSet[str] = frozenset({"riverbank"})


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

    Args:
        row: A namedtuple row from GeoDataFrame.itertuples().

    Returns:
        str or None: Bucket name, or None if the row doesn't match any zone.
    """
    landuse = _safe_tag(row, "landuse")
    leisure = _safe_tag(row, "leisure")
    natural = _safe_tag(row, "natural")
    waterway = _safe_tag(row, "waterway")

    if landuse in _RESIDENTIAL_LANDUSE:
        return "residential"
    if landuse in _COMMERCIAL_LANDUSE:
        return "commercial"
    if landuse in _INDUSTRIAL_LANDUSE:
        return "industrial"
    if leisure in _PARK_LEISURE or natural in _PARK_NATURAL or landuse in _PARK_LANDUSE:
        return "park"
    if natural in _WATER_NATURAL or waterway in _WATER_WATERWAY:
        return "water"
    return None


def _safe_tag(row: t.Any, attr: str) -> str:
    """Safely read an OSM tag attribute from a row, returning empty string if absent.

    Args:
        row: A namedtuple row from GeoDataFrame.itertuples().
        attr (str): Attribute name to read.

    Returns:
        str: Tag value, or empty string if missing or NaN.
    """
    value = getattr(row, attr, None)
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

    if not houses:
        synthetic_footprints = _build_synthetic_lots(enu_boundary, house_config)
        houses = _build_houses_from_footprints(
            synthetic_footprints, house_config, rng, color_pool
        )

    return ResidentialZone(boundary=enu_boundary, houses=houses)


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
) -> t.List[Polygon]:
    """Subdivide a zone boundary into synthetic rectangular house footprints.

    Uses a simple grid of lots based on lot_width and lot_depth from config.
    Only lots whose centroid falls inside the zone boundary are kept.
    Each lot produces a smaller footprint polygon scaled by footprint_coverage.

    Args:
        zone_boundary (Polygon): Zone boundary polygon in ENU metres.
        house_config (HouseConfig): Lot dimensions and footprint coverage ratio.

    Returns:
        List[Polygon]: Rectangular house footprint polygons in ENU metres.
    """
    min_x, min_y, max_x, max_y = zone_boundary.bounds
    lot_w = house_config.lot_width
    lot_d = house_config.lot_depth
    coverage = house_config.footprint_coverage

    candidate_lots = _grid_lots(min_x, min_y, max_x, max_y, lot_w, lot_d, coverage)
    return [fp for fp in candidate_lots if zone_boundary.contains(fp.centroid)]


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
    footprints = []
    x = min_x
    while x + lot_w <= max_x:
        footprints.extend(_column_lots(x, min_y, max_y, lot_w, lot_d, coverage))
        x += lot_w
    return footprints


def _column_lots(
    x: float,
    min_y: float,
    max_y: float,
    lot_w: float,
    lot_d: float,
    coverage: float,
) -> t.List[Polygon]:
    """Generate a vertical column of lot footprints at a given x position.

    Args:
        x (float): Left edge of this column in ENU metres.
        min_y (float): Bottom of the column.
        max_y (float): Top of the column.
        lot_w (float): Lot width in metres.
        lot_d (float): Lot depth in metres.
        coverage (float): Footprint coverage fraction (0–1).

    Returns:
        List[Polygon]: Footprint polygons in this column.
    """
    footprints = []
    y = min_y
    while y + lot_d <= max_y:
        footprints.append(_make_lot_footprint(x, y, lot_w, lot_d, coverage))
        y += lot_d
    return footprints


def _make_lot_footprint(
    lot_x: float,
    lot_y: float,
    lot_w: float,
    lot_d: float,
    coverage: float,
) -> Polygon:
    """Build a house footprint polygon centered within a lot.

    The footprint is inset from the lot edges using the coverage ratio,
    centered within the lot rectangle.

    Args:
        lot_x (float): Lot lower-left X in ENU metres.
        lot_y (float): Lot lower-left Y in ENU metres.
        lot_w (float): Lot width in metres.
        lot_d (float): Lot depth in metres.
        coverage (float): Fraction of lot covered by footprint (0–1).

    Returns:
        Polygon: House footprint polygon in ENU metres.
    """
    fp_w = lot_w * coverage
    fp_d = lot_d * coverage
    margin_x = (lot_w - fp_w) / 2.0
    margin_y = (lot_d - fp_d) / 2.0

    x0 = lot_x + margin_x
    y0 = lot_y + margin_y
    x1 = x0 + fp_w
    y1 = y0 + fp_d

    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


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

    Args:
        enu_boundary (Polygon): Zone boundary in ENU metres.
        enu_footprints (List[Polygon]): Building footprints in ENU metres.

    Returns:
        List[Polygon]: Remaining open-area polygons after subtracting buildings.
    """
    if not enu_footprints:
        return [enu_boundary]

    remaining = enu_boundary
    for footprint in enu_footprints:
        try:
            remaining = remaining.difference(footprint)
        except Exception:
            continue

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

    Shapely stores coordinates as (x=lon, y=lat). The transformer expects
    (lat, lon), so coordinates are swapped before conversion.

    Args:
        polygon (Polygon): Input polygon with WGS84 (lon, lat) coordinates.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        Polygon or None: ENU polygon, or None if conversion fails or produces
            fewer than 3 vertices.
    """
    try:
        coords = np.array(polygon.exterior.coords)
        if len(coords) < 3:
            return None
        lons = coords[:, 0]
        lats = coords[:, 1]
        enu_coords = wgs84_to_enu_batch(transformer, lats, lons, alt_m=0.0)
        xy_coords = [(pt[0], pt[1]) for pt in enu_coords]
        return Polygon(xy_coords)
    except Exception:
        return None


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
    """Find building rows whose geometry intersects the given zone polygon.

    Uses the GeoDataFrame spatial index (sindex) when available for O(log n)
    candidate selection, then confirms with a shapely intersects check.
    Falls back to iterating all rows if the spatial index is unavailable.

    Args:
        zone_polygon (Polygon): Zone boundary polygon in WGS84 lon/lat.
        buildings_gdf (GeoDataFrame): Building footprints in WGS84.

    Returns:
        List: GeoDataFrame rows whose geometry intersects the zone boundary.
    """
    if buildings_gdf.empty:
        return []

    try:
        candidate_indices = buildings_gdf.sindex.query(zone_polygon, predicate="intersects")
        candidates = buildings_gdf.iloc[candidate_indices]
        return list(candidates.itertuples())
    except Exception:
        return _get_buildings_within_fallback(zone_polygon, buildings_gdf)


def _get_buildings_within_fallback(
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

    Handles Polygon and MultiPolygon geometries. Rows that fail conversion
    are silently skipped.

    Args:
        building_rows (List): GeoDataFrame rows with WGS84 geometry.
        transformer (CoordinateTransformer): WGS84 → ENU converter.

    Returns:
        List[Polygon]: Valid ENU footprint polygons.
    """
    footprints = []
    for row in building_rows:
        geom = getattr(row, "geometry", None)
        if geom is None:
            continue
        for polygon in _iter_polygons(geom):
            footprints.extend(_as_list(_polygon_wgs84_to_enu(polygon, transformer)))
    return footprints
