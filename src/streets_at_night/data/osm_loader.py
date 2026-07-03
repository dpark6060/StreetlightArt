"""Download and cache OSM road, land use, and building data via osmnx."""

import hashlib
import typing as t
from pathlib import Path

import geopandas as gpd
import osmnx as ox


# Bump whenever keep-cols or on-disk normalization change, so stale caches
# written by an older version of this module are bypassed automatically.
_CACHE_SCHEMA_VERSION: int = 2

_ROADS_PARAMS: dict = {"network_type": "drive"}

_LAND_USE_PARAMS: dict = {
    "tags": {
        "landuse": True,
        "leisure": ["park", "garden", "nature_reserve"],
        "natural": ["water", "wood", "grassland"],
        "waterway": ["riverbank"],
    }
}

_BUILDINGS_PARAMS: dict = {"tags": {"building": True}}


def load_roads(bbox: t.Tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    """Download or load cached road network edges for the given bounding box.

    Args:
        bbox (tuple): (north, south, east, west) bounding box in WGS84 degrees.
        cache_dir (Path): Region cache directory; osm/ and osmnx/ subdirs are
            created automatically.

    Returns:
        GeoDataFrame: Road network edges with OSM tags.
    """
    _configure_osmnx_cache(cache_dir)
    cache_path = _cache_path(cache_dir, "roads", bbox, _ROADS_PARAMS)
    if cache_path.exists():
        return gpd.read_file(cache_path)
    gdf = _download_roads(bbox)
    _save_to_gpkg(gdf, cache_path, keep_cols=_ROADS_KEEP_COLS)
    return gpd.read_file(cache_path)


def load_land_use(bbox: t.Tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    """Download or load cached land use polygons for the given bounding box.

    Args:
        bbox (tuple): (north, south, east, west) bounding box in WGS84 degrees.
        cache_dir (Path): Region cache directory; osm/ and osmnx/ subdirs are
            created automatically.

    Returns:
        GeoDataFrame: Land use polygons with OSM tags.
    """
    _configure_osmnx_cache(cache_dir)
    cache_path = _cache_path(cache_dir, "land_use", bbox, _LAND_USE_PARAMS)
    if cache_path.exists():
        return gpd.read_file(cache_path)
    gdf = _download_land_use(bbox)
    _save_to_gpkg(gdf, cache_path)
    return gpd.read_file(cache_path)


def load_buildings(bbox: t.Tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    """Download or load cached building footprints for the given bounding box.

    Args:
        bbox (tuple): (north, south, east, west) bounding box in WGS84 degrees.
        cache_dir (Path): Region cache directory; osm/ and osmnx/ subdirs are
            created automatically.

    Returns:
        GeoDataFrame: Building footprint polygons with OSM tags.
    """
    _configure_osmnx_cache(cache_dir)
    cache_path = _cache_path(cache_dir, "buildings", bbox, _BUILDINGS_PARAMS)
    if cache_path.exists():
        return gpd.read_file(cache_path)
    gdf = _download_buildings(bbox)
    _save_to_gpkg(gdf, cache_path, keep_cols=_BUILDINGS_KEEP_COLS)
    return gpd.read_file(cache_path)


def _download_roads(bbox: t.Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Download road network from OSM and convert to a GeoDataFrame of edges.

    osmnx 2.x expects bbox as (left, bottom, right, top) = (west, south, east, north).

    Args:
        bbox (tuple): (north, south, east, west) — our internal convention.

    Returns:
        GeoDataFrame: Road edges.
    """
    north, south, east, west = bbox
    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type=_ROADS_PARAMS["network_type"],
    )
    _, edges = ox.graph_to_gdfs(graph)
    return edges


def _download_land_use(bbox: t.Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Download land use features from OSM.

    osmnx 2.x expects bbox as (left, bottom, right, top) = (west, south, east, north).

    Args:
        bbox (tuple): (north, south, east, west) — our internal convention.

    Returns:
        GeoDataFrame: Land use polygons.
    """
    north, south, east, west = bbox
    return ox.features_from_bbox(
        bbox=(west, south, east, north), tags=_LAND_USE_PARAMS["tags"]
    )


def _download_buildings(bbox: t.Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Download building footprints from OSM.

    osmnx 2.x expects bbox as (left, bottom, right, top) = (west, south, east, north).

    Args:
        bbox (tuple): (north, south, east, west) — our internal convention.

    Returns:
        GeoDataFrame: Building footprint polygons.
    """
    north, south, east, west = bbox
    return ox.features_from_bbox(
        bbox=(west, south, east, north), tags=_BUILDINGS_PARAMS["tags"]
    )


_FEATURES_KEEP_COLS: t.List[str] = [
    "geometry",
    "landuse",
    "leisure",
    "natural",
    "waterway",
    "name",
]

_BUILDINGS_KEEP_COLS: t.List[str] = [
    "geometry",
    "building",
    "name",
]

_ROADS_KEEP_COLS: t.List[str] = [
    "geometry",
    "highway",
    "name",
    "osmid",
    "oneway",
    "length",
    "maxspeed",
]


def _save_to_gpkg(
    gdf: gpd.GeoDataFrame, path: Path, keep_cols: t.Optional[t.List[str]] = None
) -> None:
    """Save a GeoDataFrame to a GeoPackage, retaining only the needed columns.

    OSM features GeoDataFrames contain hundreds of columns, many with names
    that are invalid in GeoPackage format (colons, case collisions). We only
    keep the columns needed downstream. List-valued cells (OSM merged-way
    tags such as `highway=['residential', 'unclassified']`) are scalarized
    to their first element before writing, since the GPKG round-trip would
    otherwise stringify them (e.g. to `"['residential', 'unclassified']"`),
    which fails downstream tag matching and silently drops the row.

    Args:
        gdf (GeoDataFrame): Data to save.
        path (Path): Destination .gpkg file path.
        keep_cols (list[str] | None): Columns to retain. Columns not present
            in the GeoDataFrame are silently skipped.
    """
    if keep_cols is None:
        keep_cols = _FEATURES_KEEP_COLS
    present = [c for c in keep_cols if c in gdf.columns]
    to_save = _scalarize_list_cells(gdf[present].copy(), present)
    to_save.to_file(path, driver="GPKG")


def _scalarize_list_cells(gdf: gpd.GeoDataFrame, cols: t.List[str]) -> gpd.GeoDataFrame:
    """Replace list-valued cells with their first element (OSM merged-way convention).

    Mirrors `zone_builder._safe_tag`'s handling of list-valued OSM tags: an
    empty list falls through to None, the same value a missing/None cell
    would already produce, rather than raising an IndexError.

    Args:
        gdf (GeoDataFrame): Data whose columns may contain list-valued cells.
        cols (list[str]): Column names to scalarize. Callers pass only
            columns already known to be present in `gdf`.

    Returns:
        GeoDataFrame: Same frame with list cells replaced by their first element.
    """
    for col in cols:
        gdf[col] = gdf[col].map(lambda v: (v[0] if v else None) if isinstance(v, list) else v)
    return gdf


def _configure_osmnx_cache(cache_dir: Path) -> None:
    """Point osmnx's HTTP response cache at cache_dir/osmnx/.

    Args:
        cache_dir (Path): Region cache directory.
    """
    osmnx_dir = cache_dir / "osmnx"
    osmnx_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(osmnx_dir)
    ox.settings.use_cache = True


def _cache_path(
    cache_dir: Path,
    prefix: str,
    bbox: t.Tuple[float, float, float, float],
    params: dict,
) -> Path:
    """Build the .gpkg cache file path under cache_dir/osm/.

    The key incorporates the schema version, the rounded bbox, and the
    per-loader download params (e.g. network_type, tags), so changing any
    of those — or bumping `_CACHE_SCHEMA_VERSION` after a keep-cols or
    normalization change — invalidates old caches instead of silently
    serving stale data.

    Args:
        cache_dir (Path): Region cache directory.
        prefix (str): File name prefix (e.g. "roads", "land_use").
        bbox (tuple): Bounding box used as part of the cache key.
        params (dict): Per-loader download parameters used as part of the
            cache key (e.g. network_type, tags).

    Returns:
        Path: Full path to the .gpkg cache file.
    """
    osm_dir = cache_dir / "osm"
    osm_dir.mkdir(parents=True, exist_ok=True)
    rounded_bbox = tuple(round(v, 6) for v in bbox)
    key_src = repr((_CACHE_SCHEMA_VERSION, rounded_bbox, sorted(params.items())))
    key = hashlib.md5(key_src.encode()).hexdigest()[:8]
    return osm_dir / f"{prefix}_{key}.gpkg"
