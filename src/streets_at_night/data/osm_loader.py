"""Download and cache OSM road, land use, and building data via osmnx."""

import hashlib
import typing as t
from pathlib import Path

import geopandas as gpd
import osmnx as ox


def load_roads(bbox: t.Tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    """Download or load cached road network edges for the given bounding box.

    Args:
        bbox (tuple): (north, south, east, west) bounding box in WGS84 degrees.
        cache_dir (Path): Directory for .gpkg cache files.

    Returns:
        GeoDataFrame: Road network edges with OSM tags.
    """
    cache_path = _cache_path(cache_dir, "roads", bbox)
    if cache_path.exists():
        return gpd.read_file(cache_path)
    gdf = _download_roads(bbox)
    _save_to_gpkg(gdf, cache_path, keep_cols=_ROADS_KEEP_COLS)
    return gdf


def load_land_use(bbox: t.Tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    """Download or load cached land use polygons for the given bounding box.

    Args:
        bbox (tuple): (north, south, east, west) bounding box in WGS84 degrees.
        cache_dir (Path): Directory for .gpkg cache files.

    Returns:
        GeoDataFrame: Land use polygons with OSM tags.
    """
    cache_path = _cache_path(cache_dir, "land_use", bbox)
    if cache_path.exists():
        return gpd.read_file(cache_path)
    gdf = _download_land_use(bbox)
    _save_to_gpkg(gdf, cache_path)
    return gdf


def load_buildings(bbox: t.Tuple[float, float, float, float], cache_dir: Path) -> gpd.GeoDataFrame:
    """Download or load cached building footprints for the given bounding box.

    Args:
        bbox (tuple): (north, south, east, west) bounding box in WGS84 degrees.
        cache_dir (Path): Directory for .gpkg cache files.

    Returns:
        GeoDataFrame: Building footprint polygons with OSM tags.
    """
    cache_path = _cache_path(cache_dir, "buildings", bbox)
    if cache_path.exists():
        return gpd.read_file(cache_path)
    gdf = _download_buildings(bbox)
    _save_to_gpkg(gdf, cache_path, keep_cols=_BUILDINGS_KEEP_COLS)
    return gdf


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
        network_type="drive",
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
    tags = {
        "landuse": True,
        "leisure": ["park", "garden", "nature_reserve"],
        "natural": ["water", "wood", "grassland"],
        "waterway": ["riverbank"],
    }
    return ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)


def _download_buildings(bbox: t.Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Download building footprints from OSM.

    osmnx 2.x expects bbox as (left, bottom, right, top) = (west, south, east, north).

    Args:
        bbox (tuple): (north, south, east, west) — our internal convention.

    Returns:
        GeoDataFrame: Building footprint polygons.
    """
    north, south, east, west = bbox
    return ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})


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
    keep the columns needed downstream.

    Args:
        gdf (GeoDataFrame): Data to save.
        path (Path): Destination .gpkg file path.
        keep_cols (list[str] | None): Columns to retain. Columns not present
            in the GeoDataFrame are silently skipped.
    """
    if keep_cols is None:
        keep_cols = _FEATURES_KEEP_COLS
    present = [c for c in keep_cols if c in gdf.columns]
    gdf[present].to_file(path, driver="GPKG")


def _cache_path(cache_dir: Path, prefix: str, bbox: t.Tuple[float, float, float, float]) -> Path:
    """Build the cache file path for a given prefix and bounding box.

    Args:
        cache_dir (Path): Cache directory.
        prefix (str): File name prefix (e.g. "roads", "land_use").
        bbox (tuple): Bounding box used as the cache key.

    Returns:
        Path: Full path to the .gpkg cache file.
    """
    key = hashlib.md5(str(bbox).encode()).hexdigest()[:8]
    return cache_dir / f"{prefix}_{key}.gpkg"
