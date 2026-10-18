"""WGS84 (lat/lon/alt) to local East-North-Up (ENU) meter coordinate conversion.

Uses the UTM zone containing the local origin as an intermediate CRS (e.g.
zone 11N for San Bernardino, zone 18N for Rochester, NY). Projecting through
a fixed far-away zone rotates and stretches the whole scene, so the zone is
always derived from the origin longitude/latitude. All output coordinates are
offsets in meters from a configurable local origin.

ENU convention:
    X = East  (meters)
    Y = North (meters)
    Z = Up    (meters, converted from feet where needed)
"""

import typing as t

import geopandas as gpd
import numpy as np
from pyproj import Transformer


_FEET_TO_METERS: float = 0.3048
_WGS84_EPSG: str = "EPSG:4326"


class CoordinateTransformer:
    """Converts between WGS84 lat/lon and local ENU meters.

    The local origin is defined at construction time. All ENU coordinates
    are offsets in meters (East, North, Up) from that origin.

    Use create_transformer() to construct an instance — do not instantiate
    directly unless you already have a valid pyproj Transformer object.

    Attributes:
        origin_lat: Latitude of the local origin in WGS84 degrees.
        origin_lon: Longitude of the local origin in WGS84 degrees.
    """

    def __init__(
        self,
        origin_lat: float,
        origin_lon: float,
        origin_easting: float,
        origin_northing: float,
        to_utm: Transformer,
        from_utm: Transformer,
        utm_epsg: str,
    ) -> None:
        """Initialise a CoordinateTransformer.

        Args:
            origin_lat (float): Latitude of the local origin in WGS84 degrees.
            origin_lon (float): Longitude of the local origin in WGS84 degrees.
            origin_easting (float): UTM easting of the origin in meters.
            origin_northing (float): UTM northing of the origin in meters.
            to_utm (Transformer): pyproj Transformer from WGS84 to the origin's UTM zone.
            from_utm (Transformer): pyproj Transformer from the origin's UTM zone to WGS84.
            utm_epsg (str): EPSG code of the intermediate UTM zone (e.g. "EPSG:32611").
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.utm_epsg = utm_epsg
        self._origin_easting = origin_easting
        self._origin_northing = origin_northing
        self._to_utm = to_utm
        self._from_utm = from_utm

    def wgs84_to_enu(
        self,
        lat: float,
        lon: float,
        alt_m: float = 0.0,
    ) -> t.Tuple[float, float, float]:
        """Convert a single WGS84 lat/lon/alt point to local ENU meters.

        Args:
            lat (float): Latitude in WGS84 degrees.
            lon (float): Longitude in WGS84 degrees.
            alt_m (float): Altitude in meters above the reference ellipsoid.

        Returns:
            tuple[float, float, float]: (east_m, north_m, up_m) in meters from origin.
        """
        easting, northing = self._to_utm.transform(lon, lat)
        east_m = easting - self._origin_easting
        north_m = northing - self._origin_northing
        return float(east_m), float(north_m), float(alt_m)

    def wgs84_to_enu_batch(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        alt_m: float = 0.0,
    ) -> np.ndarray:
        """Convert arrays of WGS84 lat/lon to local ENU meter coordinates.

        Args:
            lats (np.ndarray): Array of latitudes in WGS84 degrees.
            lons (np.ndarray): Array of longitudes in WGS84 degrees.
            alt_m (float): Altitude in meters, applied uniformly to all points.

        Returns:
            np.ndarray: Shape (N, 3) array of (east_m, north_m, up_m) values.
        """
        eastings, northings = self._to_utm.transform(lons, lats)
        east_m = np.asarray(eastings) - self._origin_easting
        north_m = np.asarray(northings) - self._origin_northing
        up_m = np.full_like(east_m, alt_m)
        return np.column_stack([east_m, north_m, up_m])

    def enu_to_wgs84(
        self,
        east_m: float,
        north_m: float,
        up_m: float = 0.0,
    ) -> t.Tuple[float, float, float]:
        """Convert local ENU meters back to WGS84 lat/lon/alt.

        Reverses the WGS84 → UTM → ENU conversion by adding the origin offset
        back and projecting through the inverse UTM transformer.

        Args:
            east_m (float): East offset in meters from origin.
            north_m (float): North offset in meters from origin.
            up_m (float): Up (altitude) in meters.

        Returns:
            tuple[float, float, float]: (lat, lon, alt_m) in WGS84 degrees and meters.
        """
        easting = east_m + self._origin_easting
        northing = north_m + self._origin_northing
        lon, lat = self._from_utm.transform(easting, northing)
        return float(lat), float(lon), float(up_m)


def get_utm_epsg(lat: float, lon: float) -> str:
    """Return the EPSG code of the UTM zone containing a WGS84 point.

    Args:
        lat (float): Latitude in WGS84 degrees (selects the N/S EPSG band).
        lon (float): Longitude in WGS84 degrees (selects the zone number).

    Returns:
        str: EPSG code string, e.g. "EPSG:32611" (north) or "EPSG:32756" (south).
    """
    zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
    band = 32600 if lat >= 0.0 else 32700
    return f"EPSG:{band + zone}"


def create_transformer(origin_lat: float, origin_lon: float) -> CoordinateTransformer:
    """Create a coordinate transformer centered on the given lat/lon origin.

    The intermediate UTM zone is derived from the origin so scenes anywhere on
    the globe project without the rotation/stretch of an out-of-zone CRS.

    Args:
        origin_lat (float): Latitude of the local origin in WGS84 degrees.
        origin_lon (float): Longitude of the local origin in WGS84 degrees.

    Returns:
        CoordinateTransformer: Configured transformer ready for conversion.
    """
    utm_epsg = get_utm_epsg(origin_lat, origin_lon)
    to_utm = Transformer.from_crs(_WGS84_EPSG, utm_epsg, always_xy=True)
    from_utm = Transformer.from_crs(utm_epsg, _WGS84_EPSG, always_xy=True)
    origin_easting, origin_northing = to_utm.transform(origin_lon, origin_lat)
    return CoordinateTransformer(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        origin_easting=origin_easting,
        origin_northing=origin_northing,
        to_utm=to_utm,
        from_utm=from_utm,
        utm_epsg=utm_epsg,
    )


def wgs84_to_enu(
    transformer: CoordinateTransformer,
    lat: float,
    lon: float,
    alt_m: float = 0.0,
) -> t.Tuple[float, float, float]:
    """Convert a single WGS84 lat/lon/alt point to local ENU meters.

    Args:
        transformer (CoordinateTransformer): Configured transformer.
        lat (float): Latitude in WGS84 degrees.
        lon (float): Longitude in WGS84 degrees.
        alt_m (float): Altitude in meters above the reference ellipsoid.

    Returns:
        tuple[float, float, float]: (east_m, north_m, up_m) in meters from origin.
    """
    return transformer.wgs84_to_enu(lat, lon, alt_m)


def wgs84_to_enu_batch(
    transformer: CoordinateTransformer,
    lats: np.ndarray,
    lons: np.ndarray,
    alt_m: float = 0.0,
) -> np.ndarray:
    """Convert arrays of WGS84 lat/lon to local ENU meter coordinates.

    Args:
        transformer (CoordinateTransformer): Configured transformer.
        lats (np.ndarray): Array of latitudes in WGS84 degrees.
        lons (np.ndarray): Array of longitudes in WGS84 degrees.
        alt_m (float): Altitude in meters, applied uniformly to all points.

    Returns:
        np.ndarray: Shape (N, 3) array of (east_m, north_m, up_m) values.
    """
    return transformer.wgs84_to_enu_batch(lats, lons, alt_m)


def convert_road_geometry(
    transformer: CoordinateTransformer,
    roads_gdf: gpd.GeoDataFrame,
) -> t.List[np.ndarray]:
    """Convert road GeoDataFrame geometries from WGS84 to ENU meter coordinates.

    Each LineString geometry in the GeoDataFrame is converted to a (N, 3)
    array of ENU meter coordinates. The Z coordinate is set to 0 (ground level).
    Geometries that are None or not LineString-like are skipped.

    All valid segments are converted in a single batched pyproj transform
    (coordinates from every segment concatenated, transformed once, then
    split back by segment length) rather than one transform call per segment.

    Args:
        transformer (CoordinateTransformer): Configured transformer.
        roads_gdf (GeoDataFrame): Road features with WGS84 geometry column.

    Returns:
        list[np.ndarray]: One (N, 3) array per road segment. Z is always 0.
    """
    valid_coords = _collect_valid_road_coords(roads_gdf)
    if not valid_coords:
        return []

    lengths = [len(coords) for coords in valid_coords]
    all_coords = np.concatenate(valid_coords, axis=0)
    enu_all = wgs84_to_enu_batch(transformer, all_coords[:, 1], all_coords[:, 0], alt_m=0.0)

    split_points = np.cumsum(lengths)[:-1]
    return np.split(enu_all, split_points)


def _collect_valid_road_coords(roads_gdf: gpd.GeoDataFrame) -> t.List[np.ndarray]:
    """Collect WGS84 (lon, lat) coordinate arrays for every valid road geometry.

    Args:
        roads_gdf (GeoDataFrame): Road features with WGS84 geometry column.

    Returns:
        list[np.ndarray]: One (N, 2) lon/lat coordinate array per road
            geometry with at least 2 vertices. Geometries that are None or
            lack `.coords` are skipped.
    """
    valid_coords = []
    for geom in roads_gdf.geometry:
        if geom is None or not hasattr(geom, "coords"):
            continue
        coords = np.array(geom.coords)
        if coords.shape[0] < 2:
            continue
        valid_coords.append(coords)
    return valid_coords


def feet_to_meters(altitude_ft: float) -> float:
    """Convert altitude in feet to meters.

    Args:
        altitude_ft (float): Altitude in feet.

    Returns:
        float: Altitude in meters.
    """
    return altitude_ft * _FEET_TO_METERS


def meters_to_feet(altitude_m: float) -> float:
    """Convert altitude in meters to feet.

    Args:
        altitude_m (float): Altitude in meters.

    Returns:
        float: Altitude in feet.
    """
    return altitude_m / _FEET_TO_METERS


def enu_to_wgs84(
    transformer: CoordinateTransformer,
    east_m: float,
    north_m: float,
    up_m: float = 0.0,
) -> t.Tuple[float, float, float]:
    """Convert local ENU meters back to WGS84 lat/lon/alt.

    Reverses the WGS84 → UTM → ENU conversion by adding the origin offset back
    and projecting through the inverse UTM transformer.

    Args:
        transformer (CoordinateTransformer): Configured coordinate transformer.
        east_m (float): East offset in meters from origin.
        north_m (float): North offset in meters from origin.
        up_m (float): Up (altitude) in meters.

    Returns:
        tuple[float, float, float]: (lat, lon, alt_m) in WGS84 degrees and meters.
    """
    return transformer.enu_to_wgs84(east_m, north_m, up_m)
