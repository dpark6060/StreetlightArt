"""Scene facade — world state aggregated into a single object.

The Scene holds road segments, zone data, camera, config, and lights.
Lights and car pairs are populated by the Stage 5 placement engine.

Scene construction is handled by placement.scene_builder.build_scene.
"""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.coordinates import CoordinateTransformer
from ..config.schema import SceneConfig
from ..models.building import Building
from ..models.road_segment import RoadSegment
from ..models.zones import CommercialZone, DarkZone, ResidentialZone
from .spatial_index import SpatialIndex

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource


class Scene:
    """World state facade. The single object passed to the render pipeline.

    Holds road segments, zone data (residential, commercial, dark), camera,
    config, lights, car pairs, and optionally the coordinate transformer for
    ENU ↔ WGS84 conversion. Roads, zones, and lights are each kept in their
    own SpatialIndex so queries for one kind never scan or filter items of
    another.

    Attributes:
        road_segments: All road segments in the scene.
        residential_zones: Residential zone boundaries and their house collections.
        commercial_zones: Commercial zone boundaries with building footprints and parking.
        dark_zones: Dark zones (parks, forests, water) with base colors.
        buildings: Building volumes (OSM footprints plus house-derived fill-in).
        lights: All light sources in the scene, as a flat list.
        car_pairs: CarLightPair references for render-time facing resolution.
        camera: The configured 3D camera.
        config: Scene configuration.
        transformer: Optional coordinate transformer for ENU ↔ WGS84 conversion.
    """

    def __init__(
        self,
        road_segments: t.List[RoadSegment],
        camera: Camera,
        config: SceneConfig,
        residential_zones: t.Optional[t.List[ResidentialZone]] = None,
        commercial_zones: t.Optional[t.List[CommercialZone]] = None,
        dark_zones: t.Optional[t.List[DarkZone]] = None,
        transformer: t.Optional[CoordinateTransformer] = None,
        buildings: t.Optional[t.List[Building]] = None,
    ) -> None:
        """Initialise the Scene.

        Args:
            road_segments (list[RoadSegment]): All road segments in the scene.
            camera (Camera): The configured 3D camera.
            config (SceneConfig): Loaded scene configuration.
            residential_zones (list[ResidentialZone], optional): Residential zones.
                Defaults to an empty list.
            commercial_zones (list[CommercialZone], optional): Commercial zones.
                Defaults to an empty list.
            dark_zones (list[DarkZone], optional): Dark zones (parks, water, forest).
                Defaults to an empty list.
            transformer (CoordinateTransformer, optional): Coordinate transformer for
                ENU ↔ WGS84 conversion. Required to use save-camera functionality.
                Defaults to None.
            buildings (list[Building], optional): Building volumes. Defaults to
                an empty list.
        """
        self.road_segments = road_segments
        self.camera = camera
        self.config = config
        self.residential_zones: t.List[ResidentialZone] = residential_zones or []
        self.commercial_zones: t.List[CommercialZone] = commercial_zones or []
        self.dark_zones: t.List[DarkZone] = dark_zones or []
        self.buildings: t.List[Building] = buildings or []
        self.transformer = transformer
        self.lights: t.List["LightSource"] = []
        self.car_pairs: t.List[t.Any] = []  # CarLightPair, typed as Any to avoid circular import
        self._road_index: t.Optional[SpatialIndex] = None
        self._zone_index: t.Optional[SpatialIndex] = None
        self._light_index: t.Optional[SpatialIndex] = None
        self._building_index: t.Optional[SpatialIndex] = None
        self._index_bounds: t.Optional[t.Tuple[float, float, float, float]] = None

    def build_spatial_index(self) -> None:
        """Build the spatial indexes over road segments, zones, and buildings.

        Computes world bounds from road geometries and zone boundaries, then
        creates and populates a separate grid index each for road segments,
        zone boundaries, and buildings. Call once after construction before
        rendering. The light index is created lazily by ``add_lights`` since
        lights are typically populated after this call.
        """
        if not self.road_segments:
            return

        bounds = _compute_combined_bounds(
            self.road_segments,
            self.residential_zones,
            self.commercial_zones,
            self.dark_zones,
        )
        min_x, min_y, max_x, max_y = bounds
        self._index_bounds = bounds
        self._road_index = SpatialIndex(min_x, min_y, max_x, max_y, cell_size=500.0)
        self._zone_index = SpatialIndex(min_x, min_y, max_x, max_y, cell_size=500.0)
        self._building_index = SpatialIndex(min_x, min_y, max_x, max_y, cell_size=500.0)

        self._index_road_segments()
        self._index_zones()
        self._index_buildings()

    def add_lights(self, lights: t.List["LightSource"]) -> None:
        """Add lights to the scene and index them spatially.

        Creates the light index lazily on first use, reusing the same world
        bounds as the road/zone indexes so light queries stay consistent
        with the rest of the scene.

        Args:
            lights (list[LightSource]): Light sources to add.
        """
        self.lights.extend(lights)
        if self._index_bounds is None:
            return

        if self._light_index is None:
            min_x, min_y, max_x, max_y = self._index_bounds
            self._light_index = SpatialIndex(min_x, min_y, max_x, max_y, cell_size=500.0)
        for light in lights:
            pos = light.position
            self._light_index.insert(light, pos[0], pos[1], pos[0], pos[1])

    def add_car_pairs(self, car_pairs: list) -> None:
        """Register car light pairs for render-time facing resolution.

        Args:
            car_pairs (list): CarLightPair objects to register.
        """
        self.car_pairs.extend(car_pairs)

    def get_roads_in_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> t.List[RoadSegment]:
        """Return road segments whose bounding box overlaps the given rectangle.

        Uses the road spatial index if built; falls back to all segments
        otherwise. Deduplicates results since the spatial index may return
        the same segment from multiple cells.

        Args:
            min_x (float): Rectangle minimum x in meters.
            min_y (float): Rectangle minimum y in meters.
            max_x (float): Rectangle maximum x in meters.
            max_y (float): Rectangle maximum y in meters.

        Returns:
            list[RoadSegment]: Candidate segments overlapping the query area.
        """
        if self._road_index is None:
            return self.road_segments

        raw = self._road_index.query_rect(min_x, min_y, max_x, max_y)
        return _deduplicate_by_identity(raw)

    def get_lights_near(
        self, x: float, y: float, radius: float
    ) -> t.List["LightSource"]:
        """Return lights within radius of (x, y) using the spatial index.

        Uses the light spatial index if built; falls back to all lights
        otherwise. The index uses a conservative bounding box so results may
        extend slightly beyond the exact radius.

        Args:
            x (float): Query center x in meters.
            y (float): Query center y in meters.
            radius (float): Search radius in meters.

        Returns:
            list[LightSource]: Candidate lights within the search radius.
        """
        if self._light_index is None:
            return self.lights

        return self._light_index.query_radius(x, y, radius)

    def get_zones_in_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> t.Tuple[t.List[ResidentialZone], t.List[CommercialZone], t.List[DarkZone]]:
        """Return (residential, commercial, dark) zones overlapping the rectangle.

        Uses the zone spatial index if built; falls back to all zones
        otherwise. Deduplicates results since zones may span multiple index
        cells. Zones of all three kinds share one index since callers always
        query them together.

        Args:
            min_x (float): Rectangle minimum x in meters.
            min_y (float): Rectangle minimum y in meters.
            max_x (float): Rectangle maximum x in meters.
            max_y (float): Rectangle maximum y in meters.

        Returns:
            tuple[list[ResidentialZone], list[CommercialZone], list[DarkZone]]:
                Zones of each type that overlap the query rectangle.
        """
        if self._zone_index is None:
            return self.residential_zones, self.commercial_zones, self.dark_zones

        raw = self._zone_index.query_rect(min_x, min_y, max_x, max_y)
        residential = _deduplicate_by_identity(
            [z for z in raw if isinstance(z, ResidentialZone)]
        )
        commercial = _deduplicate_by_identity(
            [z for z in raw if isinstance(z, CommercialZone)]
        )
        dark = _deduplicate_by_identity(
            [z for z in raw if isinstance(z, DarkZone)]
        )
        return residential, commercial, dark

    def get_buildings_in_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> t.List[Building]:
        """Return buildings whose footprint AABB overlaps the given rectangle.

        Uses the building spatial index if built; falls back to all
        buildings otherwise. Deduplicates results since the spatial index
        may return the same building from multiple cells.

        Args:
            min_x (float): Rectangle minimum x in meters.
            min_y (float): Rectangle minimum y in meters.
            max_x (float): Rectangle maximum x in meters.
            max_y (float): Rectangle maximum y in meters.

        Returns:
            list[Building]: Candidate buildings overlapping the query area.
        """
        if self._building_index is None:
            return self.buildings

        raw = self._building_index.query_rect(min_x, min_y, max_x, max_y)
        return _deduplicate_by_identity(raw)

    def _index_road_segments(self) -> None:
        """Insert all road segments into the road spatial index."""
        for segment in self.road_segments:
            seg_min_x, seg_min_y, seg_max_x, seg_max_y = _segment_bounds(segment)
            self._road_index.insert(
                segment, seg_min_x, seg_min_y, seg_max_x, seg_max_y
            )

    def _index_zones(self) -> None:
        """Insert all zone boundaries into the zone spatial index."""
        all_zone_lists = [
            self.residential_zones,
            self.commercial_zones,
            self.dark_zones,
        ]
        for zone_list in all_zone_lists:
            for zone in zone_list:
                if hasattr(zone, "boundary") and zone.boundary is not None:
                    bounds = zone.boundary.bounds  # shapely (minx, miny, maxx, maxy)
                    self._zone_index.insert(
                        zone, bounds[0], bounds[1], bounds[2], bounds[3]
                    )

    def _index_buildings(self) -> None:
        """Insert all buildings into the building spatial index by footprint AABB."""
        for building in self.buildings:
            b_min_x, b_min_y, b_max_x, b_max_y = _building_bounds(building)
            self._building_index.insert(building, b_min_x, b_min_y, b_max_x, b_max_y)


def _compute_combined_bounds(
    segments: t.List[RoadSegment],
    residential_zones: t.List[ResidentialZone],
    commercial_zones: t.List[CommercialZone],
    dark_zones: t.List[DarkZone],
) -> t.Tuple[float, float, float, float]:
    """Compute world bounds from road segments and zone boundaries combined.

    Uses road segment geometry as the required baseline, then expands to
    include any zone boundaries that extend beyond the road network.

    Args:
        segments (list[RoadSegment]): All road segments (must be non-empty).
        residential_zones (list[ResidentialZone]): Residential zones.
        commercial_zones (list[CommercialZone]): Commercial zones.
        dark_zones (list[DarkZone]): Dark zones.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in meters.

    Raises:
        ValueError: If no segments have any geometry points.
    """
    min_x, min_y, max_x, max_y = _compute_world_bounds(segments)

    all_zone_lists: t.List[t.Any] = [
        *residential_zones,
        *commercial_zones,
        *dark_zones,
    ]
    for zone in all_zone_lists:
        if hasattr(zone, "boundary") and zone.boundary is not None:
            z_min_x, z_min_y, z_max_x, z_max_y = zone.boundary.bounds
            min_x = min(min_x, z_min_x)
            min_y = min(min_y, z_min_y)
            max_x = max(max_x, z_max_x)
            max_y = max(max_y, z_max_y)

    return min_x, min_y, max_x, max_y


def _compute_world_bounds(
    segments: t.List[RoadSegment],
) -> t.Tuple[float, float, float, float]:
    """Compute overall world bounds from all road segment geometries.

    Only uses segments with at least one point. Raises if none have geometry.

    Args:
        segments (list[RoadSegment]): All road segments.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in meters.

    Raises:
        ValueError: If no segments have any geometry points.
    """
    valid_geoms = [s.geometry for s in segments if s.geometry.shape[0] > 0]
    if not valid_geoms:
        raise ValueError("Cannot build spatial index: no road segments have geometry.")
    all_points = np.vstack(valid_geoms)
    min_x = float(all_points[:, 0].min())
    min_y = float(all_points[:, 1].min())
    max_x = float(all_points[:, 0].max())
    max_y = float(all_points[:, 1].max())
    return min_x, min_y, max_x, max_y


def _deduplicate_by_identity(items: t.List[t.Any]) -> t.List[t.Any]:
    """Return items deduplicated by object identity (id), preserving order.

    Works for objects that are not hashable (e.g. contain numpy arrays).

    Args:
        items (list): Possibly duplicated list of objects.

    Returns:
        list: Items with duplicates removed, preserving first-occurrence order.
    """
    seen_ids: t.Set[int] = set()
    result = []
    for item in items:
        item_id = id(item)
        if item_id not in seen_ids:
            seen_ids.add(item_id)
            result.append(item)
    return result


def _segment_bounds(
    segment: RoadSegment,
) -> t.Tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box for a single road segment.

    Args:
        segment (RoadSegment): The road segment.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in meters.
    """
    pts = segment.geometry
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def _building_bounds(building: Building) -> t.Tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box for a single building footprint.

    Args:
        building (Building): The building.

    Returns:
        tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) in meters.
    """
    pts = building.footprint
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )
