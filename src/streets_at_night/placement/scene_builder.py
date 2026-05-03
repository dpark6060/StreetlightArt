"""Scene construction orchestrator.

Builds a fully initialised Scene from classified road data and optional
land use / building GeoDataFrames. When land use and building data are
provided, zone_builder is invoked to populate zone lists on the Scene.
After the scene is built, populate_lights runs all placement engines to
add streetlights, cars, commercial lights, and house lights.
"""

import random
import typing as t

import geopandas as gpd
import numpy as np

from ..camera.camera import Camera
from ..camera.coordinates import CoordinateTransformer, convert_road_geometry
from ..config.schema import SceneConfig
from ..data.zone_builder import build_zones
from ..models.road_segment import RoadSegment, get_default_width
from ..models.road_types import RoadType
from .car_placer import place_cars
from .commercial_placer import place_commercial_lights
from .house_light_placer import collect_house_lights
from .streetlight_placer import place_streetlights
from ..scene.scene import Scene


def build_scene(
    roads_gdf: gpd.GeoDataFrame,
    transformer: CoordinateTransformer,
    camera: Camera,
    config: SceneConfig,
    land_use_gdf: t.Optional[gpd.GeoDataFrame] = None,
    buildings_gdf: t.Optional[gpd.GeoDataFrame] = None,
) -> Scene:
    """Build a Scene from a classified roads GeoDataFrame.

    Converts road geometry to ENU coordinates, creates RoadSegment objects,
    builds the spatial index, populates zone lists (if land use and building
    data are provided), and runs all light placers via populate_lights.

    Args:
        roads_gdf (GeoDataFrame): Classified road features with road_type column.
        transformer (CoordinateTransformer): Configured coordinate transformer.
        camera (Camera): The configured 3D camera.
        config (SceneConfig): Loaded scene configuration.
        land_use_gdf (GeoDataFrame, optional): Land use polygons from OSM.
            If omitted, zones are not built and the Scene has empty zone lists.
        buildings_gdf (GeoDataFrame, optional): Building footprints from OSM.
            If omitted, zones are not built and the Scene has empty zone lists.

    Returns:
        Scene: Fully constructed scene with spatial index, zones, and lights.

    Raises:
        AssertionError: If road type count and geometry count do not match.
    """
    geom_arrays = convert_road_geometry(transformer, roads_gdf)
    road_types = list(roads_gdf["road_type"])

    assert len(road_types) == len(geom_arrays), (
        f"Road type count ({len(road_types)}) != geometry count ({len(geom_arrays)})"
    )

    segments = _build_road_segments(geom_arrays, road_types)
    residential_zones, commercial_zones, dark_zones = _build_zones_if_available(
        land_use_gdf, buildings_gdf, transformer, config
    )

    scene = Scene(
        segments,
        camera,
        config,
        residential_zones=residential_zones,
        commercial_zones=commercial_zones,
        dark_zones=dark_zones,
        transformer=transformer,
    )
    scene.build_spatial_index()
    streetlights, car_lights, commercial_lights, house_lights = populate_lights(
        scene, config, random.Random(config.seed)
    )
    _print_light_summary(streetlights, car_lights, commercial_lights, house_lights)
    return scene


def populate_lights(
    scene: Scene, config: SceneConfig, rng: random.Random
) -> t.Tuple[t.List, t.List, t.List, t.List]:
    """Run all light placers and add results to the scene.

    Creates deterministic child RNGs per placer from the master seed, then
    runs each placer in turn. Results are added to the scene via add_lights
    and add_car_pairs.

    Args:
        scene (Scene): The scene to populate. Must have road_segments and
            zone lists set before calling.
        config (SceneConfig): Loaded scene configuration with lights section.
        rng (random.Random): Master RNG seeded from config.seed. Child RNGs
            are derived from this to keep each placer independent.

    Returns:
        Tuple[List, List, List, List]: (streetlights, car_lights,
            commercial_lights, house_lights) — all generated LightSources
            by type, already added to the scene.
    """
    streetlight_rng, car_rng, commercial_rng, house_rng = _create_child_rngs(rng)

    streetlights = _run_streetlight_placer(scene, config, streetlight_rng)
    car_lights, car_pairs = _run_car_placer(scene, config, car_rng)
    commercial_lights = _run_commercial_placer(scene, config, commercial_rng)
    house_lights = _run_house_light_placer(scene, config, house_rng)

    scene.add_lights(streetlights)
    scene.add_lights(car_lights)
    scene.add_lights(commercial_lights)
    scene.add_lights(house_lights)
    scene.add_car_pairs(car_pairs)

    return streetlights, car_lights, commercial_lights, house_lights


def _create_child_rngs(
    rng: random.Random,
) -> t.Tuple[random.Random, random.Random, random.Random, random.Random]:
    """Create four independent child RNGs from a master RNG.

    Each child is seeded from a unique integer drawn from the master, so
    placer order does not affect any other placer's random sequence.

    Args:
        rng (random.Random): Master RNG to draw seeds from.

    Returns:
        Tuple[Random, Random, Random, Random]:
            (streetlight_rng, car_rng, commercial_rng, house_rng)
    """
    return (
        random.Random(rng.randint(0, 2**32)),
        random.Random(rng.randint(0, 2**32)),
        random.Random(rng.randint(0, 2**32)),
        random.Random(rng.randint(0, 2**32)),
    )


def _run_streetlight_placer(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
) -> t.List:
    """Run the streetlight placer and return generated lights.

    Args:
        scene (Scene): Scene providing road segments.
        config (SceneConfig): Scene configuration with lights section.
        rng (random.Random): Seeded RNG for this placer.

    Returns:
        List[LightSource]: All streetlight LightSources.
    """
    return place_streetlights(scene.road_segments, config.lights, rng)


def _run_car_placer(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
) -> t.Tuple[t.List, t.List]:
    """Run the car placer and return lights and car pairs.

    Args:
        scene (Scene): Scene providing road segments.
        config (SceneConfig): Scene configuration with lights section.
        rng (random.Random): Seeded RNG for this placer.

    Returns:
        Tuple[List[LightSource], List[CarLightPair]]:
            Flat list of car lights and all CarLightPair objects.
    """
    return place_cars(scene.road_segments, config.lights, rng)


def _run_commercial_placer(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
) -> t.List:
    """Run the commercial light placer and return generated lights.

    Args:
        scene (Scene): Scene providing commercial zones.
        config (SceneConfig): Scene configuration with lights section.
        rng (random.Random): Seeded RNG for this placer.

    Returns:
        List[LightSource]: All commercial LightSources.
    """
    return place_commercial_lights(scene.commercial_zones, config.lights, rng)


def _run_house_light_placer(
    scene: Scene,
    config: SceneConfig,
    rng: random.Random,
) -> t.List:
    """Run the house light collector and return generated lights.

    Args:
        scene (Scene): Scene providing residential zones.
        config (SceneConfig): Scene configuration with lights section.
        rng (random.Random): Seeded RNG for this placer.

    Returns:
        List[LightSource]: All house LightSources.
    """
    return collect_house_lights(scene.residential_zones, config.lights, rng)


def _print_light_summary(
    streetlights: t.List,
    car_lights: t.List,
    commercial_lights: t.List,
    house_lights: t.List,
) -> None:
    """Print a breakdown of lights placed by type and a total.

    Args:
        streetlights (List): Streetlight LightSources.
        car_lights (List): Car LightSources.
        commercial_lights (List): Commercial LightSources.
        house_lights (List): House LightSources.
    """
    total = len(streetlights) + len(car_lights) + len(commercial_lights) + len(house_lights)
    print(f"Lights placed: {total} total")
    print(f"  Streetlights : {len(streetlights)}")
    print(f"  Car lights   : {len(car_lights)}")
    print(f"  Commercial   : {len(commercial_lights)}")
    print(f"  House lights : {len(house_lights)}")


def _build_zones_if_available(
    land_use_gdf: t.Optional[gpd.GeoDataFrame],
    buildings_gdf: t.Optional[gpd.GeoDataFrame],
    transformer: CoordinateTransformer,
    config: SceneConfig,
) -> t.Tuple[t.List, t.List, t.List]:
    """Build zone lists from land use and building data when both are provided.

    Returns empty lists for all zone types if either GeoDataFrame is absent,
    preserving backward compatibility for callers that only supply road data.

    Args:
        land_use_gdf (GeoDataFrame or None): Land use polygons from OSM.
        buildings_gdf (GeoDataFrame or None): Building footprints from OSM.
        transformer (CoordinateTransformer): WGS84 → ENU converter.
        config (SceneConfig): Loaded scene configuration.

    Returns:
        Tuple[List, List, List]: (residential_zones, commercial_zones, dark_zones).
            All three lists are empty if land use or building data is absent.
    """
    if land_use_gdf is None or buildings_gdf is None:
        return [], [], []

    rng = random.Random(config.seed)
    return build_zones(land_use_gdf, buildings_gdf, transformer, config.zones, rng)


def _build_road_segments(
    geom_arrays: t.List[np.ndarray],
    road_types: t.List[RoadType],
) -> t.List[RoadSegment]:
    """Create RoadSegment objects from geometry arrays and road types.

    Args:
        geom_arrays (list[np.ndarray]): ENU geometry arrays, one per segment.
        road_types (list[RoadType]): Road type for each segment.

    Returns:
        list[RoadSegment]: Constructed road segments.
    """
    segments = []
    for geom, road_type in zip(geom_arrays, road_types):
        width = get_default_width(road_type)
        segments.append(RoadSegment(geometry=geom, road_type=road_type, width=width))
    return segments
