"""Configuration dataclass definitions for Streets at Night."""

import typing as t
from dataclasses import dataclass


@dataclass
class CameraConfig:
    """Camera position and orientation parameters."""

    lat: float
    lon: float
    altitude_ft: float
    target_lat: float
    target_lon: float
    target_altitude_ft: float
    fov: float
    near: float = 100.0
    far: float = 60000.0


@dataclass
class QualityPreset:
    """Output resolution and quality multipliers for a named quality level."""

    width: int
    height: int
    illumination_resolution_multiplier: float = 1.0
    ground_tile_resolution: float = 2.0


@dataclass
class BoundsConfig:
    """Geographic bounding box for OSM data download."""

    north: float
    south: float
    east: float
    west: float


@dataclass
class ColorPoolConfig:
    """Configuration for a weighted color pool."""

    colors: t.List[t.Tuple[str, float]]


@dataclass
class HouseConfig:
    """Configuration for house generation within residential zones."""

    exterior_colors: ColorPoolConfig
    yard_base_color: str
    yard_variation: int
    min_lights: int
    max_lights: int
    light_weights: t.List[float]
    footprint_coverage: float
    lot_width: float
    lot_depth: float
    setback: float


@dataclass
class ZoneColorsConfig:
    """Base colors for zone types representing very dark night ambient."""

    commercial: str
    industrial: str
    park_forest: str
    water: str
    road_surface: str
    default_ground: str


@dataclass
class ZoneConfig:
    """Configuration for zone generation and ground plane rendering."""

    house: HouseConfig
    colors: ZoneColorsConfig
    tile_size_meters: float
    tile_resolution_px: int


@dataclass
class FixtureProfileConfig:
    """Configuration for a single streetlight fixture profile."""

    color: str
    height_min: float
    height_max: float
    brightness_min: float
    brightness_max: float
    glow_radius: float
    illumination_radius: float
    illumination_intensity: float
    spacing_min: float
    spacing_max: float
    side: str


@dataclass
class FixturePoolEntry:
    """A weighted entry in a fixture pool."""

    weight: float
    profile: t.Optional[FixtureProfileConfig]


@dataclass
class StreetLightConfig:
    """Streetlight configuration per road type, as weighted fixture pools."""

    highway: t.List[FixturePoolEntry]
    major_road: t.List[FixturePoolEntry]
    collector: t.List[FixturePoolEntry]
    residential: t.List[FixturePoolEntry]
    service: t.List[FixturePoolEntry]


@dataclass
class CarConfig:
    """Car light configuration including colors and per-road-type densities."""

    headlight_colors: ColorPoolConfig
    taillight_colors: ColorPoolConfig
    separation_min: float
    separation_max: float
    density_highway_min: float
    density_highway_max: float
    density_major_min: float
    density_major_max: float
    density_collector_min: float
    density_collector_max: float
    density_residential_min: float
    density_residential_max: float
    density_service_min: float
    density_service_max: float
    headlight_glow_radius: float
    headlight_illumination_radius: float
    headlight_illumination_intensity: float
    taillight_glow_radius: float
    taillight_illumination_radius: float
    taillight_illumination_intensity: float


@dataclass
class CommercialLightConfig:
    """Commercial zone light configuration for signs, parking, and storefronts."""

    sign_colors: ColorPoolConfig
    sign_brightness_min: float
    sign_brightness_max: float
    sign_glow_radius: float
    sign_illumination_radius: float
    sign_illumination_intensity: float
    parking_colors: ColorPoolConfig
    parking_brightness_min: float
    parking_brightness_max: float
    parking_glow_radius: float
    parking_illumination_radius: float
    parking_illumination_intensity: float
    parking_grid_spacing_min: float
    parking_grid_spacing_max: float
    storefront_colors: ColorPoolConfig
    storefront_brightness_min: float
    storefront_brightness_max: float
    storefront_glow_radius: float
    storefront_illumination_radius: float
    storefront_illumination_intensity: float
    storefront_spacing_min: float
    storefront_spacing_max: float


@dataclass
class HouseLightConfig:
    """House light configuration for residential window, porch, and interior lights."""

    colors: ColorPoolConfig
    brightness_min: float
    brightness_max: float
    glow_radius: float
    illumination_radius: float
    illumination_intensity: float


@dataclass
class LightConfig:
    """Top-level light configuration for all light types in the scene."""

    streetlights: StreetLightConfig
    cars: CarConfig
    commercial: CommercialLightConfig
    house_lights: HouseLightConfig


@dataclass
class RoadStyleEntry:
    """Color and base line width for a single road type."""

    color: str
    width: int


@dataclass
class RoadStyleConfig:
    """Road rendering styles for export and preview, per road type."""

    highway: RoadStyleEntry
    major_road: RoadStyleEntry
    collector: RoadStyleEntry
    residential: RoadStyleEntry
    service: RoadStyleEntry
    minor: RoadStyleEntry


@dataclass
class RoadsConfig:
    """Road rendering configuration for export and preview paths."""

    export: RoadStyleConfig
    preview: RoadStyleConfig


@dataclass
class AtmosphereConfig:
    """Atmospheric perspective parameters for distance-based light rendering."""

    falloff_distance: float
    haze_color: t.Tuple[int, int, int]
    scatter_distance: float
    max_distance: float
    glow_expansion_enabled: bool = False
    glow_max_alpha: int = 80
    glow_falloff_steepness: float = 1.0
    min_brightness_threshold: float = 0.05
    illumination_brightness: int = 60
    illumination_min_radius: float = 15.0


@dataclass
class DevmodeCameraConfig:
    """ENU-space camera configuration for the devmode synthetic scene."""

    pos_x: float
    pos_y: float
    pos_z: float
    target_x: float
    target_y: float
    target_z: float
    fov: float
    near: float
    far: float


@dataclass
class DevmodeConfig:
    """Configuration for the devmode parameter tuning tool."""

    camera: DevmodeCameraConfig
    render_width: int
    render_height: int


@dataclass
class SceneConfig:
    """Top-level scene configuration.

    Holds all parameters needed to load data, build the scene, and
    produce a verification plot.
    """

    camera: CameraConfig
    bounds: BoundsConfig
    quality_presets: t.Dict[str, QualityPreset]
    seed: int
    zones: ZoneConfig
    lights: LightConfig
    roads: RoadsConfig
    atmosphere: AtmosphereConfig
    devmode: t.Optional["DevmodeConfig"] = None
