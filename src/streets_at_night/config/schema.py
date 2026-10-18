"""Configuration dataclass definitions for Streets at Night."""

import typing as t
from dataclasses import dataclass, field


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
    """Configuration for a single streetlight fixture profile.

    Color is authored as a blackbody temperature plus a saturation knob
    (see utils.colors.kelvin_to_rgb). The ground pool is derived physically
    from pole height and ``beam_angle_deg`` in the illumination pass, so the
    legacy cone shape parameters are not part of a streetlight profile.
    ``bulb_fraction`` and ``reflector_falloff`` split the pool into a direct
    bulb zone and a fading reflector zone (see ``BeamIllumination``); their
    defaults give the single-cutoff pool, so older configs load unchanged.
    """

    color_temp_k: float
    height_min: float
    height_max: float
    brightness_min: float
    brightness_max: float
    glow_radius: float
    beam_angle_deg: float
    illumination_intensity: float
    spacing_min: float
    spacing_max: float
    side: str
    saturation: float = 1.0
    bulb_fraction: float = 1.0
    reflector_falloff: float = 1.0


@dataclass
class FixturePoolEntry:
    """A weighted entry in a fixture pool."""

    weight: float
    profile: t.Optional[FixtureProfileConfig] = None


@dataclass
class StreetLightConfig:
    """Streetlight configuration per road type, as weighted fixture pools.

    ``color_temp_shift_k`` is a city-wide master lever: an additive Kelvin
    offset applied to every streetlight's authored temperature, so the whole
    scene can be shifted from old sodium toward LED retrofit in one edit.
    """

    highway: t.List[FixturePoolEntry]
    major_road: t.List[FixturePoolEntry]
    collector: t.List[FixturePoolEntry]
    residential: t.List[FixturePoolEntry]
    service: t.List[FixturePoolEntry]
    color_temp_shift_k: float = 0.0


@dataclass
class CarConfig:
    """Car light configuration including colors and per-road-type densities.

    Each car casts a single directional headlight cone along its travel
    bearing: ``headlight_beam_length`` sets the cone reach in meters and
    ``headlight_beam_angle_deg`` sets the full horizontal cone angle.
    """

    headlight_colors: ColorPoolConfig
    taillight_colors: ColorPoolConfig
    separation_min: float
    separation_max: float
    headlight_brightness_min: float
    headlight_brightness_max: float
    taillight_brightness_min: float
    taillight_brightness_max: float
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
    headlight_illumination_intensity: float
    headlight_illumination_falloff: float
    headlight_illumination_inner_fraction: float
    taillight_glow_radius: float
    taillight_illumination_radius: float
    taillight_illumination_intensity: float
    taillight_illumination_falloff: float
    taillight_illumination_inner_fraction: float
    headlight_beam_length: float = 25.0
    headlight_beam_angle_deg: float = 40.0


@dataclass
class CommercialLightConfig:
    """Commercial zone light configuration for signs, parking, and storefronts.

    Signs and storefronts use an authored cone ground pool. Parking lot lights
    are pole-mounted, so their pool is physical like a streetlight's: derived
    from the sampled pole height (``parking_height_min``/``max``) and the
    ``parking_beam_angle_deg`` / ``parking_bulb_fraction`` /
    ``parking_reflector_falloff`` fixture shape.
    """

    sign_colors: ColorPoolConfig
    sign_brightness_min: float
    sign_brightness_max: float
    sign_glow_radius: float
    sign_illumination_radius: float
    sign_illumination_intensity: float
    sign_illumination_falloff: float
    sign_illumination_inner_fraction: float
    parking_colors: ColorPoolConfig
    parking_brightness_min: float
    parking_brightness_max: float
    parking_glow_radius: float
    parking_height_min: float
    parking_height_max: float
    parking_beam_angle_deg: float
    parking_bulb_fraction: float
    parking_reflector_falloff: float
    parking_illumination_intensity: float
    parking_grid_spacing_min: float
    parking_grid_spacing_max: float
    storefront_colors: ColorPoolConfig
    storefront_brightness_min: float
    storefront_brightness_max: float
    storefront_glow_radius: float
    storefront_illumination_radius: float
    storefront_illumination_intensity: float
    storefront_illumination_falloff: float
    storefront_illumination_inner_fraction: float
    storefront_spacing_min: float
    storefront_spacing_max: float


@dataclass
class IlluminationConfig:
    """Ground illumination-map post-processing parameters.

    The soft knee compresses stacked (overlapping) light pools: values up to
    ``knee_start`` pass through unchanged; above it they compress smoothly
    and asymptotically approach ``knee_max``, so overlapping pools still read
    brighter than a single pool but can never blow out. Setting
    ``knee_max <= knee_start`` disables the knee.
    """

    knee_start: float = 1.5
    knee_max: float = 4.0


@dataclass
class LightConfig:
    """Top-level light configuration for all light types in the scene."""

    streetlights: StreetLightConfig
    cars: CarConfig
    commercial: CommercialLightConfig
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)


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
    haze_color: t.Tuple[int, int, int] = field(metadata={"serde": "hex_color"})
    scatter_distance: float
    glow_expansion_enabled: bool = False
    glow_max_alpha: int = 80
    glow_falloff_steepness: float = 1.0
    glow_wing_exponent: float = 2.5
    glow_tail_strength: float = 0.15
    glow_cutoff_mult: float = 8.0
    min_brightness_threshold: float = 0.05
    illumination_brightness: int = 60
    illumination_min_radius: float = 15.0
    horizon_glow_color: str = "#28180A"
    horizon_glow_width_px: int = 120
    horizon_glow_peak_alpha: int = 90
    haze_enabled: bool = True
    haze_alpha: int = 30
    vignette_strength: float = 0.35
    vignette_radius: float = 0.7


@dataclass
class GradeConfig:
    """Final color-grade parameters for the linear-light pipeline.

    Attributes:
        exposure: Exposure adjustment in stops (EV), applied as a linear
            multiplier of 2**exposure before tone mapping. 0.0 is neutral.
        tonemap: Tone-mapping operator name. "aces" applies the Narkowicz ACES
            filmic curve; any other value falls back to a linear clamp.
    """

    exposure: float = 0.0
    tonemap: str = "aces"


@dataclass
class StarsConfig:
    """Configuration for optional star rendering in the sky region."""

    stars_enabled: bool
    count: int
    brightness_min: float
    brightness_max: float
    color_variation: int
    radius: float = 1.0


@dataclass
class HeatHazeLayerConfig:
    """One atmospheric turbulence layer in the heat haze stack.

    The eddy size in pixels of a layer is approximately
    ``L0 / pixel_scale_m``. Keep this well below the image width to avoid
    the whole frame bulging as one big eddy.
    """

    L0: float
    pixel_scale_m: float
    weight: float
    wind_x_pxs: float
    wind_y_pxs: float


@dataclass
class HeatHazeConfig:
    """Configuration for the multi-layer heat haze atmospheric distortion."""

    enabled: bool
    amplitude_px: float
    max_distance_m: float
    distance_exponent: float
    sky_distance_factor: float
    seed: int
    screen_width_px: int
    screen_height_px: int
    layers: t.List[HeatHazeLayerConfig]


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
class BuildingConfig:
    """Configuration for building mass rendering and lit windows.

    Phase A (data + model) only consumes ``enabled``, ``floor_height_m``,
    ``height_jitter``, and ``height_defaults`` — the remaining fields exist
    now so the full ``[buildings]`` TOML section round-trips, and are
    consumed by the Phase B (mass render) and Phase C (windows) render
    passes.
    """

    enabled: bool
    floor_height_m: float
    height_jitter: float
    building_max_distance_m: float
    window_max_distance_m: float
    wall_color: str
    roof_color: str
    face_shade_strength: float
    window_lit_fraction: float
    window_lit_fraction_residential: float
    window_width_m: float
    window_height_m: float
    window_pitch_m: float
    window_side_margin_m: float
    window_sill_margin_m: float
    window_roof_margin_m: float
    window_color_temp_min_k: float
    window_color_temp_max_k: float
    window_saturation: float
    window_brightness_min: float
    window_brightness_max: float
    window_glow_px: float
    window_extinction_strength: float
    height_defaults: t.Dict[str, float]


@dataclass
class SceneConfig:
    """Top-level scene configuration.

    Holds all parameters needed to load data, build the scene, and
    produce a verification plot.
    """

    camera: CameraConfig
    bounds: BoundsConfig
    quality_presets: t.Dict[str, QualityPreset] = field(metadata={"toml_key": "quality"})
    seed: int
    zones: ZoneConfig
    lights: LightConfig
    roads: RoadsConfig
    atmosphere: AtmosphereConfig
    grade: GradeConfig = field(default_factory=GradeConfig)
    stars: t.Optional[StarsConfig] = None
    devmode: t.Optional["DevmodeConfig"] = None
    heat_haze: t.Optional["HeatHazeConfig"] = None
    buildings: t.Optional["BuildingConfig"] = None
    region_name: t.Optional[str] = None
    # Shared look file this config layers over, relative to the config file
    # (see config.layers). None for a self-contained config.
    look: t.Optional[str] = None
