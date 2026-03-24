"""TOML config loader — reads scene_config.toml and returns a SceneConfig."""

import copy
import tomllib
import typing as t
from pathlib import Path

from .defaults import DEFAULTS
from .schema import (
    BoundsConfig,
    CameraConfig,
    CarConfig,
    ColorPoolConfig,
    CommercialLightConfig,
    FixturePoolEntry,
    FixtureProfileConfig,
    HouseConfig,
    HouseLightConfig,
    LightConfig,
    QualityPreset,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    SceneConfig,
    StreetLightConfig,
    ZoneColorsConfig,
    ZoneConfig,
)


def load_config(toml_path: Path) -> SceneConfig:
    """Load a SceneConfig from a TOML file overlaid onto built-in defaults.

    If the file does not exist, the default values are used as-is.
    Quality preset selection is handled by the caller via
    config.quality_presets["low"] (or "medium" / "high").

    Args:
        toml_path (Path): Path to the TOML config file.

    Returns:
        SceneConfig: Populated config dataclass.
    """
    merged = _load_merged(toml_path)
    return _build_scene_config(merged)


def _load_merged(toml_path: Path) -> dict:
    """Merge user TOML values onto the built-in defaults.

    Args:
        toml_path (Path): Path to the TOML config file.

    Returns:
        dict: Merged configuration dictionary.
    """
    merged = copy.deepcopy(DEFAULTS)
    if not toml_path.exists():
        return merged
    with open(toml_path, "rb") as fh:
        user_data = tomllib.load(fh)
    _deep_merge(merged, user_data)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override values into base in-place.

    Args:
        base (dict): The base dictionary to mutate.
        override (dict): Values to overlay on top of base.
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _build_scene_config(data: dict) -> SceneConfig:
    """Construct a SceneConfig dataclass from a merged config dictionary.

    Args:
        data (dict): Merged configuration dictionary.

    Returns:
        SceneConfig: Fully populated config dataclass.
    """
    camera = _build_camera_config(data["camera"])
    bounds = _build_bounds_config(data["bounds"])
    quality_presets = _build_quality_presets(data["quality"])
    zones = _build_zone_config(data["zones"])
    lights = _build_light_config(data["lights"])
    roads = _build_roads_config(data["roads"])
    return SceneConfig(
        camera=camera,
        bounds=bounds,
        quality_presets=quality_presets,
        seed=data["seed"],
        zones=zones,
        lights=lights,
        roads=roads,
    )


def _build_camera_config(data: dict) -> CameraConfig:
    """Build a CameraConfig from a raw dict.

    Args:
        data (dict): Camera section from the config.

    Returns:
        CameraConfig: Populated camera config.
    """
    return CameraConfig(
        lat=data["lat"],
        lon=data["lon"],
        altitude_ft=data["altitude_ft"],
        target_lat=data["target_lat"],
        target_lon=data["target_lon"],
        target_altitude_ft=data["target_altitude_ft"],
        fov=data["fov"],
        near=data.get("near", 100.0),
        far=data.get("far", 60000.0),
    )


def _build_bounds_config(data: dict) -> BoundsConfig:
    """Build a BoundsConfig from a raw dict.

    Args:
        data (dict): Bounds section from the config.

    Returns:
        BoundsConfig: Populated bounds config.
    """
    return BoundsConfig(
        north=data["north"],
        south=data["south"],
        east=data["east"],
        west=data["west"],
    )


def _build_quality_presets(data: dict) -> t.Dict[str, QualityPreset]:
    """Build a dict of QualityPreset objects from the quality section.

    Args:
        data (dict): Quality section from the config, keyed by preset name.

    Returns:
        dict[str, QualityPreset]: Quality presets indexed by name.
    """
    return {
        name: QualityPreset(
            width=preset["width"],
            height=preset["height"],
            illumination_resolution_multiplier=preset.get(
                "illumination_resolution_multiplier", 1.0
            ),
            ground_tile_resolution=preset.get("ground_tile_resolution", 2.0),
        )
        for name, preset in data.items()
    }


def _build_zone_config(data: dict) -> ZoneConfig:
    """Build a ZoneConfig from the zones section of the config.

    Args:
        data (dict): Zones section from the config.

    Returns:
        ZoneConfig: Populated zone config.
    """
    house = _build_house_config(data["house"])
    colors = _build_zone_colors_config(data["colors"])
    return ZoneConfig(
        house=house,
        colors=colors,
        tile_size_meters=data["tile_size_meters"],
        tile_resolution_px=data["tile_resolution_px"],
    )


def _build_house_config(data: dict) -> HouseConfig:
    """Build a HouseConfig from the house section of the zones config.

    Args:
        data (dict): House section from the zones config.

    Returns:
        HouseConfig: Populated house config.
    """
    exterior_colors = _build_color_pool_config(data["exterior_colors"])
    return HouseConfig(
        exterior_colors=exterior_colors,
        yard_base_color=data["yard_base_color"],
        yard_variation=data["yard_variation"],
        min_lights=data["min_lights"],
        max_lights=data["max_lights"],
        light_weights=list(data["light_weights"]),
        footprint_coverage=data["footprint_coverage"],
        lot_width=data["lot_width"],
        lot_depth=data["lot_depth"],
        setback=data["setback"],
    )


def _build_zone_colors_config(data: dict) -> ZoneColorsConfig:
    """Build a ZoneColorsConfig from the colors section of the zones config.

    Args:
        data (dict): Colors section from the zones config.

    Returns:
        ZoneColorsConfig: Populated zone colors config.
    """
    return ZoneColorsConfig(
        commercial=data["commercial"],
        industrial=data["industrial"],
        park_forest=data["park_forest"],
        water=data["water"],
        road_surface=data["road_surface"],
        default_ground=data["default_ground"],
    )


def _build_color_pool_config(data: dict) -> ColorPoolConfig:
    """Build a ColorPoolConfig from a color pool section of the config.

    Args:
        data (dict): Color pool section containing a list of (hex, weight) pairs.

    Returns:
        ColorPoolConfig: Populated color pool config.
    """
    colors = [tuple(entry) for entry in data["colors"]]
    return ColorPoolConfig(colors=colors)


def _build_light_config(data: dict) -> LightConfig:
    """Build a LightConfig from the lights section of the config.

    Args:
        data (dict): Lights section from the config.

    Returns:
        LightConfig: Fully populated light config.
    """
    streetlights = _build_streetlight_config(data["streetlights"])
    cars = _build_car_config(data["cars"])
    commercial = _build_commercial_light_config(data["commercial"])
    house_lights = _build_house_light_config(data["house_lights"])
    return LightConfig(
        streetlights=streetlights,
        cars=cars,
        commercial=commercial,
        house_lights=house_lights,
    )


def _build_streetlight_config(data: dict) -> StreetLightConfig:
    """Build a StreetLightConfig from the streetlights section of the config.

    Args:
        data (dict): Streetlights section containing per-road-type fixture pools.

    Returns:
        StreetLightConfig: Populated streetlight config.
    """
    return StreetLightConfig(
        highway=_build_fixture_pool_entries(data["highway"]),
        major_road=_build_fixture_pool_entries(data["major_road"]),
        collector=_build_fixture_pool_entries(data["collector"]),
        residential=_build_fixture_pool_entries(data["residential"]),
        service=_build_fixture_pool_entries(data["service"]),
    )


def _build_fixture_pool_entries(data: t.List[dict]) -> t.List[FixturePoolEntry]:
    """Build a list of FixturePoolEntry objects from a fixture pool list.

    Args:
        data (list[dict]): List of pool entry dicts with weight and optional profile.

    Returns:
        list[FixturePoolEntry]: Populated fixture pool entries.
    """
    return [
        FixturePoolEntry(
            weight=entry["weight"],
            profile=_build_fixture_profile_config(entry["profile"]),
        )
        for entry in data
    ]


def _build_fixture_profile_config(
    data: t.Optional[dict],
) -> t.Optional[FixtureProfileConfig]:
    """Build a FixtureProfileConfig from a profile dict, or return None.

    Args:
        data (dict or None): Profile section, or None if this pool entry has no lights.

    Returns:
        FixtureProfileConfig or None: Populated profile, or None for no-lights entries.
    """
    if data is None:
        return None
    return FixtureProfileConfig(
        color=data["color"],
        height_min=data["height_min"],
        height_max=data["height_max"],
        brightness_min=data["brightness_min"],
        brightness_max=data["brightness_max"],
        glow_radius=data["glow_radius"],
        illumination_radius=data["illumination_radius"],
        illumination_intensity=data["illumination_intensity"],
        spacing_min=data["spacing_min"],
        spacing_max=data["spacing_max"],
        side=data["side"],
    )


def _build_car_config(data: dict) -> CarConfig:
    """Build a CarConfig from the cars section of the config.

    Args:
        data (dict): Cars section from the config.

    Returns:
        CarConfig: Populated car config.
    """
    return CarConfig(
        headlight_colors=_build_color_pool_config(data["headlight_colors"]),
        taillight_colors=_build_color_pool_config(data["taillight_colors"]),
        separation_min=data["separation_min"],
        separation_max=data["separation_max"],
        density_highway_min=data["density_highway_min"],
        density_highway_max=data["density_highway_max"],
        density_major_min=data["density_major_min"],
        density_major_max=data["density_major_max"],
        density_collector_min=data["density_collector_min"],
        density_collector_max=data["density_collector_max"],
        density_residential_min=data["density_residential_min"],
        density_residential_max=data["density_residential_max"],
        density_service_min=data["density_service_min"],
        density_service_max=data["density_service_max"],
    )


def _build_commercial_light_config(data: dict) -> CommercialLightConfig:
    """Build a CommercialLightConfig from the commercial section of the config.

    Args:
        data (dict): Commercial section from the config.

    Returns:
        CommercialLightConfig: Populated commercial light config.
    """
    return CommercialLightConfig(
        sign_colors=_build_color_pool_config(data["sign_colors"]),
        sign_brightness_min=data["sign_brightness_min"],
        sign_brightness_max=data["sign_brightness_max"],
        sign_glow_radius=data["sign_glow_radius"],
        sign_illumination_radius=data["sign_illumination_radius"],
        parking_colors=_build_color_pool_config(data["parking_colors"]),
        parking_brightness_min=data["parking_brightness_min"],
        parking_brightness_max=data["parking_brightness_max"],
        parking_glow_radius=data["parking_glow_radius"],
        parking_illumination_radius=data["parking_illumination_radius"],
        parking_grid_spacing_min=data["parking_grid_spacing_min"],
        parking_grid_spacing_max=data["parking_grid_spacing_max"],
        storefront_colors=_build_color_pool_config(data["storefront_colors"]),
        storefront_brightness_min=data["storefront_brightness_min"],
        storefront_brightness_max=data["storefront_brightness_max"],
        storefront_glow_radius=data["storefront_glow_radius"],
        storefront_illumination_radius=data["storefront_illumination_radius"],
        storefront_spacing_min=data["storefront_spacing_min"],
        storefront_spacing_max=data["storefront_spacing_max"],
    )


def _build_house_light_config(data: dict) -> HouseLightConfig:
    """Build a HouseLightConfig from the house_lights section of the config.

    Args:
        data (dict): House lights section from the config.

    Returns:
        HouseLightConfig: Populated house light config.
    """
    return HouseLightConfig(
        colors=_build_color_pool_config(data["colors"]),
        brightness_min=data["brightness_min"],
        brightness_max=data["brightness_max"],
        glow_radius=data["glow_radius"],
        illumination_radius=data["illumination_radius"],
        illumination_intensity=data["illumination_intensity"],
    )


def _build_roads_config(data: dict) -> RoadsConfig:
    """Build a RoadsConfig from the roads section of the config.

    Args:
        data (dict): Roads section with export and preview sub-sections.

    Returns:
        RoadsConfig: Populated roads config.
    """
    return RoadsConfig(
        export=_build_road_style_config(data["export"]),
        preview=_build_road_style_config(data["preview"]),
    )


def _build_road_style_config(data: dict) -> RoadStyleConfig:
    """Build a RoadStyleConfig from a style sub-section (export or preview).

    Args:
        data (dict): Style section keyed by road type name.

    Returns:
        RoadStyleConfig: Populated road style config.
    """
    return RoadStyleConfig(
        highway=_build_road_style_entry(data["highway"]),
        major_road=_build_road_style_entry(data["major_road"]),
        collector=_build_road_style_entry(data["collector"]),
        residential=_build_road_style_entry(data["residential"]),
        service=_build_road_style_entry(data["service"]),
        minor=_build_road_style_entry(data["minor"]),
    )


def _build_road_style_entry(data: dict) -> RoadStyleEntry:
    """Build a RoadStyleEntry from a single road type entry.

    Args:
        data (dict): Entry with 'color' (hex string) and 'width' (int).

    Returns:
        RoadStyleEntry: Populated entry.
    """
    return RoadStyleEntry(color=data["color"], width=int(data["width"]))
