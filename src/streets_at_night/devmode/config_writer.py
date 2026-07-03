"""Serializes SceneConfig back to a TOML-compatible dict and writes to file."""

import typing as t
from pathlib import Path

import tomli_w

from ..config.schema import (
    AtmosphereConfig,
    BoundsConfig,
    CameraConfig,
    CarConfig,
    ColorPoolConfig,
    CommercialLightConfig,
    DevmodeConfig,
    FixturePoolEntry,
    FixtureProfileConfig,
    GradeConfig,
    HeatHazeConfig,
    HeatHazeLayerConfig,
    HouseConfig,
    HouseLightConfig,
    LightConfig,
    QualityPreset,
    RoadStyleConfig,
    RoadStyleEntry,
    RoadsConfig,
    SceneConfig,
    StarsConfig,
    StreetLightConfig,
    ZoneColorsConfig,
    ZoneConfig,
)


def write_config(config: SceneConfig, path: Path) -> None:
    """Serialize a SceneConfig to a TOML file.

    Args:
        config (SceneConfig): The config to serialize.
        path (Path): Destination file path. Will be created or overwritten.
    """
    data = config_to_dict(config)
    with open(path, "wb") as fh:
        tomli_w.dump(data, fh)


def config_to_dict(config: SceneConfig) -> dict:
    """Convert a SceneConfig to a nested TOML-compatible dict.

    Key structure and naming exactly matches what loader.py expects on
    re-load, so that write_config(config, path); load_config(path) round-trips.

    Args:
        config (SceneConfig): The config to convert.

    Returns:
        dict: Nested dict ready for TOML serialization.
    """
    data: dict = {
        "seed": config.seed,
        "camera": _camera_to_dict(config.camera),
        "bounds": _bounds_to_dict(config.bounds),
        "quality": _quality_presets_to_dict(config.quality_presets),
        "zones": _zone_to_dict(config.zones),
        "lights": _lights_to_dict(config.lights),
        "roads": _roads_to_dict(config.roads),
        "atmosphere": _atmosphere_to_dict(config.atmosphere),
        "grade": _grade_to_dict(config.grade),
    }
    if config.stars is not None:
        data["stars"] = _stars_to_dict(config.stars)
    if config.heat_haze is not None:
        data["heat_haze"] = _heat_haze_to_dict(config.heat_haze)
    if config.devmode is not None:
        data["devmode"] = _devmode_to_dict(config.devmode)
    if config.region_name:
        data["region_name"] = config.region_name
    return data


# ---------------------------------------------------------------------------
# Section serializers
# ---------------------------------------------------------------------------


def _camera_to_dict(camera: CameraConfig) -> dict:
    """Serialize a CameraConfig to a dict.

    Args:
        camera (CameraConfig): Camera configuration.

    Returns:
        dict: Serialized camera section.
    """
    return {
        "lat": camera.lat,
        "lon": camera.lon,
        "altitude_ft": camera.altitude_ft,
        "target_lat": camera.target_lat,
        "target_lon": camera.target_lon,
        "target_altitude_ft": camera.target_altitude_ft,
        "fov": camera.fov,
        "near": camera.near,
        "far": camera.far,
    }


def _bounds_to_dict(bounds: BoundsConfig) -> dict:
    """Serialize a BoundsConfig to a dict.

    Args:
        bounds (BoundsConfig): Geographic bounding box.

    Returns:
        dict: Serialized bounds section.
    """
    return {
        "north": bounds.north,
        "south": bounds.south,
        "east": bounds.east,
        "west": bounds.west,
    }


def _quality_presets_to_dict(presets: t.Dict[str, QualityPreset]) -> dict:
    """Serialize quality presets to a dict keyed by preset name.

    Args:
        presets (dict[str, QualityPreset]): Quality presets.

    Returns:
        dict: Serialized quality section.
    """
    return {
        name: {
            "width": preset.width,
            "height": preset.height,
            "illumination_resolution_multiplier": preset.illumination_resolution_multiplier,
            "ground_tile_resolution": preset.ground_tile_resolution,
        }
        for name, preset in presets.items()
    }


def _zone_to_dict(zone: ZoneConfig) -> dict:
    """Serialize a ZoneConfig to a dict.

    Args:
        zone (ZoneConfig): Zone configuration.

    Returns:
        dict: Serialized zones section.
    """
    return {
        "house": _house_to_dict(zone.house),
        "colors": _zone_colors_to_dict(zone.colors),
        "tile_size_meters": zone.tile_size_meters,
        "tile_resolution_px": zone.tile_resolution_px,
    }


def _house_to_dict(house: HouseConfig) -> dict:
    """Serialize a HouseConfig to a dict.

    Args:
        house (HouseConfig): House configuration.

    Returns:
        dict: Serialized house section.
    """
    return {
        "exterior_colors": _color_pool_to_dict(house.exterior_colors),
        "yard_base_color": house.yard_base_color,
        "yard_variation": house.yard_variation,
        "min_lights": house.min_lights,
        "max_lights": house.max_lights,
        "light_weights": list(house.light_weights),
        "footprint_coverage": house.footprint_coverage,
        "lot_width": house.lot_width,
        "lot_depth": house.lot_depth,
        "setback": house.setback,
    }


def _zone_colors_to_dict(colors: ZoneColorsConfig) -> dict:
    """Serialize a ZoneColorsConfig to a dict.

    Args:
        colors (ZoneColorsConfig): Zone color configuration.

    Returns:
        dict: Serialized zone colors section.
    """
    return {
        "commercial": colors.commercial,
        "industrial": colors.industrial,
        "park_forest": colors.park_forest,
        "water": colors.water,
        "road_surface": colors.road_surface,
        "default_ground": colors.default_ground,
    }


def _color_pool_to_dict(pool: ColorPoolConfig) -> dict:
    """Serialize a ColorPoolConfig to its TOML representation.

    The loader reads colors as a list of [hex_str, weight] pairs.

    Args:
        pool (ColorPoolConfig): Color pool configuration.

    Returns:
        dict: Serialized color pool section with a 'colors' list.
    """
    return {"colors": [[color, weight] for color, weight in pool.colors]}


def _lights_to_dict(lights: LightConfig) -> dict:
    """Serialize a LightConfig to a dict.

    Args:
        lights (LightConfig): Top-level light configuration.

    Returns:
        dict: Serialized lights section.
    """
    return {
        "streetlights": _streetlights_to_dict(lights.streetlights),
        "cars": _cars_to_dict(lights.cars),
        "commercial": _commercial_lights_to_dict(lights.commercial),
        "house_lights": _house_lights_to_dict(lights.house_lights),
    }


def _streetlights_to_dict(streetlights: StreetLightConfig) -> dict:
    """Serialize a StreetLightConfig to a dict of fixture pool lists.

    Args:
        streetlights (StreetLightConfig): Streetlight configuration.

    Returns:
        dict: Serialized streetlights section.
    """
    return {
        "color_temp_shift_k": streetlights.color_temp_shift_k,
        "highway": _fixture_pool_entries_to_list(streetlights.highway),
        "major_road": _fixture_pool_entries_to_list(streetlights.major_road),
        "collector": _fixture_pool_entries_to_list(streetlights.collector),
        "residential": _fixture_pool_entries_to_list(streetlights.residential),
        "service": _fixture_pool_entries_to_list(streetlights.service),
    }


def _fixture_pool_entries_to_list(entries: t.List[FixturePoolEntry]) -> t.List[dict]:
    """Serialize fixture pool entries to a list of dicts.

    Entries with profile=None produce a dict with only 'weight', matching
    the loader's use of entry.get("profile") for nullable profiles.

    Args:
        entries (list[FixturePoolEntry]): Fixture pool entries.

    Returns:
        list[dict]: Serialized pool entries.
    """
    return [_fixture_pool_entry_to_dict(entry) for entry in entries]


def _fixture_pool_entry_to_dict(entry: FixturePoolEntry) -> dict:
    """Serialize a single FixturePoolEntry to a dict.

    Args:
        entry (FixturePoolEntry): A single pool entry.

    Returns:
        dict: Serialized entry. No 'profile' key if profile is None.
    """
    result: dict = {"weight": entry.weight}
    if entry.profile is not None:
        result["profile"] = _fixture_profile_to_dict(entry.profile)
    return result


def _fixture_profile_to_dict(profile: FixtureProfileConfig) -> dict:
    """Serialize a FixtureProfileConfig to a dict.

    Args:
        profile (FixtureProfileConfig): Fixture profile configuration.

    Returns:
        dict: Serialized fixture profile section.
    """
    return {
        "color_temp_k": profile.color_temp_k,
        "saturation": profile.saturation,
        "height_min": profile.height_min,
        "height_max": profile.height_max,
        "brightness_min": profile.brightness_min,
        "brightness_max": profile.brightness_max,
        "glow_radius": profile.glow_radius,
        "beam_angle_deg": profile.beam_angle_deg,
        "illumination_intensity": profile.illumination_intensity,
        "spacing_min": profile.spacing_min,
        "spacing_max": profile.spacing_max,
        "side": profile.side,
    }


def _cars_to_dict(cars: CarConfig) -> dict:
    """Serialize a CarConfig to a dict.

    Args:
        cars (CarConfig): Car light configuration.

    Returns:
        dict: Serialized cars section.
    """
    return {
        "headlight_colors": _color_pool_to_dict(cars.headlight_colors),
        "taillight_colors": _color_pool_to_dict(cars.taillight_colors),
        "separation_min": cars.separation_min,
        "separation_max": cars.separation_max,
        "headlight_brightness_min": cars.headlight_brightness_min,
        "headlight_brightness_max": cars.headlight_brightness_max,
        "taillight_brightness_min": cars.taillight_brightness_min,
        "taillight_brightness_max": cars.taillight_brightness_max,
        "density_highway_min": cars.density_highway_min,
        "density_highway_max": cars.density_highway_max,
        "density_major_min": cars.density_major_min,
        "density_major_max": cars.density_major_max,
        "density_collector_min": cars.density_collector_min,
        "density_collector_max": cars.density_collector_max,
        "density_residential_min": cars.density_residential_min,
        "density_residential_max": cars.density_residential_max,
        "density_service_min": cars.density_service_min,
        "density_service_max": cars.density_service_max,
        "headlight_glow_radius": cars.headlight_glow_radius,
        "headlight_illumination_radius": cars.headlight_illumination_radius,
        "headlight_illumination_intensity": cars.headlight_illumination_intensity,
        "headlight_illumination_falloff": cars.headlight_illumination_falloff,
        "headlight_illumination_inner_fraction": cars.headlight_illumination_inner_fraction,
        "taillight_glow_radius": cars.taillight_glow_radius,
        "taillight_illumination_radius": cars.taillight_illumination_radius,
        "taillight_illumination_intensity": cars.taillight_illumination_intensity,
        "taillight_illumination_falloff": cars.taillight_illumination_falloff,
        "taillight_illumination_inner_fraction": cars.taillight_illumination_inner_fraction,
    }


def _commercial_lights_to_dict(commercial: CommercialLightConfig) -> dict:
    """Serialize a CommercialLightConfig to a dict.

    Args:
        commercial (CommercialLightConfig): Commercial light configuration.

    Returns:
        dict: Serialized commercial section.
    """
    return {
        "sign_colors": _color_pool_to_dict(commercial.sign_colors),
        "sign_brightness_min": commercial.sign_brightness_min,
        "sign_brightness_max": commercial.sign_brightness_max,
        "sign_glow_radius": commercial.sign_glow_radius,
        "sign_illumination_radius": commercial.sign_illumination_radius,
        "sign_illumination_intensity": commercial.sign_illumination_intensity,
        "sign_illumination_falloff": commercial.sign_illumination_falloff,
        "sign_illumination_inner_fraction": commercial.sign_illumination_inner_fraction,
        "parking_colors": _color_pool_to_dict(commercial.parking_colors),
        "parking_brightness_min": commercial.parking_brightness_min,
        "parking_brightness_max": commercial.parking_brightness_max,
        "parking_glow_radius": commercial.parking_glow_radius,
        "parking_illumination_radius": commercial.parking_illumination_radius,
        "parking_illumination_intensity": commercial.parking_illumination_intensity,
        "parking_illumination_falloff": commercial.parking_illumination_falloff,
        "parking_illumination_inner_fraction": commercial.parking_illumination_inner_fraction,
        "parking_grid_spacing_min": commercial.parking_grid_spacing_min,
        "parking_grid_spacing_max": commercial.parking_grid_spacing_max,
        "storefront_colors": _color_pool_to_dict(commercial.storefront_colors),
        "storefront_brightness_min": commercial.storefront_brightness_min,
        "storefront_brightness_max": commercial.storefront_brightness_max,
        "storefront_glow_radius": commercial.storefront_glow_radius,
        "storefront_illumination_radius": commercial.storefront_illumination_radius,
        "storefront_illumination_intensity": commercial.storefront_illumination_intensity,
        "storefront_illumination_falloff": commercial.storefront_illumination_falloff,
        "storefront_illumination_inner_fraction": commercial.storefront_illumination_inner_fraction,
        "storefront_spacing_min": commercial.storefront_spacing_min,
        "storefront_spacing_max": commercial.storefront_spacing_max,
    }


def _house_lights_to_dict(house_lights: HouseLightConfig) -> dict:
    """Serialize a HouseLightConfig to a dict.

    Args:
        house_lights (HouseLightConfig): House light configuration.

    Returns:
        dict: Serialized house_lights section.
    """
    return {
        "colors": _color_pool_to_dict(house_lights.colors),
        "brightness_min": house_lights.brightness_min,
        "brightness_max": house_lights.brightness_max,
        "glow_radius": house_lights.glow_radius,
        "illumination_radius": house_lights.illumination_radius,
        "illumination_intensity": house_lights.illumination_intensity,
        "illumination_falloff": house_lights.illumination_falloff,
        "illumination_inner_fraction": house_lights.illumination_inner_fraction,
    }


def _roads_to_dict(roads: RoadsConfig) -> dict:
    """Serialize a RoadsConfig to a dict.

    Args:
        roads (RoadsConfig): Road rendering configuration.

    Returns:
        dict: Serialized roads section.
    """
    return {
        "export": _road_style_config_to_dict(roads.export),
        "preview": _road_style_config_to_dict(roads.preview),
    }


def _road_style_config_to_dict(style: RoadStyleConfig) -> dict:
    """Serialize a RoadStyleConfig to a dict.

    Args:
        style (RoadStyleConfig): Road style configuration.

    Returns:
        dict: Serialized road style section.
    """
    return {
        "highway": _road_style_entry_to_dict(style.highway),
        "major_road": _road_style_entry_to_dict(style.major_road),
        "collector": _road_style_entry_to_dict(style.collector),
        "residential": _road_style_entry_to_dict(style.residential),
        "service": _road_style_entry_to_dict(style.service),
        "minor": _road_style_entry_to_dict(style.minor),
    }


def _road_style_entry_to_dict(entry: RoadStyleEntry) -> dict:
    """Serialize a RoadStyleEntry to a dict.

    Args:
        entry (RoadStyleEntry): Road style entry.

    Returns:
        dict: Serialized road style entry.
    """
    return {"color": entry.color, "width": entry.width}


def _atmosphere_to_dict(atmosphere: AtmosphereConfig) -> dict:
    """Serialize an AtmosphereConfig to a dict.

    Converts the (R, G, B) haze_color tuple back to a '#RRGGBB' hex string
    to match what the loader expects on re-load.

    Args:
        atmosphere (AtmosphereConfig): Atmosphere configuration.

    Returns:
        dict: Serialized atmosphere section.
    """
    r, g, b = atmosphere.haze_color
    haze_hex = f"#{r:02X}{g:02X}{b:02X}"
    return {
        "falloff_distance": atmosphere.falloff_distance,
        "haze_color": haze_hex,
        "scatter_distance": atmosphere.scatter_distance,
        "glow_expansion_enabled": atmosphere.glow_expansion_enabled,
        "glow_max_alpha": atmosphere.glow_max_alpha,
        "glow_falloff_steepness": atmosphere.glow_falloff_steepness,
        "glow_wing_exponent": atmosphere.glow_wing_exponent,
        "glow_tail_strength": atmosphere.glow_tail_strength,
        "glow_cutoff_mult": atmosphere.glow_cutoff_mult,
        "min_brightness_threshold": atmosphere.min_brightness_threshold,
        "illumination_brightness": atmosphere.illumination_brightness,
        "illumination_min_radius": atmosphere.illumination_min_radius,
        "horizon_glow_color": atmosphere.horizon_glow_color,
        "horizon_glow_width_px": atmosphere.horizon_glow_width_px,
        "horizon_glow_peak_alpha": atmosphere.horizon_glow_peak_alpha,
        "haze_enabled": atmosphere.haze_enabled,
        "haze_alpha": atmosphere.haze_alpha,
        "vignette_strength": atmosphere.vignette_strength,
        "vignette_radius": atmosphere.vignette_radius,
    }


def _grade_to_dict(grade: GradeConfig) -> dict:
    """Serialize a GradeConfig to a dict.

    Args:
        grade (GradeConfig): Final color-grade configuration.

    Returns:
        dict: Serialized grade section.
    """
    return {
        "exposure": grade.exposure,
        "tonemap": grade.tonemap,
    }


def _stars_to_dict(stars: StarsConfig) -> dict:
    """Serialize a StarsConfig to a dict.

    Args:
        stars (StarsConfig): Star rendering configuration.

    Returns:
        dict: Serialized stars section.
    """
    return {
        "stars_enabled": stars.stars_enabled,
        "count": stars.count,
        "brightness_min": stars.brightness_min,
        "brightness_max": stars.brightness_max,
        "color_variation": stars.color_variation,
        "radius": stars.radius,
    }


def _heat_haze_to_dict(heat_haze: HeatHazeConfig) -> dict:
    """Serialize a HeatHazeConfig to a dict.

    Args:
        heat_haze (HeatHazeConfig): Multi-layer heat haze configuration.

    Returns:
        dict: Serialized heat_haze section, including the [[layers]]
            array-of-tables.
    """
    return {
        "enabled": heat_haze.enabled,
        "amplitude_px": heat_haze.amplitude_px,
        "max_distance_m": heat_haze.max_distance_m,
        "distance_exponent": heat_haze.distance_exponent,
        "sky_distance_factor": heat_haze.sky_distance_factor,
        "seed": heat_haze.seed,
        "screen_width_px": heat_haze.screen_width_px,
        "screen_height_px": heat_haze.screen_height_px,
        "layers": [_heat_haze_layer_to_dict(layer) for layer in heat_haze.layers],
    }


def _heat_haze_layer_to_dict(layer: HeatHazeLayerConfig) -> dict:
    """Serialize a single HeatHazeLayerConfig to a dict.

    Args:
        layer (HeatHazeLayerConfig): One atmospheric turbulence layer.

    Returns:
        dict: Serialized layer entry.
    """
    return {
        "r0": layer.r0,
        "L0": layer.L0,
        "pixel_scale_m": layer.pixel_scale_m,
        "weight": layer.weight,
        "wind_x_pxs": layer.wind_x_pxs,
        "wind_y_pxs": layer.wind_y_pxs,
    }


def _devmode_to_dict(devmode: DevmodeConfig) -> dict:
    """Serialize a DevmodeConfig to a dict.

    Args:
        devmode (DevmodeConfig): Devmode configuration.

    Returns:
        dict: Serialized devmode section.
    """
    return {
        "render_width": devmode.render_width,
        "render_height": devmode.render_height,
        "camera": {
            "pos_x": devmode.camera.pos_x,
            "pos_y": devmode.camera.pos_y,
            "pos_z": devmode.camera.pos_z,
            "target_x": devmode.camera.target_x,
            "target_y": devmode.camera.target_y,
            "target_z": devmode.camera.target_z,
            "fov": devmode.camera.fov,
            "near": devmode.camera.near,
            "far": devmode.camera.far,
        },
    }
