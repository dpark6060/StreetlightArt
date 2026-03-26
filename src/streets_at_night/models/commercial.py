"""Factory functions for commercial zone light sources."""

import random

import numpy as np

from .color_pool import ColorPool
from .light_source import LightSource
from ..config.schema import CommercialLightConfig


def create_sign_light(
    position: np.ndarray,
    config: CommercialLightConfig,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing an illuminated commercial sign.

    Signs are placed along road-facing edges of commercial zone buildings.
    They are bright with a large illumination radius due to ambient spill.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
        config (CommercialLightConfig): Commercial light configuration providing
            color pools, brightness range, and glow/illumination radii.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured sign light.
    """
    pool = ColorPool(config.sign_colors.colors, rng)
    color = pool.pick()
    brightness = rng.uniform(config.sign_brightness_min, config.sign_brightness_max)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.5,
        glow_radius=config.sign_glow_radius,
        illumination_radius=config.sign_illumination_radius,
        illumination_intensity=config.sign_illumination_intensity,
    )


def create_parking_lot_light(
    position: np.ndarray,
    config: CommercialLightConfig,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing a parking lot pole light.

    Parking lot lights are placed in grids within commercial zone parking
    areas. They are bright with wide illumination coverage.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
            The z component should reflect the pole height.
        config (CommercialLightConfig): Commercial light configuration providing
            color pools, brightness range, and glow/illumination radii.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured parking lot light.
    """
    pool = ColorPool(config.parking_colors.colors, rng)
    color = pool.pick()
    brightness = rng.uniform(config.parking_brightness_min, config.parking_brightness_max)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.3,
        glow_radius=config.parking_glow_radius,
        illumination_radius=config.parking_illumination_radius,
        illumination_intensity=config.parking_illumination_intensity,
    )


def create_storefront_light(
    position: np.ndarray,
    config: CommercialLightConfig,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing a storefront window or awning light.

    Storefront lights are placed along building faces in commercial zones.
    They are dimmer than signs but cast warm ambient light onto the sidewalk.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
        config (CommercialLightConfig): Commercial light configuration providing
            color pools, brightness range, and glow/illumination radii.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured storefront light.
    """
    pool = ColorPool(config.storefront_colors.colors, rng)
    color = pool.pick()
    brightness = rng.uniform(config.storefront_brightness_min, config.storefront_brightness_max)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.25,
        glow_radius=config.storefront_glow_radius,
        illumination_radius=config.storefront_illumination_radius,
        illumination_intensity=config.storefront_illumination_intensity,
    )
