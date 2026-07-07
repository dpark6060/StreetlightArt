"""Factory function for house window and porch light sources."""

import random
import typing as t

import numpy as np

from .color_pool import ColorPool
from .illumination import ConeIllumination
from .light_source import LightSource
from ..config.schema import HouseLightConfig


def create_house_light(
    position: np.ndarray,
    config: HouseLightConfig,
    rng: random.Random,
    pool: t.Optional[ColorPool] = None,
) -> LightSource:
    """Create a LightSource representing a house window, porch, or interior light.

    House lights are dim and warm, representing the ambient glow of residential
    activity through windows and on porches.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
        config (HouseLightConfig): House light configuration providing color pool,
            brightness range, glow radius, illumination radius, and intensity.
        rng (random.Random): Seeded RNG for color and parameter sampling.
        pool (ColorPool, optional): Pre-built color pool. If None, one is
            created from config.colors. Pass a shared instance when creating
            many house lights in the same run to avoid re-parsing the palette.

    Returns:
        LightSource: Configured house light.
    """
    pool = pool if pool is not None else ColorPool(config.colors.colors, rng)
    color = pool.pick()
    brightness = rng.uniform(config.brightness_min, config.brightness_max)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.2,
        glow_radius=config.glow_radius,
        illumination_intensity=config.illumination_intensity,
        illumination=ConeIllumination(
            config.illumination_radius,
            config.illumination_falloff,
            config.illumination_inner_fraction,
        ),
    )
