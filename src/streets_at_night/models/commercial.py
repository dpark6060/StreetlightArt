"""Factory functions for commercial zone light sources."""

import random
import typing as t

import numpy as np

from .color_pool import ColorPool
from .light_source import LightSource


# ---------------------------------------------------------------------------
# Sign light
# ---------------------------------------------------------------------------

_SIGN_COLORS: t.List[t.Tuple[str, float]] = [
    ("#FFE8CC", 35.0),  # warm white
    ("#FF2020", 12.0),  # red
    ("#2040FF", 12.0),  # blue
    ("#20CC40", 10.0),  # green
    ("#FFE020", 10.0),  # yellow
    ("#8020CC", 8.0),   # purple
    ("#FF8020", 8.0),   # orange
    ("#FF60A0", 5.0),   # pink
]


def create_sign_light(
    position: np.ndarray,
    config: dict,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing an illuminated commercial sign.

    Signs are placed along road-facing edges of commercial zone buildings.
    They are bright with a large illumination radius due to ambient spill.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
        config (dict): Configuration overrides. Currently unused; reserved
            for TOML-driven customisation.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured sign light.
    """
    pool = ColorPool(_SIGN_COLORS, rng)
    color = pool.pick()
    brightness = rng.uniform(0.6, 0.9)
    glow_radius = rng.uniform(6.0, 14.0)
    illumination_radius = rng.uniform(15.0, 25.0)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.5,
        glow_radius=glow_radius,
        illumination_radius=illumination_radius,
        illumination_intensity=0.7,
    )


# ---------------------------------------------------------------------------
# Parking lot light
# ---------------------------------------------------------------------------

_PARKING_LOT_COLORS: t.List[t.Tuple[str, float]] = [
    ("#E0E8FF", 70.0),  # LED white
    ("#FFA040", 30.0),  # sodium
]


def create_parking_lot_light(
    position: np.ndarray,
    config: dict,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing a parking lot pole light.

    Parking lot lights are placed in grids within commercial zone parking
    areas. They are bright with wide illumination coverage.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
            The z component should reflect the pole height.
        config (dict): Configuration overrides. Currently unused; reserved
            for TOML-driven customisation.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured parking lot light.
    """
    pool = ColorPool(_PARKING_LOT_COLORS, rng)
    color = pool.pick()
    brightness = rng.uniform(0.75, 0.9)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.3,
        glow_radius=20.0,
        illumination_radius=30.0,
        illumination_intensity=0.8,
    )


# ---------------------------------------------------------------------------
# Storefront light
# ---------------------------------------------------------------------------

_STOREFRONT_COLORS: t.List[t.Tuple[str, float]] = [
    ("#FFE8CC", 60.0),  # warm white
    ("#FFE080", 25.0),  # warm yellow
    ("#F0F0FF", 15.0),  # fluorescent
]


def create_storefront_light(
    position: np.ndarray,
    config: dict,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing a storefront window or awning light.

    Storefront lights are placed along building faces in commercial zones.
    They are dimmer than signs but cast warm ambient light onto the sidewalk.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
        config (dict): Configuration overrides. Currently unused; reserved
            for TOML-driven customisation.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured storefront light.
    """
    pool = ColorPool(_STOREFRONT_COLORS, rng)
    color = pool.pick()
    brightness = rng.uniform(0.5, 0.7)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.25,
        glow_radius=8.0,
        illumination_radius=12.0,
        illumination_intensity=0.5,
    )
