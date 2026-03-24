"""Factory function for house window and porch light sources."""

import random
import typing as t

import numpy as np

from .color_pool import ColorPool
from .light_source import LightSource


# ---------------------------------------------------------------------------
# House light color pool
# ---------------------------------------------------------------------------

_HOUSE_LIGHT_COLORS: t.List[t.Tuple[str, float]] = [
    ("#FFE4A0", 45.0),  # warm yellow (most common interior/window light)
    ("#FFF0D0", 25.0),  # warm white (LED or incandescent)
    ("#FFCC80", 15.0),  # amber (porch/accent)
    ("#D0E0FF", 15.0),  # TV blue (cool blue interior glow)
]


def create_house_light(
    position: np.ndarray,
    config: dict,
    rng: random.Random,
) -> LightSource:
    """Create a LightSource representing a house window, porch, or interior light.

    House lights are dim and warm, representing the ambient glow of residential
    activity through windows and on porches.

    Args:
        position (np.ndarray): 3-element (x, y, z) ENU position in meters.
        config (dict): Configuration overrides. Currently unused; reserved
            for TOML-driven customisation.
        rng (random.Random): Seeded RNG for color and parameter sampling.

    Returns:
        LightSource: Configured house light.
    """
    pool = ColorPool(_HOUSE_LIGHT_COLORS, rng)
    color = pool.pick()
    brightness = rng.uniform(0.3, 0.6)
    return LightSource(
        position=position,
        color=color,
        brightness=brightness,
        size=0.2,
        glow_radius=5.0,
        illumination_radius=8.0,
        illumination_intensity=0.3,
    )
