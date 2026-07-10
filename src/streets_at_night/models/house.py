"""House data model — footprint and colors."""

import random
import typing as t

from shapely.geometry import Polygon

from .color_pool import ColorPool
from ..config.schema import HouseConfig
from ..utils.colors import hex_to_rgb


class House:
    """A single house with a footprint and base colors.

    On construction the house picks its visual properties (exterior color,
    yard color) using the supplied seeded RNG.
    """

    def __init__(
        self,
        footprint: Polygon,
        config: HouseConfig,
        rng: random.Random,
        color_pool: t.Optional[ColorPool] = None,
    ) -> None:
        """Initialize a House from a footprint polygon, config, and seeded RNG.

        Args:
            footprint (Polygon): Shapely polygon representing the house
                footprint in local ENU metres.
            config (HouseConfig): House generation parameters (colors,
                lot dimensions).
            rng (random.Random): Seeded RNG instance for reproducible
                property selection.
            color_pool (ColorPool, optional): Pre-built color pool for exterior
                color selection.  If None, one is created from
                config.exterior_colors.  Pass a shared instance when
                constructing many Houses in the same zone to avoid re-parsing
                the color list on every instantiation.
        """
        self.footprint = footprint
        self.base_color = self._pick_base_color(config, rng, color_pool)
        self.yard_color = self._pick_yard_color(config, rng)

    # ------------------------------------------------------------------
    # Private init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_base_color(
        config: HouseConfig,
        rng: random.Random,
        color_pool: t.Optional[ColorPool] = None,
    ) -> t.Tuple[int, int, int]:
        """Pick the house exterior color from the configured color pool.

        Args:
            config (HouseConfig): House config containing exterior_colors pool.
            rng (random.Random): Seeded RNG for reproducibility.
            color_pool (ColorPool, optional): Pre-built pool to use.  If None,
                one is constructed from config.exterior_colors.

        Returns:
            Tuple[int, int, int]: (R, G, B) exterior color in [0, 255].
        """
        pool = color_pool if color_pool is not None else ColorPool(config.exterior_colors.colors, rng)
        return pool.pick()

    @staticmethod
    def _pick_yard_color(
        config: HouseConfig,
        rng: random.Random,
    ) -> t.Tuple[int, int, int]:
        """Pick a yard color by applying random variation to the base yard color.

        Args:
            config (HouseConfig): House config with yard_base_color and
                yard_variation.
            rng (random.Random): Seeded RNG for reproducibility.

        Returns:
            Tuple[int, int, int]: (R, G, B) yard color in [0, 255].
        """
        base = hex_to_rgb(config.yard_base_color)
        variation = config.yard_variation
        r = _clamp(base[0] + rng.randint(-variation, variation))
        g = _clamp(base[1] + rng.randint(-variation, variation))
        b = _clamp(base[2] + rng.randint(-variation, variation))
        return r, g, b


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _clamp(value: int, lo: int = 0, hi: int = 255) -> int:
    """Clamp an integer to [lo, hi].

    Args:
        value (int): Value to clamp.
        lo (int): Lower bound (inclusive). Defaults to 0.
        hi (int): Upper bound (inclusive). Defaults to 255.

    Returns:
        int: Clamped value.
    """
    return max(lo, min(hi, value))
