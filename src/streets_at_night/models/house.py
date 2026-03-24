"""House data model — footprint, colors, and light positions."""

import random
import typing as t

import numpy as np
from shapely.geometry import Polygon

from .color_pool import ColorPool
from ..config.schema import HouseConfig
from ..utils.colors import hex_to_rgb


class House:
    """A single house with a footprint, base colors, and light positions.

    On construction the house picks its visual properties (exterior color,
    yard color, light count) using the supplied seeded RNG.  Light positions
    along the footprint perimeter are computed immediately.  Call
    ``get_lights(rng)`` to create LightSource objects at those positions.
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
                light count distribution, lot dimensions).
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
        self.num_lights = self._pick_num_lights(config, rng)
        self.light_positions = self._compute_light_positions(footprint, self.num_lights)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_lights(self, rng: random.Random) -> t.List["LightSource"]:
        """Create LightSource objects for this house's lights.

        Args:
            rng (random.Random): Seeded RNG passed through to the house
                light factory for reproducible color selection.

        Returns:
            List[LightSource]: One LightSource per pre-computed light
                position. Returns an empty list if num_lights is 0.
        """
        from .house_light import create_house_light
        from .light_source import LightSource

        lights: t.List[LightSource] = []
        for pos in self.light_positions:
            position_3d = np.array([pos[0], pos[1], 0.0])
            lights.append(create_house_light(position_3d, {}, rng))
        return lights

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

    @staticmethod
    def _pick_num_lights(
        config: HouseConfig,
        rng: random.Random,
    ) -> int:
        """Draw the number of lights from a weighted distribution.

        Args:
            config (HouseConfig): House config with min_lights, max_lights,
                and light_weights.
            rng (random.Random): Seeded RNG for reproducibility.

        Returns:
            int: Number of lights in [min_lights, max_lights].
        """
        choices = list(range(config.min_lights, config.max_lights + 1))
        result = rng.choices(choices, weights=config.light_weights, k=1)
        return result[0]

    @staticmethod
    def _compute_light_positions(
        footprint: Polygon,
        num_lights: int,
    ) -> t.List[t.Tuple[float, float]]:
        """Compute evenly-spaced light positions along the footprint perimeter.

        Positions are placed at equal parametric intervals around the exterior
        ring and offset ~1 m inward toward the centroid so they sit just
        inside the building edge.

        Args:
            footprint (Polygon): House footprint polygon in ENU metres.
            num_lights (int): Number of positions to compute.

        Returns:
            List[Tuple[float, float]]: (x, y) positions in ENU metres.
        """
        if num_lights == 0:
            return []
        return _place_along_perimeter(footprint, num_lights)


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


def _place_along_perimeter(
    footprint: Polygon,
    num_positions: int,
    inset_m: float = 1.0,
) -> t.List[t.Tuple[float, float]]:
    """Place ``num_positions`` evenly-spaced points around a polygon perimeter.

    Each point is offset slightly inward toward the centroid so that it sits
    just inside the building edge rather than exactly on it.

    Args:
        footprint (Polygon): The house footprint polygon.
        num_positions (int): Number of points to place.
        inset_m (float): Distance in metres to pull each point inward.
            Defaults to 1.0 m.

    Returns:
        List[Tuple[float, float]]: (x, y) ENU positions.
    """
    perimeter = footprint.exterior.length
    centroid_x = footprint.centroid.x
    centroid_y = footprint.centroid.y

    positions = []
    for i in range(num_positions):
        t_param = i / num_positions
        point = footprint.exterior.interpolate(t_param, normalized=True)
        inset_pos = _inset_toward_centroid(
            point.x, point.y, centroid_x, centroid_y, inset_m, perimeter
        )
        positions.append(inset_pos)
    return positions


def _inset_toward_centroid(
    px: float,
    py: float,
    cx: float,
    cy: float,
    inset_m: float,
    perimeter: float,
) -> t.Tuple[float, float]:
    """Shift a perimeter point slightly toward the polygon centroid.

    Args:
        px (float): Point x coordinate.
        py (float): Point y coordinate.
        cx (float): Centroid x coordinate.
        cy (float): Centroid y coordinate.
        inset_m (float): Desired inset distance in metres.
        perimeter (float): Polygon perimeter length — used to guard against
            degenerate (zero-area) footprints.

    Returns:
        Tuple[float, float]: Inset (x, y) position.
    """
    dx = cx - px
    dy = cy - py
    dist = (dx * dx + dy * dy) ** 0.5

    if dist < 1e-6 or perimeter < 1e-6:
        return px, py

    step = min(inset_m, dist * 0.5)
    nx = dx / dist
    ny = dy / dist
    return px + nx * step, py + ny * step
