"""Weighted random color selection from a named palette."""

import random
import typing as t

from ..utils.colors import hex_to_rgb


class ColorPool:
    """Selects colors randomly from a weighted palette.

    All color selection in the project goes through this class, ensuring
    reproducible results when a seeded RNG is provided.
    """

    def __init__(
        self,
        colors: t.List[t.Tuple[str, float]],
        rng: random.Random,
    ) -> None:
        """Initialize a ColorPool with weighted hex colors and an RNG.

        Args:
            colors (List[Tuple[str, float]]): List of (hex_color, weight) pairs.
                Hex color strings must be in "#RRGGBB" format (e.g. "#0D0A08").
                Weights are relative — they do not need to sum to 1.
            rng (random.Random): Seeded RNG instance for reproducible selection.
        """
        self._rng = rng
        self._rgb_colors, self._weights = self._parse_colors(colors)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def pick(self) -> t.Tuple[int, int, int]:
        """Pick a color from the pool using weighted random selection.

        Returns:
            Tuple[int, int, int]: An (R, G, B) tuple with values in [0, 255].
        """
        chosen = self._rng.choices(self._rgb_colors, weights=self._weights, k=1)
        return chosen[0]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_colors(
        self,
        colors: t.List[t.Tuple[str, float]],
    ) -> t.Tuple[t.List[t.Tuple[int, int, int]], t.List[float]]:
        """Unzip and convert hex color strings to RGB tuples.

        Args:
            colors (List[Tuple[str, float]]): Raw (hex_color, weight) pairs.

        Returns:
            Tuple[List[Tuple[int,int,int]], List[float]]:
                Parallel lists of RGB tuples and weights.
        """
        rgb_list = [hex_to_rgb(hex_color) for hex_color, _ in colors]
        weight_list = [weight for _, weight in colors]
        return rgb_list, weight_list

