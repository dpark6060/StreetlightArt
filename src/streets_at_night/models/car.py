"""CarLightPair — a paired headlight/taillight set for a single car."""

import math
import random
import typing as t

import numpy as np

from .color_pool import ColorPool
from .illumination import ConeIllumination
from .light_source import LightSource
from ..config.schema import CarConfig


# Placeholder color for unresolved lights (replaced on resolve_facing call)
_PLACEHOLDER_COLOR: t.Tuple[int, int, int] = (255, 255, 255)

# Car light physical constants
_LIGHT_SIZE: float = 0.15


class CarLightPair:
    """A paired set of car lights — always two, never one.

    Stores absolute travel bearing so headlight vs taillight assignment
    can be deferred to render time via ``resolve_facing()``. This keeps
    placement valid when the camera moves during preview.

    Car light positions are offset perpendicular to the travel direction
    to approximate the separation between the two headlights (or taillights).
    """

    def __init__(
        self,
        position: np.ndarray,
        travel_bearing: float,
        config: CarConfig,
        rng: random.Random,
        headlight_pool: t.Optional[ColorPool] = None,
        taillight_pool: t.Optional[ColorPool] = None,
    ) -> None:
        """Initialize a CarLightPair at a road position.

        Args:
            position (np.ndarray): 3-element (x, y, z) center position on
                the road in ENU meters.
            travel_bearing (float): Absolute travel direction in radians.
                0 = East, pi/2 = North. NOT camera-relative.
            config (CarConfig): Car light configuration providing headlight
                and taillight color pools and separation range.
            rng (random.Random): Seeded RNG for separation sampling.
            headlight_pool (ColorPool, optional): Pre-built headlight color
                pool. If None, one is created from config.headlight_colors.
                Pass a shared instance when placing many cars to avoid
                re-parsing the palette per car.
            taillight_pool (ColorPool, optional): Pre-built taillight color
                pool. If None, one is created from config.taillight_colors.
        """
        self.travel_bearing = travel_bearing
        self._rng = rng
        self._config = config
        headlight_pool = headlight_pool if headlight_pool is not None else ColorPool(config.headlight_colors.colors, rng)
        taillight_pool = taillight_pool if taillight_pool is not None else ColorPool(config.taillight_colors.colors, rng)
        self._headlight_color = headlight_pool.pick()
        self._headlight_brightness = rng.uniform(config.headlight_brightness_min, config.headlight_brightness_max)
        self._taillight_color = taillight_pool.pick()
        self._taillight_brightness = rng.uniform(config.taillight_brightness_min, config.taillight_brightness_max)
        self._lights = self._create_lights(
            position,
            travel_bearing,
            rng,
            config.separation_min,
            config.separation_max,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def resolve_facing(self, camera_forward: np.ndarray) -> None:
        """Update light colors and intensities based on camera orientation.

        A travel direction opposing the camera forward vector means the
        car is approaching the viewer, so the viewer sees its headlights.
        A travel direction aligned with the camera forward vector means
        the car is moving away, so the viewer sees its taillights.

        Args:
            camera_forward (np.ndarray): Camera look direction as a
                3-element unit vector in ENU space.
        """
        travel_dir = self._get_travel_direction()
        dot = float(np.dot(travel_dir, camera_forward))

        if dot < 0:
            self._apply_headlight_properties()
        else:
            self._apply_taillight_properties()

    def get_lights(self) -> t.List[LightSource]:
        """Return the two LightSource instances for this car.

        Returns:
            List[LightSource]: Always exactly two lights.
        """
        return list(self._lights)

    # ------------------------------------------------------------------
    # Private init helpers
    # ------------------------------------------------------------------

    def _create_lights(
        self,
        position: np.ndarray,
        travel_bearing: float,
        rng: random.Random,
        separation_min: float,
        separation_max: float,
    ) -> t.Tuple[LightSource, LightSource]:
        """Create two LightSource instances offset perpendicular to travel.

        Args:
            position (np.ndarray): Road center position in ENU meters.
            travel_bearing (float): Travel direction in radians.
            rng (random.Random): RNG for separation sampling.
            separation_min (float): Minimum light separation in meters.
            separation_max (float): Maximum light separation in meters.

        Returns:
            Tuple[LightSource, LightSource]: Left and right light positions.
        """
        separation = rng.uniform(separation_min, separation_max)
        left_pos, right_pos = self._compute_light_positions(
            position, travel_bearing, separation
        )
        left_light = self._make_placeholder_light(left_pos)
        right_light = self._make_placeholder_light(right_pos)
        return left_light, right_light

    @staticmethod
    def _compute_light_positions(
        center: np.ndarray,
        bearing: float,
        separation: float,
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Compute left and right light positions perpendicular to travel.

        Args:
            center (np.ndarray): Road center position.
            bearing (float): Travel direction in radians.
            separation (float): Total width between the two lights in meters.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Left and right positions.
        """
        # Perpendicular direction is bearing rotated 90 degrees
        perp_x = -math.sin(bearing)
        perp_y = math.cos(bearing)
        perp = np.array([perp_x, perp_y, 0.0])
        half = separation / 2.0
        left_pos = center + perp * half
        right_pos = center - perp * half
        return left_pos, right_pos

    @staticmethod
    def _make_placeholder_light(position: np.ndarray) -> LightSource:
        """Create a LightSource with placeholder values at a given position.

        Args:
            position (np.ndarray): 3-element ENU position.

        Returns:
            LightSource: Placeholder light — color and intensity are
                replaced when resolve_facing() is called.
        """
        return LightSource(
            position=position,
            color=_PLACEHOLDER_COLOR,
            brightness=0.0,
            size=_LIGHT_SIZE,
            glow_radius=0.0,
            illumination_intensity=0.0,
            illumination=None,
        )

    # ------------------------------------------------------------------
    # Private resolve helpers
    # ------------------------------------------------------------------

    def _get_travel_direction(self) -> np.ndarray:
        """Compute travel direction unit vector from the stored bearing.

        Returns:
            np.ndarray: 3-element unit vector in ENU space.
        """
        travel_x = math.cos(self.travel_bearing)
        travel_y = math.sin(self.travel_bearing)
        return np.array([travel_x, travel_y, 0.0])

    def _apply_headlight_properties(self) -> None:
        """Set both lights to the pre-sampled headlight color, brightness, and glow."""
        for light in self._lights:
            light.color = self._headlight_color
            light.brightness = self._headlight_brightness
            light.glow_radius = self._config.headlight_glow_radius
            light.illumination_intensity = self._config.headlight_illumination_intensity
            light.illumination = ConeIllumination(
                self._config.headlight_illumination_radius,
                self._config.headlight_illumination_falloff,
                self._config.headlight_illumination_inner_fraction,
            )

    def _apply_taillight_properties(self) -> None:
        """Set both lights to the pre-sampled taillight color, brightness, and glow."""
        for light in self._lights:
            light.color = self._taillight_color
            light.brightness = self._taillight_brightness
            light.glow_radius = self._config.taillight_glow_radius
            light.illumination_intensity = self._config.taillight_illumination_intensity
            light.illumination = ConeIllumination(
                self._config.taillight_illumination_radius,
                self._config.taillight_illumination_falloff,
                self._config.taillight_illumination_inner_fraction,
            )
