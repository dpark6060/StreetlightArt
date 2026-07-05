"""FixtureProfile, FixturePool, and StreetLight factory for road-mounted lights."""

import math
import random
import typing as t
from dataclasses import dataclass

import numpy as np

from ..config.schema import FixturePoolEntry, FixtureProfileConfig, StreetLightConfig
from ..utils.colors import kelvin_to_rgb
from .light_source import LightSource
from .road_types import RoadType


# ---------------------------------------------------------------------------
# FixtureProfile
# ---------------------------------------------------------------------------


@dataclass
class FixtureProfile:
    """Defines the lighting character of an entire road segment.

    All streetlights on a given road segment share the same FixtureProfile.
    Variation happens between segments, not within them.

    Attributes:
        fixture_color: RGB color for every light on this segment.
        height: Pole height in meters.
        brightness: Light intensity scalar in [0, 1].
        glow_radius: Atmospheric glow halo radius in meters.
        beam_angle_deg: Full luminaire cone angle in degrees. The ground pool
            radius and falloff are derived from this plus the pole height.
        illumination_intensity: Ground illumination strength in [0, 1].
        spacing: Distance between consecutive lights in meters.
        side: Where lights are placed — "both", "one", or "alternating".
    """

    fixture_color: t.Tuple[int, int, int]
    height: float
    brightness: float
    glow_radius: float
    beam_angle_deg: float
    illumination_intensity: float
    spacing: float
    side: str


# ---------------------------------------------------------------------------
# FixturePool
# ---------------------------------------------------------------------------


class FixturePool:
    """Weighted collection of FixtureProfileConfigs (or None) for a road type.

    None entries represent segments with no streetlights. The weight of
    a None entry controls how frequently a road type has dark segments.

    Each entry stores a config, not a materialized FixtureProfile. Numeric
    ranges (height, brightness, spacing) are sampled fresh on every pick()
    call, so each road segment gets its own values instead of the whole
    road type collapsing onto a handful of pre-built variants.
    """

    def __init__(
        self,
        entries_with_weights: t.List[t.Tuple[t.Optional[FixtureProfileConfig], float]],
        rng: random.Random,
        color_temp_shift_k: float,
    ) -> None:
        """Initialize a FixturePool with weighted config entries and a seeded RNG.

        Args:
            entries_with_weights (List[Tuple[Optional[FixtureProfileConfig], float]]):
                List of (config, weight) pairs. A None config means no lights.
                Weights are relative and do not need to sum to 1.
            rng (random.Random): Seeded RNG for reproducible selection and
                range sampling.
            color_temp_shift_k (float): City-wide additive Kelvin offset
                applied to every sampled profile's authored color temperature.
        """
        self._rng = rng
        self._color_temp_shift_k = color_temp_shift_k
        self._entries = [e for e, _ in entries_with_weights]
        self._weights = [w for _, w in entries_with_weights]

    def pick(self) -> t.Optional[FixtureProfile]:
        """Pick a config using weighted random selection and sample a profile.

        Returns:
            Optional[FixtureProfile]: A freshly sampled profile, or None if
                this segment should have no streetlights.
        """
        chosen = self._rng.choices(self._entries, weights=self._weights, k=1)[0]
        if chosen is None:
            return None
        return config_to_fixture_profile(chosen, self._rng, self._color_temp_shift_k)


# ---------------------------------------------------------------------------
# StreetLight factory
# ---------------------------------------------------------------------------


class StreetLight:
    """Factory for creating LightSource objects from a FixtureProfile.

    All StreetLight behavior is captured in FixtureProfile. This class
    provides the static factory method that converts a profile + position
    into a renderer-ready LightSource.
    """

    @staticmethod
    def create_from_profile(
        fixture_profile: FixtureProfile,
        position: np.ndarray,
        road_type: RoadType,
    ) -> LightSource:
        """Create a LightSource from a FixtureProfile and world position.

        Args:
            fixture_profile (FixtureProfile): The profile that defines this
                light's visual properties.
            position (np.ndarray): 3-element (x, y, z) ENU position in meters.
                The z component should reflect the pole height.
            road_type (RoadType): The road type this light belongs to.
                Stored for context but does not alter the created LightSource.

        Returns:
            LightSource: A fully configured light ready for rendering.
        """
        half_angle = math.radians(fixture_profile.beam_angle_deg / 2.0)
        derived_radius = fixture_profile.height * math.tan(half_angle)
        return LightSource(
            position=position,
            color=fixture_profile.fixture_color,
            brightness=fixture_profile.brightness,
            size=0.3,
            glow_radius=fixture_profile.glow_radius,
            # illumination_radius is derived from height + beam angle and kept as
            # a fallback; the physical branch in the illumination pass recomputes
            # it. illumination_falloff / inner_fraction are unused when beam is set.
            illumination_radius=derived_radius,
            illumination_intensity=fixture_profile.illumination_intensity,
            illumination_falloff=2.0,
            illumination_inner_fraction=0.0,
            beam_angle_deg=fixture_profile.beam_angle_deg,
        )


# ---------------------------------------------------------------------------
# FixturePool factory
# ---------------------------------------------------------------------------

# Maps RoadType to the StreetLightConfig attribute name.
# MINOR falls back to the residential pool since it has no dedicated config entry.
_ROAD_TYPE_TO_CONFIG_ATTR: t.Dict[RoadType, str] = {
    RoadType.HIGHWAY: "highway",
    RoadType.MAJOR_ROAD: "major_road",
    RoadType.COLLECTOR: "collector",
    RoadType.RESIDENTIAL: "residential",
    RoadType.SERVICE: "service",
    RoadType.MINOR: "residential",
}


def create_fixture_pool(
    road_type: RoadType,
    config: StreetLightConfig,
    rng: random.Random,
) -> FixturePool:
    """Create a fixture pool for the given road type from config.

    Reads the weighted pool entries for the road type from the provided
    StreetLightConfig and wraps their configs (not materialized profiles)
    in a FixturePool. Each entry's height, brightness, and spacing ranges
    are sampled fresh every time the pool is picked from, not once here.
    Entries with no profile (None) become dark-segment slots in the pool.

    MINOR roads use the residential pool since no dedicated config exists.

    Args:
        road_type (RoadType): The road classification.
        config (StreetLightConfig): Streetlight config with pool entries per road type.
        rng (random.Random): Seeded RNG shared for selection and range sampling.

    Returns:
        FixturePool: Pool appropriate for the road type.
    """
    pool_entries = _get_pool_entries_for_road_type(road_type, config)
    entries_with_weights = [(entry.profile, entry.weight) for entry in pool_entries]
    return FixturePool(entries_with_weights, rng, config.color_temp_shift_k)


def _get_pool_entries_for_road_type(
    road_type: RoadType,
    config: StreetLightConfig,
) -> t.List[FixturePoolEntry]:
    """Return the FixturePoolEntry list for the given road type.

    Args:
        road_type (RoadType): The road classification.
        config (StreetLightConfig): Streetlight config to read from.

    Returns:
        List[FixturePoolEntry]: Pool entries for the road type.
    """
    attr = _ROAD_TYPE_TO_CONFIG_ATTR.get(road_type, "residential")
    return getattr(config, attr)


def config_to_fixture_profile(
    cfg: FixtureProfileConfig,
    rng: random.Random,
    color_temp_shift_k: float,
) -> FixtureProfile:
    """Convert a config profile to a runtime FixtureProfile, sampling ranges.

    Height, brightness, and spacing are sampled uniformly from their
    [min, max] ranges defined in the config. The fixture color is resolved
    from its blackbody temperature (plus the city-wide shift) and saturation.

    Args:
        cfg (FixtureProfileConfig): Config profile with range fields.
        rng (random.Random): RNG used to sample range parameters.
        color_temp_shift_k (float): Additive Kelvin offset applied to the
            authored color temperature before conversion to RGB.

    Returns:
        FixtureProfile: Constructed fixture profile with sampled values.
    """
    return FixtureProfile(
        fixture_color=kelvin_to_rgb(cfg.color_temp_k + color_temp_shift_k, cfg.saturation),
        height=rng.uniform(cfg.height_min, cfg.height_max),
        brightness=rng.uniform(cfg.brightness_min, cfg.brightness_max),
        glow_radius=cfg.glow_radius,
        beam_angle_deg=cfg.beam_angle_deg,
        illumination_intensity=cfg.illumination_intensity,
        spacing=rng.uniform(cfg.spacing_min, cfg.spacing_max),
        side=cfg.side,
    )
