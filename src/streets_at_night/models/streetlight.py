"""FixtureProfile, FixturePool, and StreetLight factory for road-mounted lights."""

import random
import typing as t
from dataclasses import dataclass

import numpy as np

from ..config.schema import FixturePoolEntry, FixtureProfileConfig, StreetLightConfig
from ..utils.colors import hex_to_rgb
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
        illumination_radius: Ground illumination radius in meters.
        illumination_intensity: Ground illumination strength in [0, 1].
        illumination_falloff: Cone falloff exponent controlling the rate of
            intensity drop from the inner edge to the outer radius.
        illumination_inner_fraction: Fraction of illumination_radius at full
            brightness before falloff begins, in [0, 1].
        spacing: Distance between consecutive lights in meters.
        side: Where lights are placed — "both", "one", or "alternating".
    """

    fixture_color: t.Tuple[int, int, int]
    height: float
    brightness: float
    glow_radius: float
    illumination_radius: float
    illumination_intensity: float
    illumination_falloff: float
    illumination_inner_fraction: float
    spacing: float
    side: str


# ---------------------------------------------------------------------------
# FixturePool
# ---------------------------------------------------------------------------


class FixturePool:
    """Weighted collection of FixtureProfiles (or None) for a road type.

    None entries represent segments with no streetlights. The weight of
    a None entry controls how frequently a road type has dark segments.
    """

    def __init__(
        self,
        profiles_with_weights: t.List[t.Tuple[t.Optional[FixtureProfile], float]],
        rng: random.Random,
    ) -> None:
        """Initialize a FixturePool with weighted profiles and a seeded RNG.

        Args:
            profiles_with_weights (List[Tuple[Optional[FixtureProfile], float]]):
                List of (profile, weight) pairs. A None profile means no lights.
                Weights are relative and do not need to sum to 1.
            rng (random.Random): Seeded RNG for reproducible selection.
        """
        self._rng = rng
        self._profiles = [p for p, _ in profiles_with_weights]
        self._weights = [w for _, w in profiles_with_weights]

    def pick(self) -> t.Optional[FixtureProfile]:
        """Pick a profile using weighted random selection.

        Returns:
            Optional[FixtureProfile]: A profile, or None if this segment
                should have no streetlights.
        """
        chosen = self._rng.choices(self._profiles, weights=self._weights, k=1)
        return chosen[0]


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
        return LightSource(
            position=position,
            color=fixture_profile.fixture_color,
            brightness=fixture_profile.brightness,
            size=0.3,
            glow_radius=fixture_profile.glow_radius,
            illumination_radius=fixture_profile.illumination_radius,
            illumination_intensity=fixture_profile.illumination_intensity,
            illumination_falloff=fixture_profile.illumination_falloff,
            illumination_inner_fraction=fixture_profile.illumination_inner_fraction,
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
    StreetLightConfig. Each entry with a profile is converted to a
    FixtureProfile by sampling its height, brightness, and spacing ranges.
    Entries with no profile (None) become dark-segment slots in the pool.

    MINOR roads use the residential pool since no dedicated config exists.

    Args:
        road_type (RoadType): The road classification.
        config (StreetLightConfig): Streetlight config with pool entries per road type.
        rng (random.Random): Seeded RNG for sampling profile parameter ranges.

    Returns:
        FixturePool: Pool appropriate for the road type.
    """
    pool_entries = _get_pool_entries_for_road_type(road_type, config)
    profiles_with_weights = _build_profiles_with_weights(pool_entries, rng)
    return FixturePool(profiles_with_weights, rng)


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


def _build_profiles_with_weights(
    pool_entries: t.List[FixturePoolEntry],
    rng: random.Random,
) -> t.List[t.Tuple[t.Optional[FixtureProfile], float]]:
    """Convert pool entries to (profile, weight) pairs for FixturePool.

    Args:
        pool_entries (List[FixturePoolEntry]): Weighted pool entries from config.
        rng (random.Random): Seeded RNG for sampling parameter ranges.

    Returns:
        List[Tuple[Optional[FixtureProfile], float]]: Pairs ready for FixturePool.
    """
    result = []
    for entry in pool_entries:
        if entry.profile is None:
            result.append((None, entry.weight))
        else:
            profile = _config_to_fixture_profile(entry.profile, rng)
            result.append((profile, entry.weight))
    return result


def _config_to_fixture_profile(
    cfg: FixtureProfileConfig,
    rng: random.Random,
) -> FixtureProfile:
    """Convert a config profile to a runtime FixtureProfile, sampling ranges.

    Height, brightness, and spacing are sampled uniformly from their
    [min, max] ranges defined in the config. All other fields are fixed values.

    Args:
        cfg (FixtureProfileConfig): Config profile with range fields.
        rng (random.Random): RNG used to sample range parameters.

    Returns:
        FixtureProfile: Constructed fixture profile with sampled values.
    """
    return FixtureProfile(
        fixture_color=hex_to_rgb(cfg.color),
        height=rng.uniform(cfg.height_min, cfg.height_max),
        brightness=rng.uniform(cfg.brightness_min, cfg.brightness_max),
        glow_radius=cfg.glow_radius,
        illumination_radius=cfg.illumination_radius,
        illumination_intensity=cfg.illumination_intensity,
        illumination_falloff=cfg.illumination_falloff,
        illumination_inner_fraction=cfg.illumination_inner_fraction,
        spacing=rng.uniform(cfg.spacing_min, cfg.spacing_max),
        side=cfg.side,
    )
