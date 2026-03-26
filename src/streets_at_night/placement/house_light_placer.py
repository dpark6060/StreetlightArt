"""Residential zone house light collection — activates lights on each House."""

import random
import typing as t

from ..config.schema import HouseLightConfig, LightConfig
from ..models.light_source import LightSource
from ..models.zones import ResidentialZone


def collect_house_lights(
    residential_zones: t.List[ResidentialZone],
    config: LightConfig,
    rng: random.Random,
) -> t.List[LightSource]:
    """Collect light sources from all houses across all residential zones.

    Each House already knows its light positions from Stage 4 initialisation.
    This function calls get_lights() on every house and gathers the results.
    Houses with num_lights == 0 produce no lights and are silently skipped.

    Args:
        residential_zones (List[ResidentialZone]): Zones whose houses to
            activate.
        config (LightConfig): Top-level light configuration. Passed for
            interface consistency; light parameters are embedded in each
            House via its HouseConfig.
        rng (random.Random): Seeded RNG passed through to house light
            factory for reproducible color selection.

    Returns:
        List[LightSource]: All house lights across all residential zones.
    """
    lights: t.List[LightSource] = []
    for zone in residential_zones:
        lights.extend(_collect_zone_lights(zone, rng, config.house_lights))
    return lights


def _collect_zone_lights(
    zone: ResidentialZone,
    rng: random.Random,
    house_light_config: HouseLightConfig,
) -> t.List[LightSource]:
    """Collect light sources from all houses in a single residential zone.

    Args:
        zone (ResidentialZone): The zone to process.
        rng (random.Random): Seeded RNG for house light color selection.
        house_light_config (HouseLightConfig): House light configuration
            passed through to each house's get_lights call.

    Returns:
        List[LightSource]: All lights from houses in this zone.
    """
    lights: t.List[LightSource] = []
    for house in zone.houses:
        lights.extend(house.get_lights(rng, house_light_config))
    return lights
