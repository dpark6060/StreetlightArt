"""Enumerates every user-editable light kind the scene produces.

The catalog drives the Devmode tab's section list. Each ``LightKind.target``
references the live config dataclass instance, so editing a kind's fields
mutates the working config directly — that identity is what makes "edit a
value, switch to another kind, switch back" preserve the change with no extra
bookkeeping. Rebuild the catalog whenever the working config object is replaced
(a config-bar load), so targets point into the new config.
"""

import typing as t
from dataclasses import dataclass

from ..config.schema import FixturePoolEntry, SceneConfig

# Streetlight road-type keys in display order. pretty() derives the label.
_STREETLIGHT_ROAD_KEYS: t.Tuple[str, ...] = (
    "highway",
    "major_road",
    "collector",
    "residential",
    "service",
)


@dataclass
class LightKind:
    """One user-editable light category.

    Attributes:
        label: Display name, e.g. "Streetlight: Highway (pool 1)".
        target: The config dataclass instance holding the editable fields.
        field_prefix: When set, only dataclass fields whose name starts with
            this prefix are shown, with the prefix stripped from the display
            label. None shows all fields.
    """

    label: str
    target: t.Any
    field_prefix: t.Optional[str] = None


def _pretty(road_key: str) -> str:
    """Convert a road-type config key to a display name.

    Args:
        road_key (str): Attribute name on StreetLightConfig (e.g. "major_road").

    Returns:
        str: Title-cased display name (e.g. "Major Road").
    """
    return road_key.replace("_", " ").title()


def _build_streetlight_kinds(config: SceneConfig) -> t.List[LightKind]:
    """Build one LightKind per profiled fixture pool entry, per road type.

    Pool index is counted only over entries that have a profile, so a road type
    whose first entry is a no-lights entry still starts at "(pool 0)".

    Args:
        config (SceneConfig): Working config.

    Returns:
        list[LightKind]: Streetlight kinds in road-type / pool order.
    """
    kinds: t.List[LightKind] = []
    for road_key in _STREETLIGHT_ROAD_KEYS:
        entries: t.List[FixturePoolEntry] = getattr(config.lights.streetlights, road_key)
        pool_index = 0
        for entry in entries:
            if entry.profile is None:
                continue
            kinds.append(
                LightKind(
                    label=f"Streetlight: {_pretty(road_key)} (pool {pool_index})",
                    target=entry.profile,
                    field_prefix=None,
                )
            )
            pool_index += 1
    return kinds


def build_light_catalog(config: SceneConfig) -> t.List[LightKind]:
    """Enumerate every editable light kind in the config, in display order.

    Args:
        config (SceneConfig): Working config whose light sections are enumerated.

    Returns:
        list[LightKind]: All editable light kinds, targets pointing into config.
    """
    kinds: t.List[LightKind] = []
    kinds.extend(_build_streetlight_kinds(config))

    cars = config.lights.cars
    kinds.append(LightKind("Cars: Headlights", cars, "headlight_"))
    kinds.append(LightKind("Cars: Taillights", cars, "taillight_"))
    kinds.append(LightKind("Cars: Density", cars, "density_"))
    kinds.append(LightKind("Cars: Separation", cars, "separation_"))

    kinds.append(LightKind("House Lights", config.lights.house_lights, None))

    commercial = config.lights.commercial
    kinds.append(LightKind("Commercial: Signs", commercial, "sign_"))
    kinds.append(LightKind("Commercial: Parking", commercial, "parking_"))
    kinds.append(LightKind("Commercial: Storefronts", commercial, "storefront_"))

    # Lit-window controls, shown only when the config has a [buildings] section.
    # The window_ prefix scopes the section to the emissive-window fields (lit
    # fraction, brightness, color temperature, saturation, sizing, glow).
    if config.buildings is not None:
        kinds.append(LightKind("Buildings: Windows", config.buildings, "window_"))

    # City-wide color_temp_shift_k master lever, which no per-pool entry covers.
    kinds.append(LightKind("Streetlights: Global", config.lights.streetlights, "color_temp_shift"))

    return kinds
