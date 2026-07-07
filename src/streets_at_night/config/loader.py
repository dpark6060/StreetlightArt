"""TOML config loader — reads scene_config.toml and returns a SceneConfig.

The load path is: read TOML -> deep-merge over the built-in DEFAULTS ->
reflect the merged dict onto the schema dataclasses (``config.serde``). There
is no hand-written per-section reader; ``schema.py`` is the single source of
truth for the config surface.
"""

import copy
import tomllib
from pathlib import Path

from .defaults import DEFAULTS
from .schema import DevmodeCameraConfig, DevmodeConfig, SceneConfig
from .serde import dataclass_from_dict


def load_config(toml_path: Path) -> SceneConfig:
    """Load a SceneConfig from a TOML file overlaid onto built-in defaults.

    If the file does not exist, the default values are used as-is.
    Quality preset selection is handled by the caller via
    config.quality_presets["low"] (or "medium" / "high").

    Args:
        toml_path (Path): Path to the TOML config file.

    Returns:
        SceneConfig: Populated config dataclass.
    """
    merged = _load_merged(toml_path)
    return dataclass_from_dict(SceneConfig, merged)


def _load_merged(toml_path: Path) -> dict:
    """Merge user TOML values onto the built-in defaults.

    Args:
        toml_path (Path): Path to the TOML config file.

    Returns:
        dict: Merged configuration dictionary.
    """
    merged = copy.deepcopy(DEFAULTS)
    if not toml_path.exists():
        return merged
    with open(toml_path, "rb") as fh:
        user_data = tomllib.load(fh)
    _deep_merge(merged, user_data)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override values into base in-place.

    Note: only dict values are merged recursively. List values (e.g. fixture
    pool entries, heat-haze layers) replace the base wholesale — this is
    load-bearing user behavior, not an oversight.

    Args:
        base (dict): The base dictionary to mutate.
        override (dict): Values to overlay on top of base.
    """
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def build_devmode_camera_config(data: dict) -> DevmodeCameraConfig:
    """Build a DevmodeCameraConfig from the devmode camera section.

    Thin wrapper over the reflection walker, kept as a named entry point for
    the devmode scene builders that construct a camera config directly from
    ``DEFAULTS["devmode"]["camera"]``.

    Args:
        data (dict): Camera section from the devmode config.

    Returns:
        DevmodeCameraConfig: Populated devmode camera config.
    """
    return dataclass_from_dict(DevmodeCameraConfig, data)


def _build_devmode_config(data: dict) -> DevmodeConfig:
    """Build a DevmodeConfig from the devmode section of the config.

    Args:
        data (dict): Devmode section from the config.

    Returns:
        DevmodeConfig: Populated devmode config.
    """
    return dataclass_from_dict(DevmodeConfig, data)
