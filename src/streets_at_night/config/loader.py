"""TOML config loader — reads scene_config.toml and returns a SceneConfig.

The load path is: read TOML (plus its shared look file, if it names one; see
``config.layers``) -> deep-merge over the built-in DEFAULTS -> reflect the
merged dict onto the schema dataclasses (``config.serde``). There
is no hand-written per-section reader; ``schema.py`` is the single source of
truth for the config surface.
"""

from pathlib import Path

from .layers import load_layered
from .schema import DevmodeCameraConfig, DevmodeConfig, SceneConfig
from .serde import dataclass_from_dict


def load_config(toml_path: Path) -> SceneConfig:
    """Load a SceneConfig from a TOML file overlaid onto built-in defaults.

    If the file does not exist, the default values are used as-is. If it
    names a ``look`` file, that file is merged in underneath it.
    Quality preset selection is handled by the caller via
    config.quality_presets["low"] (or "medium" / "high").

    Args:
        toml_path (Path): Path to the TOML config file.

    Returns:
        SceneConfig: Populated config dataclass.
    """
    merged = load_layered(toml_path)
    return dataclass_from_dict(SceneConfig, merged)


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
