"""Serializes SceneConfig back to a TOML-compatible dict and writes to file.

Both functions delegate to the reflection walker in ``config.serde``; the
dataclass definitions in ``schema.py`` are the single source of truth for the
serialized shape, so there is no hand-written per-section serializer to keep
in sync with the loader.
"""

from pathlib import Path

import tomli_w

from ..config.schema import SceneConfig
from ..config.serde import dataclass_to_dict


def write_config(config: SceneConfig, path: Path) -> None:
    """Serialize a SceneConfig to a TOML file.

    Args:
        config (SceneConfig): The config to serialize.
        path (Path): Destination file path. Will be created or overwritten.
    """
    data = config_to_dict(config)
    with open(path, "wb") as fh:
        tomli_w.dump(data, fh)


def config_to_dict(config: SceneConfig) -> dict:
    """Convert a SceneConfig to a nested TOML-compatible dict.

    Key structure and naming exactly match what loader.py expects on re-load,
    so that ``write_config(config, path); load_config(path)`` round-trips.

    Args:
        config (SceneConfig): The config to convert.

    Returns:
        dict: Nested dict ready for TOML serialization.
    """
    return dataclass_to_dict(config)
