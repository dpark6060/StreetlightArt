"""Serializes SceneConfig back to a TOML-compatible dict and writes to file.

Both functions delegate to the reflection walker in ``config.serde``; the
dataclass definitions in ``schema.py`` are the single source of truth for the
serialized shape, so there is no hand-written per-section serializer to keep
in sync with the loader.

A config that layers over a shared look file (``config.look``) is written back
as two files: place sections to ``path``, look sections to the look file (see
``config.layers.split_for_save``). Every save path in the app goes through
``write_config``, so devmode, the tune window, and Save As all route the same way.
"""

from pathlib import Path

import tomli_w

from ..config.layers import look_path_for, read_toml, split_for_save
from ..config.schema import SceneConfig
from ..config.serde import dataclass_to_dict


def write_config(config: SceneConfig, path: Path) -> None:
    """Serialize a SceneConfig to TOML, splitting place and look if layered.

    Args:
        config (SceneConfig): The config to serialize.
        path (Path): Destination (place) file path. Created or overwritten.
            A layered config's look path resolves relative to this file.
    """
    data = config_to_dict(config)
    if config.look is None:
        _dump(data, path)
        return

    look_path = look_path_for(path, config.look)
    place, look = split_for_save(data, read_toml(path), read_toml(look_path))
    _dump(place, path)
    _dump(look, look_path)


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


def _dump(data: dict, path: Path) -> None:
    """Write a dict as TOML, creating the parent directory if needed.

    Args:
        data (dict): TOML-serializable dict.
        path (Path): Destination file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        tomli_w.dump(data, fh)
