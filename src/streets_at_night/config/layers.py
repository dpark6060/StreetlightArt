"""Two-layer configs: a per-city place file over a shared look file.

A place file may name a look file with a top-level ``look = "looks/night.toml"``
(path relative to the place file). Loading merges DEFAULTS <- look <- place,
so a city can still override any single look value.

Saving splits the merged config back apart: place-owned sections go to the
place file, everything else to the look file, except values the place file
already overrides, which stay in the place file so a city's override never
leaks into the shared look.
"""

import copy
import tomllib
import typing as t
from pathlib import Path

from .defaults import DEFAULTS

# Top-level keys that describe where and how a city is framed. Everything
# else (lights, atmosphere, grade, buildings, roads, zones, ...) is the look.
PLACE_KEYS: t.FrozenSet[str] = frozenset(
    {"look", "region_name", "bounds", "camera", "devmode", "seed"}
)

LOOK_KEY: str = "look"

_KeyPath = t.Tuple[str, ...]


def deep_merge(base: dict, override: dict) -> None:
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
            deep_merge(base[key], value)
        else:
            base[key] = value


def read_toml(path: Path) -> dict:
    """Read a TOML file, or return {} if it does not exist.

    Args:
        path (Path): TOML file path.

    Returns:
        dict: Parsed TOML.
    """
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def look_path_for(place_path: Path, look: str) -> Path:
    """Resolve a place file's ``look`` value to a path.

    Args:
        place_path (Path): The place file.
        look (str): The ``look`` value, relative to the place file.

    Returns:
        Path: The look file path.
    """
    return place_path.parent / look


def load_layered(place_path: Path) -> dict:
    """Return DEFAULTS merged with the look file (if any) and the place file.

    Args:
        place_path (Path): The config file passed on the command line.

    Returns:
        dict: The merged configuration dictionary.

    Raises:
        FileNotFoundError: If the place file names a look file that is missing.
        ValueError: If the look file itself names a look (no nesting).
    """
    merged = copy.deepcopy(DEFAULTS)
    place = read_toml(place_path)
    look = place.get(LOOK_KEY)
    if look is not None:
        look_path = look_path_for(place_path, look)
        if not look_path.exists():
            raise FileNotFoundError(f"look file not found: {look_path} (from {place_path})")
        look_data = read_toml(look_path)
        if LOOK_KEY in look_data:
            raise ValueError(f"look file {look_path} names its own look; looks can't nest")
        deep_merge(merged, look_data)
    deep_merge(merged, place)
    return merged


def split_for_save(full: dict, place_raw: dict, look_raw: dict) -> t.Tuple[dict, dict]:
    """Split a full config dict into place and look file contents.

    Args:
        full (dict): The whole config, as serialized from SceneConfig.
        place_raw (dict): The place file's current contents (its overrides).
        look_raw (dict): The look file's current contents.

    Returns:
        tuple[dict, dict]: (place file dict, look file dict).
    """
    place = {k: copy.deepcopy(v) for k, v in full.items() if k in PLACE_KEYS}
    look = {k: copy.deepcopy(v) for k, v in full.items() if k not in PLACE_KEYS}

    # What the look alone resolves to, so an overridden value can be restored.
    look_resolved = copy.deepcopy(DEFAULTS)
    deep_merge(look_resolved, look_raw)

    for path in _leaf_paths(place_raw):
        if path[0] in PLACE_KEYS or not _has_path(full, path):
            continue
        _set_path(place, path, copy.deepcopy(_get_path(full, path)))
        if _has_path(look_resolved, path):
            _set_path(look, path, copy.deepcopy(_get_path(look_resolved, path)))
    return place, look


def _leaf_paths(data: dict, prefix: _KeyPath = ()) -> t.Iterator[_KeyPath]:
    """Yield key paths to every non-dict value (lists count as leaves).

    Args:
        data (dict): Nested dict.
        prefix (tuple[str, ...]): Path so far.

    Yields:
        tuple[str, ...]: Key path to a leaf value.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _leaf_paths(value, (*prefix, key))
        else:
            yield (*prefix, key)


def _has_path(data: dict, path: _KeyPath) -> bool:
    """True if every key along path exists."""
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _get_path(data: dict, path: _KeyPath) -> t.Any:
    """Return the value at path (must exist)."""
    node = data
    for key in path:
        node = node[key]
    return node


def _set_path(data: dict, path: _KeyPath, value: t.Any) -> None:
    """Set the value at path, creating intermediate dicts."""
    node = data
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
