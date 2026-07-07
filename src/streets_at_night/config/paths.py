"""Shared cache-path and coordinate-origin helpers.

Both ``main.py`` and the devmode modules need to resolve the effective cache
directory and compute the ENU coordinate origin from a ``SceneConfig``. This
module is the single source of truth for both so neither side has to import
from the other (which previously required a lazy in-function import in
``devmode/cmd_devmode.py`` to dodge a circular import).
"""

import typing as t
from pathlib import Path

from .schema import SceneConfig


def resolve_cache_dir(base_dir: Path, config: SceneConfig) -> Path:
    """Return the effective cache directory, applying region_name if set.

    Args:
        base_dir (Path): Base cache directory from CLI (default: cache/).
        config (SceneConfig): Loaded scene config.

    Returns:
        Path: base_dir / region_name if region_name is set, else base_dir.
    """
    if config.region_name:
        return base_dir / config.region_name
    return base_dir


def compute_origin(config: SceneConfig) -> t.Tuple[float, float]:
    """Compute the local coordinate origin as the midpoint between camera and target.

    Args:
        config (SceneConfig): Scene configuration with camera section.

    Returns:
        tuple[float, float]: (origin_lat, origin_lon) in WGS84 degrees.
    """
    origin_lat = (config.camera.lat + config.camera.target_lat) / 2.0
    origin_lon = (config.camera.lon + config.camera.target_lon) / 2.0
    return origin_lat, origin_lon
