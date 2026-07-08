"""Disk cache for built Scene content, keyed on scene-determining inputs.

A Scene's geometry, zones, buildings, and lights are a pure function of the OSM
data (bbox), the ENU origin, the RNG seed, and the ``[lights]``/``[zones]``/
``[buildings]`` config sections. The ENU origin is the camera/target lat-lon
midpoint (see ``config.paths.compute_origin``), so it captures the only part of
the camera the scene content depends on — camera altitude, FOV, near/far,
quality, atmosphere, grade, stars, and heat-haze do NOT change scene content and
can be tuned freely without invalidating the cache. That exposure/look tuning
loop is exactly what this cache accelerates: it turns a ~20-35 s rebuild into a
~5 s unpickle.

Correctness rule (this repo's audit calls under-keyed caches its single biggest
bug source): the key MUST include everything the content is a function of.
Missing a field serves a stale scene silently. The key here covers the schema
version, seed, region name, bounds, the ENU origin, the full lights/zones/
buildings config subtrees, and the signatures (size + mtime) of the OSM .gpkg
files the scene was built from — so re-downloaded OSM data invalidates it too.
"""

import hashlib
import json
import pickle
import typing as t
from pathlib import Path

from ..config.paths import compute_origin
from ..config.schema import SceneConfig
from ..config.serde import dataclass_to_dict

# Bump when the cached content's structure changes (new Scene fields, changed
# model shapes) so old pickles are ignored instead of unpickled wrongly.
SCENE_CACHE_VERSION = 1

# Keep at most this many scene pickles (each ~a few hundred MB) — newest kept,
# older evicted — so tuning across many configs can't grow the cache unbounded.
_MAX_CACHED_SCENES = 5

_SUBDIR = "scene_cache"


def compute_scene_cache_key(
    config: SceneConfig,
    bbox: t.Tuple[float, float, float, float],
    cache_dir: Path,
) -> str:
    """Compute the cache key for the scene a config + bbox would build.

    Args:
        config (SceneConfig): The working scene configuration.
        bbox (tuple): (north, south, east, west) bounds used for OSM loading.
        cache_dir (Path): Cache directory (holds the OSM .gpkg files hashed
            into the key).

    Returns:
        str: Hex sha256 digest uniquely identifying the scene content.
    """
    origin_lat, origin_lon = compute_origin(config)
    payload = {
        "version": SCENE_CACHE_VERSION,
        "seed": config.seed,
        "region_name": config.region_name,
        "bounds": [
            config.bounds.north,
            config.bounds.south,
            config.bounds.east,
            config.bounds.west,
        ],
        # Origin is the only camera-derived quantity the content depends on.
        "origin": [round(origin_lat, 9), round(origin_lon, 9)],
        "lights": dataclass_to_dict(config.lights),
        "zones": dataclass_to_dict(config.zones),
        "buildings": (
            dataclass_to_dict(config.buildings) if config.buildings is not None else None
        ),
        "osm": _osm_signatures(cache_dir),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_scene_content(cache_dir: Path, key: str) -> t.Optional[dict]:
    """Load cached scene content for a key, or None if absent/unreadable.

    A corrupt or incompatible pickle returns None (rather than raising) so the
    caller falls back to a fresh build.

    Args:
        cache_dir (Path): Cache directory.
        key (str): Scene cache key from ``compute_scene_cache_key``.

    Returns:
        dict | None: The cached content dict, or None on miss/failure.
    """
    path = cache_dir / _SUBDIR / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_scene_content(cache_dir: Path, key: str, content: dict) -> None:
    """Persist scene content atomically and evict old entries.

    Args:
        cache_dir (Path): Cache directory.
        key (str): Scene cache key from ``compute_scene_cache_key``.
        content (dict): The scene content to pickle.
    """
    directory = cache_dir / _SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.pkl"
    tmp = directory / f"{key}.pkl.tmp"

    with open(tmp, "wb") as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)  # atomic swap so a crash mid-write can't leave a partial

    _evict_old_scenes(directory)


def _evict_old_scenes(directory: Path) -> None:
    """Delete all but the newest ``_MAX_CACHED_SCENES`` scene pickles.

    Args:
        directory (Path): The scene_cache directory.
    """
    pickles = sorted(
        directory.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for old in pickles[_MAX_CACHED_SCENES:]:
        old.unlink(missing_ok=True)


def _osm_signatures(cache_dir: Path) -> t.List[t.List[t.Any]]:
    """Signature of every OSM .gpkg in the cache: [name, size, mtime].

    Including these means re-downloaded OSM data (new file size/mtime)
    invalidates the scene cache even at the same bbox.

    Args:
        cache_dir (Path): Cache directory whose ``osm/`` subdir holds the gpkgs.

    Returns:
        list[list]: Sorted [name, size, mtime_int] triples.
    """
    osm_dir = cache_dir / "osm"
    if not osm_dir.is_dir():
        return []
    sigs = []
    for gpkg in sorted(osm_dir.glob("*.gpkg")):
        st = gpkg.stat()
        sigs.append([gpkg.name, st.st_size, int(st.st_mtime)])
    return sigs
