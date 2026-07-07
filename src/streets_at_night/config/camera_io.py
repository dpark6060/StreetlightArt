"""Read/write camera and bounds sections of the scene TOML config.

Extracted from ``main.py`` so both the CLI (preview ``C`` key) and the UI tabs
can persist camera and bounds edits back to the config file. Only these two
sections are touched; every other section in the file is preserved on write.

Import direction: this module imports from ``config.schema`` and
``camera.coordinates`` only. It must never import ``main``.
"""

import tomllib
import typing as t
from pathlib import Path

import tomli_w

from ..camera.coordinates import enu_to_wgs84, meters_to_feet
from .schema import BoundsConfig

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def build_camera_toml_dict(scene: "Scene") -> t.Dict[str, float]:
    """Convert the current camera ENU state to a TOML-ready dict.

    Args:
        scene (Scene): The current world scene with transformer set.

    Returns:
        dict[str, float]: Camera fields ready to write into the TOML [camera] section.
    """
    camera = scene.camera
    transformer = scene.transformer

    pos = camera.position
    lat, lon, alt_m = enu_to_wgs84(transformer, pos[0], pos[1], pos[2])
    alt_ft = meters_to_feet(alt_m)

    tgt = camera.target
    target_lat, target_lon, target_alt_m = enu_to_wgs84(
        transformer, tgt[0], tgt[1], tgt[2]
    )
    target_alt_ft = meters_to_feet(target_alt_m)

    return {
        "lat": lat,
        "lon": lon,
        "altitude_ft": alt_ft,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "target_altitude_ft": target_alt_ft,
        "fov": camera.fov_deg,
    }


def write_camera_to_toml(toml_path: Path, camera_data: t.Dict[str, float]) -> None:
    """Read the existing TOML file, replace [camera] values, and write it back.

    Creates the [camera] table if the file lacks one (a UI-created minimal
    config might not have it).

    Args:
        toml_path (Path): Path to the TOML config file.
        camera_data (dict[str, float]): Updated camera field values.
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    data.setdefault("camera", {}).update(camera_data)

    with open(toml_path, "wb") as f:
        tomli_w.dump(data, f)


def save_camera_position(scene: "Scene", toml_path: Path) -> None:
    """Save current camera position and target to the TOML config file.

    Converts camera ENU position and target back to WGS84 lat/lon and altitude
    in feet, then updates the [camera] section in the TOML file. All other
    sections are preserved.

    Prints an error if the scene has no transformer (camera cannot be converted).

    Args:
        scene (Scene): The current world scene (must have transformer set).
        toml_path (Path): Path to the TOML config file to update.
    """
    if scene.transformer is None:
        print("Cannot save camera: scene has no coordinate transformer.")
        return

    camera_data = build_camera_toml_dict(scene)
    write_camera_to_toml(toml_path, camera_data)
    print(f"Camera position saved to {toml_path}")


def write_bounds_to_toml(toml_path: Path, bounds: BoundsConfig) -> None:
    """Read the TOML file, replace the [bounds] section values, write it back.

    Mirrors write_camera_to_toml: tomllib.load, update data["bounds"] with
    north/south/east/west floats, tomli_w.dump. Creates the [bounds] table if
    the file lacks one. Raises FileNotFoundError if toml_path does not exist
    (callers handle the no-config save-as flow).

    Args:
        toml_path (Path): Path to the TOML config file.
        bounds (BoundsConfig): Geographic bounds to persist.
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    data.setdefault("bounds", {}).update(
        {
            "north": float(bounds.north),
            "south": float(bounds.south),
            "east": float(bounds.east),
            "west": float(bounds.west),
        }
    )

    with open(toml_path, "wb") as f:
        tomli_w.dump(data, f)
