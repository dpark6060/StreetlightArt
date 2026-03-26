"""Builds an ENU-space Camera from DevmodeCameraConfig."""

import numpy as np

from ..camera.camera import Camera
from ..config.schema import DevmodeCameraConfig


def build_devmode_camera(
    config: DevmodeCameraConfig,
    width: int,
    height: int,
) -> Camera:
    """Construct a Camera from a DevmodeCameraConfig and render dimensions.

    All position values in the config are already in ENU meters, so no
    coordinate transformation is required.

    Args:
        config (DevmodeCameraConfig): ENU-space camera parameters.
        width (int): Render surface width in pixels.
        height (int): Render surface height in pixels.

    Returns:
        Camera: Configured camera for the devmode synthetic scene.
    """
    position = np.array([config.pos_x, config.pos_y, config.pos_z])
    target = np.array([config.target_x, config.target_y, config.target_z])
    return Camera(
        position=position,
        target=target,
        fov_deg=config.fov,
        aspect_ratio=width / height,
        near=config.near,
        far=config.far,
    )
