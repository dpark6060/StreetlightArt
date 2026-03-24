"""Shared render utility functions used across multiple render passes."""

import math
import typing as t

import numpy as np

if t.TYPE_CHECKING:
    from ..camera.camera import Camera
    from ..models.light_source import LightSource


def extract_positions(lights: t.List["LightSource"]) -> np.ndarray:
    """Stack light positions into a (N, 3) array for batch projection.

    Args:
        lights (list[LightSource]): Light sources to extract positions from.

    Returns:
        np.ndarray: (N, 3) float array of ENU positions.
    """
    return np.array([light.position for light in lights], dtype=float)


def camera_distance(light_pos: np.ndarray, cam_pos: np.ndarray) -> float:
    """Compute Euclidean distance from the camera to a light source.

    Args:
        light_pos (np.ndarray): (3,) light position in ENU.
        cam_pos (np.ndarray): (3,) camera position in ENU.

    Returns:
        float: Distance in meters, clamped to at least 1.0.
    """
    return max(1.0, float(np.linalg.norm(light_pos - cam_pos)))


def compute_focal_length(camera: "Camera", screen_width: int) -> float:
    """Derive a focal length in pixels from camera FOV and screen width.

    Args:
        camera (Camera): The active camera.
        screen_width (int): Screen width in pixels.

    Returns:
        float: Focal length in pixels.
    """
    fov_rad = math.radians(camera.fov_deg)
    return screen_width / (2.0 * math.tan(fov_rad / 2.0))
