"""Shared render utility functions used across multiple render passes."""

import math
import typing as t

import numpy as np

if t.TYPE_CHECKING:
    from ..camera.camera import Camera
    from ..models.light_source import LightSource


def compute_horizon_y(
    camera: "Camera",
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> int:
    """Approximate the horizon screen y-coordinate.

    Projects a point far ahead on the ground plane (z=0) to find where the
    horizon falls in screen space. Falls back to the image quarter-point if
    the far point is behind the camera.

    Args:
        camera (Camera): The camera.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        int: Screen y-coordinate of the horizon line.
    """
    from ..camera.projection import project_points

    # Move far ahead in the camera's actual horizontal look direction.
    # Using +Y (north) was wrong for south-facing cameras: the point ended up
    # behind the camera, project_points returned invalid, and the fallback
    # height//4 placed the horizon ~25% from the top regardless of camera tilt.
    forward = camera.get_forward()
    horiz = np.array([forward[0], forward[1], 0.0])
    horiz_len = np.linalg.norm(horiz)
    if horiz_len > 1e-10:
        horiz /= horiz_len
    else:
        horiz = np.array([0.0, 1.0, 0.0])  # camera pointing straight down

    far_ahead = camera.position + horiz * camera.far * 0.9
    far_ahead[2] = 0.0
    coords, valid = project_points(
        far_ahead.reshape(1, 3), view_matrix, proj_matrix, (width, height)
    )
    if valid[0]:
        return int(np.clip(coords[0, 1], 0, height - 1))
    return height // 4


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


def apply_brightness_falloff(brightness: float, distance: float, falloff_distance: float) -> float:
    """Apply exponential brightness attenuation based on distance from camera.

    Args:
        brightness (float): Base brightness scalar.
        distance (float): Distance from camera in meters.
        falloff_distance (float): Characteristic decay distance in meters.

    Returns:
        float: Attenuated brightness scalar.
    """
    return brightness * math.exp(-distance / falloff_distance)
