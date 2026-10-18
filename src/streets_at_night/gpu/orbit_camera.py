"""Orbit camera for the GPU tuning scene: rotate, pan, and zoom around a ground pivot."""

import math
from dataclasses import dataclass, field

import numpy as np

from ..camera.camera import Camera

# Degrees of rotation per pixel of mouse drag.
_ROTATE_DEG_PER_PX: float = 0.3

# Pitch is kept off the horizon and off straight-down, where yaw degenerates.
_PITCH_MIN_DEG: float = 2.0
_PITCH_MAX_DEG: float = 89.0

# Distance multiplier per scroll notch, and the allowed zoom range in meters.
_ZOOM_STEP: float = 1.1
_DISTANCE_MIN_M: float = 2.0
_DISTANCE_MAX_M: float = 1500.0

# Default view: from the south, low enough that the first pole's light,
# the person, and the car are all in frame.
_DEFAULT_PIVOT: tuple = (8.0, 2.0, 0.0)
_DEFAULT_YAW_DEG: float = -70.0
_DEFAULT_PITCH_DEG: float = 15.0
_DEFAULT_DISTANCE_M: float = 45.0


@dataclass
class OrbitCamera:
    """Camera that orbits a pivot point on the ground plane.

    Attributes:
        pivot: (3,) ENU point the camera looks at and orbits around.
        yaw_deg: Heading of the camera *from* the pivot, counterclockwise from
            east (0 = camera east of the pivot looking west).
        pitch_deg: Elevation angle of the camera above the ground plane.
        distance: Camera distance from the pivot in meters.
        fov_deg: Horizontal field of view in degrees.
        near: Near clip distance in meters.
        far: Far clip distance in meters.
    """

    pivot: np.ndarray = field(default_factory=lambda: np.array(_DEFAULT_PIVOT))
    yaw_deg: float = _DEFAULT_YAW_DEG
    pitch_deg: float = _DEFAULT_PITCH_DEG
    distance: float = _DEFAULT_DISTANCE_M
    fov_deg: float = 60.0
    near: float = 0.1
    far: float = 2000.0

    def eye_position(self) -> np.ndarray:
        """Return the camera position in ENU meters.

        Returns:
            np.ndarray: (3,) camera position.
        """
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        offset = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ])
        return self.pivot + offset * self.distance

    def rotate(self, dx_px: float, dy_px: float) -> None:
        """Orbit around the pivot from a mouse drag.

        Dragging right swings the camera to the right around the pivot;
        dragging down raises the camera.

        Args:
            dx_px (float): Horizontal drag in pixels.
            dy_px (float): Vertical drag in pixels (screen y grows downward).
        """
        self.yaw_deg = (self.yaw_deg - dx_px * _ROTATE_DEG_PER_PX) % 360.0
        pitch = self.pitch_deg + dy_px * _ROTATE_DEG_PER_PX
        self.pitch_deg = min(max(pitch, _PITCH_MIN_DEG), _PITCH_MAX_DEG)

    def pan(self, dx_px: float, dy_px: float, viewport_width_px: int) -> None:
        """Slide the pivot across the ground so the scene follows the mouse.

        Movement is scaled so a point at the pivot's depth tracks the cursor.

        Args:
            dx_px (float): Horizontal drag in pixels.
            dy_px (float): Vertical drag in pixels (screen y grows downward).
            viewport_width_px (int): Viewport width in pixels.
        """
        meters_per_px = (
            2.0 * self.distance * math.tan(math.radians(self.fov_deg) / 2.0)
            / max(viewport_width_px, 1)
        )
        yaw = math.radians(self.yaw_deg)
        # Ground-plane right and away-from-camera directions.
        right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        away = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])
        self.pivot = self.pivot + (-dx_px * right + dy_px * away) * meters_per_px

    def zoom(self, notches: float) -> None:
        """Move toward (positive) or away from (negative) the pivot.

        Args:
            notches (float): Scroll amount; each notch scales distance by 10%.
        """
        distance = self.distance / (_ZOOM_STEP ** notches)
        self.distance = min(max(distance, _DISTANCE_MIN_M), _DISTANCE_MAX_M)

    def reset(self) -> None:
        """Restore the default view of the person, car, and first pole."""
        self.pivot = np.array(_DEFAULT_PIVOT)
        self.yaw_deg = _DEFAULT_YAW_DEG
        self.pitch_deg = _DEFAULT_PITCH_DEG
        self.distance = _DEFAULT_DISTANCE_M

    def to_camera(self, aspect_ratio: float) -> Camera:
        """Build the renderer's Camera for the current orbit state.

        Args:
            aspect_ratio (float): Viewport width / height.

        Returns:
            Camera: Camera at the orbit position looking at the pivot.
        """
        return Camera(
            position=self.eye_position(),
            target=self.pivot.copy(),
            fov_deg=self.fov_deg,
            aspect_ratio=aspect_ratio,
            near=self.near,
            far=self.far,
        )
