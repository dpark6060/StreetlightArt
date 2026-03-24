"""3D pinhole camera with perspective projection matrices and frustum planes.

The camera uses a standard look-at model:
    - Forward: normalize(target - position)
    - Right:   normalize(forward × world_up)   (world_up = Z axis)
    - Up:      normalize(right × forward)       (camera-local up)

The view matrix transforms world coordinates to camera space where the camera
looks down the -Z axis (OpenGL convention).

The projection matrix follows OpenGL conventions: right-handed, maps depth
to the [-1, 1] NDC range.
"""

import numpy as np


_WORLD_UP: np.ndarray = np.array([0.0, 0.0, 1.0])


class Camera:
    """3D pinhole camera with perspective projection.

    All positions are in local ENU meters. Provides view matrix, projection
    matrix, frustum planes, and mutation helpers for interactive preview.

    Attributes:
        position: (3,) camera position in ENU meters.
        target: (3,) look-at point in ENU meters.
        fov_deg: Horizontal field of view in degrees.
        aspect_ratio: Image width divided by height.
        near: Near clip plane distance in meters.
        far: Far clip plane distance in meters.
    """

    def __init__(
        self,
        position: np.ndarray,
        target: np.ndarray,
        fov_deg: float,
        aspect_ratio: float,
        near: float,
        far: float,
    ) -> None:
        """Initialise a Camera.

        Args:
            position (np.ndarray): (3,) camera position in ENU meters.
            target (np.ndarray): (3,) look-at point in ENU meters.
            fov_deg (float): Horizontal field of view in degrees.
            aspect_ratio (float): Image width / height.
            near (float): Near clip plane distance in meters.
            far (float): Far clip plane distance in meters.
        """
        self.position = np.asarray(position, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.fov_deg = float(fov_deg)
        self.aspect_ratio = float(aspect_ratio)
        self.near = float(near)
        self.far = float(far)

    # ------------------------------------------------------------------
    # Basis vectors
    # ------------------------------------------------------------------

    def get_forward(self) -> np.ndarray:
        """Return the normalized forward vector (position → target).

        Returns:
            np.ndarray: (3,) normalized forward direction.
        """
        return _normalize(self.target - self.position)

    def get_right(self) -> np.ndarray:
        """Return the normalized right vector (forward × world_up).

        Returns:
            np.ndarray: (3,) normalized right direction.
        """
        return _normalize(np.cross(self.get_forward(), _WORLD_UP))

    def get_up(self) -> np.ndarray:
        """Return the normalized camera-local up vector (right × forward).

        Returns:
            np.ndarray: (3,) normalized camera-up direction.
        """
        return _normalize(np.cross(self.get_right(), self.get_forward()))

    # ------------------------------------------------------------------
    # Matrices and frustum
    # ------------------------------------------------------------------

    def get_view_matrix(self) -> np.ndarray:
        """Compute the 4×4 view matrix (world space to camera space).

        Camera space has the camera at the origin looking down -Z, with +Y up
        and +X right (standard OpenGL look-at convention).

        Returns:
            np.ndarray: (4, 4) view matrix.
        """
        forward = self.get_forward()
        right = self.get_right()
        up = self.get_up()
        return _build_view_matrix(forward, right, up, self.position)

    def get_projection_matrix(self) -> np.ndarray:
        """Compute the 4×4 perspective projection matrix.

        Follows OpenGL conventions: right-handed coordinate system, NDC depth
        range [-1, 1].

        Returns:
            np.ndarray: (4, 4) projection matrix.
        """
        fov_rad = np.radians(self.fov_deg)
        f = 1.0 / np.tan(fov_rad / 2.0)
        fy = f * self.aspect_ratio
        return _build_projection_matrix(f, fy, self.near, self.far)

    def get_frustum_planes(self) -> np.ndarray:
        """Compute the 6 frustum plane equations in world space.

        Each plane is represented as (nx, ny, nz, d) where the plane equation
        is: dot(normal, point) + d >= 0 means the point is inside the frustum.

        Plane order: near, far, left, right, bottom, top.

        Returns:
            np.ndarray: (6, 4) array of plane equations [nx, ny, nz, d].
        """
        view_proj = self.get_projection_matrix() @ self.get_view_matrix()
        return _extract_frustum_planes(view_proj)

    # ------------------------------------------------------------------
    # Mutation helpers for interactive preview
    # ------------------------------------------------------------------

    def move(self, delta: np.ndarray) -> None:
        """Translate the camera by delta, keeping the look direction fixed.

        Both position and target are shifted so the camera moves without
        rotating.

        Args:
            delta (np.ndarray): (3,) displacement in ENU meters.
        """
        delta = np.asarray(delta, dtype=float)
        self.position = self.position + delta
        self.target = self.target + delta

    def rotate(self, yaw_deg: float, pitch_deg: float) -> None:
        """Rotate the look direction by yaw and pitch, keeping position fixed.

        Yaw rotates around the world Z-axis; pitch rotates around the
        camera-local right axis.  The target is updated to reflect the new
        look direction at the original distance.

        Args:
            yaw_deg (float): Horizontal rotation in degrees (positive = left).
            pitch_deg (float): Vertical rotation in degrees (positive = up).
        """
        forward = self.get_forward()
        distance = float(np.linalg.norm(self.target - self.position))

        forward = _apply_yaw(forward, yaw_deg)
        right = _normalize(np.cross(forward, _WORLD_UP))
        forward = _apply_pitch(forward, right, pitch_deg)

        self.target = self.position + forward * distance


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_camera(
    position: np.ndarray,
    target: np.ndarray,
    fov_deg: float,
    aspect_ratio: float,
    near: float,
    far: float,
) -> Camera:
    """Construct a Camera from the given parameters.

    Args:
        position (np.ndarray): (3,) camera position in ENU meters.
        target (np.ndarray): (3,) look-at point in ENU meters.
        fov_deg (float): Horizontal field of view in degrees.
        aspect_ratio (float): Image width / height.
        near (float): Near clip plane distance in meters.
        far (float): Far clip plane distance in meters.

    Returns:
        Camera: Configured camera.
    """
    return Camera(
        position=position,
        target=target,
        fov_deg=fov_deg,
        aspect_ratio=aspect_ratio,
        near=near,
        far=far,
    )


# ---------------------------------------------------------------------------
# Backward-compatible free-function wrappers
# ---------------------------------------------------------------------------


def get_forward(camera: Camera) -> np.ndarray:
    """Compute the normalized forward vector of the camera.

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (3,) normalized forward direction.
    """
    return camera.get_forward()


def get_right(camera: Camera) -> np.ndarray:
    """Compute the normalized right vector of the camera.

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (3,) normalized right direction.
    """
    return camera.get_right()


def get_up(camera: Camera) -> np.ndarray:
    """Compute the camera-local up vector (not world up).

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (3,) normalized camera-up direction.
    """
    return camera.get_up()


def get_view_matrix(camera: Camera) -> np.ndarray:
    """Compute the 4×4 view matrix (world space to camera space).

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (4, 4) view matrix.
    """
    return camera.get_view_matrix()


def get_projection_matrix(camera: Camera) -> np.ndarray:
    """Compute the 4×4 perspective projection matrix.

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (4, 4) projection matrix.
    """
    return camera.get_projection_matrix()


def get_frustum_planes(camera: Camera) -> np.ndarray:
    """Compute the 6 frustum plane equations in world space.

    Args:
        camera (Camera): The camera.

    Returns:
        np.ndarray: (6, 4) array of plane equations [nx, ny, nz, d].
    """
    return camera.get_frustum_planes()


# ---------------------------------------------------------------------------
# Private math helpers
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return the normalized (unit length) version of vector v.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Unit vector in the same direction.

    Raises:
        ValueError: If the vector has zero length.
    """
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError(f"Cannot normalize a zero-length vector: {v}")
    return v / norm


def _apply_yaw(forward: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a forward vector around the world Z-axis by yaw_deg.

    Args:
        forward (np.ndarray): (3,) current forward direction.
        yaw_deg (float): Rotation in degrees.

    Returns:
        np.ndarray: (3,) new normalized forward direction.
    """
    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    rot = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y,  cos_y, 0.0],
        [0.0,    0.0,   1.0],
    ])
    return _normalize(rot @ forward)


def _apply_pitch(
    forward: np.ndarray, right: np.ndarray, pitch_deg: float
) -> np.ndarray:
    """Rotate a forward vector around the camera right axis by pitch_deg.

    Uses Rodrigues' rotation formula.

    Args:
        forward (np.ndarray): (3,) current forward direction.
        right (np.ndarray): (3,) normalized camera right axis.
        pitch_deg (float): Rotation in degrees (positive = up).

    Returns:
        np.ndarray: (3,) new normalized forward direction.
    """
    pitch_rad = np.radians(pitch_deg)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    rotated = forward * cos_p + np.cross(right, forward) * sin_p
    return _normalize(rotated)


def _build_view_matrix(
    forward: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """Assemble the 4×4 view matrix from basis vectors and camera position.

    Args:
        forward (np.ndarray): (3,) normalized forward vector.
        right (np.ndarray): (3,) normalized right vector.
        up (np.ndarray): (3,) normalized camera-up vector.
        position (np.ndarray): (3,) camera position in world space.

    Returns:
        np.ndarray: (4, 4) view matrix.
    """
    view = np.eye(4, dtype=float)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, position)
    view[1, 3] = -np.dot(up, position)
    view[2, 3] = np.dot(forward, position)
    return view


def _build_projection_matrix(f: float, fy: float, near: float, far: float) -> np.ndarray:
    """Assemble the 4×4 OpenGL perspective projection matrix.

    Args:
        f (float): Horizontal focal length factor (1/tan(fov_h/2)).
        fy (float): Vertical focal length factor (f * aspect_ratio).
        near (float): Near clip plane distance.
        far (float): Far clip plane distance.

    Returns:
        np.ndarray: (4, 4) perspective projection matrix.
    """
    proj = np.zeros((4, 4), dtype=float)
    proj[0, 0] = f
    proj[1, 1] = fy
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    return proj


def _extract_frustum_planes(view_proj: np.ndarray) -> np.ndarray:
    """Extract the 6 frustum planes from a combined view-projection matrix.

    Uses the Gribb-Hartmann method to extract planes from the combined
    view-projection matrix. Each plane is normalized.

    Args:
        view_proj (np.ndarray): (4, 4) combined view-projection matrix.

    Returns:
        np.ndarray: (6, 4) array of normalized plane equations [nx, ny, nz, d].
    """
    m = view_proj
    planes = np.array([
        m[3] + m[2],   # near
        m[3] - m[2],   # far
        m[3] + m[0],   # left
        m[3] - m[0],   # right
        m[3] + m[1],   # bottom
        m[3] - m[1],   # top
    ])
    norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
    return planes / norms
