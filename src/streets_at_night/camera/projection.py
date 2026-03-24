"""Vectorized 3D-to-2D projection, near-plane line clipping, and frustum culling.

Performance note: project_points processes all points as a single numpy batch.
Line clipping and frustum culling operate per-segment since segment counts are
typically in the tens of thousands — Python-level loops are acceptable there.

Coordinate conventions:
    - World space: ENU (X=East, Y=North, Z=Up), meters.
    - Camera space: camera at origin, looking down -Z, +Y up, +X right.
    - NDC: x ∈ [-1, 1], y ∈ [-1, 1] after perspective divide.
    - Screen: (0,0) at top-left, (width-1, height-1) at bottom-right.
"""

import typing as t

import numpy as np


def project_points(
    points_3d: np.ndarray,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    screen_size: t.Tuple[int, int],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Project 3D world points to 2D screen coordinates.

    Applies the view matrix then the projection matrix, performs the perspective
    divide, and maps NDC coordinates to pixel coordinates.

    Args:
        points_3d (np.ndarray): Shape (N, 3) world coordinates in ENU meters.
        view_matrix (np.ndarray): (4, 4) world-to-camera matrix.
        proj_matrix (np.ndarray): (4, 4) perspective projection matrix.
        screen_size (tuple[int, int]): (width, height) of the output image.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - screen_coords: (N, 2) pixel coordinates (float).
            - valid_mask: (N,) bool — True if the point is in front of camera
              and within NDC screen bounds (x, y ∈ [-1, 1]).
    """
    width, height = screen_size
    homogeneous = _to_homogeneous(points_3d)
    view_coords = _apply_matrix(view_matrix, homogeneous)
    clip_coords = _apply_matrix(proj_matrix, view_coords)
    ndc, w = _perspective_divide(clip_coords)
    screen_coords = _ndc_to_screen(ndc, width, height)
    valid_mask = _compute_valid_mask(view_coords, w, ndc)
    return screen_coords, valid_mask


def clip_line_to_near_plane(
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    view_matrix: np.ndarray,
    near: float,
) -> t.Optional[t.Tuple[np.ndarray, np.ndarray]]:
    """Clip a line segment to the camera near plane.

    Returns the clipped segment in world coordinates, or None if the entire
    segment is behind the camera (camera-space z >= -near for both points).

    If one endpoint is behind the near plane, the segment is clipped at the
    intersection with the plane by linear interpolation in world space.

    Args:
        p1_world (np.ndarray): (3,) first endpoint in world coordinates.
        p2_world (np.ndarray): (3,) second endpoint in world coordinates.
        view_matrix (np.ndarray): (4, 4) world-to-camera matrix.
        near (float): Near clip plane distance in meters (positive value).

    Returns:
        tuple[np.ndarray, np.ndarray] | None:
            Clipped (p1, p2) world coordinates, or None if fully behind.
    """
    z1 = _camera_z(p1_world, view_matrix)
    z2 = _camera_z(p2_world, view_matrix)

    p1_in_front = z1 < -near
    p2_in_front = z2 < -near

    if not p1_in_front and not p2_in_front:
        return None
    if p1_in_front and p2_in_front:
        return (p1_world, p2_world)
    return _clip_one_endpoint(p1_world, p2_world, z1, z2, near, p1_in_front)


def frustum_cull_segments(
    segments: t.List[t.Tuple[np.ndarray, np.ndarray]],
    view_matrix: np.ndarray,
    near: float,
    far: float,
) -> t.List[t.Tuple[np.ndarray, np.ndarray]]:
    """Filter and clip road segments to the view frustum.

    Removes segments where both endpoints are beyond the far plane or both are
    behind the near plane. Clips segments that cross the near plane.

    Note: No clipping is performed against the far plane — segments partially
    beyond far are kept (endpoints beyond far will be culled at projection time).

    Args:
        segments (list): List of (p1, p2) world-space endpoint pairs (np.ndarray).
        view_matrix (np.ndarray): (4, 4) world-to-camera matrix.
        near (float): Near clip plane distance in meters.
        far (float): Far clip plane distance in meters.

    Returns:
        list: Filtered and near-clipped (p1, p2) world coordinate pairs.
    """
    result = []
    for p1, p2 in segments:
        visible = _process_segment(p1, p2, view_matrix, near, far)
        if visible is not None:
            result.append(visible)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a w=1 column to convert (N, 3) points to (N, 4) homogeneous.

    Args:
        points (np.ndarray): Shape (N, 3).

    Returns:
        np.ndarray: Shape (N, 4) with w=1 appended.
    """
    ones = np.ones((points.shape[0], 1), dtype=float)
    return np.hstack([points, ones])


def _apply_matrix(matrix: np.ndarray, homogeneous: np.ndarray) -> np.ndarray:
    """Transform (N, 4) homogeneous points by a (4, 4) matrix.

    Args:
        matrix (np.ndarray): (4, 4) transformation matrix.
        homogeneous (np.ndarray): (N, 4) homogeneous points.

    Returns:
        np.ndarray: (N, 4) transformed points.
    """
    return (matrix @ homogeneous.T).T


def _perspective_divide(clip_coords: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
    """Divide clip coordinates by w to produce NDC coordinates.

    Args:
        clip_coords (np.ndarray): (N, 4) clip space coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - ndc: (N, 3) normalized device coordinates.
            - w: (N,) w values before divide.
    """
    w = clip_coords[:, 3:4]
    safe_w = np.where(np.abs(w) < 1e-10, 1e-10, w)
    ndc = clip_coords[:, :3] / safe_w
    return ndc, clip_coords[:, 3]


def _ndc_to_screen(ndc: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert NDC coordinates to pixel screen coordinates.

    NDC x ∈ [-1, 1] maps to screen x ∈ [0, width].
    NDC y ∈ [-1, 1] maps to screen y ∈ [height, 0] (y is flipped).

    Args:
        ndc (np.ndarray): (N, 3) normalized device coordinates.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.

    Returns:
        np.ndarray: (N, 2) screen pixel coordinates.
    """
    screen_x = (ndc[:, 0] + 1.0) * width / 2.0
    screen_y = (1.0 - ndc[:, 1]) * height / 2.0
    return np.column_stack([screen_x, screen_y])


def _compute_valid_mask(
    view_coords: np.ndarray,
    w: np.ndarray,
    ndc: np.ndarray,
) -> np.ndarray:
    """Compute a boolean mask indicating which projected points are valid.

    A point is valid if it is in front of the camera (view_z < 0 and w > 0)
    and its NDC x and y coordinates fall within [-1, 1] (i.e. on screen).

    Args:
        view_coords (np.ndarray): (N, 4) camera-space coordinates.
        w (np.ndarray): (N,) clip-space w values (positive = in front).
        ndc (np.ndarray): (N, 3) normalized device coordinates after perspective divide.

    Returns:
        np.ndarray: (N,) boolean mask.
    """
    view_z = view_coords[:, 2]
    in_front = view_z < 0.0
    w_positive = w > 0.0
    in_screen_x = (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
    in_screen_y = (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)
    return in_front & w_positive & in_screen_x & in_screen_y


def _camera_z(point_world: np.ndarray, view_matrix: np.ndarray) -> float:
    """Compute the camera-space z coordinate of a single world point.

    Args:
        point_world (np.ndarray): (3,) world coordinates.
        view_matrix (np.ndarray): (4, 4) view matrix.

    Returns:
        float: Camera-space z coordinate.
    """
    homogeneous = np.append(point_world, 1.0)
    return float(view_matrix[2] @ homogeneous)


def _clip_one_endpoint(
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    z1: float,
    z2: float,
    near: float,
    p1_in_front: bool,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Clip the behind-camera endpoint to the near plane boundary.

    Interpolates in world space to find where the segment crosses the near
    plane (camera z == -near).

    Args:
        p1_world (np.ndarray): (3,) first endpoint in world space.
        p2_world (np.ndarray): (3,) second endpoint in world space.
        z1 (float): Camera-space z of p1.
        z2 (float): Camera-space z of p2.
        near (float): Near plane distance (positive).
        p1_in_front (bool): True if p1 is in front of the near plane.

    Returns:
        tuple[np.ndarray, np.ndarray]: Clipped (p1, p2) world coordinates.
    """
    near_z = -near
    t_val = (near_z - z1) / (z2 - z1)
    intersection = p1_world + t_val * (p2_world - p1_world)
    if p1_in_front:
        return (p1_world, intersection)
    return (intersection, p2_world)


def _process_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    view_matrix: np.ndarray,
    near: float,
    far: float,
) -> t.Optional[t.Tuple[np.ndarray, np.ndarray]]:
    """Apply culling and near-plane clipping to a single segment.

    Rejects the segment if both endpoints are entirely beyond the far plane.
    Clips to the near plane if one endpoint is behind it.

    Args:
        p1 (np.ndarray): (3,) first endpoint in world space.
        p2 (np.ndarray): (3,) second endpoint in world space.
        view_matrix (np.ndarray): (4, 4) view matrix.
        near (float): Near plane distance in meters.
        far (float): Far plane distance in meters.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Visible segment, or None.
    """
    z1 = _camera_z(p1, view_matrix)
    z2 = _camera_z(p2, view_matrix)

    both_beyond_far = (z1 < -far) and (z2 < -far)
    if both_beyond_far:
        return None

    return clip_line_to_near_plane(p1, p2, view_matrix, near)
