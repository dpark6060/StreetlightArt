"""Road render pass — project and draw classified road segments onto a backend."""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import frustum_cull_segments, project_points_full
from ..config.schema import RoadStyleConfig
from ..models.road_segment import RoadSegment
from ..models.road_types import RoadType
from ..utils.colors import hex_to_rgb
from .backend import RenderBackend

if t.TYPE_CHECKING:
    from ..scene.scene import Scene

StyleTable = t.Dict[RoadType, t.Tuple[t.Tuple[int, int, int], int]]

# Hardcoded fallback — only used if scene.config.roads is somehow missing.
_FALLBACK_PREVIEW_STYLE: StyleTable = {
    RoadType.HIGHWAY:      ((255, 215,   0), 4),
    RoadType.MAJOR_ROAD:   ((255, 107,  53), 3),
    RoadType.COLLECTOR:    ((255, 255, 255), 2),
    RoadType.RESIDENTIAL:  ((136, 136, 136), 1),
    RoadType.SERVICE:      (( 85,  85,  85), 1),
    RoadType.MINOR:        ((100, 100, 120), 1),
}

_FALLBACK_EXPORT_STYLE: StyleTable = {
    RoadType.HIGHWAY:      (( 20,  18,  16), 4),
    RoadType.MAJOR_ROAD:   (( 18,  16,  16), 3),
    RoadType.COLLECTOR:    (( 16,  14,  14), 2),
    RoadType.RESIDENTIAL:  (( 14,  12,  12), 1),
    RoadType.SERVICE:      (( 12,  10,  10), 1),
    RoadType.MINOR:        (( 10,   9,   8), 1),
}

# Distance (meters) at which the base line width applies
_REFERENCE_DISTANCE_M: float = 5000.0


def _build_style_table(style_config: RoadStyleConfig) -> StyleTable:
    """Build a RoadType → (RGB, width) table from a RoadStyleConfig.

    Args:
        style_config (RoadStyleConfig): Per-road-type color and width config.

    Returns:
        StyleTable: Mapping from RoadType to (color_tuple, width).
    """
    return {
        RoadType.HIGHWAY:    (hex_to_rgb(style_config.highway.color), style_config.highway.width),
        RoadType.MAJOR_ROAD: (hex_to_rgb(style_config.major_road.color), style_config.major_road.width),
        RoadType.COLLECTOR:  (hex_to_rgb(style_config.collector.color), style_config.collector.width),
        RoadType.RESIDENTIAL:(hex_to_rgb(style_config.residential.color), style_config.residential.width),
        RoadType.SERVICE:    (hex_to_rgb(style_config.service.color), style_config.service.width),
        RoadType.MINOR:      (hex_to_rgb(style_config.minor.color), style_config.minor.width),
    }


def render_roads_from_scene(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> int:
    """Render visible road segments using preview colors from config.

    Queries the scene's spatial index with the camera's far-plane footprint,
    sorts candidates back-to-front by camera depth, then projects and draws
    each segment using the preview color palette from scene config.

    Args:
        scene (Scene): The world scene with spatial index built.
        camera (Camera): Camera used for projection and culling.
        backend (RenderBackend): Target render surface.

    Returns:
        int: Number of road segments drawn.
    """
    roads_config = getattr(scene.config, "roads", None)
    if roads_config is not None:
        style_table = _build_style_table(roads_config.preview)
    else:
        style_table = _FALLBACK_PREVIEW_STYLE

    return _render_roads(scene, camera, backend, style_table)


def render_roads_export(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> int:
    """Render visible road segments using dark export colors from config.

    Same projection and culling as the preview path, but uses the dark
    asphalt color palette intended for final export renders. Roads are
    nearly invisible without illumination, creating a realistic night look.

    Args:
        scene (Scene): The world scene with spatial index built.
        camera (Camera): Camera used for projection and culling.
        backend (RenderBackend): Target render surface.

    Returns:
        int: Number of road segments drawn.
    """
    roads_config = getattr(scene.config, "roads", None)
    if roads_config is not None:
        style_table = _build_style_table(roads_config.export)
    else:
        style_table = _FALLBACK_EXPORT_STYLE

    return _render_roads(scene, camera, backend, style_table)


def _render_roads(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    style_table: StyleTable,
) -> int:
    """Render visible road segments with the given style table.

    Args:
        scene (Scene): The world scene with spatial index built.
        camera (Camera): Camera used for projection and culling.
        backend (RenderBackend): Target render surface.
        style_table (StyleTable): RoadType → (color, width) mapping.

    Returns:
        int: Number of road segments drawn.
    """
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()

    candidates = _query_camera_frustum(scene, camera)
    candidates.sort(key=lambda s: _get_segment_depth(s, view_matrix), reverse=True)

    drawn = 0
    for segment in candidates:
        was_drawn = _draw_segment(
            segment.geometry,
            segment.road_type,
            style_table,
            view_matrix,
            proj_matrix,
            camera.near,
            camera.far,
            (width, height),
            backend,
        )
        if was_drawn:
            drawn += 1

    return drawn


def _get_segment_depth(segment: RoadSegment, view_matrix: np.ndarray) -> float:
    """Return the camera-space depth of a road segment's midpoint.

    A larger positive value means the segment is farther from the camera.
    Used to sort segments back-to-front for correct depth ordering.

    Args:
        segment (RoadSegment): The road segment.
        view_matrix (np.ndarray): (4, 4) view matrix.

    Returns:
        float: Camera-space depth (positive = farther from camera).
    """
    midpoint = segment.geometry[len(segment.geometry) // 2]
    return -_camera_depth(midpoint, view_matrix)


def _query_camera_frustum(scene: "Scene", camera: Camera) -> t.List[t.Any]:
    """Query the scene for road segments within the camera's view range.

    Uses a conservative square AABB around the camera position up to the
    far plane distance.

    Args:
        scene (Scene): The world scene.
        camera (Camera): The camera.

    Returns:
        list[RoadSegment]: Candidate segments in the camera's view area.
    """
    cam_x = camera.position[0]
    cam_y = camera.position[1]
    far = camera.far
    return scene.get_roads_in_rect(
        cam_x - far, cam_y - far, cam_x + far, cam_y + far
    )


def _draw_segment(
    segment_geom: np.ndarray,
    road_type: RoadType,
    style_table: t.Dict[RoadType, t.Tuple[t.Tuple[int, int, int], int]],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    near: float,
    far: float,
    screen_size: t.Tuple[int, int],
    backend: RenderBackend,
) -> bool:
    """Project and draw one road geometry array using the given style table.

    Applies frustum culling, projects visible edges, and draws them onto the
    backend. Returns True if at least one edge was drawn.

    Args:
        segment_geom (np.ndarray): (N, 3) ENU coordinates of the road.
        road_type (RoadType): Classification of this road.
        style_table (dict): Map from RoadType to (color, base_width).
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        near (float): Camera near plane distance.
        far (float): Camera far plane distance.
        screen_size (tuple[int, int]): (width, height) in pixels.
        backend (RenderBackend): Target render surface.

    Returns:
        bool: True if at least one edge was drawn.
    """
    if segment_geom.shape[0] < 2:
        return False

    style = style_table.get(road_type)
    if style is None:
        return False

    color, base_width = style
    width, height = screen_size
    pairs = _build_segment_pairs(segment_geom)
    visible = frustum_cull_segments(pairs, view_matrix, near, far)

    if not visible:
        return False

    screen_coords, in_front_mask = _project_segment_pairs(
        visible, view_matrix, proj_matrix, (width, height)
    )
    return _draw_visible_edges(
        visible, screen_coords, in_front_mask, color, base_width,
        view_matrix, width, height, backend,
    )


def _project_segment_pairs(
    pairs: t.List[t.Tuple[np.ndarray, np.ndarray]],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    screen_size: t.Tuple[int, int],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Project all endpoints of a list of edge pairs in a single batch call.

    Stacks every pair's two endpoints into one (2K, 3) array so the
    perspective transform runs once for the whole segment instead of once
    per edge.

    Args:
        pairs (list[tuple[np.ndarray, np.ndarray]]): Near-clipped world-space
            edge endpoint pairs.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        screen_size (tuple[int, int]): (width, height) in pixels.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - screen_coords: (2K, 2) pixel coordinates, endpoints interleaved
              as [p1_0, p2_0, p1_1, p2_1, ...].
            - in_front_mask: (2K,) bool — True where the point is in front of
              the camera (on-screen bounds are not required for lines).
    """
    points = np.array([point for pair in pairs for point in pair])
    screen_coords, in_front_mask, _ = project_points_full(
        points, view_matrix, proj_matrix, screen_size
    )
    return screen_coords, in_front_mask


def _draw_visible_edges(
    pairs: t.List[t.Tuple[np.ndarray, np.ndarray]],
    screen_coords: np.ndarray,
    in_front_mask: np.ndarray,
    color: t.Tuple[int, int, int],
    base_width: int,
    view_matrix: np.ndarray,
    width: int,
    height: int,
    backend: RenderBackend,
) -> bool:
    """Draw each edge whose endpoints are both in front of the camera.

    Off-screen endpoints are still drawn — both PIL and pygame clip line
    primitives to the surface, so passing off-screen pixel coordinates is
    safe. A cheap same-side-of-screen reject skips edges that plainly cannot
    intersect the visible area, avoiding pathological huge-coordinate draws.

    Args:
        pairs (list[tuple[np.ndarray, np.ndarray]]): World-space edge pairs,
            in the same order used to build screen_coords.
        screen_coords (np.ndarray): (2K, 2) projected pixel coordinates.
        in_front_mask (np.ndarray): (2K,) bool, True if in front of camera.
        color (tuple[int, int, int]): RGB line color.
        base_width (int): Base line width at the reference distance.
        view_matrix (np.ndarray): (4, 4) view matrix.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        backend (RenderBackend): Target render surface.

    Returns:
        bool: True if at least one edge was drawn.
    """
    drawn = False
    for i, (p1, p2) in enumerate(pairs):
        idx1, idx2 = 2 * i, 2 * i + 1
        if not (in_front_mask[idx1] and in_front_mask[idx2]):
            continue

        sx1, sy1 = screen_coords[idx1]
        sx2, sy2 = screen_coords[idx2]
        if _is_off_same_side(sx1, sy1, sx2, sy2, width, height):
            continue

        line_width = _compute_line_width(p1, p2, view_matrix, base_width)
        backend.draw_line(
            (int(sx1), int(sy1)), (int(sx2), int(sy2)), color, line_width
        )
        drawn = True
    return drawn


def _is_off_same_side(
    sx1: float, sy1: float, sx2: float, sy2: float, width: int, height: int
) -> bool:
    """Check whether both edge endpoints lie off-screen on the same side.

    A cheap 4-comparison reject: if both endpoints are left of, right of,
    above, or below the screen rect, the edge cannot cross the visible area.

    Args:
        sx1 (float): First endpoint screen x.
        sy1 (float): First endpoint screen y.
        sx2 (float): Second endpoint screen x.
        sy2 (float): Second endpoint screen y.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.

    Returns:
        bool: True if the edge is entirely off-screen on one side.
    """
    if sx1 < 0 and sx2 < 0:
        return True
    if sx1 > width and sx2 > width:
        return True
    if sy1 < 0 and sy2 < 0:
        return True
    if sy1 > height and sy2 > height:
        return True
    return False


def _build_segment_pairs(
    points: np.ndarray,
) -> t.List[t.Tuple[np.ndarray, np.ndarray]]:
    """Convert a (N, 3) point array to a list of consecutive (p1, p2) pairs.

    Args:
        points (np.ndarray): (N, 3) ENU coordinate array.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Edge pairs.
    """
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _compute_line_width(
    p1: np.ndarray,
    p2: np.ndarray,
    view_matrix: np.ndarray,
    base_width: int,
) -> int:
    """Compute perspective-scaled line width for an edge.

    Uses the average absolute camera-space z depth of the two endpoints.
    Closer segments get thicker lines.

    Args:
        p1 (np.ndarray): (3,) first endpoint in world space.
        p2 (np.ndarray): (3,) second endpoint in world space.
        view_matrix (np.ndarray): (4, 4) view matrix.
        base_width (int): Width at the reference distance.

    Returns:
        int: Clamped line width in pixels (minimum 1).
    """
    z1 = abs(_camera_depth(p1, view_matrix))
    z2 = abs(_camera_depth(p2, view_matrix))
    avg_depth = (z1 + z2) / 2.0
    if avg_depth < 1.0:
        return base_width
    scaled = base_width * _REFERENCE_DISTANCE_M / avg_depth
    return max(1, int(scaled))


def _camera_depth(point_world: np.ndarray, view_matrix: np.ndarray) -> float:
    """Get the camera-space z coordinate (depth) of a world point.

    Args:
        point_world (np.ndarray): (3,) world coordinates.
        view_matrix (np.ndarray): (4, 4) view matrix.

    Returns:
        float: Camera-space z (negative = in front of camera).
    """
    h = np.append(point_world, 1.0)
    return float(view_matrix[2] @ h)
