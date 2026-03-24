"""Road render pass — project and draw classified road segments onto a backend."""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import frustum_cull_segments, project_points
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

    for p1, p2 in visible:
        _project_and_draw_edge(
            p1, p2, color, base_width,
            view_matrix, proj_matrix, backend, width, height,
        )
    return True


def _project_and_draw_edge(
    p1: np.ndarray,
    p2: np.ndarray,
    color: t.Tuple[int, int, int],
    base_width: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    backend: RenderBackend,
    width: int,
    height: int,
) -> None:
    """Project two 3D endpoints and draw a line between them.

    Skips drawing if either projected point falls outside the valid screen area.
    Scales line width by distance for a perspective effect.

    Args:
        p1 (np.ndarray): (3,) first endpoint in world ENU.
        p2 (np.ndarray): (3,) second endpoint in world ENU.
        color (tuple[int, int, int]): RGB line color.
        base_width (int): Base line width at the reference distance.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        backend (RenderBackend): Target render surface.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    """
    pair = np.array([p1, p2])
    screen_coords, valid_mask = project_points(
        pair, view_matrix, proj_matrix, (width, height)
    )
    if not (valid_mask[0] and valid_mask[1]):
        return

    line_width = _compute_line_width(p1, p2, view_matrix, base_width)
    sx1, sy1 = int(screen_coords[0, 0]), int(screen_coords[0, 1])
    sx2, sy2 = int(screen_coords[1, 0]), int(screen_coords[1, 1])
    backend.draw_line((sx1, sy1), (sx2, sy2), color, line_width)


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
