"""Building mass render pass — extruded building walls and roofs.

Draws opaque wall and roof polygons for each visible building footprint into
the backend's sRGB substrate (via ``backend.draw_polygon``). This is purely
geometry — no emissive light is added here. Lit windows (an additive,
linear-buffer emission pass) are a separate later phase and are not part of
this module.
"""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import clip_polygon_to_near_plane, project_points
from ..utils.colors import hex_to_rgb
from .backend import RenderBackend
from .raycast import GroundHit, visible_ground_aabb

if t.TYPE_CHECKING:
    from ..models.building import Building
    from ..scene.scene import Scene

# Fixed pseudo-moon direction (2D, XY) used to shade wall faces relative to
# each other. Not a real light source — purely a cosmetic face-separation cue.
_MOON_DIRECTION_2D: np.ndarray = np.array([0.7071067811865476, 0.7071067811865476])

# Minimum on-screen roof bounding-box size (in either dimension) required to
# bother drawing a building at all.
_MIN_ROOF_PIXELS: float = 2.0

# Minimum number of vertices required to draw a projected face polygon.
_MIN_POLYGON_VERTICES: int = 3


def render_buildings(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    gbuffer: t.Optional[GroundHit] = None,
) -> t.Dict[str, t.Any]:
    """Render visible building volumes as extruded wall and roof polygons.

    Queries the scene's building spatial index against the camera's visible
    ground footprint, drops buildings beyond the configured max distance,
    sorts the remainder back-to-front by footprint-centroid camera distance,
    then draws each building's camera-facing walls and roof.

    Args:
        scene (Scene): The world scene with the building spatial index built.
        camera (Camera): Camera used for culling and projection.
        backend (RenderBackend): Target render surface.
        gbuffer (GroundHit | None): Shared per-pixel ground raycast buffer.
            Not used for mass rendering in this phase; accepted so this pass
            has the same signature as the other export passes and can share
            the pipeline's cached G-buffer.

    Returns:
        dict[str, Any]: {"buildings_drawn": int} — number of buildings for
        which at least the roof was drawn.
    """
    building_config = getattr(scene.config, "buildings", None)
    if building_config is None or not building_config.enabled:
        return {"buildings_drawn": 0}

    candidates = _get_visible_buildings(scene, camera, building_config.building_max_distance_m)
    if not candidates:
        return {"buildings_drawn": 0}

    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()
    wall_color = hex_to_rgb(building_config.wall_color)
    roof_color = hex_to_rgb(building_config.roof_color)

    drawn = 0
    for building, _distance in candidates:
        was_drawn = _draw_building(
            building,
            camera,
            view_matrix,
            proj_matrix,
            camera.near,
            width,
            height,
            wall_color,
            roof_color,
            building_config.face_shade_strength,
            backend,
        )
        if was_drawn:
            drawn += 1

    return {"buildings_drawn": drawn}


def _get_visible_buildings(
    scene: "Scene",
    camera: Camera,
    max_distance_m: float,
) -> t.List[t.Tuple["Building", float]]:
    """Query, distance-cull, and back-to-front sort visible buildings.

    Args:
        scene (Scene): The world scene with the building spatial index built.
        camera (Camera): Camera used for the visible-ground query and
            distance culling.
        max_distance_m (float): Buildings farther than this from the camera
            (in the XY plane) are dropped.

    Returns:
        list[tuple[Building, float]]: (building, distance) pairs sorted
        farthest-first (painter's algorithm draw order).
    """
    min_x, min_y, max_x, max_y = visible_ground_aabb(camera)
    candidates = scene.get_buildings_in_rect(min_x, min_y, max_x, max_y)

    cam_xy = camera.position[:2]
    with_distance = [
        (building, _get_centroid_distance(building, cam_xy)) for building in candidates
    ]
    in_range = [pair for pair in with_distance if pair[1] <= max_distance_m]
    in_range.sort(key=lambda pair: pair[1], reverse=True)
    return in_range


def _get_centroid_distance(building: "Building", cam_xy: np.ndarray) -> float:
    """Compute the XY distance from the camera to a building footprint centroid.

    Args:
        building (Building): The building.
        cam_xy (np.ndarray): (2,) camera position projected to the XY plane.

    Returns:
        float: Euclidean distance in meters.
    """
    centroid = building.footprint.mean(axis=0)
    return float(np.linalg.norm(centroid - cam_xy))


def _draw_building(
    building: "Building",
    camera: Camera,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    near: float,
    width: int,
    height: int,
    wall_color: t.Tuple[int, int, int],
    roof_color: t.Tuple[int, int, int],
    shade_strength: float,
    backend: RenderBackend,
) -> bool:
    """Project and draw one building's walls and roof.

    Draws walls first, then the roof, matching the camera-above-roofline
    convention used throughout the plan. Skips the whole building if its
    projected roof bounding box is smaller than the tiny-building pixel
    threshold in both dimensions.

    Args:
        building (Building): The building to draw.
        camera (Camera): Camera used for backface culling.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        near (float): Camera near plane distance.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        wall_color (tuple[int, int, int]): Base wall RGB color.
        roof_color (tuple[int, int, int]): Roof RGB color.
        shade_strength (float): Per-face shading strength (0-1).
        backend (RenderBackend): Target render surface.

    Returns:
        bool: True if the roof (and possibly walls) were drawn.
    """
    roof_vertices_3d = _build_roof_vertices(building)
    roof_screen = _project_face(roof_vertices_3d, view_matrix, proj_matrix, near, width, height)
    if roof_screen is None:
        return False
    if _is_below_pixel_threshold(roof_screen):
        return False

    _draw_walls(
        building, camera, view_matrix, proj_matrix, near, width, height,
        wall_color, shade_strength, backend,
    )
    backend.draw_polygon(roof_screen, roof_color)
    return True


def _build_roof_vertices(building: "Building") -> t.List[np.ndarray]:
    """Build the roof ring's world-space vertices at z = building height.

    Args:
        building (Building): The building.

    Returns:
        list[np.ndarray]: (3,) world-space vertices, no closing duplicate.
    """
    return [
        np.array([x, y, building.height])
        for x, y in building.footprint
    ]


def _draw_walls(
    building: "Building",
    camera: Camera,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    near: float,
    width: int,
    height: int,
    wall_color: t.Tuple[int, int, int],
    shade_strength: float,
    backend: RenderBackend,
) -> int:
    """Backface-cull, project, and draw each camera-facing wall quad.

    Args:
        building (Building): The building whose walls to draw.
        camera (Camera): Camera used to determine which walls face it.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        near (float): Camera near plane distance.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        wall_color (tuple[int, int, int]): Base wall RGB color.
        shade_strength (float): Per-face shading strength (0-1).
        backend (RenderBackend): Target render surface.

    Returns:
        int: Number of wall quads drawn.
    """
    footprint = building.footprint
    n = len(footprint)
    cam_xy = camera.position[:2]

    drawn = 0
    for i in range(n):
        p0 = footprint[i]
        p1 = footprint[(i + 1) % n]
        normal = _get_outward_normal(p0, p1)
        if not _is_wall_facing_camera(normal, p0, p1, cam_xy):
            continue

        wall_vertices_3d = _build_wall_vertices(p0, p1, building.height)
        wall_screen = _project_face(wall_vertices_3d, view_matrix, proj_matrix, near, width, height)
        if wall_screen is None:
            continue

        shade = _get_face_shade(normal, shade_strength)
        backend.draw_polygon(wall_screen, _apply_shade(wall_color, shade))
        drawn += 1

    return drawn


def _build_wall_vertices(p0: np.ndarray, p1: np.ndarray, height: float) -> t.List[np.ndarray]:
    """Build a wall quad's world-space vertices for one footprint edge.

    Args:
        p0 (np.ndarray): (2,) edge start point.
        p1 (np.ndarray): (2,) edge end point.
        height (float): Building height in meters.

    Returns:
        list[np.ndarray]: Four (3,) vertices: bottom-p0, bottom-p1, top-p1, top-p0.
    """
    return [
        np.array([p0[0], p0[1], 0.0]),
        np.array([p1[0], p1[1], 0.0]),
        np.array([p1[0], p1[1], height]),
        np.array([p0[0], p0[1], height]),
    ]


def _get_outward_normal(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Compute the outward 2D unit normal of a CCW footprint edge.

    For a counter-clockwise ring, the outward normal of edge p0->p1 is the
    edge vector rotated -90 degrees (clockwise): normal = (dy, -dx).

    Args:
        p0 (np.ndarray): (2,) edge start point.
        p1 (np.ndarray): (2,) edge end point.

    Returns:
        np.ndarray: (2,) outward unit normal, or the zero vector if the edge
        is degenerate.
    """
    edge = p1 - p0
    normal = np.array([edge[1], -edge[0]])
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-9:
        return normal
    return normal / magnitude


def _is_wall_facing_camera(
    normal: np.ndarray, p0: np.ndarray, p1: np.ndarray, cam_xy: np.ndarray
) -> bool:
    """Check whether a wall faces the camera (i.e. is not backface-culled).

    A wall faces the camera when its outward normal points back toward the
    camera rather than away from it — equivalently, when
    dot(normal, wall_center - camera_xy) <= 0.

    Args:
        normal (np.ndarray): (2,) outward unit normal of the wall.
        p0 (np.ndarray): (2,) edge start point.
        p1 (np.ndarray): (2,) edge end point.
        cam_xy (np.ndarray): (2,) camera position projected to the XY plane.

    Returns:
        bool: True if the wall should be drawn (faces the camera).
    """
    wall_center = (p0 + p1) / 2.0
    return bool(np.dot(normal, wall_center - cam_xy) <= 0.0)


def _get_face_shade(normal: np.ndarray, shade_strength: float) -> float:
    """Compute a face brightness multiplier from its outward normal.

    Faces pointing toward the fixed pseudo-moon direction are brightened
    relative to faces pointing away, giving adjacent walls visually distinct
    shading without a real light source.

    Args:
        normal (np.ndarray): (2,) outward unit normal of the face.
        shade_strength (float): Strength of the directional component (0-1).

    Returns:
        float: Brightness multiplier, in [1 - shade_strength, 1.0].
    """
    alignment = max(0.0, float(np.dot(normal, _MOON_DIRECTION_2D)))
    return (1.0 - shade_strength) + shade_strength * alignment


def _apply_shade(
    color: t.Tuple[int, int, int], shade: float
) -> t.Tuple[int, int, int]:
    """Scale an RGB color by a brightness multiplier, clamped to [0, 255].

    Args:
        color (tuple[int, int, int]): Base RGB color.
        shade (float): Brightness multiplier.

    Returns:
        tuple[int, int, int]: Shaded RGB color.
    """
    return tuple(int(max(0.0, min(255.0, channel * shade))) for channel in color)


def _project_face(
    vertices_3d: t.List[np.ndarray],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    near: float,
    width: int,
    height: int,
) -> t.Optional[t.List[t.Tuple[float, float]]]:
    """Near-clip and project a face's world-space vertices to screen space.

    Every clipped vertex is in front of the camera by construction, so all
    projected vertices are returned — including ones that land off-screen.
    Both PIL and pygame clip polygon primitives to the surface themselves.

    Args:
        vertices_3d (list[np.ndarray]): World-space face vertices, no
            closing duplicate.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        near (float): Camera near plane distance.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.

    Returns:
        list[tuple[float, float]] | None: Screen-space vertices, or None if
        the face is fully behind the near plane or degenerate after clipping.
    """
    clipped = clip_polygon_to_near_plane(vertices_3d, view_matrix, near)
    if len(clipped) < _MIN_POLYGON_VERTICES:
        return None

    points_3d = np.stack(clipped, axis=0)
    screen_coords, _ = project_points(points_3d, view_matrix, proj_matrix, (width, height))
    return [(float(x), float(y)) for x, y in screen_coords]


def _is_below_pixel_threshold(screen_vertices: t.List[t.Tuple[float, float]]) -> bool:
    """Check whether a projected polygon's bounding box is too small to draw.

    Args:
        screen_vertices (list[tuple[float, float]]): Projected 2D vertices.

    Returns:
        bool: True if the bounding box is smaller than the minimum roof
        pixel threshold in both width and height.
    """
    xs = [x for x, _y in screen_vertices]
    ys = [y for _x, y in screen_vertices]
    bbox_width = max(xs) - min(xs)
    bbox_height = max(ys) - min(ys)
    return bbox_width < _MIN_ROOF_PIXELS and bbox_height < _MIN_ROOF_PIXELS
