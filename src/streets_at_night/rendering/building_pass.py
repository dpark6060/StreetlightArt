"""Building mass render pass — extruded building walls, roofs, and windows.

Draws opaque wall and roof polygons for each visible building footprint into
the backend's sRGB substrate (via ``backend.draw_polygon``) — this part is
purely geometry, no emissive light. Lit windows are a separate additive step
within the same pass: they reuse the camera-facing wall quads computed for
the mass render, derive a deterministic per-wall window grid, light-test and
color each cell with a seeded RNG, and accumulate emission straight into a
float32 linear buffer via ``backend.add_linear`` — no bloom, no ground pool,
just an emissive rectangle the compositor's tone map rolls off naturally.
"""

import random
import typing as t
from dataclasses import dataclass

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import clip_polygon_to_near_plane, project_points, project_points_full
from ..config.schema import BuildingConfig
from ..utils.colors import hex_to_rgb, kelvin_to_rgb
from .backend import RenderBackend
from .color_space import srgb255_to_linear
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

# How far outward along the wall normal a window's corners are inset, so the
# window rectangle doesn't sit exactly coplanar with the wall fill.
_WINDOW_NORMAL_INSET_M: float = 0.05

# Beyond this camera distance, windows always use the single-pixel sub-pixel
# accumulation path, regardless of their projected screen size.
_WINDOW_SUBPIXEL_DISTANCE_M: float = 2000.0

# A projected window quad whose screen-space bounding box is smaller than
# this in both dimensions is accumulated as a single sub-pixel point instead
# of a rasterized rectangle.
_WINDOW_SUBPIXEL_PIXEL_THRESHOLD: float = 1.0

# Building kinds that use the residential window lit fraction. Synthetic-lot
# house footprints are tagged "house" by building_builder, so they're covered.
_RESIDENTIAL_KINDS: t.FrozenSet[str] = frozenset({"house", "detached", "residential"})


@dataclass
class FacingWall:
    """One camera-facing wall edge of a building, precomputed once per frame.

    Shared by the wall-fill draw step and the window placement step so the
    backface-cull math (an expensive-ish per-edge computation across
    potentially thousands of buildings) runs exactly once per wall.

    Attributes:
        edge_index: Index of the footprint edge this wall was built from.
        p0: (2,) edge start point in ENU meters.
        p1: (2,) edge end point in ENU meters.
        normal: (2,) outward unit normal of the wall.
    """

    edge_index: int
    p0: np.ndarray
    p1: np.ndarray
    normal: np.ndarray


@dataclass
class WindowCell:
    """One window's world-space rectangle and its reproducible RNG key.

    Attributes:
        corners: Four (3,) world-space points on the wall plane, in order
            bottom-left, bottom-right, top-right, top-left.
        rng_key: ``(window_seed, edge_index, row, col)`` — reproducible
            regardless of iteration order or culling.
    """

    corners: t.List[np.ndarray]
    rng_key: t.Tuple[int, int, int, int]


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
    window_buffer: t.Optional[np.ndarray] = None
    windows_drawn = 0
    for building, distance in candidates:
        facing_walls = _get_facing_walls(building, camera)
        was_drawn = _draw_building(
            building,
            facing_walls,
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
        if not was_drawn:
            continue
        drawn += 1

        if distance > building_config.window_max_distance_m:
            continue
        if window_buffer is None:
            window_buffer = np.zeros((height, width, 3), dtype=np.float32)
        windows_drawn += _accumulate_building_windows(
            building, facing_walls, distance, view_matrix, proj_matrix, width, height,
            building_config, window_buffer,
        )

    if window_buffer is not None and windows_drawn > 0:
        window_buffer = _apply_window_glow(window_buffer, building_config.window_glow_px)
        backend.add_linear(window_buffer)

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
    if not candidates:
        return []

    # Vectorize the centroid-distance cull and sort over all candidates rather
    # than computing footprint.mean + norm per building in Python. Centroids are
    # precomputed on each Building, so this is one array gather + one norm.
    cam_xy = camera.position[:2]
    centroids = np.array([building.centroid for building in candidates])
    distances = np.linalg.norm(centroids - cam_xy, axis=1)

    in_range = np.nonzero(distances <= max_distance_m)[0]
    # Stable descending sort (farthest first) so ties keep candidate order,
    # matching Python's stable list.sort(reverse=True) painter ordering.
    order = in_range[np.argsort(-distances[in_range], kind="stable")]
    return [(candidates[i], float(distances[i])) for i in order]


def _draw_building(
    building: "Building",
    facing_walls: t.List[FacingWall],
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
        facing_walls (list[FacingWall]): This building's camera-facing wall
            edges, already backface-culled.
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
    faces_3d = _build_building_faces(building, facing_walls)
    faces_screen = _project_faces(faces_3d, view_matrix, proj_matrix, near, width, height)

    roof_screen = faces_screen[0]
    if roof_screen is None:
        return False
    if _is_below_pixel_threshold(roof_screen):
        return False

    for wall, wall_screen in zip(facing_walls, faces_screen[1:]):
        if wall_screen is None:
            continue
        shade = _get_face_shade(wall.normal, shade_strength)
        backend.draw_polygon(wall_screen, _apply_shade(wall_color, shade))

    backend.draw_polygon(roof_screen, roof_color)
    return True


def _build_building_faces(
    building: "Building", facing_walls: t.List[FacingWall]
) -> t.List[np.ndarray]:
    """Build world-space vertex arrays for a building's roof and facing walls.

    The first entry is the roof ring (footprint at z = height); the rest are
    the camera-facing wall quads in ``facing_walls`` order. Vertex order within
    each face matches the earlier per-face builders, so the projected polygons
    fill identically — only the arrays are built in one place so the whole
    building can be projected in a single batch.

    Args:
        building (Building): The building whose faces to build.
        facing_walls (list[FacingWall]): This building's camera-facing wall
            edges, already backface-culled.

    Returns:
        list[np.ndarray]: [(N, 3) roof, (4, 3) wall, ...] world-space faces.
    """
    footprint = building.footprint
    height = building.height
    n = len(footprint)

    roof = np.empty((n, 3), dtype=float)
    roof[:, 0] = footprint[:, 0]
    roof[:, 1] = footprint[:, 1]
    roof[:, 2] = height
    faces: t.List[np.ndarray] = [roof]

    for wall in facing_walls:
        p0, p1 = wall.p0, wall.p1
        faces.append(
            np.array([
                [p0[0], p0[1], 0.0],
                [p1[0], p1[1], 0.0],
                [p1[0], p1[1], height],
                [p0[0], p0[1], height],
            ])
        )
    return faces


def _project_faces(
    faces_3d: t.List[np.ndarray],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    near: float,
    width: int,
    height: int,
) -> t.List[t.Optional[t.List[t.Tuple[float, float]]]]:
    """Project every face of one building, batching when nothing crosses near.

    Fast path — when every vertex of every face is in front of the near plane
    (the common case for an aerial camera), near-plane clipping is a no-op, so
    all faces are projected in a single matmul and sliced back apart per face.
    Otherwise each face falls back to the per-face near-clip + project path,
    identical to the pre-batch behavior.

    Args:
        faces_3d (list[np.ndarray]): Per-face (K, 3) world-space vertex arrays.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        near (float): Camera near plane distance.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.

    Returns:
        list[list[tuple[float, float]] | None]: Screen-space vertices per face,
        aligned with ``faces_3d``; None where a face is behind near or
        degenerate after clipping.
    """
    all_vertices = np.concatenate(faces_3d, axis=0)
    cam_z = all_vertices @ view_matrix[2, :3] + view_matrix[2, 3]
    if not np.all(cam_z < -near):
        return [
            _project_face(list(face), view_matrix, proj_matrix, near, width, height)
            for face in faces_3d
        ]

    screen_coords, _, _ = project_points_full(
        all_vertices, view_matrix, proj_matrix, (width, height)
    )
    result: t.List[t.Optional[t.List[t.Tuple[float, float]]]] = []
    offset = 0
    for face in faces_3d:
        count = len(face)
        chunk = screen_coords[offset:offset + count]
        verts = [(float(x), float(y)) for x, y in chunk]
        # The all-inside Sutherland-Hodgman pass in clip_polygon_to_near_plane
        # (the slow path) emits vertices rotated left by one. Reproduce that
        # rotation so the polygon handed to draw_polygon is byte-for-byte the
        # same list the per-face path produced — PIL's fill can break boundary
        # ties differently for a different start vertex, so an identical
        # polygon in a different rotation is not pixel-identical.
        result.append(verts[1:] + verts[:1])
        offset += count
    return result


def _get_facing_walls(building: "Building", camera: Camera) -> t.List[FacingWall]:
    """Backface-cull a building's footprint edges against the camera.

    Computed once per building per frame and reused for both wall drawing
    and window placement, so the backface test runs exactly once per edge.

    Args:
        building (Building): The building whose footprint edges to test.
        camera (Camera): Camera used to determine which walls face it.

    Returns:
        list[FacingWall]: The subset of footprint edges that face the camera.
    """
    footprint = building.footprint
    n = len(footprint)
    cam_xy = camera.position[:2]

    facing_walls = []
    for i in range(n):
        p0 = footprint[i]
        p1 = footprint[(i + 1) % n]
        normal = _get_outward_normal(p0, p1)
        if _is_wall_facing_camera(normal, p0, p1, cam_xy):
            facing_walls.append(FacingWall(edge_index=i, p0=p0, p1=p1, normal=normal))

    return facing_walls


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


# ---------------------------------------------------------------------------
# Window grid derivation — pure geometry, no camera/RNG/rendering
# ---------------------------------------------------------------------------


def build_window_grid(
    p0: np.ndarray,
    p1: np.ndarray,
    height: float,
    normal: np.ndarray,
    edge_index: int,
    window_seed: int,
    config: BuildingConfig,
) -> t.List[WindowCell]:
    """Derive the deterministic window grid for one camera-facing wall.

    Pure geometry: given a wall edge, the building height, and the window
    config, computes how many window rows/columns fit and where each
    window's four world-space corners land on the wall plane. Rows are
    spaced by ``floor_height_m`` within the vertical band between
    ``window_sill_margin_m`` and ``window_roof_margin_m``; columns are
    spaced by ``window_pitch_m`` within the horizontal band inset by
    ``window_side_margin_m`` on each side. Any leftover space is centered
    as a margin on both ends of each axis. Walls too small to fit at least
    one row and one column produce no cells.

    Args:
        p0 (np.ndarray): (2,) wall edge start point in ENU meters.
        p1 (np.ndarray): (2,) wall edge end point in ENU meters.
        height (float): Building height in meters.
        normal (np.ndarray): (2,) outward unit normal of the wall.
        edge_index (int): Index of this footprint edge, used in the RNG key.
        window_seed (int): The building's per-building window seed.
        config (BuildingConfig): Window sizing/spacing/margin configuration.

    Returns:
        list[WindowCell]: One entry per window cell, empty if the wall is
        too small to fit any window.
    """
    edge_vector = p1 - p0
    wall_width = float(np.linalg.norm(edge_vector))
    if wall_width < 1e-9:
        return []
    direction = edge_vector / wall_width

    rows = _get_window_row_count(height, config)
    cols = _get_window_col_count(wall_width, config)
    if rows < 1 or cols < 1:
        return []

    row_centers = _get_window_row_centers(height, rows, config)
    col_offsets = _get_window_col_offsets(wall_width, cols, config)

    cells = []
    for row, z_center in enumerate(row_centers):
        for col, offset in enumerate(col_offsets):
            corners = _build_window_corners(p0, direction, normal, offset, z_center, config)
            cells.append(WindowCell(corners=corners, rng_key=(window_seed, edge_index, row, col)))

    return cells


def _get_window_row_count(height: float, config: BuildingConfig) -> int:
    """Compute how many window rows fit in a wall's vertical band.

    Buildings too short for a full floor still get one centered row when a
    window physically fits in the band, so one-story houses render lit
    windows instead of dark boxes.

    Args:
        height (float): Building height in meters.
        config (BuildingConfig): Window margin/spacing configuration.

    Returns:
        int: Number of rows that fit (0 when not even one window fits).
    """
    usable = height - config.window_roof_margin_m - config.window_sill_margin_m
    if usable <= 0.0:
        return 0
    rows = int(np.floor(usable / config.floor_height_m))
    if rows == 0 and usable >= config.window_height_m:
        return 1
    return rows


def _get_window_col_count(wall_width: float, config: BuildingConfig) -> int:
    """Compute how many window columns fit in a wall's horizontal band.

    Args:
        wall_width (float): Wall edge length in meters.
        config (BuildingConfig): Window margin/spacing configuration.

    Returns:
        int: Number of columns that fit (may be 0 if too narrow).
    """
    usable = wall_width - 2.0 * config.window_side_margin_m
    if usable <= 0.0:
        return 0
    return int(np.floor(usable / config.window_pitch_m))


def _get_window_row_centers(height: float, rows: int, config: BuildingConfig) -> t.List[float]:
    """Compute each window row's vertical center, centered within the band.

    Args:
        height (float): Building height in meters.
        rows (int): Number of window rows (must be >= 1).
        config (BuildingConfig): Window margin/spacing configuration.

    Returns:
        list[float]: Height (z) of each row's center, in meters.
    """
    usable = height - config.window_roof_margin_m - config.window_sill_margin_m
    span = rows * config.floor_height_m
    base_offset = config.window_sill_margin_m + (usable - span) / 2.0
    return [base_offset + config.floor_height_m * (row + 0.5) for row in range(rows)]


def _get_window_col_offsets(wall_width: float, cols: int, config: BuildingConfig) -> t.List[float]:
    """Compute each window column's along-wall offset, centered within the band.

    Args:
        wall_width (float): Wall edge length in meters.
        cols (int): Number of window columns (must be >= 1).
        config (BuildingConfig): Window margin/spacing configuration.

    Returns:
        list[float]: Distance from ``p0`` along the wall edge to each
        column's center, in meters.
    """
    usable = wall_width - 2.0 * config.window_side_margin_m
    span = cols * config.window_pitch_m
    base_offset = config.window_side_margin_m + (usable - span) / 2.0
    return [base_offset + config.window_pitch_m * (col + 0.5) for col in range(cols)]


def _build_window_corners(
    p0: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    offset_along_wall: float,
    z_center: float,
    config: BuildingConfig,
) -> t.List[np.ndarray]:
    """Build a window's four world-space corners on the wall plane.

    Args:
        p0 (np.ndarray): (2,) wall edge start point in ENU meters.
        direction (np.ndarray): (2,) unit vector from ``p0`` toward ``p1``.
        normal (np.ndarray): (2,) outward unit normal of the wall.
        offset_along_wall (float): Distance from ``p0`` to the window's
            horizontal center, in meters.
        z_center (float): Vertical center of the window, in meters.
        config (BuildingConfig): Window sizing configuration.

    Returns:
        list[np.ndarray]: Four (3,) world-space corners, in order
        bottom-left, bottom-right, top-right, top-left.
    """
    half_width = config.window_width_m / 2.0
    half_height = config.window_height_m / 2.0
    center_xy = p0 + direction * offset_along_wall + normal * _WINDOW_NORMAL_INSET_M
    left_xy = center_xy - direction * half_width
    right_xy = center_xy + direction * half_width

    return [
        np.array([left_xy[0], left_xy[1], z_center - half_height]),
        np.array([right_xy[0], right_xy[1], z_center - half_height]),
        np.array([right_xy[0], right_xy[1], z_center + half_height]),
        np.array([left_xy[0], left_xy[1], z_center + half_height]),
    ]


# ---------------------------------------------------------------------------
# Window lighting + emission accumulation
# ---------------------------------------------------------------------------


def _accumulate_building_windows(
    building: "Building",
    facing_walls: t.List[FacingWall],
    distance_m: float,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    config: BuildingConfig,
    window_buffer: np.ndarray,
) -> int:
    """Derive, light-test, and accumulate every lit window on a building.

    Args:
        building (Building): The building whose facing walls to process.
        facing_walls (list[FacingWall]): This building's camera-facing wall
            edges, already backface-culled.
        distance_m (float): Camera distance to the building's footprint
            centroid, in meters — beyond ``_WINDOW_SUBPIXEL_DISTANCE_M`` all
            windows use the sub-pixel accumulation path.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        config (BuildingConfig): Window sizing/color/brightness configuration.
        window_buffer (np.ndarray): Float32 (H, W, 3) linear buffer,
            modified in place.

    Returns:
        int: Number of windows that were lit and accumulated.
    """
    force_subpixel = distance_m >= _WINDOW_SUBPIXEL_DISTANCE_M
    lit_fraction = _get_lit_fraction(building.building_kind, config)
    lit_count = 0
    for wall in facing_walls:
        cells = build_window_grid(
            wall.p0, wall.p1, building.height, wall.normal, wall.edge_index,
            building.window_seed, config,
        )
        for cell in cells:
            if _accumulate_window_cell(
                cell, view_matrix, proj_matrix, width, height, config, lit_fraction,
                force_subpixel, window_buffer,
            ):
                lit_count += 1

    return lit_count


def _get_lit_fraction(building_kind: str, config: BuildingConfig) -> float:
    """Return the window lit fraction for a building's kind.

    Args:
        building_kind (str): Raw OSM ``building`` tag value.
        config (BuildingConfig): Window configuration.

    Returns:
        float: ``window_lit_fraction_residential`` for house-like kinds,
        ``window_lit_fraction`` for everything else.
    """
    if building_kind in _RESIDENTIAL_KINDS:
        return config.window_lit_fraction_residential
    return config.window_lit_fraction


def _accumulate_window_cell(
    cell: WindowCell,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
    config: BuildingConfig,
    lit_fraction: float,
    force_subpixel: bool,
    window_buffer: np.ndarray,
) -> bool:
    """Light-test one window cell and accumulate its emission if lit.

    Args:
        cell (WindowCell): The window's corners and RNG key.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        config (BuildingConfig): Window color/brightness configuration.
        lit_fraction (float): Fraction of windows on this building that are
            lit, already resolved for the building's kind.
        force_subpixel (bool): If True, always use the sub-pixel path
            regardless of the projected quad's screen size.
        window_buffer (np.ndarray): Float32 (H, W, 3) linear buffer,
            modified in place.

    Returns:
        bool: True if the window was lit and its emission was accumulated.
    """
    rng = _get_window_rng(cell.rng_key)
    if not _is_window_lit(rng, lit_fraction):
        return False

    linear_color = _get_window_linear_color(rng, config)
    brightness = _get_window_brightness(rng, config)

    screen_coords = _project_window(cell.corners, view_matrix, proj_matrix, width, height)
    if screen_coords is None:
        return False

    if force_subpixel or _is_window_subpixel(screen_coords):
        centroid = screen_coords.mean(axis=0)
        _accumulate_subpixel_window(window_buffer, centroid[0], centroid[1], linear_color, brightness)
    else:
        _accumulate_window_rect(window_buffer, screen_coords, linear_color, brightness)

    return True


def _get_window_rng(rng_key: t.Tuple[int, int, int, int]) -> random.Random:
    """Build a reproducible RNG from a window's tuple seed key.

    ``random.Random`` does not accept tuples directly, so the key is joined
    into a string first. ``random.seed`` hashes string seeds via SHA-512
    rather than the process's salted ``hash()``, so this is stable across
    runs and interpreter processes regardless of ``PYTHONHASHSEED``.

    Args:
        rng_key (tuple[int, int, int, int]): ``(window_seed, edge_index,
            row, col)``.

    Returns:
        random.Random: A fresh RNG seeded deterministically from the key.
    """
    return random.Random(":".join(str(component) for component in rng_key))


def _is_window_lit(rng: random.Random, lit_fraction: float) -> bool:
    """Check whether a window's seeded RNG draw falls under the lit fraction.

    Args:
        rng (random.Random): This window's seeded RNG.
        lit_fraction (float): Fraction of windows that are lit, in [0, 1].

    Returns:
        bool: True if the window is lit.
    """
    return rng.random() < lit_fraction


def _get_window_linear_color(
    rng: random.Random, config: BuildingConfig
) -> t.Tuple[float, float, float]:
    """Draw a random window color temperature and convert it to linear RGB.

    Args:
        rng (random.Random): This window's seeded RNG.
        config (BuildingConfig): Color temperature range and saturation.

    Returns:
        tuple[float, float, float]: Linear (R, G, B) each in [0, 1].
    """
    kelvin = rng.uniform(config.window_color_temp_min_k, config.window_color_temp_max_k)
    srgb_color = kelvin_to_rgb(kelvin, config.window_saturation)
    return srgb255_to_linear(srgb_color)


def _get_window_brightness(rng: random.Random, config: BuildingConfig) -> float:
    """Draw a random window brightness multiplier.

    Args:
        rng (random.Random): This window's seeded RNG.
        config (BuildingConfig): Brightness range configuration.

    Returns:
        float: Linear brightness multiplier for the window.
    """
    return rng.uniform(config.window_brightness_min, config.window_brightness_max)


def _project_window(
    corners: t.List[np.ndarray],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> t.Optional[np.ndarray]:
    """Project a window's four world-space corners to screen space.

    Unlike wall/roof faces, windows are not near-clipped — they are small
    enough, and already known to sit on an already-visible wall, that a
    simple in-front-of-camera check is sufficient.

    Args:
        corners (list[np.ndarray]): Four (3,) world-space corners.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.

    Returns:
        np.ndarray | None: (4, 2) screen-space corners, or None if any
        corner is behind the camera.
    """
    points_3d = np.stack(corners, axis=0)
    screen_coords, in_front_mask, _ = project_points_full(
        points_3d, view_matrix, proj_matrix, (width, height)
    )
    if not bool(np.all(in_front_mask)):
        return None
    return screen_coords


def _is_window_subpixel(screen_coords: np.ndarray) -> bool:
    """Check whether a projected window quad is smaller than one pixel.

    Args:
        screen_coords (np.ndarray): (4, 2) projected screen-space corners.

    Returns:
        bool: True if the bounding box is below threshold in both dimensions.
    """
    bbox_width = float(screen_coords[:, 0].max() - screen_coords[:, 0].min())
    bbox_height = float(screen_coords[:, 1].max() - screen_coords[:, 1].min())
    return (
        bbox_width < _WINDOW_SUBPIXEL_PIXEL_THRESHOLD
        and bbox_height < _WINDOW_SUBPIXEL_PIXEL_THRESHOLD
    )


def _accumulate_subpixel_window(
    buffer: np.ndarray,
    cx: float,
    cy: float,
    linear_color: t.Tuple[float, float, float],
    brightness: float,
) -> None:
    """Add a single-pixel emissive point to the linear accumulation buffer.

    Used for windows whose projected size is below one pixel — the common
    case beyond a few kilometers — so distant buildings still sparkle
    instead of vanishing entirely to sub-pixel rounding.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer, modified in place.
        cx (float): Screen-space center x.
        cy (float): Screen-space center y.
        linear_color (tuple[float, float, float]): Linear RGB color in [0, 1].
        brightness (float): Linear brightness multiplier.
    """
    h, w, _ = buffer.shape
    xi = int(round(cx))
    yi = int(round(cy))
    if not (0 <= xi < w and 0 <= yi < h):
        return

    for c, channel_value in enumerate(linear_color):
        buffer[yi, xi, c] += channel_value * brightness


def _accumulate_window_rect(
    buffer: np.ndarray,
    screen_coords: np.ndarray,
    linear_color: t.Tuple[float, float, float],
    brightness: float,
) -> None:
    """Add a flat-filled emissive rectangle to the linear accumulation buffer.

    Fills the screen-space bounding box of the projected window quad
    uniformly — windows are small enough on screen that an exact polygon
    rasterization isn't worth the cost.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer, modified in place.
        screen_coords (np.ndarray): (4, 2) projected screen-space corners.
        linear_color (tuple[float, float, float]): Linear RGB color in [0, 1].
        brightness (float): Linear brightness multiplier.
    """
    h, w, _ = buffer.shape
    x1 = max(0, int(np.floor(screen_coords[:, 0].min())))
    x2 = min(w, int(np.ceil(screen_coords[:, 0].max())) + 1)
    y1 = max(0, int(np.floor(screen_coords[:, 1].min())))
    y2 = min(h, int(np.ceil(screen_coords[:, 1].max())) + 1)
    if x1 >= x2 or y1 >= y2:
        return

    for c, channel_value in enumerate(linear_color):
        buffer[y1:y2, x1:x2, c] += channel_value * brightness


def _apply_window_glow(buffer: np.ndarray, glow_px: float) -> np.ndarray:
    """Optionally soften the window emission buffer with a small box blur.

    Kept off by default (``glow_px == 0``) since lit windows are meant to be
    sharp emissive rectangles, not bloom sources. When enabled, gives a
    subtle "diffuse through glass" softness without per-window PSFs.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer.
        glow_px (float): Blur radius in pixels; values <= 0 are a no-op.

    Returns:
        np.ndarray: The blurred buffer, or ``buffer`` unchanged if
        ``glow_px <= 0``.
    """
    if glow_px <= 0.0:
        return buffer

    kernel_size = max(1, int(round(glow_px))) * 2 + 1
    blurred = _box_blur_axis(buffer, kernel_size, axis=0)
    blurred = _box_blur_axis(blurred, kernel_size, axis=1)
    return blurred


def _box_blur_axis(buffer: np.ndarray, kernel_size: int, axis: int) -> np.ndarray:
    """Apply a 1D box blur of the given kernel size along one axis.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer.
        kernel_size (int): Odd number of samples averaged per output pixel.
        axis (int): Axis to blur along (0 = rows, 1 = columns).

    Returns:
        np.ndarray: Float32 array, same shape as ``buffer``.
    """
    pad = kernel_size // 2
    pad_widths = [(0, 0)] * buffer.ndim
    pad_widths[axis] = (pad, pad)
    padded = np.pad(buffer, pad_widths, mode="edge")

    cumulative = np.cumsum(padded, axis=axis)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate([np.zeros(zero_shape, dtype=cumulative.dtype), cumulative], axis=axis)

    hi = [slice(None)] * buffer.ndim
    lo = [slice(None)] * buffer.ndim
    hi[axis] = slice(kernel_size, kernel_size + buffer.shape[axis])
    lo[axis] = slice(0, buffer.shape[axis])
    summed = cumulative[tuple(hi)] - cumulative[tuple(lo)]
    return (summed / kernel_size).astype(np.float32)
