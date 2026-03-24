"""Stage 2 perspective road renderer.

Produces a static PIL image showing classified roads projected from the 3D
camera onto a 2D frame. Roads are colored by type and drawn with perspective-
correct line widths (closer = thicker).

Color palette and width choices are intentionally vivid for verification
purposes — they make it easy to confirm that each road type is projecting
correctly before the final artistic pass in later stages.
"""

import typing as t
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw

from ..camera.camera import Camera, get_view_matrix, get_projection_matrix
from ..camera.projection import frustum_cull_segments, project_points
from ..config.schema import QualityPreset
from ..models.road_types import RoadType


# Sky and ground background colors
_SKY_COLOR: t.Tuple[int, int, int] = (5, 5, 15)
_GROUND_COLOR: t.Tuple[int, int, int] = (15, 20, 15)

# Horizon glow band color (additive tint)
_HORIZON_GLOW_COLOR: t.Tuple[int, int, int, int] = (40, 25, 10, 60)

# Road type → (RGB color, base line width in pixels)
_ROAD_STYLE: t.Dict[RoadType, t.Tuple[t.Tuple[int, int, int], int]] = {
    RoadType.HIGHWAY:      ((255, 100, 100), 4),
    RoadType.MAJOR_ROAD:   ((255, 165,  50), 3),
    RoadType.COLLECTOR:    ((220, 200,  50), 2),
    RoadType.RESIDENTIAL:  ((100, 200, 100), 1),
    RoadType.SERVICE:      ((100, 180, 220), 1),
    RoadType.MINOR:        ((180, 140, 200), 1),
}

# Distance (meters) at which the base line width applies
_REFERENCE_DISTANCE_M: float = 5000.0

# Number of semi-transparent lines for the horizon glow band
_HORIZON_GLOW_LINES: int = 10


def render_roads_perspective(
    roads_gdf: gpd.GeoDataFrame,
    road_segments_enu: t.List[np.ndarray],
    camera: Camera,
    preset: QualityPreset,
    output_path: Path,
) -> None:
    """Render a perspective road image and save to output_path.

    Fills sky and ground regions, draws a subtle horizon glow, then projects
    and draws each road segment using its classified color and a
    distance-scaled line width.

    Args:
        roads_gdf (GeoDataFrame): Classified road features with road_type column.
        road_segments_enu (list[np.ndarray]): ENU coordinate arrays from
            convert_road_geometry, one (N, 3) array per road.
        camera (Camera): Configured 3D camera.
        preset (QualityPreset): Resolution settings for the output image.
        output_path (Path): Where to save the resulting PNG.
    """
    width, height = preset.width, preset.height
    view_matrix = get_view_matrix(camera)
    proj_matrix = get_projection_matrix(camera)

    image = Image.new("RGB", (width, height), _SKY_COLOR)
    draw = ImageDraw.Draw(image, "RGBA")

    horizon_y = _compute_horizon_y(camera, view_matrix, proj_matrix, width, height)
    _fill_ground(draw, width, height, horizon_y)
    _draw_horizon_glow(draw, width, horizon_y)
    _draw_all_roads(
        draw, roads_gdf, road_segments_enu, view_matrix, proj_matrix,
        camera, width, height,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    print(f"Saved perspective render to {output_path}")


def _compute_horizon_y(
    camera: Camera,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    width: int,
    height: int,
) -> int:
    """Approximate the horizon screen y-coordinate.

    Projects a point far ahead on the ground plane (z=0) to find where the
    horizon falls in screen space. Falls back to the image midpoint if the
    far point is behind the camera.

    Args:
        camera (Camera): The camera.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        width (int): Image width in pixels.
        height (int): Image height in pixels.

    Returns:
        int: Screen y-coordinate of the horizon line.
    """
    far_ahead = camera.position + np.array([0.0, camera.far * 0.9, 0.0])
    far_ahead[2] = 0.0
    coords, valid = project_points(
        far_ahead.reshape(1, 3), view_matrix, proj_matrix, (width, height)
    )
    if valid[0]:
        return int(np.clip(coords[0, 1], 0, height - 1))
    return height // 4


def _fill_ground(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    horizon_y: int,
) -> None:
    """Fill everything below the horizon with the ground color.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        width (int): Image width.
        height (int): Image height.
        horizon_y (int): Screen y of the horizon.
    """
    if horizon_y >= height:
        return
    draw.rectangle([(0, horizon_y), (width - 1, height - 1)], fill=_GROUND_COLOR)


def _draw_horizon_glow(
    draw: ImageDraw.ImageDraw,
    width: int,
    horizon_y: int,
) -> None:
    """Draw a subtle glow band centered on the horizon line.

    Draws a series of semi-transparent lines above and below the horizon,
    fading in opacity toward the edges of the band.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context with RGBA mode.
        width (int): Image width.
        horizon_y (int): Screen y of the horizon.
    """
    r, g, b, max_alpha = _HORIZON_GLOW_COLOR
    half = _HORIZON_GLOW_LINES // 2
    for offset in range(-half, half + 1):
        fade = 1.0 - abs(offset) / (half + 1)
        alpha = int(max_alpha * fade)
        y = horizon_y + offset
        draw.line([(0, y), (width - 1, y)], fill=(r, g, b, alpha), width=1)


def _draw_all_roads(
    draw: ImageDraw.ImageDraw,
    roads_gdf: gpd.GeoDataFrame,
    road_segments_enu: t.List[np.ndarray],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
) -> None:
    """Project and draw all road segments.

    Pairs each ENU segment array with its road_type from the GeoDataFrame,
    applies frustum culling, projects, and draws.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        roads_gdf (GeoDataFrame): Classified roads with road_type column.
        road_segments_enu (list[np.ndarray]): ENU coordinate arrays.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        camera (Camera): The camera (for near/far values).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    """
    road_types = list(roads_gdf["road_type"])
    road_count = min(len(road_types), len(road_segments_enu))

    for i in range(road_count):
        road_type = road_types[i]
        segment_points = road_segments_enu[i]
        style = _ROAD_STYLE.get(road_type)
        if style is None:
            continue
        color, base_width = style
        _draw_road_segment(
            draw, segment_points, color, base_width,
            view_matrix, proj_matrix, camera, width, height,
        )


def _draw_road_segment(
    draw: ImageDraw.ImageDraw,
    segment_points: np.ndarray,
    color: t.Tuple[int, int, int],
    base_width: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
) -> None:
    """Cull, project, and draw a single road polyline.

    Iterates through consecutive point pairs in the segment, applies
    frustum culling and near-plane clipping to each edge, projects the
    endpoints, and draws with perspective-scaled width.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        segment_points (np.ndarray): (N, 3) ENU coordinates of the road.
        color (tuple[int, int, int]): RGB color for this road type.
        base_width (int): Base line width at reference distance.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        camera (Camera): Camera (for near/far).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    """
    if segment_points.shape[0] < 2:
        return

    pairs = _build_segment_pairs(segment_points)
    visible = frustum_cull_segments(pairs, view_matrix, camera.near, camera.far)

    for p1, p2 in visible:
        _project_and_draw_edge(
            draw, p1, p2, color, base_width,
            view_matrix, proj_matrix, camera, width, height,
        )


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


def _project_and_draw_edge(
    draw: ImageDraw.ImageDraw,
    p1: np.ndarray,
    p2: np.ndarray,
    color: t.Tuple[int, int, int],
    base_width: int,
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    camera: Camera,
    width: int,
    height: int,
) -> None:
    """Project two 3D endpoints and draw a line between them.

    Skips drawing if either projected point falls outside the valid screen area.
    Scales line width by distance for perspective effect.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context.
        p1 (np.ndarray): (3,) first endpoint in world ENU.
        p2 (np.ndarray): (3,) second endpoint in world ENU.
        color (tuple[int, int, int]): RGB line color.
        base_width (int): Base line width at reference distance.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        camera (Camera): Camera (for near plane).
        width (int): Image width.
        height (int): Image height.
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
    draw.line([(sx1, sy1), (sx2, sy2)], fill=color, width=line_width)


def _compute_line_width(
    p1: np.ndarray,
    p2: np.ndarray,
    view_matrix: np.ndarray,
    base_width: int,
) -> int:
    """Compute perspective-scaled line width for an edge.

    Uses the average absolute camera-space z depth of the two endpoints as
    the distance metric. Closer segments get thicker lines.

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
