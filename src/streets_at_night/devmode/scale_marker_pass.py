"""Render pass that draws a car-length scale reference line at each row start.

A single red line of length CAR_LENGTH_M is drawn transverse to each synthetic
row (along Y, centered on the row centerline) near the row's beginning, so the
scale of lights, pools, roads, and houses can be judged against a familiar
object. Devmode-only: registered exclusively by devmode render helpers.
"""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import frustum_cull_segments, project_points_full
from ..rendering.backend import RenderBackend
from ..rendering.render_pipeline import RenderPass

if t.TYPE_CHECKING:
    from ..scene.scene import Scene

# Length of an average passenger car in meters (Corolla 4.63, Civic 4.67,
# Camry 4.88 — 4.7 sits mid-band).
CAR_LENGTH_M: float = 4.7

# X inset from the row start so the marker clears the row label anchor at x=0.
_MARKER_X_M: float = 30.0

# Near-pure red reads as annotation — nothing in the night palette is red.
_MARKER_COLOR: t.Tuple[int, int, int] = (255, 60, 60)

# Fixed stroke width; the marker's projected LENGTH carries the scale info.
_MARKER_WIDTH_PX: int = 2


def make_scale_marker_pass(
    row_labels: t.List[t.Tuple[str, float]],
) -> RenderPass:
    """Create a render pass drawing a car-length reference line per row.

    Each row gets one transverse line of length CAR_LENGTH_M centered on the
    row centerline at x=_MARKER_X_M, z=0. Lines are near-plane clipped,
    projected, and drawn directly via backend.draw_line, so this pass must run
    after render_composite (it draws onto the resolved sRGB frame).

    Args:
        row_labels (list[tuple[str, float]]): (label_text, y_offset) pairs;
            only the y offsets are used.

    Returns:
        RenderPass: Callable conforming to (Scene, Camera, RenderBackend) -> Any.
    """
    def _scale_marker_pass(scene: "Scene", cam: Camera, backend: RenderBackend) -> None:
        """Draw the reference line for every row visible to the camera.

        Args:
            scene (Scene): The scene being rendered (unused; RenderPass protocol).
            cam (Camera): Camera providing view/projection matrices.
            backend (RenderBackend): Target render surface.
        """
        pairs = _build_marker_pairs(row_labels)
        view_matrix = cam.get_view_matrix()
        proj_matrix = cam.get_projection_matrix()
        visible = frustum_cull_segments(pairs, view_matrix, cam.near, cam.far)
        if not visible:
            return
        _draw_marker_edges(visible, view_matrix, proj_matrix, backend)

    return _scale_marker_pass


def _build_marker_pairs(
    row_labels: t.List[t.Tuple[str, float]],
) -> t.List[t.Tuple[np.ndarray, np.ndarray]]:
    """Build world-space endpoint pairs for each row's reference line.

    Args:
        row_labels (list[tuple[str, float]]): Row label definitions.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: One (p1, p2) pair per row; each
            line runs along Y, length CAR_LENGTH_M, centered on the row
            centerline at x=_MARKER_X_M, z=0.
    """
    half = CAR_LENGTH_M / 2.0
    return [
        (
            np.array([_MARKER_X_M, y - half, 0.0]),
            np.array([_MARKER_X_M, y + half, 0.0]),
        )
        for _, y in row_labels
    ]


def _draw_marker_edges(
    pairs: t.List[t.Tuple[np.ndarray, np.ndarray]],
    view_matrix: np.ndarray,
    proj_matrix: np.ndarray,
    backend: RenderBackend,
) -> None:
    """Project near-clipped marker edges and draw the in-front ones.

    Uses the in-front mask (not the on-screen mask) — both backends clip line
    primitives to the surface, matching road_pass behavior.

    Args:
        pairs (list[tuple[np.ndarray, np.ndarray]]): Near-clipped world edges.
        view_matrix (np.ndarray): (4, 4) view matrix.
        proj_matrix (np.ndarray): (4, 4) projection matrix.
        backend (RenderBackend): Target render surface.
    """
    screen_size = backend.get_surface_size()
    points = np.array([point for pair in pairs for point in pair])
    screen_coords, in_front_mask, _ = project_points_full(
        points, view_matrix, proj_matrix, screen_size
    )
    for i in range(len(pairs)):
        idx1, idx2 = 2 * i, 2 * i + 1
        if not (in_front_mask[idx1] and in_front_mask[idx2]):
            continue
        sx1, sy1 = screen_coords[idx1]
        sx2, sy2 = screen_coords[idx2]
        backend.draw_line(
            (int(sx1), int(sy1)), (int(sx2), int(sy2)),
            _MARKER_COLOR, _MARKER_WIDTH_PX,
        )
