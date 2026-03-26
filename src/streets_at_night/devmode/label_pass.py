"""Render pass that draws row identification labels onto the scene."""

import typing as t

import numpy as np
from PIL import ImageDraw, ImageFont

from ..camera.camera import Camera
from ..camera.projection import project_points
from ..rendering.backend import RenderBackend
from ..rendering.render_pipeline import RenderPass

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


def make_label_pass(
    row_labels: t.List[t.Tuple[str, float]],
    camera: Camera,
) -> RenderPass:
    """Create a render pass that draws row labels at their world positions.

    Projects each label's anchor position to screen space and draws the label
    text if the projected point falls within the visible frame.

    Args:
        row_labels (list[tuple[str, float]]): List of (label_text, y_offset)
            pairs. Each label is anchored at world position [0, y_offset, 0].
        camera (Camera): Camera used for projection. Captured in closure.

    Returns:
        RenderPass: Callable conforming to (Scene, Camera, RenderBackend) -> Any.
    """
    def _label_pass(scene: "Scene", cam: Camera, backend: RenderBackend) -> None:
        """Draw all row labels onto the backend surface.

        Args:
            scene (Scene): The scene being rendered (unused, but required by
                the RenderPass protocol).
            cam (Camera): Camera providing current view/projection matrices.
            backend (RenderBackend): Target render surface.
        """
        world_points = _build_label_world_points(row_labels)
        screen_size = backend.get_surface_size()
        screen_coords, valid_mask = _project_label_points(cam, world_points, screen_size)

        image = backend.get_image()
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        _draw_visible_labels(draw, font, row_labels, screen_coords, valid_mask)

        backend.paste_image(image)

    return _label_pass


def _build_label_world_points(
    row_labels: t.List[t.Tuple[str, float]],
) -> np.ndarray:
    """Build an (N, 3) array of world-space anchor points for each label.

    Each label is anchored at [0, y_offset, 0].

    Args:
        row_labels (list[tuple[str, float]]): Row label definitions.

    Returns:
        np.ndarray: (N, 3) array of ENU world positions.
    """
    return np.array([[0.0, y, 0.0] for _, y in row_labels])


def _project_label_points(
    camera: Camera,
    world_points: np.ndarray,
    screen_size: t.Tuple[int, int],
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Project world-space label anchors to screen coordinates.

    Args:
        camera (Camera): The current camera.
        world_points (np.ndarray): (N, 3) ENU world positions.
        screen_size (tuple[int, int]): (width, height) of the render surface.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - screen_coords: (N, 2) pixel positions.
            - valid_mask: (N,) bool array indicating on-screen points.
    """
    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    return project_points(world_points, view_matrix, proj_matrix, screen_size)


def _draw_visible_labels(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    row_labels: t.List[t.Tuple[str, float]],
    screen_coords: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Draw label text at each valid (on-screen) projected position.

    Args:
        draw (ImageDraw.ImageDraw): PIL draw context for the current image.
        font (ImageFont.FreeTypeFont): Font for rendering text.
        row_labels (list[tuple[str, float]]): Label text and y-offsets.
        screen_coords (np.ndarray): (N, 2) projected screen positions.
        valid_mask (np.ndarray): (N,) bool — True if the point is on-screen.
    """
    for i, (label_text, _) in enumerate(row_labels):
        if not valid_mask[i]:
            continue
        x = float(screen_coords[i, 0])
        y = float(screen_coords[i, 1])
        draw.text((x + 4, y - 8), label_text, fill=(255, 255, 100), font=font)
