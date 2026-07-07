"""Devmode scene render helpers shared by the CLI tool and the UI Devmode tab.

Public versions of the render/scene helpers that ``cmd_devmode`` keeps as
private copies. Both consume these; ``cmd_devmode`` is removed in change 09.
These helpers return PIL images (never pygame surfaces) — fitting the image to
a viewport and converting it to a surface is the caller's job (``ui.imaging``
for the tab). Nothing here imports ``ui`` so the ``ui -> devmode`` dependency
direction holds.
"""

import random
import typing as t

import numpy as np
from PIL import Image

from ..config.schema import SceneConfig
from ..rendering.atmosphere_pass import render_atmosphere
from ..rendering.compositor import render_composite
from ..rendering.ground_pass import render_ground
from ..rendering.illumination_pass import render_illumination
from ..rendering.light_pass import render_lights_export
from ..rendering.pillow_backend import PillowBackend
from ..rendering.raycast import build_ground_gbuffer
from ..rendering.render_pipeline import render_scene_frame
from ..rendering.road_pass import render_roads_export
from .label_pass import make_label_pass
from .synthetic_scene import ROW_LABELS

# Full-quality re-render only fires once camera movement has been still for
# this many seconds; while moving, a cheap roads-only frame is shown instead.
DEBOUNCE_SECONDS = 0.15


def ensure_devmode_config(config: SceneConfig) -> None:
    """Initialise config.devmode from DEFAULTS if it is None.

    Args:
        config (SceneConfig): Config to populate in-place if devmode is absent.
    """
    if config.devmode is not None:
        return
    from ..config.defaults import DEFAULTS
    from ..config.loader import _build_devmode_config
    config.devmode = _build_devmode_config(DEFAULTS["devmode"])


def sync_camera_from_config(scene: t.Any, config: SceneConfig) -> None:
    """Update scene.camera position, target, and FOV from config.devmode.camera.

    Used for real-tiles mode where the scene is not rebuilt on re-render —
    only the lights are replaced. Without this, camera slider changes have
    no effect because the Camera object on the scene is never touched.

    Args:
        scene: The scene whose camera to update.
        config (SceneConfig): Config with updated devmode.camera values.
    """
    if config.devmode is None:
        return
    cam = config.devmode.camera
    scene.camera.position = np.array([cam.pos_x, cam.pos_y, cam.pos_z])
    scene.camera.target = np.array([cam.target_x, cam.target_y, cam.target_z])
    scene.camera.fov_deg = cam.fov


def rebuild_real_lights(scene: t.Any, config: SceneConfig) -> t.Any:
    """Clear and re-place all lights on the existing real strip scene.

    Keeps road geometry, zones, and camera fixed. Only the spatial index and
    light lists are rebuilt so that updated config parameters take effect.

    Args:
        scene: The real strip scene to update in-place.
        config (SceneConfig): Updated scene configuration.

    Returns:
        The same scene object with refreshed lights.
    """
    from ..placement.scene_builder import populate_lights

    scene.lights = []
    scene.car_pairs = []
    scene.build_spatial_index()
    populate_lights(scene, config, random.Random(config.seed))
    return scene


def should_render_full(
    last_move_t: t.Optional[float],
    now: float,
    dirty: bool,
) -> bool:
    """Decide whether to run the full render pipeline this frame.

    While the camera is actively moving (dirty), a cheap preview frame is
    shown instead and this returns False. Once movement stops, this returns
    True exactly once movement has been still for at least DEBOUNCE_SECONDS —
    callers should reset last_move_t to None after honoring a True result so
    the full render does not repeat every frame thereafter.

    Args:
        last_move_t (float | None): Elapsed-time timestamp of the last camera
            movement, or None if the camera has not moved since the last full
            render.
        now (float): Current elapsed-time timestamp.
        dirty (bool): True if the camera moved this frame.

    Returns:
        bool: True if a full-quality render should run this frame.
    """
    if dirty:
        return False
    if last_move_t is None:
        return False
    return (now - last_move_t) >= DEBOUNCE_SECONDS


def build_render_passes(
    scene: t.Any,
    gbuffer: t.Any,
    use_labels: bool,
) -> list:
    """Build the devmode render pass list, matching the export pipeline order.

    Illumination and atmosphere are wrapped to receive the shared G-buffer.
    The composite pass must be present and after the light passes so the
    accumulated linear light buffer is resolved into the visible image; labels
    come last so they overlay the final composited frame.

    Args:
        scene: The scene to render (used for the label pass camera).
        gbuffer: Shared ground raycast for illumination and atmosphere.
        use_labels (bool): When True, append the synthetic-row label pass.

    Returns:
        list: Ordered render passes for render_scene_frame.
    """
    def _illumination_pass(s: t.Any, c: t.Any, b: t.Any) -> t.Any:
        return render_illumination(s, c, b, gbuffer=gbuffer)

    def _atmosphere_pass(s: t.Any, c: t.Any, b: t.Any) -> None:
        render_atmosphere(s, c, b, gbuffer=gbuffer)

    passes = [
        render_ground,
        render_roads_export,
        _illumination_pass,
        render_lights_export,
        _atmosphere_pass,
        render_composite,
    ]
    if use_labels:
        passes.append(make_label_pass(ROW_LABELS, scene.camera))
    return passes


def render_devmode_scene(
    scene: t.Any,
    config: SceneConfig,
    use_labels: bool = True,
) -> Image.Image:
    """Render the scene using the Pillow backend and return the image.

    Mirrors the export pipeline so devmode shows the same result: ground →
    roads → illumination → lights → atmosphere → composite, then optional
    labels. Renders at config.devmode.render_width/render_height.

    Args:
        scene: The scene to render.
        config (SceneConfig): Current scene configuration.
        use_labels (bool): When True, append a label pass for synthetic rows.

    Returns:
        PIL.Image.Image: The rendered frame at devmode render resolution.
    """
    for car_pair in scene.car_pairs:
        car_pair.resolve_facing(scene.camera.get_forward())
    render_width = config.devmode.render_width
    render_height = config.devmode.render_height
    backend = PillowBackend(render_width, render_height)
    gbuffer = build_ground_gbuffer(scene.camera, render_width, render_height)
    passes = build_render_passes(scene, gbuffer, use_labels)
    render_scene_frame(scene, backend, passes=passes)
    return backend.get_image()


def render_cheap(scene: t.Any, config: SceneConfig) -> Image.Image:
    """Render a cheap ground+roads-only preview frame for use while moving.

    Skips illumination, lights, atmosphere, and composite so navigation stays
    responsive. Renders at config.devmode.render_width/render_height so this
    transient frame has identical framing to the settled full-quality frame.

    Args:
        scene: The scene to render.
        config (SceneConfig): Current scene configuration.

    Returns:
        PIL.Image.Image: The cheap preview frame at devmode render resolution.
    """
    render_width = config.devmode.render_width
    render_height = config.devmode.render_height
    backend = PillowBackend(render_width, render_height)
    passes = [render_ground, render_roads_export]
    render_scene_frame(scene, backend, passes=passes)
    return backend.get_image()
