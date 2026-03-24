"""Render pipeline orchestrator — runs all passes in order.

Passes are represented as callables with the signature:
    (scene: Scene, camera: Camera, backend: RenderBackend) -> Any

New passes for Stage 5+ can be added by extending the list returned by
get_preview_passes() or get_export_passes() without modifying this module.

The warp-preview path (get_preview_passes_warp) is used for the interactive
pygame preview. It replaces per-segment projection with a single PIL
perspective warp of the pre-rendered flat map. The flat map and its metadata
must be generated before calling render_scene_frame with the warp passes.
"""

import typing as t
from pathlib import Path

from PIL import Image

from ..camera.camera import Camera
from .backend import RenderBackend
from .flat_map import FlatMapMeta

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


RenderPass = t.Callable[["Scene", Camera, RenderBackend], t.Any]


def get_preview_passes() -> t.List[RenderPass]:
    """Return the ordered list of render passes for the pygame preview.

    Includes ground fill, road projection, and light dot rendering. Light
    dots are colored circles — no glow or illumination is applied in preview
    since those effects are too costly for real-time rendering.

    Returns:
        list[RenderPass]: Passes to run each preview frame, in order.
    """
    from .ground_pass import render_ground
    from .light_pass import render_lights_preview
    from .road_pass import render_roads_from_scene
    return [render_ground, render_roads_from_scene, render_lights_preview]


def get_preview_passes_warp(
    flat_map: Image.Image,
    flat_meta: FlatMapMeta,
) -> t.List[RenderPass]:
    """Return the warp-based preview pass list.

    Returns a single pass that perspective-warps the flat road map to screen
    each frame. Much faster than the segment-projection preview path.
    The flat_map and flat_meta are captured in the closure.

    Args:
        flat_map (Image.Image): Pre-rendered top-down road map.
        flat_meta (FlatMapMeta): World bounds and resolution metadata.

    Returns:
        list[RenderPass]: Single-element list containing the warp render pass.
    """
    from .warp_preview import render_warp_preview

    def _warp_pass(
        scene: "Scene", camera: Camera, backend: RenderBackend
    ) -> None:
        render_warp_preview(scene, camera, backend, flat_map, flat_meta)

    return [_warp_pass]


def get_export_passes(cache_dir: Path = Path("cache")) -> t.List[RenderPass]:
    """Return the ordered list of render passes for Pillow export.

    Uses tiled ground rendering for per-house color variation. Runs the full
    Stage 5 pipeline: ground, roads, ground illumination pools, then light
    cores and glow halos. The cache_dir is captured in a closure so the ground
    export pass conforms to the standard (scene, camera, backend) signature.

    Pass order:
        1. Ground (tiled export) — sky, horizon, and textured ground.
        2. Roads — per-segment 3D projection with night-palette colors.
        3. Illumination — soft light-pool splats, gaussian-blurred, additive.
        4. Lights — bright cores and gaussian glow halos, additive.

    Args:
        cache_dir (Path): Directory for tile and OSM cache files.

    Returns:
        list[RenderPass]: Passes to run for still image export, in order.
    """
    from .ground_pass import render_ground_export
    from .illumination_pass import render_illumination
    from .light_pass import render_lights_export
    from .road_pass import render_roads_export

    def _ground_export_pass(
        scene: "Scene", camera: Camera, backend: RenderBackend
    ) -> None:
        render_ground_export(scene, camera, backend, cache_dir)

    return [_ground_export_pass, render_roads_export, render_illumination, render_lights_export]


def render_scene_frame(
    scene: "Scene",
    backend: RenderBackend,
    passes: t.Optional[t.List[RenderPass]] = None,
) -> t.Dict[str, t.Any]:
    """Run all render passes in order and return accumulated stats.

    Each pass receives (scene, camera, backend). If a pass returns a dict,
    its keys are merged into the stats. If it returns an int, it is stored
    under 'segments_drawn'.

    Args:
        scene (Scene): The world scene with camera and road segments.
        backend (RenderBackend): Target render surface.
        passes (list[RenderPass] | None): Ordered passes to run. Defaults to
            get_export_passes() if None.

    Returns:
        dict[str, Any]: Accumulated stats from all passes.
    """
    if passes is None:
        passes = get_export_passes()

    stats: t.Dict[str, t.Any] = {}
    for render_pass in passes:
        result = render_pass(scene, scene.camera, backend)
        if isinstance(result, dict):
            stats.update(result)
        elif isinstance(result, int):
            stats["segments_drawn"] = result

    return stats
