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
from ..config.schema import QualityPreset
from .backend import RenderBackend
from .flat_map import FlatMapMeta

if t.TYPE_CHECKING:
    from ..scene.scene import Scene


RenderPass = t.Callable[["Scene", Camera, RenderBackend], t.Any]


def get_preview_passes(no_lights: bool = False) -> t.List[RenderPass]:
    """Return the ordered list of render passes for the pygame preview.

    Includes ground fill, road projection, and optionally light dot rendering.
    Light dots are colored circles — no glow or illumination is applied in
    preview since those effects are too costly for real-time rendering.

    Args:
        no_lights (bool): When True, omit the light preview pass. Defaults to False.

    Returns:
        list[RenderPass]: Passes to run each preview frame, in order.
    """
    from .ground_pass import render_ground
    from .road_pass import render_roads_from_scene

    passes: t.List[RenderPass] = [render_ground, render_roads_from_scene]
    if not no_lights:
        from .light_pass import render_lights_preview
        passes.append(render_lights_preview)
    return passes


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


def get_export_passes(
    cache_dir: Path = Path("cache"),
    no_lights: bool = False,
    quality_preset: t.Optional[QualityPreset] = None,
) -> t.List[RenderPass]:
    """Return the ordered list of render passes for Pillow export.

    Uses tiled ground rendering for per-house color variation. Runs the full
    Stage 5 pipeline: ground, roads, ground illumination pools, then light
    cores and glow halos. The cache_dir and quality_preset are captured in
    closures so all passes conform to the standard (scene, camera, backend)
    signature.

    Pass order:
        1. Ground (tiled export) — sky, horizon, and textured ground.
        2. Roads — per-segment 3D projection with night-palette colors.
        3. Illumination — world-space Gaussian map, raycast-sampled, additive.
           (omitted when no_lights=True)
        4. Lights — bright cores and gaussian glow halos, additive.
           (omitted when no_lights=True)
        5. Atmosphere — horizon glow gradient and optional distance haze.
        6. Compositor — vignette darkening, optional stars, color grade stub.

    Args:
        cache_dir (Path): Directory for tile and OSM cache files.
        no_lights (bool): When True, omit illumination and light passes.
            Defaults to False.
        quality_preset (QualityPreset | None): Active quality preset used to
            scale the illumination map resolution. When None, the illumination
            pass uses its default multiplier of 1.0.

    Returns:
        list[RenderPass]: Passes to run for still image export, in order.
    """
    from .atmosphere_pass import render_atmosphere
    from .compositor import render_composite
    from .ground_pass import render_ground_export
    from .road_pass import render_roads_export

    def _ground_export_pass(
        scene: "Scene", camera: Camera, backend: RenderBackend
    ) -> None:
        render_ground_export(scene, camera, backend, cache_dir)

    passes: t.List[RenderPass] = [_ground_export_pass, render_roads_export]
    if not no_lights:
        from .illumination_pass import render_illumination
        from .light_pass import render_lights_export

        multiplier = (
            quality_preset.illumination_resolution_multiplier
            if quality_preset is not None
            else 1.0
        )

        def _illumination_pass(
            scene: "Scene", camera: Camera, backend: RenderBackend
        ) -> t.Dict[str, t.Any]:
            return render_illumination(scene, camera, backend, multiplier)

        passes.extend([_illumination_pass, render_lights_export])

    passes.extend([render_atmosphere, render_composite])
    return passes


def get_export_pass_names(no_lights: bool = False) -> t.List[str]:
    """Return human-readable names for each export pass in pipeline order.

    The returned list matches the pass order produced by get_export_passes()
    for the same no_lights argument.

    Args:
        no_lights (bool): When True, omit illumination and light pass names.
            Defaults to False.

    Returns:
        list[str]: Ordered pass name strings.
    """
    names = ["ground", "roads"]
    if not no_lights:
        names.extend(["illumination", "lights"])
    names.extend(["atmosphere", "compositor"])
    return names


def render_scene_frame_with_progress(
    scene: "Scene",
    backend: RenderBackend,
    passes: t.List[RenderPass],
    pass_names: t.List[str],
) -> t.Dict[str, t.Any]:
    """Run all render passes in order, printing progress before each pass.

    Prints "Rendering pass N/total: name..." to stdout before each pass.
    Return value format matches render_scene_frame.

    Args:
        scene (Scene): The world scene with camera and road segments.
        backend (RenderBackend): Target render surface.
        passes (list[RenderPass]): Ordered passes to run.
        pass_names (list[str]): Human-readable names matching pass order.

    Returns:
        dict[str, Any]: Accumulated stats from all passes.
    """
    total = len(passes)
    stats: t.Dict[str, t.Any] = {}

    for i, (render_pass, name) in enumerate(zip(passes, pass_names)):
        print(f"Rendering pass {i + 1}/{total}: {name}...")
        result = render_pass(scene, scene.camera, backend)
        if isinstance(result, dict):
            stats.update(result)
        elif isinstance(result, int):
            stats["segments_drawn"] = result

    return stats


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
