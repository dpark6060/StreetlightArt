"""Light render passes — dots for preview, core + glow halos for export."""

import typing as t

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import project_points
from .backend import RenderBackend
from .color_space import srgb255_to_linear
from .render_utils import apply_brightness_falloff, camera_distance, compute_focal_length, extract_positions

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Minimum glow radius in screen pixels before we skip drawing the halo
_MIN_GLOW_RADIUS_PX: float = 2.0


def render_lights_preview(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> t.Dict[str, int]:
    """Render lights as colored dots for interactive preview.

    Projects every light in the scene to screen space. Draws a filled circle
    for each visible light. Circle radius scales with brightness (attenuated by
    atmospheric falloff) but no glow halos or color shift are applied — this
    pass is intended for real-time pygame preview.

    Args:
        scene (Scene): The world scene containing all light sources.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface.

    Returns:
        dict[str, int]: Stats with key "lights_drawn" reporting visible count.
    """
    if not scene.lights:
        return {"lights_drawn": 0}

    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()
    atmos = scene.config.atmosphere

    positions = extract_positions(scene.lights)
    screen_coords, valid_mask = project_points(
        positions, view_matrix, proj_matrix, (width, height)
    )

    drawn = 0
    for i, light in enumerate(scene.lights):
        if not valid_mask[i]:
            continue
        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        distance = camera_distance(light.position, camera.position)
        effective_brightness = _apply_brightness_falloff(
            light.brightness, distance, atmos.falloff_distance
        )
        radius = _preview_radius(effective_brightness)
        backend.draw_circle(center, radius, light.color)
        drawn += 1

    return {"lights_drawn": drawn}


def render_lights_export(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
) -> t.Dict[str, int]:
    """Render lights as bright cores plus gaussian glow halos, in linear light.

    For each visible light, applies atmospheric perspective:
      - Brightness falloff: exponential attenuation by distance.
      - Color shift: blends toward warm haze color at far distances.
      - Glow expansion: optionally grows glow radius with scatter if enabled.
    Lights below min_brightness_threshold are skipped entirely.

    Both the core and the glow are emission, accumulated into a single float32
    linear-radiance buffer (cores and halos add together; order-independent, so
    no far-to-near sorting hack is needed) and handed to the backend's linear
    light buffer via add_linear. The tone map later rolls bright overlaps off to
    white instead of hard-clipping.

    Args:
        scene (Scene): The world scene containing all light sources.
        camera (Camera): The active camera.
        backend (RenderBackend): Target render surface (must support add_linear).

    Returns:
        dict[str, int]: Stats with key "lights_drawn" reporting visible count.
    """
    if not scene.lights:
        return {"lights_drawn": 0}

    view_matrix = camera.get_view_matrix()
    proj_matrix = camera.get_projection_matrix()
    width, height = backend.get_surface_size()

    positions = extract_positions(scene.lights)
    screen_coords, valid_mask = project_points(
        positions, view_matrix, proj_matrix, (width, height)
    )

    focal_length = compute_focal_length(camera, width)
    atmos = scene.config.atmosphere
    light_buffer = np.zeros((height, width, 3), dtype=np.float32)

    visible_indices = _get_visible_indices(scene.lights, valid_mask, camera, atmos)

    drawn = 0
    for i in visible_indices:
        light = scene.lights[i]
        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        distance = camera_distance(light.position, camera.position)

        effective_brightness = _apply_brightness_falloff(
            light.brightness, distance, atmos.falloff_distance
        )
        effective_color = _apply_haze_color(
            light.color, distance, atmos.haze_color, atmos.max_distance
        )
        linear_color = srgb255_to_linear(effective_color)
        effective_glow_radius = _resolve_glow_radius(light, distance, atmos)

        core_radius = _core_radius(light, distance, focal_length, effective_brightness)
        radius_px = _glow_radius_px(effective_glow_radius, distance, focal_length)

        _accumulate_core(
            light_buffer, center[0], center[1], core_radius,
            linear_color, effective_brightness,
        )

        if radius_px >= _MIN_GLOW_RADIUS_PX:
            _accumulate_glow(
                light_buffer,
                center[0],
                center[1],
                linear_color,
                radius_px,
                (atmos.glow_max_alpha / 255.0) * effective_brightness,
                atmos.glow_falloff_steepness,
            )
        drawn += 1

    backend.add_linear(light_buffer)
    return {"lights_drawn": drawn}


# ---------------------------------------------------------------------------
# Private helpers — export loop support
# ---------------------------------------------------------------------------


def _get_visible_indices(
    lights: t.List["LightSource"],
    valid_mask: np.ndarray,
    camera: Camera,
    atmos: t.Any,
) -> t.List[int]:
    """Collect indices of lights that pass the mask and brightness threshold.

    Returns indices sorted by descending camera distance so far lights are
    drawn first and closer lights paint over them.

    Args:
        lights (list[LightSource]): All lights in the scene.
        valid_mask (np.ndarray): Boolean mask from project_points.
        camera (Camera): Active camera (used to compute distances).
        atmos: AtmosphereConfig with falloff_distance and min_brightness_threshold.

    Returns:
        list[int]: Sorted indices of drawable lights, farthest first.
    """
    candidates = []
    for i, light in enumerate(lights):
        if not valid_mask[i]:
            continue
        distance = camera_distance(light.position, camera.position)
        effective_brightness = _apply_brightness_falloff(
            light.brightness, distance, atmos.falloff_distance
        )
        if effective_brightness < atmos.min_brightness_threshold:
            continue
        candidates.append((distance, i))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [i for _, i in candidates]


def _resolve_glow_radius(
    light: "LightSource",
    distance: float,
    atmos: t.Any,
) -> float:
    """Return effective glow radius, expanding with distance only if configured.

    Args:
        light (LightSource): The light source.
        distance (float): Distance from camera to light in meters.
        atmos: AtmosphereConfig with glow_expansion_enabled and scatter_distance.

    Returns:
        float: Effective glow radius in meters.
    """
    if atmos.glow_expansion_enabled:
        return _apply_glow_expansion(light.glow_radius, distance, atmos.scatter_distance)
    return light.glow_radius


# ---------------------------------------------------------------------------
# Private helpers — atmospheric perspective
# ---------------------------------------------------------------------------


def _apply_brightness_falloff(
    brightness: float, distance: float, falloff_distance: float
) -> float:
    """Apply exponential brightness attenuation based on distance.

    Args:
        brightness (float): Base brightness scalar.
        distance (float): Distance from camera in meters.
        falloff_distance (float): Characteristic decay distance in meters.

    Returns:
        float: Attenuated brightness.
    """
    return apply_brightness_falloff(brightness, distance, falloff_distance)


def _apply_haze_color(
    color: t.Tuple[int, int, int],
    distance: float,
    haze_color: t.Tuple[int, int, int],
    max_distance: float,
) -> t.Tuple[int, int, int]:
    """Blend light color toward warm haze as distance increases.

    Args:
        color (tuple[int, int, int]): Original RGB light color.
        distance (float): Distance from camera in meters.
        haze_color (tuple[int, int, int]): Target RGB haze color.
        max_distance (float): Distance at which haze is fully applied.

    Returns:
        tuple[int, int, int]: Blended RGB color.
    """
    t_blend = min(1.0, distance / max_distance)
    blended = tuple(
        max(0, min(255, int(c * (1.0 - t_blend) + h * t_blend)))
        for c, h in zip(color, haze_color)
    )
    return (blended[0], blended[1], blended[2])


def _apply_glow_expansion(
    glow_radius: float, distance: float, scatter_distance: float
) -> float:
    """Expand glow radius with distance to simulate atmospheric scatter.

    Args:
        glow_radius (float): Base world-space glow radius in meters.
        distance (float): Distance from camera in meters.
        scatter_distance (float): Distance at which the radius doubles.

    Returns:
        float: Expanded glow radius in meters.
    """
    return glow_radius * (1.0 + distance / scatter_distance)


# ---------------------------------------------------------------------------
# Private helpers — preview
# ---------------------------------------------------------------------------


def _preview_radius(brightness: float) -> float:
    """Compute a screen-space dot radius for the preview pass.

    Args:
        brightness (float): Light brightness scalar (1.0 = normal).

    Returns:
        float: Radius in pixels, at least 1.
    """
    return max(1.0, brightness * 3.0)


# ---------------------------------------------------------------------------
# Private helpers — export
# ---------------------------------------------------------------------------


def _core_radius(
    light: "LightSource",
    distance: float,
    focal_length: float,
    effective_brightness: float,
) -> float:
    """Compute the bright-core circle radius in screen pixels.

    Args:
        light (LightSource): The light source.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.
        effective_brightness (float): Brightness after atmospheric falloff.

    Returns:
        float: Core radius in pixels, at least 1.
    """
    screen_size = light.size / distance * focal_length
    return max(1.0, screen_size * effective_brightness)


def _glow_radius_px(glow_radius: float, distance: float, focal_length: float) -> float:
    """Compute the glow halo radius in screen pixels.

    Args:
        glow_radius (float): World-space glow radius in meters.
        distance (float): Distance from camera to light in meters.
        focal_length (float): Focal length in pixels.

    Returns:
        float: Glow radius in pixels.
    """
    return glow_radius / distance * focal_length


def _accumulate_core(
    buffer: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
    color_linear: t.Tuple[float, float, float],
    brightness: float,
) -> None:
    """Add a bright emissive core disc to the linear accumulation buffer.

    Paints a filled disc of ``color_linear * brightness`` with a 1px soft edge
    for anti-aliasing. Sub-pixel cores (radius < 1) still light their centre
    pixel via the +0.5 coverage term, so no separate point-drawing path is
    needed.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer, modified in place.
        cx (float): Screen-space center x.
        cy (float): Screen-space center y.
        radius_px (float): Core radius in screen pixels.
        color_linear (tuple[float, float, float]): Linear RGB core color in [0, 1].
        brightness (float): Linear brightness multiplier for the core.
    """
    h, w, _ = buffer.shape
    margin = int(radius_px) + 2
    x1 = max(0, int(cx) - margin)
    x2 = min(w, int(cx) + margin + 1)
    y1 = max(0, int(cy) - margin)
    y2 = min(h, int(cy) + margin + 1)
    if x1 >= x2 or y1 >= y2:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    coverage = np.clip(radius_px - dist + 0.5, 0.0, 1.0)
    contrib = coverage * brightness

    for c, col_val in enumerate(color_linear):
        buffer[y1:y2, x1:x2, c] += col_val * contrib


def _accumulate_glow(
    buffer: np.ndarray,
    cx: float,
    cy: float,
    color_linear: t.Tuple[float, float, float],
    radius_px: float,
    peak: float,
    steepness: float,
) -> None:
    """Add a gaussian glow contribution to the linear accumulation buffer.

    Operates only on the bounding box of the glow for efficiency. Contributions
    from multiple lights accumulate additively — overlapping halos get brighter,
    unbounded, and are rolled off by the tone map later.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer, modified in place.
        cx (float): Screen-space center x.
        cy (float): Screen-space center y.
        color_linear (tuple[float, float, float]): Linear RGB glow color in [0, 1].
        radius_px (float): Glow radius in screen pixels.
        peak (float): Peak glow intensity in linear radiance at the center.
        steepness (float): Gaussian sigma multiplier — higher = sharper falloff.
    """
    h, w, _ = buffer.shape
    margin = int(radius_px * 2.5) + 2
    x1 = max(0, int(cx) - margin)
    x2 = min(w, int(cx) + margin + 1)
    y1 = max(0, int(cy) - margin)
    y2 = min(h, int(cy) + margin + 1)
    if x1 >= x2 or y1 >= y2:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = max(radius_px / 2.0 * steepness, 0.5)
    intensity = peak * np.exp(-dist_sq / (2.0 * sigma * sigma))

    for c, col_val in enumerate(color_linear):
        buffer[y1:y2, x1:x2, c] += col_val * intensity
