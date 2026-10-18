"""Light render passes — dots for preview, core + glow halos for export."""

import typing as t
from dataclasses import dataclass

import numpy as np

from ..camera.camera import Camera
from ..camera.projection import project_points
from .backend import RenderBackend
from .color_space import srgb255_to_linear
from .render_utils import apply_brightness_falloff, compute_focal_length, extract_positions

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Minimum glow radius in screen pixels before we skip drawing the halo
_MIN_GLOW_RADIUS_PX: float = 2.0


@dataclass(frozen=True)
class LightSprite:
    """Screen-space core + glow parameters for one light at one camera.

    Attributes:
        linear_color: Linear (R, G, B) emission color.
        core_radius_px: Core disc radius in pixels (at least 1).
        core_brightness: Linear core radiance (brightness after extinction).
        glow_radius_px: Glow halo radius in pixels; 0.0 when the halo is too
            small to draw.
        glow_peak: Linear glow radiance at the halo center.
    """

    linear_color: t.Tuple[float, float, float]
    core_radius_px: float
    core_brightness: float
    glow_radius_px: float
    glow_peak: float


def compute_light_sprite(
    light: "LightSource",
    distance: float,
    effective_brightness: float,
    focal_length: float,
    atmos: t.Any,
) -> LightSprite:
    """Compute a light's on-screen core and glow sizes and intensities.

    Shared by the CPU export pass and the GPU tuning scene so both draw
    identical cores and halos.

    Args:
        light (LightSource): The light source.
        distance (float): Camera distance to the light in meters.
        effective_brightness (float): Brightness after atmospheric extinction.
        focal_length (float): Focal length in pixels.
        atmos: AtmosphereConfig (glow expansion, glow_max_alpha).

    Returns:
        LightSprite: Core and glow parameters for this frame.
    """
    # A light's emitted radiance dims by extinction (effective_brightness)
    # but keeps its own color; the haze veil over it is added per-pixel by
    # the airlight term in the atmosphere pass.
    glow_radius_px = _glow_radius_px(
        _resolve_glow_radius(light, distance, atmos), distance, focal_length
    )
    if glow_radius_px < _MIN_GLOW_RADIUS_PX:
        glow_radius_px = 0.0
    return LightSprite(
        linear_color=srgb255_to_linear(light.color),
        core_radius_px=_core_radius(light, distance, focal_length, effective_brightness),
        core_brightness=effective_brightness,
        glow_radius_px=glow_radius_px,
        glow_peak=(atmos.glow_max_alpha / 255.0) * effective_brightness,
    )


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

    Distances and falloff-attenuated brightness are computed once for every
    light with vectorized numpy operations, then indexed inside the draw loop.

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

    distances = _compute_light_distances(positions, camera.position)
    eff_brightness = _compute_effective_brightness(scene.lights, distances, atmos.falloff_distance)

    drawn = 0
    for i in np.nonzero(valid_mask)[0]:
        light = scene.lights[i]
        if not light.emits_core:
            continue
        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        radius = _preview_radius(float(eff_brightness[i]))
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

    distances = _compute_light_distances(positions, camera.position)
    eff_brightness = _compute_effective_brightness(scene.lights, distances, atmos.falloff_distance)
    visible_indices = _get_visible_indices(valid_mask, eff_brightness, atmos.min_brightness_threshold)

    drawn = 0
    for i in visible_indices:
        light = scene.lights[i]
        if not light.emits_core:
            continue
        center = (float(screen_coords[i, 0]), float(screen_coords[i, 1]))
        sprite = compute_light_sprite(
            light, float(distances[i]), float(eff_brightness[i]), focal_length, atmos
        )

        _accumulate_core(
            light_buffer, center[0], center[1], sprite.core_radius_px,
            sprite.linear_color, sprite.core_brightness,
        )

        if sprite.glow_radius_px > 0.0:
            _accumulate_glow(
                light_buffer,
                center[0],
                center[1],
                sprite.linear_color,
                sprite.glow_radius_px,
                sprite.glow_peak,
                atmos.glow_falloff_steepness,
                atmos.glow_wing_exponent,
                atmos.glow_tail_strength,
                atmos.glow_cutoff_mult,
            )
        drawn += 1

    backend.add_linear(light_buffer)
    return {"lights_drawn": drawn}


# ---------------------------------------------------------------------------
# Private helpers — export loop support
# ---------------------------------------------------------------------------


def _compute_light_distances(positions: np.ndarray, camera_position: np.ndarray) -> np.ndarray:
    """Compute per-light camera distance for every light in one vectorized call.

    Args:
        positions (np.ndarray): (N, 3) light positions in ENU metres.
        camera_position (np.ndarray): (3,) camera position in ENU metres.

    Returns:
        np.ndarray: (N,) distances in metres, each clamped to at least 1.0.
    """
    return np.maximum(1.0, np.linalg.norm(positions - camera_position, axis=1))


def _compute_effective_brightness(
    lights: t.List["LightSource"],
    distances: np.ndarray,
    falloff_distance: float,
) -> np.ndarray:
    """Compute atmospheric-falloff-attenuated brightness for every light at once.

    Args:
        lights (list[LightSource]): All lights in the scene.
        distances (np.ndarray): (N,) per-light camera distances in metres.
        falloff_distance (float): Characteristic decay distance in metres.

    Returns:
        np.ndarray: (N,) attenuated brightness values.
    """
    brightnesses = np.fromiter((light.brightness for light in lights), dtype=float, count=len(lights))
    return brightnesses * np.exp(-distances / falloff_distance)


def _get_visible_indices(
    valid_mask: np.ndarray,
    eff_brightness: np.ndarray,
    min_brightness_threshold: float,
) -> np.ndarray:
    """Collect indices of lights that pass the mask and brightness threshold.

    The export light buffer is a float32 linear accumulation buffer — cores
    and glows from different lights simply add together, so paint order does
    not affect the result. Indices are returned in natural (ascending) order;
    no far-to-near sort is needed.

    Args:
        valid_mask (np.ndarray): Boolean mask from project_points.
        eff_brightness (np.ndarray): Per-light brightness after atmospheric falloff.
        min_brightness_threshold (float): Minimum effective brightness required to draw.

    Returns:
        np.ndarray: Indices of drawable lights, in natural order.
    """
    visible = valid_mask & (eff_brightness >= min_brightness_threshold)
    return np.nonzero(visible)[0]


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
    wing_exponent: float,
    tail_strength: float,
    cutoff_mult: float,
) -> None:
    """Add a glare glow contribution to the linear accumulation buffer.

    The glow is the sum of two components: a tight gaussian core (the sharp
    centre of the halo) and a wide power-law tail (1/(1+(d/rc)^n)) that
    produces the faint, far-reaching glare real night lights show. A pure
    gaussian falls off too fast to look glary; the power-law wings are what
    give the characteristic veiling glare. The tail is bounded by
    ``cutoff_mult`` so a single light never paints the whole frame.

    Both components are 1.0 at the centre, so the centre stays at ``peak``
    regardless of the core/tail mix. Contributions from multiple lights
    accumulate additively — overlapping halos get brighter, unbounded, and
    are rolled off by the tone map later.

    Args:
        buffer (np.ndarray): Float32 (H, W, 3) linear buffer, modified in place.
        cx (float): Screen-space center x.
        cy (float): Screen-space center y.
        color_linear (tuple[float, float, float]): Linear RGB glow color in [0, 1].
        radius_px (float): Glow radius in screen pixels.
        peak (float): Peak glow intensity in linear radiance at the center.
        steepness (float): Gaussian core sigma multiplier — higher = sharper core.
        wing_exponent (float): Power-law tail exponent n in 1/(1+(d/rc)^n).
            Lower (≈2) gives wider, glarier wings; higher gives a tighter halo.
        tail_strength (float): Fraction of the peak carried by the power-law
            wings vs the gaussian core, in [0, 1].
        cutoff_mult (float): Tail extent as a multiple of radius_px, bounding
            the painted region.
    """
    h, w, _ = buffer.shape
    margin = int(radius_px * cutoff_mult) + 2
    # Absolute cap so a near (large-radius) light cannot paint the whole frame.
    margin = min(margin, max(h, w) // 2)
    x1 = max(0, int(cx) - margin)
    x2 = min(w, int(cx) + margin + 1)
    y1 = max(0, int(cy) - margin)
    y2 = min(h, int(cy) + margin + 1)
    if x1 >= x2 or y1 >= y2:
        return

    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    sigma = max(radius_px / 2.0 * steepness, 0.5)
    core = np.exp(-dist_sq / (2.0 * sigma * sigma))

    knee = max(radius_px * 0.5, 1.0)
    dist = np.sqrt(dist_sq)
    tail = 1.0 / (1.0 + (dist / knee) ** wing_exponent)

    intensity = peak * ((1.0 - tail_strength) * core + tail_strength * tail)

    for c, col_val in enumerate(color_linear):
        buffer[y1:y2, x1:x2, c] += col_val * intensity
