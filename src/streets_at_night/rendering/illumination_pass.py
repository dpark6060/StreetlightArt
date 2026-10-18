"""Illumination render pass — world-space ground-plane light pooling for export.

Builds a float32 illumination map in world (ENU) space by painting cone-model
blobs for each light source. The map is then sampled via a per-pixel raycast from
camera through screen pixels down to the z=0 ground plane, and the result is
additively composited onto the main image.

Each blob uses a two-zone cone model: an inner flat zone at full brightness up to
inner_fraction * radius, followed by a fade zone where intensity falls from 1 to 0
using (1 - fade_t)^falloff. The falloff exponent controls the curve steepness in the
fade zone; the inner fraction controls the size of the flat core.

This approach is physically grounded: illumination represents light reflecting off
the road surface in world space, so perspective foreshortening and occlusion by the
horizon are handled automatically by the raycast — no special-casing needed.
"""

import math
import typing as t

import numpy as np

from ..camera.camera import Camera
from .backend import RenderBackend
from ..models.illumination import BeamIllumination, DirectionalConeIllumination
from .color_space import srgb255_to_linear
from .raycast import GroundHit, build_ground_gbuffer
from .render_utils import apply_brightness_falloff, camera_distance

if t.TYPE_CHECKING:
    from ..models.light_source import LightSource
    from ..scene.scene import Scene


# Base illumination map resolution in meters per pixel.
# Lower values = finer map = slower but more detailed. 2.0 m/px balances quality and performance.
_BASE_ILLUMINATION_RESOLUTION: float = 2.0

# Padding added around the visible-ground bounds so lights near the edges
# still spill their pool into frame.
_BOUNDS_PADDING_M: float = 200.0

# Maximum illumination map size in pixels (H * W). Resolution is coarsened
# beyond this to cap the float32 RGB map at ~3 GB regardless of how much
# ground the camera can see (e.g. a near-horizon shot over a large bbox).
_MAX_ILLUM_MAP_PIXELS: int = 256_000_000


def render_illumination(
    scene: "Scene",
    camera: Camera,
    backend: RenderBackend,
    illumination_resolution_multiplier: float = 1.0,
    gbuffer: t.Optional[GroundHit] = None,
) -> t.Dict[str, t.Any]:
    """Render ground-plane illumination from light sources via world-space map.

    Builds a float32 linear illumination map sized to the *visible* ground —
    derived from the shared G-buffer's per-pixel hit coordinates, not the full
    road network extent — by painting per-light cone-model blobs at their
    world positions. Each blob's brightness is attenuated by exponential
    distance falloff. Each screen pixel is raycast to the z=0 ground plane;
    the hit world coordinate is sampled from the map and added to the
    backend's linear light buffer.

    Returns an empty dict if there are no lights or no pixel hits the ground
    (an all-sky frame).

    Args:
        scene (Scene): World scene containing lights and config. Uses
            scene.config.atmosphere.falloff_distance and
            min_brightness_threshold for distance attenuation and culling.
        camera (Camera): Active camera providing position, far plane, and
            matrices.
        backend (RenderBackend): Target render surface (must support add_linear).
        illumination_resolution_multiplier (float): Quality multiplier from the
            active QualityPreset. Higher values produce a finer illumination map.
            Defaults to 1.0.
        gbuffer (GroundHit | None): Precomputed ground raycast for this camera and
            size. When None, it is computed here. The pipeline shares one G-buffer
            between this pass and the ground export pass to avoid raycasting twice.

    Returns:
        dict[str, Any]: Stats dict, empty when skipped. Otherwise contains
            "map_info": (min_x, max_y, resolution) of the built illumination map.
    """
    if not scene.lights:
        return {}

    atmos = scene.config.atmosphere
    illumination_brightness = atmos.illumination_brightness
    falloff_distance = atmos.falloff_distance
    min_brightness_threshold = atmos.min_brightness_threshold

    width, height = backend.get_surface_size()
    if gbuffer is None:
        gbuffer = build_ground_gbuffer(camera, width, height)

    max_light_radius = max(
        (
            effective_pool_radius(light)
            for light in scene.lights
            if light.illumination is not None
        ),
        default=0.0,
    )
    padding = max(_BOUNDS_PADDING_M, max_light_radius)
    bounds = _compute_visible_bounds(gbuffer, camera, padding, scene.get_lights_aabb())
    if bounds is None:
        return {}

    resolution = _BASE_ILLUMINATION_RESOLUTION / illumination_resolution_multiplier
    resolution = _auto_scale_illumination_resolution(*bounds, resolution)

    illum_map, map_info = _build_illumination_map(
        scene.lights, bounds, resolution, illumination_brightness,
        camera.position, falloff_distance, min_brightness_threshold,
    )

    knee = scene.config.lights.illumination
    illum_map = _apply_soft_knee(illum_map, knee.knee_start, knee.knee_max)

    min_x, max_y, res = map_info
    illum_img = _sample_illumination_map(
        illum_map, gbuffer.hit_x, gbuffer.hit_y, gbuffer.ground_mask, min_x, max_y, res
    )

    backend.add_linear(illum_img)

    return {"map_info": map_info}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_visible_bounds(
    gbuffer: GroundHit,
    camera: Camera,
    padding: float,
    lights_aabb: t.Optional[t.Tuple[float, float, float, float]] = None,
) -> t.Optional[t.Tuple[float, float, float, float]]:
    """Compute illumination map world bounds from the visible ground G-buffer.

    Uses the actual per-pixel ground hit coordinates for this frame — not the
    full road-network extent — so the map only covers what the camera can
    see. Hit coordinates are clamped to camera.position +/- camera.far first:
    near-horizon pixels can report hits tens of kilometers away even though
    the raycast itself does not enforce the far clip plane, which would
    otherwise re-inflate the bounds.

    When ``lights_aabb`` is supplied, the result is intersected with it
    (expanded by ``padding`` so pool spill near the edge is preserved). No
    light exists beyond that box, so the cropped-out region carries zero
    illumination anyway; cropping keeps the map from ballooning to a
    near-horizon shot's tens-of-kilometers footprint over empty ground, which
    would otherwise force the pixel-budget guard to coarsen resolution on the
    pools that are actually visible.

    Args:
        gbuffer (GroundHit): Per-pixel ground raycast for this frame.
        camera (Camera): Active camera, used for the far-plane clamp.
        padding (float): Meters of padding added on every side so lights near
            the visible edge still spill their pool into frame.
        lights_aabb (tuple[float, float, float, float] | None): (min_x, min_y,
            max_x, max_y) bounding box of all light positions, or None to skip
            the intersection.

    Returns:
        tuple[float, float, float, float] | None: (min_x, min_y, max_x, max_y)
            in meters, or None if no pixel in the G-buffer hits the ground, or
            the visible region does not overlap any light.
    """
    mask = gbuffer.ground_mask
    if not np.any(mask):
        return None

    far = camera.far
    hit_x = np.clip(gbuffer.hit_x, camera.position[0] - far, camera.position[0] + far)
    hit_y = np.clip(gbuffer.hit_y, camera.position[1] - far, camera.position[1] + far)

    min_x = float(hit_x[mask].min()) - padding
    max_x = float(hit_x[mask].max()) + padding
    min_y = float(hit_y[mask].min()) - padding
    max_y = float(hit_y[mask].max()) + padding

    if lights_aabb is not None:
        lx0, ly0, lx1, ly1 = lights_aabb
        min_x = max(min_x, lx0 - padding)
        min_y = max(min_y, ly0 - padding)
        max_x = min(max_x, lx1 + padding)
        max_y = min(max_y, ly1 + padding)
        if min_x >= max_x or min_y >= max_y:
            return None

    return min_x, min_y, max_x, max_y


def _auto_scale_illumination_resolution(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    resolution: float,
) -> float:
    """Coarsen resolution (m/px) if the map would exceed the pixel budget.

    Mirrors flat_map._auto_scale_resolution's approach of growing meters-per-
    pixel until the map fits, but bounds by total pixel count
    (_MAX_ILLUM_MAP_PIXELS) rather than a per-axis dimension cap, since the
    illumination map tracks the visible ground rather than the full road
    network. Prints a one-line notice when it engages so silent quality loss
    doesn't read as full coverage.

    Args:
        min_x (float): Map min x in meters.
        min_y (float): Map min y in meters.
        max_x (float): Map max x in meters.
        max_y (float): Map max y in meters.
        resolution (float): Requested meters per pixel.

    Returns:
        float: Meters per pixel, increased just enough to fit the pixel budget.
    """
    world_w = max_x - min_x
    world_h = max_y - min_y
    n_px = (world_w / resolution) * (world_h / resolution)
    if n_px <= _MAX_ILLUM_MAP_PIXELS:
        return resolution

    scaled = resolution * math.sqrt(n_px / _MAX_ILLUM_MAP_PIXELS)
    print(
        f"Illumination map ({world_w:.0f}m x {world_h:.0f}m) exceeds pixel budget "
        f"at {resolution:.2f} m/px; coarsening to {scaled:.2f} m/px."
    )
    return scaled


def effective_pool_radius(light: "LightSource") -> float:
    """Compute the effective ground pool radius for a light, in meters.

    ``BeamIllumination`` lights derive the pool radius from the beam cone
    geometry (pole height + angle); ``ConeIllumination`` lights use the
    authored radius; ``DirectionalConeIllumination`` lights use ``length`` (the
    oriented cone fits inside a circle of that radius centered on the apex, so
    AABB culling and the paint window stay correct). Shared by AABB culling and
    blob painting so the two can't drift apart. ``None`` (no ground pool) is not
    a valid input here — callers skip those lights before reaching this helper.

    Args:
        light (LightSource): Light source with a non-None illumination model.

    Returns:
        float: Effective ground pool radius in meters.

    Raises:
        ValueError: If a BeamIllumination light has no positive pole height
            (a physical pool is undefined at ground level — fail loud rather
            than silently producing a wrong pool).
    """
    model = light.illumination
    if isinstance(model, BeamIllumination):
        height = float(light.position[2])
        if height <= 0.0:
            raise ValueError(
                "BeamIllumination light requires a positive pole height "
                f"(position z); got z={height}"
            )
        return height * math.tan(math.radians(model.beam_angle_deg / 2.0))
    if isinstance(model, DirectionalConeIllumination):
        return model.length
    return model.radius


def _is_light_visible(
    light: "LightSource",
    bounds: t.Tuple[float, float, float, float],
    camera_position: np.ndarray,
    falloff_distance: float,
    min_brightness_threshold: float,
) -> bool:
    """Check whether a light can contribute visible illumination to the map.

    Culls two cases cheaply, before any per-pixel painting work: lights whose
    blob AABB (position +/- effective pool radius) does not intersect the map
    bounds, and lights whose camera-distance-attenuated peak has fallen below
    the atmosphere's min_brightness_threshold.

    Args:
        light (LightSource): Candidate light source.
        bounds (tuple[float, float, float, float]): (min_x, min_y, max_x, max_y)
            of the illumination map in world ENU meters.
        camera_position (np.ndarray): (3,) camera position in world ENU meters.
        falloff_distance (float): Characteristic decay distance in meters.
        min_brightness_threshold (float): Minimum attenuated peak brightness
            for a light to be considered visible.

    Returns:
        bool: True if the light should be painted.
    """
    min_x, min_y, max_x, max_y = bounds
    radius = effective_pool_radius(light)
    lx, ly = float(light.position[0]), float(light.position[1])
    if lx + radius < min_x or lx - radius > max_x or ly + radius < min_y or ly - radius > max_y:
        return False

    dist = camera_distance(light.position, camera_position)
    effective_brightness = apply_brightness_falloff(light.brightness, dist, falloff_distance)
    return effective_brightness * light.illumination_intensity >= min_brightness_threshold


def _build_illumination_map(
    lights: t.List["LightSource"],
    bounds: t.Tuple[float, float, float, float],
    resolution: float,
    illumination_brightness: int,
    camera_position: np.ndarray,
    falloff_distance: float,
    min_brightness_threshold: float,
) -> t.Tuple[np.ndarray, t.Tuple[float, float, float]]:
    """Build a float32 RGB illumination map by painting cone-model blobs for each light.

    Initializes a zero-filled (H, W, 3) float32 array, then paints each
    light's contribution via _paint_illumination_blob (with distance-based
    brightness falloff applied). Lights culled by _is_light_visible — outside
    the map bounds or below min_brightness_threshold once attenuated — are
    skipped entirely, avoiding per-pixel work for lights that can't be seen.

    Args:
        lights (list[LightSource]): All light sources to consider.
        bounds (tuple[float, float, float, float]): (min_x, min_y, max_x, max_y)
            in world ENU meters.
        resolution (float): Meters per pixel.
        illumination_brightness (int): Base brightness scalar from AtmosphereConfig.
        camera_position (np.ndarray): (3,) camera position in world ENU meters, used
            for per-light distance falloff.
        falloff_distance (float): Characteristic decay distance in meters for the
            exponential brightness attenuation.
        min_brightness_threshold (float): Minimum attenuated peak brightness
            for a light to be painted.

    Returns:
        tuple[np.ndarray, tuple[float, float, float]]:
            - illum_map: (H, W, 3) float32 array of illumination RGB values.
            - map_info: (min_x, max_y, resolution) for coordinate conversion.
    """
    min_x, min_y, max_x, max_y = bounds

    width = max(1, int((max_x - min_x) / resolution))
    height = max(1, int((max_y - min_y) / resolution))

    illum_map = np.zeros((height, width, 3), dtype=np.float32)

    for light in lights:
        if light.illumination is None:
            continue
        if not _is_light_visible(
            light, bounds, camera_position, falloff_distance, min_brightness_threshold
        ):
            continue
        _paint_illumination_blob(
            illum_map, light, min_x, max_y, resolution, illumination_brightness,
            camera_position, falloff_distance,
        )

    map_info = (min_x, max_y, resolution)
    return illum_map, map_info


def _apply_soft_knee(illum_map: np.ndarray, knee_start: float, knee_max: float) -> np.ndarray:
    """Compress stacked illumination above knee_start toward knee_max.

    Identity below ``knee_start``; above it, values compress smoothly
    (C1-continuous at the knee) and asymptotically approach ``knee_max`` —
    monotonic, so brighter stacks always stay brighter. Disabled (returns
    the input unchanged) when ``knee_max <= knee_start``.

    Mutates ``illum_map`` in place (and returns it) to avoid allocating
    several full-map temporaries — on a budget-capped ~3 GB map the old
    out-of-place form churned ~10 GB transient. The only caller rebinds the
    result and never reuses the input, so in-place is safe. Behavior is
    bit-identical to the out-of-place form for float32 inputs.

    Args:
        illum_map (np.ndarray): (H, W, 3) float32 linear illumination map.
            Mutated in place.
        knee_start (float): Intensity where compression begins.
        knee_max (float): Asymptotic intensity ceiling.

    Returns:
        np.ndarray: The compressed map (the input array, mutated; returned
        unchanged when disabled).
    """
    if knee_max <= knee_start:
        return illum_map
    span = knee_max - knee_start

    # over = max(illum_map - knee_start, 0.0)
    over = illum_map - knee_start
    np.maximum(over, 0.0, out=over)
    # denom = over + span (needed before `over` is overwritten below)
    denom = over + span
    # compressed = knee_start + span * over / denom, computed in `over`
    over *= span
    over /= denom
    over += knee_start
    # Keep compressed only where above the knee; leave the rest untouched.
    np.copyto(illum_map, over, where=illum_map > knee_start)
    return illum_map


def _paint_illumination_blob(
    illum_map: np.ndarray,
    light: "LightSource",
    min_x: float,
    max_y: float,
    resolution: float,
    illumination_brightness: int,
    camera_position: np.ndarray,
    falloff_distance: float,
) -> None:
    """Paint an intensity blob for one light source into the illumination map.

    Three falloff models are supported, selected per light:

    - Physical (``BeamIllumination``): the pool radius is derived from pole
      height and beam angle, and intensity follows the inverse-square +
      cosine-incidence law for a point source at height h,
      ``1 / (1 + (r/h)^2)^1.5``, with a reflector band past the bulb cutoff
      that fades to zero at the housing cutoff (see ``_physical_falloff``). A taller pole gives a wider, flatter,
      dimmer pool for free. Used by streetlights and parking lot lights.
    - Authored two-zone cone (``ConeIllumination``): the authored ``radius``
      with an inner flat zone and ``falloff`` fade. Used by lights with no
      meaningful pole height (signs, storefronts).
    - Oriented cone (``DirectionalConeIllumination``): a wedge cast forward
      along a heading — axial two-zone fade times an angular feather, zero
      behind the apex. Used by car headlights.

    Light brightness is attenuated by exponential distance falloff from the
    camera before computing the peak, matching the formula used in the glow pass.

    Y-axis convention: top of image = max_y in world (same as FlatMapMeta).

    Args:
        illum_map (np.ndarray): (H, W, 3) float32 array to paint into (modified in-place).
        light (LightSource): Light source with position, color, brightness, and
            a BeamIllumination (physical) or ConeIllumination model.
        min_x (float): World x coordinate corresponding to pixel column 0.
        max_y (float): World y coordinate corresponding to pixel row 0.
        resolution (float): Meters per pixel.
        illumination_brightness (int): Base brightness scalar from AtmosphereConfig.
        camera_position (np.ndarray): (3,) camera position in world ENU meters.
        falloff_distance (float): Characteristic decay distance in meters for
            exponential brightness attenuation.
    """
    height = float(light.position[2])
    is_physical = isinstance(light.illumination, BeamIllumination)
    radius_m = effective_pool_radius(light)

    cx = (light.position[0] - min_x) / resolution
    cy = (max_y - light.position[1]) / resolution
    radius_px = radius_m / resolution

    if radius_px < 1.0:
        return

    map_h, map_w = illum_map.shape[:2]
    x0 = max(0, int(cx - radius_px))
    x1 = min(map_w, int(cx + radius_px + 1))
    y0 = max(0, int(cy - radius_px))
    y1 = min(map_h, int(cy + radius_px + 1))

    if x0 >= x1 or y0 >= y1:
        return

    grid_y, grid_x = np.meshgrid(
        np.arange(y0, y1, dtype=np.float32),
        np.arange(x0, x1, dtype=np.float32),
        indexing="ij",
    )

    # Signed world-space deltas from the light. NOTE the y flip: map row 0 is
    # max_y (north edge), so increasing grid_y means decreasing world y.
    # Getting this sign wrong silently mirrors every directional cone
    # north/south, which no radial test would ever catch.
    dx_m = (grid_x - cx) * resolution
    dy_m = (cy - grid_y) * resolution
    dist_m = np.hypot(dx_m, dy_m)

    model = light.illumination
    if is_physical:
        falloff = _physical_falloff(
            dist_m, height, radius_m, model.bulb_fraction, model.reflector_falloff
        )
    elif isinstance(model, DirectionalConeIllumination):
        falloff = _directional_cone_falloff(dx_m, dy_m, model)
    else:
        falloff = _cone_falloff(dist_m, radius_m, model.inner_fraction, model.falloff)

    peak = compute_pool_peak(light, illumination_brightness, camera_position, falloff_distance)
    intensity = peak * falloff

    lin_r, lin_g, lin_b = srgb255_to_linear(light.color)
    illum_map[y0:y1, x0:x1, 0] += intensity * lin_r
    illum_map[y0:y1, x0:x1, 1] += intensity * lin_g
    illum_map[y0:y1, x0:x1, 2] += intensity * lin_b


def compute_pool_peak(
    light: "LightSource",
    illumination_brightness: int,
    camera_position: np.ndarray,
    falloff_distance: float,
) -> float:
    """Compute a light's peak ground-pool radiance before the falloff curve.

    Shared by the CPU illumination map and the GPU tuning scene so both
    scale pools identically.

    Args:
        light (LightSource): Light with a non-None illumination model.
        illumination_brightness (int): Base brightness scalar from
            AtmosphereConfig, authored on a 0-255 scale.
        camera_position (np.ndarray): (3,) camera position in ENU meters.
        falloff_distance (float): Characteristic extinction distance in meters.

    Returns:
        float: Linear peak radiance; multiply by the falloff curve and the
        light's linear color to get the painted value.
    """
    dist = camera_distance(light.position, camera_position)
    effective_brightness = apply_brightness_falloff(light.brightness, dist, falloff_distance)
    # illumination_brightness is authored on a 0-255 scale; /255 maps it into
    # linear radiance so the painted values are linear-light, not sRGB codes.
    peak = (illumination_brightness / 255.0) * light.illumination_intensity * effective_brightness
    if isinstance(light.illumination, BeamIllumination):
        # Illuminance under a lamp goes as I / h^2: a shorter pole gives a
        # smaller but brighter pool, so total light depends on intensity alone.
        peak *= (_REFERENCE_POLE_HEIGHT_M / float(light.position[2])) ** 2
    return peak


# Pole height (m) at which a physical pool's peak equals its authored
# intensity. Shorter poles peak brighter by (ref / h)^2, taller ones dimmer.
# Set to a typical highway pole so tuned highway pools keep their look.
_REFERENCE_POLE_HEIGHT_M: float = 13.5

# Minimum fraction of the beam radius over which the pool edge fades to zero.
_CUTOFF_EDGE_SOFTNESS: float = 0.15


def _physical_falloff(
    dist_m: np.ndarray,
    height: float,
    radius_m: float,
    bulb_fraction: float = 1.0,
    reflector_falloff: float = 1.0,
) -> np.ndarray:
    """Compute ground illuminance falloff for a pole-mounted luminaire.

    Follows the inverse-square + cosine-incidence law, normalized to 1.0
    directly under the lamp: ``E(r)/E(0) = 1 / (1 + (r/h)^2)^1.5``. Inside the
    bulb radius (``bulb_fraction * radius_m``) that curve is used as-is. Past
    it, only the reflector lights the ground, and the curve is multiplied by
    ``(1 - smoothstep(t)) ** reflector_falloff`` as t runs 0 -> 1 from the bulb
    radius to the housing cutoff. The fade starts at the bulb-zone brightness
    with zero slope, so there is no line at the bulb cutoff, and reaches zero
    at ``radius_m``. The band spans at least the outer 15% of the beam radius
    so it never ends in a hard ring. With the defaults (bulb fills the pool,
    exponent 1) this is the single-cutoff pool.

    Args:
        dist_m (np.ndarray): Radial ground distance from the lamp, in meters.
        height (float): Pole height in meters (> 0).
        radius_m (float): Housing cutoff radius in meters.
        bulb_fraction (float): Bulb cutoff radius as a fraction of radius_m.
            Clamped to [0, 1]; a bulb can't be visible past the housing.
        reflector_falloff (float): Reflector-band fade exponent. Higher dies
            off faster. Floored just above 0 so the pool still ends at radius_m.

    Returns:
        np.ndarray: Falloff multiplier in [0, 1], same shape as dist_m.
    """
    inverse_square = 1.0 / (1.0 + (dist_m / height) ** 2) ** 1.5

    bulb_radius = min(max(bulb_fraction, 0.0), 1.0) * radius_m
    fade_start = min(bulb_radius, (1.0 - _CUTOFF_EDGE_SOFTNESS) * radius_m)
    reflector_fade = (1.0 - _smoothstep(dist_m, fade_start, radius_m)) ** max(
        reflector_falloff, 1e-6
    )
    return inverse_square * reflector_fade


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    """Return the smoothstep ramp from 0 at edge0 to 1 at edge1.

    Args:
        x (np.ndarray): Input values.
        edge0 (float): Value where the ramp starts (output 0).
        edge1 (float): Value where the ramp ends (output 1).

    Returns:
        np.ndarray: Ramp values in [0, 1], same shape as x.
    """
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _cone_falloff(
    dist_m: np.ndarray,
    radius_m: float,
    inner_fraction: float,
    exponent: float,
) -> np.ndarray:
    """Compute the legacy two-zone cone falloff.

    The inner zone [0, inner_fraction * radius] is at full brightness; the fade
    zone ramps to zero via a smoothstep raised to ``exponent``. At radius the
    falloff is zero, so there is no hard edge.

    Args:
        dist_m (np.ndarray): Radial ground distance from the light, in meters.
        radius_m (float): Outer pool radius in meters.
        inner_fraction (float): Fraction of radius at full brightness, in [0, 1].
        exponent (float): Fade-zone falloff exponent.

    Returns:
        np.ndarray: Falloff multiplier in [0, 1], same shape as dist_m.
    """
    t = np.clip(dist_m / radius_m, 0.0, 1.0)
    denom = max(1.0 - inner_fraction, 1e-6)
    fade_t = np.clip((t - inner_fraction) / denom, 0.0, 1.0)
    fade_t = fade_t * fade_t * (3.0 - 2.0 * fade_t)
    return (1.0 - fade_t) ** exponent


# Fraction of the cone half-angle over which the wedge edge feathers to zero.
_CONE_EDGE_SOFTNESS: float = 0.3


def _directional_cone_falloff(
    dx_m: np.ndarray,
    dy_m: np.ndarray,
    model: DirectionalConeIllumination,
) -> np.ndarray:
    """Compute the oriented headlight-cone falloff.

    Projects each ground offset onto the cone axis: ``along`` (meters ahead
    of the apex) drives the same two-zone axial fade as ConeIllumination;
    the off-axis angle drives a smoothstep feather over the outer
    _CONE_EDGE_SOFTNESS of the half-angle. Points behind the apex are zero.

    Args:
        dx_m (np.ndarray): Signed east offsets from the apex, in meters.
        dy_m (np.ndarray): Signed north offsets from the apex, in meters.
        model (DirectionalConeIllumination): Cone geometry and fade params.

    Returns:
        np.ndarray: Falloff multiplier in [0, 1], same shape as dx_m.
    """
    ux, uy = model.direction
    along = dx_m * ux + dy_m * uy
    lateral = -dx_m * uy + dy_m * ux

    axial = _cone_falloff(
        np.maximum(along, 0.0), model.length, model.inner_fraction, model.falloff
    )

    half_angle = math.radians(model.beam_angle_deg / 2.0)
    off_axis = np.arctan2(np.abs(lateral), np.maximum(along, 1e-6))
    edge_start = half_angle * (1.0 - _CONE_EDGE_SOFTNESS)
    edge = np.clip(
        (half_angle - off_axis) / max(half_angle - edge_start, 1e-6), 0.0, 1.0
    )
    edge = edge * edge * (3.0 - 2.0 * edge)

    return np.where(along > 0.0, axial * edge, 0.0).astype(np.float32)


def _sample_illumination_map(
    illum_map: np.ndarray,
    hit_x: np.ndarray,
    hit_y: np.ndarray,
    ground_mask: np.ndarray,
    min_x: float,
    max_y: float,
    resolution: float,
) -> np.ndarray:
    """Sample the illumination map at each ground-hit world position.

    Performs bilinear interpolation across the four surrounding map texels for
    each valid ground pixel. Pixels outside the ground mask remain zero (black).

    Args:
        illum_map (np.ndarray): (H_map, W_map, 3) float32 illumination map.
        hit_x (np.ndarray): (H, W) ENU x coordinates of ground intersections.
        hit_y (np.ndarray): (H, W) ENU y coordinates of ground intersections.
        ground_mask (np.ndarray): (H, W) bool mask of valid ground pixels.
        min_x (float): World x corresponding to column 0 of the illumination map.
        max_y (float): World y corresponding to row 0 of the illumination map.
        resolution (float): Meters per pixel in the illumination map.

    Returns:
        np.ndarray: (H, W, 3) float32 illumination values for the screen.
    """
    screen_h, screen_w = ground_mask.shape
    map_h, map_w = illum_map.shape[:2]

    result = np.zeros((screen_h, screen_w, 3), dtype=np.float32)

    # Fractional map coordinates for all pixels
    px_f = (hit_x - min_x) / resolution
    py_f = (max_y - hit_y) / resolution

    # Integer floor coordinates for the top-left texel of each 2x2 sample quad
    px0 = np.clip(np.floor(px_f).astype(int), 0, map_w - 1)
    py0 = np.clip(np.floor(py_f).astype(int), 0, map_h - 1)

    # Neighbour coordinates (clamped so we never go out of bounds)
    px1 = np.clip(px0 + 1, 0, map_w - 1)
    py1 = np.clip(py0 + 1, 0, map_h - 1)

    # Sub-texel fractional offsets, shape (H, W)
    fx = np.clip(px_f - px0, 0.0, 1.0).astype(np.float32)
    fy = np.clip(py_f - py0, 0.0, 1.0).astype(np.float32)

    m = ground_mask

    # Sample the four surrounding texels, shape (N, 3) where N = ground pixels
    c00 = illum_map[py0[m], px0[m]]   # top-left
    c10 = illum_map[py0[m], px1[m]]   # top-right
    c01 = illum_map[py1[m], px0[m]]   # bottom-left
    c11 = illum_map[py1[m], px1[m]]   # bottom-right

    # Expand fractional weights to (N, 1) for broadcast multiply over RGB
    fx_m = fx[m, np.newaxis]
    fy_m = fy[m, np.newaxis]

    result[m] = (
        c00 * (1.0 - fx_m) * (1.0 - fy_m)
        + c10 * fx_m * (1.0 - fy_m)
        + c01 * (1.0 - fx_m) * fy_m
        + c11 * fx_m * fy_m
    )

    return result
