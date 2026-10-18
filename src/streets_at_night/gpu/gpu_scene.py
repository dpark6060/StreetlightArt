"""Pack a TuningScene into the flat float32 arrays the GPU renderer uploads.

Geometry arrays are rebuilt only when the scene changes. Light arrays depend on
the camera (extinction, screen-space glow size) and are packed every frame; the
per-light numbers come from the same functions the CPU passes call
(``compute_pool_peak``, ``effective_pool_radius``, ``compute_light_sprite``).
"""

import math
import struct
import typing as t

import numpy as np

from ..camera.camera import Camera
from ..config.schema import SceneConfig
from ..models.illumination import (
    BeamIllumination,
    ConeIllumination,
    DirectionalConeIllumination,
)
from ..models.light_source import LightSource
from ..rendering.building_pass import get_all_walls, get_face_shade
from ..rendering.color_space import srgb255_to_linear
from ..rendering.illumination_pass import compute_pool_peak, effective_pool_radius
from ..rendering.light_pass import compute_light_sprite
from ..rendering.render_utils import apply_brightness_falloff, camera_distance, compute_focal_length
from .tuning_scene import ROAD_X_RANGE, ROAD_Y_RANGE, Box, TuningScene, building_colors

# Floats per vertex: position (3), linear color (3), ground flag (1).
VERTEX_FLOATS: int = 7

# Floats per ground-pool light record: 4 x vec4 (see illumination in shaders).
POOL_LIGHT_FLOATS: int = 16

# Floats per sprite instance: 3 x vec4 (see billboard shader).
SPRITE_FLOATS: int = 12

# Pool model ids shared with the WGSL ``pool_falloff`` switch.
POOL_PHYSICAL: float = 1.0
POOL_CONE: float = 2.0
POOL_DIRECTIONAL: float = 3.0

# The ground plane extends this far past the road ends in every direction.
_GROUND_HALF_EXTENT_M: float = 1500.0

# Road sits slightly above the ground plane so the two never z-fight.
_ROAD_LIFT_M: float = 0.01

# Same fixed pseudo-moon shading the CPU building pass uses; boxes reuse it.
_BOX_SHADE_STRENGTH: float = 0.35

FRAME_UNIFORM_BYTES: int = 160


def build_opaque_vertices(scene: TuningScene, config: SceneConfig) -> np.ndarray:
    """Triangulate the ground, road, building, and boxes.

    Args:
        scene (TuningScene): The scene to triangulate.
        config (SceneConfig): Working config (building colors).

    Returns:
        np.ndarray: (N, VERTEX_FLOATS) float32 triangle-list vertices.
    """
    triangles: t.List[np.ndarray] = []
    ground = srgb255_to_linear(scene.ground_color)
    road = srgb255_to_linear(scene.road_color)

    e = _GROUND_HALF_EXTENT_M
    triangles.append(_quad([(-e, -e, 0.0), (e, -e, 0.0), (e, e, 0.0), (-e, e, 0.0)], ground, 1.0))
    (x0, x1), (y0, y1) = ROAD_X_RANGE, ROAD_Y_RANGE
    z = _ROAD_LIFT_M
    triangles.append(_quad([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], road, 1.0))

    wall, roof, shade_strength = building_colors(config)
    building = scene.building
    for w in get_all_walls(building):
        shade = get_face_shade(w.normal, shade_strength)
        color = tuple(c * shade for c in srgb255_to_linear(wall))
        triangles.append(_wall_quad(w.p0, w.p1, building.height, color))
    triangles.append(_roof(building.footprint, building.height, srgb255_to_linear(roof)))

    for box in scene.boxes:
        triangles.extend(_box_triangles(box))

    return np.concatenate(triangles).astype(np.float32)


def build_window_vertices(scene: TuningScene) -> np.ndarray:
    """Triangulate lit windows as emissive quads.

    The color slot holds linear color * brightness; per-frame extinction is a
    single uniform multiplier (there is one building).

    Args:
        scene (TuningScene): The scene whose windows to triangulate.

    Returns:
        np.ndarray: (N, VERTEX_FLOATS) float32 vertices, possibly empty.
    """
    if not scene.windows:
        return np.zeros((0, VERTEX_FLOATS), dtype=np.float32)
    quads = []
    for window in scene.windows:
        emission = tuple(c * window.brightness for c in window.linear_color)
        quads.append(_quad([tuple(c) for c in window.corners], emission, 0.0))
    return np.concatenate(quads).astype(np.float32)


def window_extinction(scene: TuningScene, config: SceneConfig, camera: Camera) -> float:
    """Return the per-frame window extinction, as the CPU window pass computes it.

    Args:
        scene (TuningScene): Scene (building position).
        config (SceneConfig): Working config.
        camera (Camera): Current camera.

    Returns:
        float: Multiplier in (0, 1].
    """
    if config.buildings is None:
        return 1.0
    strength = config.buildings.window_extinction_strength
    falloff = config.atmosphere.falloff_distance
    if strength == 0.0 or falloff <= 0.0:
        return 1.0
    centroid = np.array([*scene.building.centroid, 0.0])
    distance = float(np.linalg.norm(camera.position - centroid))
    return math.exp(-distance * strength / falloff)


def pack_pool_lights(lights: t.List[LightSource], camera: Camera, config: SceneConfig) -> np.ndarray:
    """Pack every light with a ground pool into GPU records.

    Args:
        lights (list[LightSource]): Scene lights.
        camera (Camera): Current camera (extinction depends on distance).
        config (SceneConfig): Working config.

    Returns:
        np.ndarray: (K, POOL_LIGHT_FLOATS) float32 records, K >= 1 (a zero
        record is returned when there are no pools, since GPU storage
        buffers can't be empty).
    """
    atmos = config.atmosphere
    records = []
    for light in lights:
        model = light.illumination
        if model is None:
            continue
        peak = compute_pool_peak(
            light, atmos.illumination_brightness, camera.position, atmos.falloff_distance
        )
        lin = srgb255_to_linear(light.color)
        x, y, z = (float(v) for v in light.position)
        if isinstance(model, BeamIllumination):
            kind = POOL_PHYSICAL
            a = (effective_pool_radius(light), model.bulb_fraction, model.reflector_falloff, z)
            b = (0.0, 0.0, 0.0, 0.0)
        elif isinstance(model, DirectionalConeIllumination):
            kind = POOL_DIRECTIONAL
            a = (*model.direction, model.length, math.radians(model.beam_angle_deg / 2.0))
            b = (model.falloff, model.inner_fraction, 0.0, 0.0)
        elif isinstance(model, ConeIllumination):
            kind = POOL_CONE
            a = (model.radius, model.inner_fraction, model.falloff, 0.0)
            b = (0.0, 0.0, 0.0, 0.0)
        else:
            raise TypeError(f"unknown illumination model: {type(model).__name__}")
        records.append((x, y, z, kind, *lin, peak, *a, *b))
    if not records:
        return np.zeros((1, POOL_LIGHT_FLOATS), dtype=np.float32)
    return np.array(records, dtype=np.float32)


def pack_sprites(
    lights: t.List[LightSource], camera: Camera, width: int, config: SceneConfig
) -> np.ndarray:
    """Pack core + glow billboards for every visible light.

    Applies the CPU light pass's visibility rule (skip lights whose extinct
    brightness is under ``min_brightness_threshold``) and its sprite sizing.

    Args:
        lights (list[LightSource]): Scene lights.
        camera (Camera): Current camera.
        width (int): Viewport width in pixels (sets focal length).
        config (SceneConfig): Working config.

    Returns:
        np.ndarray: (S, SPRITE_FLOATS) float32 instances, possibly empty.
    """
    atmos = config.atmosphere
    focal = compute_focal_length(camera, width)
    instances = []
    for light in lights:
        if not light.emits_core:
            continue
        distance = camera_distance(light.position, camera.position)
        effective = apply_brightness_falloff(light.brightness, distance, atmos.falloff_distance)
        if effective < atmos.min_brightness_threshold:
            continue
        sprite = compute_light_sprite(light, distance, effective, focal, atmos)
        instances.append((
            *(float(v) for v in light.position), 0.0,
            *sprite.linear_color, sprite.core_radius_px,
            sprite.core_brightness, sprite.glow_radius_px, sprite.glow_peak, 0.0,
        ))
    if not instances:
        return np.zeros((0, SPRITE_FLOATS), dtype=np.float32)
    return np.array(instances, dtype=np.float32)


def pack_frame_uniforms(
    camera: Camera,
    width: int,
    height: int,
    config: SceneConfig,
    pool_count: int,
    window_ext: float,
    encode_srgb: bool,
) -> bytes:
    """Pack the per-frame uniform block (layout matches ``Frame`` in the WGSL).

    Args:
        camera (Camera): Current camera.
        width (int): Viewport width in pixels.
        height (int): Viewport height in pixels.
        config (SceneConfig): Working config.
        pool_count (int): Number of real ground-pool records.
        window_ext (float): Window extinction multiplier.
        encode_srgb (bool): True when the output texture is not an sRGB
            format, so the composite shader must encode itself.

    Returns:
        bytes: FRAME_UNIFORM_BYTES of packed little-endian float32.
    """
    view_proj = webgpu_view_projection(camera)
    atmos = config.atmosphere
    knee = config.lights.illumination
    grade = config.grade
    focal = compute_focal_length(camera, width)
    values = [
        *view_proj.T.reshape(-1),  # WGSL matrices are column-major
        *camera.position, focal,
        float(width), float(height), 0.0, 0.0,
        knee.knee_start, knee.knee_max, float(pool_count), window_ext,
        atmos.glow_falloff_steepness, atmos.glow_wing_exponent,
        atmos.glow_tail_strength, atmos.glow_cutoff_mult,
        2.0 ** grade.exposure, atmos.vignette_strength, atmos.vignette_radius,
        1.0 if encode_srgb else 0.0,
        1.0 if grade.tonemap == "aces" else 0.0, 0.0, 0.0, 0.0,
    ]
    return struct.pack(f"<{len(values)}f", *values)


def webgpu_view_projection(camera: Camera) -> np.ndarray:
    """Return projection @ view with clip depth remapped from [-1, 1] to [0, 1].

    Args:
        camera (Camera): Camera with OpenGL-convention matrices.

    Returns:
        np.ndarray: (4, 4) float32 view-projection matrix.
    """
    depth_remap = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return (depth_remap @ camera.get_projection_matrix() @ camera.get_view_matrix()).astype(np.float32)


def _quad(corners: t.Sequence[t.Sequence[float]], color: t.Sequence[float], flag: float) -> np.ndarray:
    """Two triangles for a convex quad given counterclockwise corners.

    Args:
        corners: Four (x, y, z) corners.
        color: Linear RGB.
        flag (float): 1.0 for ground-lit surfaces, 0.0 otherwise.

    Returns:
        np.ndarray: (6, VERTEX_FLOATS) vertices.
    """
    c0, c1, c2, c3 = corners
    return np.array([[*p, *color, flag] for p in (c0, c1, c2, c0, c2, c3)], dtype=np.float32)


def _wall_quad(
    p0: np.ndarray, p1: np.ndarray, height: float, color: t.Sequence[float]
) -> np.ndarray:
    """A vertical wall from ground to height along edge p0 -> p1.

    Args:
        p0 (np.ndarray): (2,) edge start.
        p1 (np.ndarray): (2,) edge end.
        height (float): Wall height in meters.
        color: Linear RGB (already shaded).

    Returns:
        np.ndarray: (6, VERTEX_FLOATS) vertices.
    """
    return _quad(
        [(p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0), (p1[0], p1[1], height), (p0[0], p0[1], height)],
        color, 0.0,
    )


def _roof(footprint: np.ndarray, height: float, color: t.Sequence[float]) -> np.ndarray:
    """Fan-triangulate a convex footprint at roof height.

    Args:
        footprint (np.ndarray): (N, 2) CCW ring.
        height (float): Roof height in meters.
        color: Linear RGB.

    Returns:
        np.ndarray: (3 * (N - 2), VERTEX_FLOATS) vertices.
    """
    pts = [(float(x), float(y), height) for x, y in footprint]
    tris = []
    for i in range(1, len(pts) - 1):
        tris.extend([pts[0], pts[i], pts[i + 1]])
    return np.array([[*p, *color, 0.0] for p in tris], dtype=np.float32)


def _box_triangles(box: Box) -> t.List[np.ndarray]:
    """Walls (pseudo-moon shaded, like buildings) and roof of a box.

    Args:
        box (Box): The box.

    Returns:
        list[np.ndarray]: Vertex arrays for the four walls and the roof.
    """
    (x0, y0), (x1, y1) = box.min_xy, box.max_xy
    ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    wall = srgb255_to_linear(box.wall_color)
    parts = []
    for i in range(4):
        p0, p1 = ring[i], ring[(i + 1) % 4]
        edge = p1 - p0
        normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
        shade = get_face_shade(normal, _BOX_SHADE_STRENGTH)
        parts.append(_wall_quad(p0, p1, box.height, tuple(c * shade for c in wall)))
    parts.append(_roof(ring, box.height, srgb255_to_linear(box.roof_color)))
    return parts
