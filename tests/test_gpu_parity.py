# Radical
"""GPU vs CPU parity: the WGSL ports must match the numpy render math.

Renders on the real GPU, so the whole module is skipped when no wgpu adapter
is available. These tests are the drift guard between the two renderers.
"""

import numpy as np
import pytest

wgpu = pytest.importorskip("wgpu")

from streets_at_night.config.defaults import DEFAULTS
from streets_at_night.config.schema import SceneConfig
from streets_at_night.config.serde import dataclass_from_dict
from streets_at_night.gpu.gpu_scene import (
    VERTEX_FLOATS,
    pack_frame_uniforms,
    pack_pool_lights,
    webgpu_view_projection,
)
from streets_at_night.gpu.orbit_camera import OrbitCamera
from streets_at_night.models.illumination import (
    BeamIllumination,
    ConeIllumination,
    DirectionalConeIllumination,
)
from streets_at_night.models.light_source import LightSource
from streets_at_night.rendering.color_space import linear_to_srgb, srgb255_to_linear, tonemap_aces
from streets_at_night.rendering.compositor import _build_vignette_scalar
from streets_at_night.rendering.illumination_pass import (
    _apply_soft_knee,
    _cone_falloff,
    _directional_cone_falloff,
    _physical_falloff,
    compute_pool_peak,
    effective_pool_radius,
)

try:
    from streets_at_night.gpu.renderer import TuningRenderer, get_device

    _DEVICE = get_device()
except Exception:  # noqa: BLE001 — any adapter failure means no GPU here
    _DEVICE = None

pytestmark = pytest.mark.skipif(_DEVICE is None, reason="no wgpu adapter")

_SIZE = (160, 120)
_BLACK = (0.0, 0.0, 0.0)


@pytest.fixture
def config() -> SceneConfig:
    """Return a SceneConfig built from project defaults."""
    return dataclass_from_dict(SceneConfig, DEFAULTS)


@pytest.fixture
def camera():
    """Return a near-top-down camera over the origin covering about 80 m."""
    orbit = OrbitCamera(
        pivot=np.array([0.0, 0.0, 0.0]), yaw_deg=-90.0, pitch_deg=88.0, distance=110.0,
        fov_deg=40.0,
    )
    return orbit.to_camera(_SIZE[0] / _SIZE[1])


@pytest.fixture
def lights():
    """Return one light of each ground-pool model, spread across the view."""
    def light(position, color, model):
        return LightSource(
            position=np.array(position), color=color, brightness=0.9, size=0.3,
            glow_radius=0.0, illumination_intensity=0.8, illumination=model,
        )
    return [
        light((-15.0, 5.0, 10.0), (255, 180, 90), BeamIllumination(140.0, 0.4, 2.5)),
        light((18.0, 8.0, 0.0), (120, 160, 255), ConeIllumination(12.0, 1.5, 0.3)),
        light((5.0, -14.0, 0.7), (255, 255, 230),
              DirectionalConeIllumination((0.6, 0.8), 25.0, 40.0, 1.5, 0.3)),
    ]


def _ground_vertices() -> np.ndarray:
    """A black ground quad flagged to receive light pools."""
    e = 500.0
    corners = [(-e, -e, 0.0), (e, -e, 0.0), (e, e, 0.0), (-e, e, 0.0)]
    return np.array(
        [[*corners[i], *_BLACK, 1.0] for i in (0, 1, 2, 0, 2, 3)], dtype=np.float32
    )


def _pixel_ground_points(camera, width, height) -> np.ndarray:
    """World (x, y) where each pixel center's ray meets z = 0."""
    inv = np.linalg.inv(webgpu_view_projection(camera).astype(np.float64))
    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    ndc_x = px / width * 2.0 - 1.0
    ndc_y = 1.0 - py / height * 2.0

    def unproject(z):
        clip = np.stack([ndc_x, ndc_y, np.full_like(ndc_x, z), np.ones_like(ndc_x)], axis=-1)
        world = clip @ inv.T
        return world[..., :3] / world[..., 3:4]

    near, far = unproject(0.0), unproject(1.0)
    t = near[..., 2] / (near[..., 2] - far[..., 2])
    return near[..., :2] + (far[..., :2] - near[..., :2]) * t[..., None]


def _cpu_pool_map(lights, camera, config, ground_xy) -> np.ndarray:
    """Evaluate the CPU falloff functions at the given ground points."""
    atmos = config.atmosphere
    total = np.zeros((*ground_xy.shape[:2], 3), dtype=np.float32)
    for light in lights:
        dx = ground_xy[..., 0] - light.position[0]
        dy = ground_xy[..., 1] - light.position[1]
        dist = np.hypot(dx, dy)
        model = light.illumination
        radius = effective_pool_radius(light)
        if isinstance(model, BeamIllumination):
            falloff = _physical_falloff(
                dist, float(light.position[2]), radius, model.bulb_fraction, model.reflector_falloff
            )
        elif isinstance(model, DirectionalConeIllumination):
            falloff = _directional_cone_falloff(dx, dy, model)
        else:
            falloff = _cone_falloff(dist, radius, model.inner_fraction, model.falloff)
        falloff = np.where(dist <= radius, falloff, 0.0)
        peak = compute_pool_peak(light, atmos.illumination_brightness, camera.position, atmos.falloff_distance)
        total += (peak * falloff)[..., None] * np.array(srgb255_to_linear(light.color), dtype=np.float32)
    knee = config.lights.illumination
    return _apply_soft_knee(total, knee.knee_start, knee.knee_max)


def _render(config, camera, lights, output_format="rgba8unorm"):
    renderer = TuningRenderer(_DEVICE, output_format)
    renderer.set_geometry(_ground_vertices(), np.zeros((0, VERTEX_FLOATS), np.float32))
    uniforms = pack_frame_uniforms(camera, *_SIZE, config, len(lights), 1.0, encode_srgb=True)
    image = renderer.render_offscreen(
        uniforms, pack_pool_lights(lights, camera, config),
        np.zeros((0, 12), np.float32), _BLACK, _SIZE,
    )
    return image, renderer.read_hdr()


def test_ground_pools_match_cpu_falloff(config, camera, lights):
    # Arrange
    expected = _cpu_pool_map(lights, camera, config, _pixel_ground_points(camera, *_SIZE))

    # Act
    _, hdr = _render(config, camera, lights)

    # Assert — float16 target: ~0.1% relative precision, plus edge pixels
    assert expected.max() > 0.1
    np.testing.assert_allclose(hdr, expected, rtol=0.01, atol=0.004)


def test_soft_knee_matches_cpu_when_pools_stack(config, camera, lights):
    # Arrange — push intensity well past the knee
    for light in lights:
        light.illumination_intensity = 40.0
    expected = _cpu_pool_map(lights, camera, config, _pixel_ground_points(camera, *_SIZE))

    # Act
    _, hdr = _render(config, camera, lights)

    # Assert
    assert expected.max() > config.lights.illumination.knee_start
    np.testing.assert_allclose(hdr, expected, rtol=0.01, atol=0.004)


def test_composite_matches_cpu_grade(config, camera, lights):
    # Arrange
    config.grade.exposure = 1.0

    # Act
    image, hdr = _render(config, camera, lights)

    # Assert — vignette, exposure, ACES, sRGB, 8-bit quantize as in compositor
    vignette = _build_vignette_scalar(
        *_SIZE, config.atmosphere.vignette_strength, config.atmosphere.vignette_radius
    )
    graded = tonemap_aces(hdr * vignette[..., None] * (2.0 ** config.grade.exposure))
    expected = (linear_to_srgb(graded) * 255.0 + 0.5).astype(np.int16)
    assert np.abs(image.astype(np.int16) - expected).max() <= 2


def test_directional_cone_is_dark_behind_apex(config, camera, lights):
    # Arrange — only the headlight cone, pointing northeast
    cone = [lights[2]]

    # Act
    _, hdr = _render(config, camera, cone)

    # Assert — sample 5 m behind the apex (opposite the heading)
    ground = _pixel_ground_points(camera, *_SIZE)
    behind = np.array([5.0 - 0.6 * 5.0, -14.0 - 0.8 * 5.0])
    row, col = np.unravel_index(
        np.argmin(np.linalg.norm(ground - behind, axis=-1)), ground.shape[:2]
    )
    assert hdr[row, col].max() == pytest.approx(0.0, abs=1e-6)
    assert hdr.max() > 0.1
