# Radical
"""Unit tests for the heat_haze pipeline."""

from unittest import mock

import numpy as np
import pytest

from streets_at_night.camera.camera import create_camera
from streets_at_night.config.schema import HeatHazeConfig, HeatHazeLayerConfig
from streets_at_night.rendering.heat_haze import (
    HazeField,
    HeatHaze,
    TorchWarper,
    build_distance_scale,
    compute_depth_map,
    kolmogorov_screen_fft,
    load_depth_map,
    save_depth_map,
    select_torch_device,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def downward_camera():
    """Return a camera 1000m up looking straight down — every ray hits ground."""
    return create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 1.0, 0.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


@pytest.fixture
def horizon_camera():
    """Return a camera looking nearly horizontal — many rays miss the ground."""
    return create_camera(
        position=np.array([0.0, 0.0, 1000.0]),
        target=np.array([0.0, 10000.0, 990.0]),
        fov_deg=60.0,
        aspect_ratio=16.0 / 9.0,
        near=10.0,
        far=20000.0,
    )


@pytest.fixture
def two_layer_config() -> HeatHazeConfig:
    """A minimal two-layer heat haze config sized for cheap tests."""
    return HeatHazeConfig(
        enabled=True,
        amplitude_px=1.0,
        max_distance_m=20000.0,
        distance_exponent=1.5,
        sky_distance_factor=0.5,
        seed=42,
        screen_width_px=512,
        screen_height_px=256,
        layers=[
            HeatHazeLayerConfig(r0=0.10, L0=2.0, pixel_scale_m=0.2,
                                weight=1.0, wind_x_pxs=50.0, wind_y_pxs=2.0),
            HeatHazeLayerConfig(r0=0.10, L0=0.5, pixel_scale_m=0.1,
                                weight=0.5, wind_x_pxs=120.0, wind_y_pxs=-3.0),
        ],
    )


# ---------------------------------------------------------------------------
# kolmogorov_screen_fft
# ---------------------------------------------------------------------------


def test_kolmogorov_screen_fft_returns_correct_shape():
    # Act
    screen = kolmogorov_screen_fft(64, 128, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=0)

    # Assert
    assert screen.shape == (64, 128)


def test_kolmogorov_screen_fft_returns_float32():
    # Act
    screen = kolmogorov_screen_fft(64, 64, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=0)

    # Assert
    assert screen.dtype == np.float32


def test_kolmogorov_screen_fft_is_zero_mean():
    # Arrange — DC component is explicitly zeroed in the PSD

    # Act
    screen = kolmogorov_screen_fft(128, 128, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=42)

    # Assert
    assert abs(screen.mean()) < 1e-3


def test_kolmogorov_screen_fft_is_deterministic_per_seed():
    # Act
    a = kolmogorov_screen_fft(32, 32, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=7)
    b = kolmogorov_screen_fft(32, 32, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=7)

    # Assert
    np.testing.assert_array_equal(a, b)


def test_kolmogorov_screen_fft_differs_across_seeds():
    # Act
    a = kolmogorov_screen_fft(32, 32, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=1)
    b = kolmogorov_screen_fft(32, 32, pixel_scale_m=0.1, r0=0.15, L0=10.0, seed=2)

    # Assert
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# compute_depth_map
# ---------------------------------------------------------------------------


def test_compute_depth_map_returns_correct_shape(downward_camera):
    # Act
    depth = compute_depth_map(
        downward_camera, width=64, height=36,
        max_distance_m=20000.0, sky_distance_factor=0.5,
    )

    # Assert
    assert depth.shape == (36, 64)
    assert depth.dtype == np.float32


def test_compute_depth_map_caps_ground_at_max_distance(downward_camera):
    # Arrange — looking straight down from 1000m, all rays terminate near 1000m
    # which is well below the max_distance ceiling.
    max_d = 20000.0

    # Act
    depth = compute_depth_map(
        downward_camera, width=64, height=36,
        max_distance_m=max_d, sky_distance_factor=0.5,
    )

    # Assert
    assert depth.max() <= max_d
    # Ground from a 1000m altitude camera is around 1000-1300m typically
    assert depth.min() >= 1000.0


def test_compute_depth_map_sky_pixels_use_sky_factor(horizon_camera):
    # Arrange — horizon camera will have some sky pixels
    max_d = 20000.0
    sky_factor = 0.4
    sky_value = sky_factor * max_d  # 8000.0

    # Act
    depth = compute_depth_map(
        horizon_camera, width=64, height=36,
        max_distance_m=max_d, sky_distance_factor=sky_factor,
    )

    # Assert — at least some pixels should be exactly the sky_value
    assert np.any(np.isclose(depth, sky_value))


def test_compute_depth_map_sky_factor_zero_zeroes_sky(horizon_camera):
    # Arrange
    max_d = 10000.0

    # Act
    depth = compute_depth_map(
        horizon_camera, width=64, height=36,
        max_distance_m=max_d, sky_distance_factor=0.0,
    )

    # Assert — the sky_value (0) should be the minimum since ground is finite
    assert depth.min() == 0.0


# ---------------------------------------------------------------------------
# build_distance_scale
# ---------------------------------------------------------------------------


def test_build_distance_scale_clips_to_unit_interval():
    # Arrange — depth values both within and beyond max_distance
    depth = np.array([[0.0, 5000.0, 10000.0, 15000.0, 99999.0]], dtype=np.float32)

    # Act
    scale = build_distance_scale(depth, max_distance_m=10000.0, exponent=1.0)

    # Assert
    assert scale.min() >= 0.0
    assert scale.max() <= 1.0


def test_build_distance_scale_exponent_one_is_linear():
    # Arrange
    depth = np.array([[0.0, 5000.0, 10000.0]], dtype=np.float32)

    # Act
    scale = build_distance_scale(depth, max_distance_m=10000.0, exponent=1.0)

    # Assert
    np.testing.assert_allclose(scale[0], [0.0, 0.5, 1.0])


def test_build_distance_scale_higher_exponent_steeper_falloff():
    # Arrange — at 0.5 of max distance, exponent 2 should give 0.25, exp 1 gives 0.5
    depth = np.array([[5000.0]], dtype=np.float32)

    # Act
    scale_e1 = build_distance_scale(depth, max_distance_m=10000.0, exponent=1.0)
    scale_e2 = build_distance_scale(depth, max_distance_m=10000.0, exponent=2.0)

    # Assert
    assert scale_e2[0, 0] < scale_e1[0, 0]


# ---------------------------------------------------------------------------
# save / load depth map
# ---------------------------------------------------------------------------


def test_save_and_load_depth_map_roundtrip(tmp_path):
    # Arrange
    original = np.linspace(0, 100, 64 * 36, dtype=np.float32).reshape(36, 64)
    out_path = tmp_path / "depth.npy"

    # Act
    save_depth_map(out_path, original)
    loaded = load_depth_map(out_path)

    # Assert
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, original)


# ---------------------------------------------------------------------------
# HazeField
# ---------------------------------------------------------------------------


def test_hazefield_sample_returns_correct_shapes(two_layer_config):
    # Arrange
    field = HazeField(
        out_height=72, out_width=128, screen_height=128, screen_width=256,
        amplitude_px=two_layer_config.amplitude_px,
        layers=two_layer_config.layers, seed=two_layer_config.seed,
    )

    # Act
    dx, dy = field.sample(t=0.0)

    # Assert
    assert dx.shape == (72, 128)
    assert dy.shape == (72, 128)
    assert dx.dtype == np.float32
    assert dy.dtype == np.float32


def test_hazefield_sample_changes_over_time(two_layer_config):
    # Arrange
    field = HazeField(
        out_height=72, out_width=128, screen_height=128, screen_width=256,
        amplitude_px=two_layer_config.amplitude_px,
        layers=two_layer_config.layers, seed=two_layer_config.seed,
    )

    # Act
    dx0, _ = field.sample(t=0.0)
    dx1, _ = field.sample(t=2.0)

    # Assert
    assert not np.array_equal(dx0, dx1)


def _reference_slide(screen: np.ndarray, x_off: float, y_off: float, w_out: int, h_out: int) -> np.ndarray:
    """Pre-optimization reference implementation of HazeField._slide.

    Copied verbatim (modulo parameter names) from the implementation prior to
    the row-gather/column-gather refactor, to pin down output equivalence.
    """
    h_scr, w_scr = screen.shape

    x_off = x_off % w_scr
    i0 = int(np.floor(x_off))
    fx = x_off - i0
    cols = (np.arange(w_out) + i0) % w_scr
    cols_p1 = (cols + 1) % w_scr

    y_off_pix = y_off % h_scr
    j0 = int(np.floor(y_off_pix))
    fy = y_off_pix - j0
    rows = (np.arange(h_out) + j0) % h_scr
    rows_p1 = (rows + 1) % h_scr

    a = screen[rows][:, cols]
    b = screen[rows][:, cols_p1]
    c = screen[rows_p1][:, cols]
    d = screen[rows_p1][:, cols_p1]
    return (
        (1 - fx) * (1 - fy) * a
        + fx * (1 - fy) * b
        + (1 - fx) * fy * c
        + fx * fy * d
    ).astype(np.float32)


def test_hazefield_slide_matches_reference_implementation(two_layer_config):
    # Arrange — fixed seed/layers/time so both implementations see identical
    # phase screens and identical (x_off, y_off) offsets.
    field = HazeField(
        out_height=72, out_width=128, screen_height=256, screen_width=512,
        amplitude_px=two_layer_config.amplitude_px,
        layers=two_layer_config.layers, seed=two_layer_config.seed,
    )
    lay = field._layers[0]
    x_off, y_off = lay["wind_x"] * 1.7, lay["wind_y"] * 1.7

    # Act
    indices = field._slide_indices(x_off, y_off, *lay["dx"].shape)
    optimized = field._slide(lay["dx"], indices)
    reference = _reference_slide(lay["dx"], x_off, y_off, w_out=128, h_out=72)

    # Assert
    np.testing.assert_allclose(optimized, reference, rtol=1e-6, atol=1e-6)


def test_hazefield_sample_matches_reference_slide_end_to_end(two_layer_config):
    # Arrange — verify the full sample() path (which hoists shared index math
    # across the dx/dy calls per layer) matches per-call reference slides.
    field = HazeField(
        out_height=72, out_width=128, screen_height=256, screen_width=512,
        amplitude_px=two_layer_config.amplitude_px,
        layers=two_layer_config.layers, seed=two_layer_config.seed,
    )
    t = 3.25

    # Act
    dx, dy = field.sample(t)

    # Assert — recompute expected dx/dy from the reference slide directly
    expected_dx = np.zeros((72, 128), dtype=np.float32)
    expected_dy = np.zeros((72, 128), dtype=np.float32)
    for lay in field._layers:
        expected_dx += _reference_slide(
            lay["dx"], lay["wind_x"] * t, lay["wind_y"] * t, w_out=128, h_out=72,
        )
        expected_dy += _reference_slide(
            lay["dy"], lay["wind_x"] * t, lay["wind_y"] * t, w_out=128, h_out=72,
        )
    expected_dx *= field.amplitude_px
    expected_dy *= field.amplitude_px

    np.testing.assert_allclose(dx, expected_dx, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(dy, expected_dy, rtol=1e-6, atol=1e-6)


def test_hazefield_sample_amplitude_scaling(two_layer_config):
    # Arrange — same layers, different amplitudes
    f1 = HazeField(
        out_height=72, out_width=128, screen_height=128, screen_width=256,
        amplitude_px=1.0, layers=two_layer_config.layers, seed=42,
    )
    f2 = HazeField(
        out_height=72, out_width=128, screen_height=128, screen_width=256,
        amplitude_px=3.0, layers=two_layer_config.layers, seed=42,
    )

    # Act
    dx1, _ = f1.sample(t=0.5)
    dx2, _ = f2.sample(t=0.5)

    # Assert — amp=3 output should be 3× amp=1 output (deterministic at same seed)
    np.testing.assert_allclose(dx2, 3.0 * dx1, rtol=1e-5)


# ---------------------------------------------------------------------------
# TorchWarper
# ---------------------------------------------------------------------------


def test_torchwarper_warp_returns_correct_shape_and_dtype():
    # Arrange
    warper = TorchWarper(device=select_torch_device())
    frame = np.full((36, 64, 3), 100, dtype=np.uint8)
    warper.set_base_frame(frame)
    dx = np.zeros((36, 64), dtype=np.float32)
    dy = np.zeros((36, 64), dtype=np.float32)

    # Act
    out = warper.warp(dx, dy)

    # Assert
    assert out.shape == (36, 64, 3)
    assert out.dtype == np.uint8


def test_torchwarper_zero_displacement_is_near_identity():
    # Arrange
    warper = TorchWarper(device=select_torch_device())
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    frame[::4, ::4] = 200
    warper.set_base_frame(frame)
    dx = np.zeros((36, 64), dtype=np.float32)
    dy = np.zeros((36, 64), dtype=np.float32)

    # Act
    out = warper.warp(dx, dy)

    # Assert — bilinear at zero displacement reproduces the input within ±1
    # (the uint8↔float32 round-trip via /255 then *255 truncates, so values
    # around 200 land on 199; that's a precision artifact, not a sampling bug).
    diff = np.abs(out.astype(np.int16) - frame.astype(np.int16))
    assert diff.max() <= 1


def test_torchwarper_raises_when_no_base_frame_set():
    # Arrange
    warper = TorchWarper(device=select_torch_device())
    dx = np.zeros((36, 64), dtype=np.float32)
    dy = np.zeros((36, 64), dtype=np.float32)

    # Act & Assert
    with pytest.raises(RuntimeError):
        warper.warp(dx, dy)


# ---------------------------------------------------------------------------
# HeatHaze (orchestrator)
# ---------------------------------------------------------------------------


def test_heathaze_apply_returns_correct_shape(two_layer_config, downward_camera):
    # Arrange
    width, height = 128, 72
    base = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    depth = compute_depth_map(
        downward_camera, width, height,
        two_layer_config.max_distance_m, two_layer_config.sky_distance_factor,
    )
    haze = HeatHaze(width, height, two_layer_config)
    haze.set_base_frame(base)
    haze.set_depth_map(depth)

    # Act
    out = haze.apply(t=0.5)

    # Assert
    assert out.shape == (height, width, 3)
    assert out.dtype == np.uint8


def test_heathaze_apply_raises_without_depth_map(two_layer_config):
    # Arrange — call apply without ever calling set_depth_map
    width, height = 128, 72
    base = np.zeros((height, width, 3), dtype=np.uint8)
    haze = HeatHaze(width, height, two_layer_config)
    haze.set_base_frame(base)

    # Act & Assert
    with pytest.raises(RuntimeError):
        haze.apply(t=0.0)


def test_heathaze_uses_warper_to_produce_output(two_layer_config, downward_camera):
    # Arrange — patch the warper so we can verify it was called
    width, height = 64, 36
    base = np.zeros((height, width, 3), dtype=np.uint8)
    depth = compute_depth_map(
        downward_camera, width, height,
        two_layer_config.max_distance_m, two_layer_config.sky_distance_factor,
    )
    haze = HeatHaze(width, height, two_layer_config)
    haze.set_base_frame(base)
    haze.set_depth_map(depth)
    expected_out = np.full((height, width, 3), 77, dtype=np.uint8)

    with mock.patch.object(haze._warper, "warp", return_value=expected_out) as mock_warp:
        # Act
        result = haze.apply(t=1.0)

    # Assert
    mock_warp.assert_called_once()
    np.testing.assert_array_equal(result, expected_out)
