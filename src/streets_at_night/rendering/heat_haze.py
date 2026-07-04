"""Atmospheric heat haze pipeline.

Produces physically-motivated heat shimmer by combining:

  1. **Per-pixel depth map** — a raycast against the z=0 ground plane gives
     the atmospheric path length each pixel sees. Sky pixels are clamped to
     a fraction of the max-haze distance so they shimmer (less than the
     furthest ground pixels) rather than being masked out.
  2. **Multi-layer Kolmogorov phase screens** — wide phase screens are
     generated once at startup via FFT-filtered Von Karman noise (same
     k^(-11/3) power law as real atmospheric turbulence). Each layer has its
     own outer scale, eddy size, weight, and wind vector.
  3. **Frozen-flow advection** — each layer's screen is advected by sliding
     a window across it (Taylor's frozen-turbulence hypothesis). Per-pixel
     dx/dy is the weighted sum across layers.
  4. **MPS-accelerated warping** — `torch.nn.functional.grid_sample` on the
     Apple GPU bilinearly resamples the base frame at the displaced
     coordinates. ~6 ms per frame at 1280x720 (the base frame stays on the
     device between frames; only dx/dy is uploaded).

The single public entry point is the `HeatHaze` class, which owns the phase
screens and the GPU-resident base frame and exposes `apply(t)` to produce a
warped frame at time t. `compute_depth_map` is a free function used by the
caller (preview path) when the camera moves and by the render command to
write a depth map alongside the rendered image (for use by `animate`).
"""

from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..camera.camera import Camera
from ..config.schema import HeatHazeConfig, HeatHazeLayerConfig
from .raycast import GroundHit, build_ground_gbuffer


# ---------------------------------------------------------------------------
# Phase-screen generation (pure-numpy FFT-filtered Von Karman noise)
# ---------------------------------------------------------------------------


def kolmogorov_screen_fft(
    height: int,
    width: int,
    pixel_scale_m: float,
    r0: float,
    L0: float,
    seed: int,
) -> np.ndarray:
    """Generate a Kolmogorov-spectrum phase screen via FFT-filtered noise.

    Implements the Von Karman PSD (Kolmogorov with finite outer scale):
        Phi_phi(k) = 0.023 * r0^(-5/3) * (k^2 + 1/L0^2)^(-11/6)
    Filters complex white noise in Fourier space by sqrt(Phi_phi(k)) and
    inverse-FFTs to a real-valued phase screen.

    Args:
        height (int): Screen height in pixels.
        width (int): Screen width in pixels.
        pixel_scale_m (float): Meters per pixel in the atmospheric phase plane.
        r0 (float): Fried parameter (m). Smaller -> more distortion.
        L0 (float): Outer scale (m). The largest eddy size; eddy size in the
            output image is approximately L0 / pixel_scale_m pixels.
        seed (int): RNG seed for reproducibility.

    Returns:
        np.ndarray: (H, W) float32 phase screen.
    """
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(height, d=pixel_scale_m)[:, None]
    fx = np.fft.fftfreq(width, d=pixel_scale_m)[None, :]
    k2 = (2 * np.pi * fx) ** 2 + (2 * np.pi * fy) ** 2
    psd = 0.023 * (r0 ** (-5.0 / 3.0)) * (k2 + (1.0 / L0) ** 2) ** (-11.0 / 6.0)
    psd[0, 0] = 0.0
    noise = rng.standard_normal((height, width)) + 1j * rng.standard_normal((height, width))
    field_F = noise * np.sqrt(psd) * np.sqrt(height * width) * pixel_scale_m
    return np.fft.ifft2(field_F).real.astype(np.float32)


# ---------------------------------------------------------------------------
# Depth map and distance-scale utilities
# ---------------------------------------------------------------------------


def compute_depth_map(
    camera: Camera,
    width: int,
    height: int,
    max_distance_m: float,
    sky_distance_factor: float,
    gbuffer: t.Optional[GroundHit] = None,
) -> np.ndarray:
    """Compute per-pixel atmospheric path length in meters.

    For every screen pixel, casts a ray through the camera and intersects with
    the z=0 ground plane. The ray parameter distance to that intersection is
    the atmospheric path length to the ground. Ground-hit pixels are capped
    at ``max_distance_m``; sky pixels (no ground intersection) are set to
    ``sky_distance_factor * max_distance_m`` so they shimmer less than the
    farthest ground.

    Args:
        camera (Camera): Active camera (provides position + view + projection).
        width (int): Screen width in pixels.
        height (int): Screen height in pixels.
        max_distance_m (float): Distance ceiling. Beyond this, atmospheric
            haze already fades the scene to nothing, so heat shimmer
            saturates at this value.
        sky_distance_factor (float): Fraction of ``max_distance_m`` used as
            the effective distance for sky pixels. Range [0, 1]; 0 disables
            sky shimmer, 1 makes sky shimmer at the full max distance.
        gbuffer (GroundHit | None): Pre-computed ground raycast for this
            camera and size. When omitted, built internally. Callers that
            also render a frame for the same camera position (e.g. the
            preview loop) should build it once and pass it to both.

    Returns:
        np.ndarray: (H, W) float32 depth array in meters, already clamped.
    """
    if gbuffer is None:
        gbuffer = build_ground_gbuffer(camera, width, height)

    sky_distance = float(sky_distance_factor) * float(max_distance_m)
    ground_clamped = np.minimum(gbuffer.hit_t, max_distance_m)
    depth = np.where(gbuffer.ground_mask, ground_clamped, sky_distance).astype(np.float32)
    return depth


def build_distance_scale(
    depth: np.ndarray,
    max_distance_m: float,
    exponent: float,
) -> np.ndarray:
    """Convert a depth map into a per-pixel distortion-strength multiplier.

    Returns ``(clip(depth / max_distance_m, 0, 1)) ** exponent``. The result
    is in [0, 1]: 0 for pixels at the camera, 1 for pixels at or beyond
    ``max_distance_m``. The exponent shapes the falloff (>1 sharpens the
    near-camera attenuation; <1 flattens it).

    Args:
        depth (np.ndarray): (H, W) depth in meters.
        max_distance_m (float): Distance at which the scale reaches 1.0.
        exponent (float): Falloff steepness.

    Returns:
        np.ndarray: (H, W) float32 in [0, 1].
    """
    norm = np.clip(depth / max_distance_m, 0.0, 1.0)
    return (norm ** exponent).astype(np.float32)


def save_depth_map(path: Path, depth: np.ndarray) -> None:
    """Save a depth map as a float32 .npy file.

    Args:
        path (Path): Output path; parent must exist.
        depth (np.ndarray): (H, W) depth map in meters.
    """
    np.save(path, depth.astype(np.float32))


def load_depth_map(path: Path) -> np.ndarray:
    """Load a depth map saved by ``save_depth_map``.

    Args:
        path (Path): Input .npy path.

    Returns:
        np.ndarray: (H, W) float32 depth map.
    """
    return np.load(path).astype(np.float32)


# ---------------------------------------------------------------------------
# HazeField — multi-layer frozen-flow displacement field
# ---------------------------------------------------------------------------


class HazeField:
    """Multi-layer atmospheric displacement field with frozen-flow animation.

    Generates one wide Kolmogorov phase screen per layer per axis (so two
    screens per layer: one for dx, one for dy) at construction time. Each
    frame, slides each layer's window across its screen at the layer's wind
    vector (Taylor's frozen turbulence) and sums the contributions into a
    final (dx, dy) field, scaled by the global amplitude.

    Attributes:
        amplitude_px (float): Final displacement std at distance_scale=1.0.
    """

    def __init__(
        self,
        out_height: int,
        out_width: int,
        screen_height: int,
        screen_width: int,
        amplitude_px: float,
        layers: t.Sequence[HeatHazeLayerConfig],
        seed: int,
    ):
        """Build phase screens for each layer.

        ``screen_height`` and ``screen_width`` should both be larger than the
        output dimensions; their ratios to ``wind_y_pxs`` and ``wind_x_pxs``
        determine the cyclic loop period. For dominantly-vertical motion
        (rising heat) make ``screen_height`` large; for dominantly-horizontal
        motion make ``screen_width`` large.

        Args:
            out_height (int): Output image height in pixels.
            out_width (int): Output image width in pixels.
            screen_height (int): Phase-screen height; must be >= out_height.
            screen_width (int): Phase-screen width; must be >= out_width.
            amplitude_px (float): Final displacement std (px) at horizon.
            layers (Sequence[HeatHazeLayerConfig]): Per-layer turbulence and
                wind parameters.
            seed (int): RNG seed; each layer/axis gets a deterministic offset.
        """
        self.amplitude_px = float(amplitude_px)
        self._h_out = int(out_height)
        self._w_out = int(out_width)
        self._h_screen = max(int(screen_height), self._h_out)
        self._w_screen = max(int(screen_width), self._w_out)
        self._layers: t.List[t.Dict[str, t.Any]] = []
        for i, lc in enumerate(layers):
            scr_dx = self._build_layer_screen(lc, seed + 2 * i)
            scr_dy = self._build_layer_screen(lc, seed + 2 * i + 1)
            self._layers.append({
                "dx": scr_dx,
                "dy": scr_dy,
                "wind_x": float(lc.wind_x_pxs),
                "wind_y": float(lc.wind_y_pxs),
            })

    def _build_layer_screen(self, lc: HeatHazeLayerConfig, seed: int) -> np.ndarray:
        """Generate a single FFT-filtered phase screen for one axis of one layer.

        The raw phase screen is renormalized so its std equals the layer's
        weight, which lets the layer weights cleanly control relative
        contribution to the summed displacement.

        Args:
            lc (HeatHazeLayerConfig): Layer parameters.
            seed (int): RNG seed for this screen.

        Returns:
            np.ndarray: (H_screen, W_screen) float32 weighted phase screen.
        """
        scr = kolmogorov_screen_fft(
            self._h_screen, self._w_screen, lc.pixel_scale_m, lc.r0, lc.L0, seed,
        )
        return (scr / (scr.std() + 1e-9) * lc.weight).astype(np.float32)

    def sample(self, t: float) -> t.Tuple[np.ndarray, np.ndarray]:
        """Compute the (dx, dy) displacement field at time t (seconds).

        Each layer's window is advected by ``(wind_x * t, wind_y * t)``
        pixels with bilinear sub-pixel interpolation; contributions are
        summed and scaled by ``amplitude_px``.

        Args:
            t (float): Elapsed seconds since animation start.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(dx, dy)``, each ``(H, W_out)``
                float32, units of pixels.
        """
        dx_total = np.zeros((self._h_out, self._w_out), dtype=np.float32)
        dy_total = np.zeros((self._h_out, self._w_out), dtype=np.float32)
        for lay in self._layers:
            # dx and dy screens share a shape and are advected by the same
            # (wind_x * t, wind_y * t) offset this frame, so the cyclic
            # row/col indices and bilinear fractions are identical for both —
            # compute them once and reuse for both slides.
            h_scr, w_scr = lay["dx"].shape
            indices = self._slide_indices(lay["wind_x"] * t, lay["wind_y"] * t, h_scr, w_scr)
            dx_total += self._slide(lay["dx"], indices)
            dy_total += self._slide(lay["dy"], indices)
        dx_total *= self.amplitude_px
        dy_total *= self.amplitude_px
        return dx_total, dy_total

    def _slide_indices(
        self, x_off: float, y_off: float, h_scr: int, w_scr: int,
    ) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Compute cyclic row/col indices and bilinear fractions for ``_slide``.

        Both offsets wrap around the screen bounds (cyclic boundary).

        Args:
            x_off (float): Horizontal offset in pixels (cyclic on w_scr).
            y_off (float): Vertical offset in pixels (cyclic on h_scr).
            h_scr (int): Source screen height.
            w_scr (int): Source screen width.

        Returns:
            tuple: ``(rows, rows_p1, cols, cols_p1, fx, fy)`` — row/col index
                arrays into the source screen and the bilinear sub-pixel
                fractions along each axis.
        """
        x_off = x_off % w_scr
        i0 = int(np.floor(x_off))
        fx = x_off - i0
        cols = (np.arange(self._w_out) + i0) % w_scr
        cols_p1 = (cols + 1) % w_scr

        y_off_pix = y_off % h_scr
        j0 = int(np.floor(y_off_pix))
        fy = y_off_pix - j0
        rows = (np.arange(self._h_out) + j0) % h_scr
        rows_p1 = (rows + 1) % h_scr

        return rows, rows_p1, cols, cols_p1, fx, fy

    def _slide(
        self,
        screen: np.ndarray,
        indices: t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float],
    ) -> np.ndarray:
        """Sample a window from ``screen`` at precomputed cyclic indices.

        Gathers each of the two needed rows once, then takes both column
        gathers from that row rather than re-gathering the full row per
        corner (the naive ``screen[rows][:, cols]`` form materializes an
        intermediate 4x per slide call).

        Args:
            screen (np.ndarray): (H_screen, W_screen) source phase screen.
            indices: ``(rows, rows_p1, cols, cols_p1, fx, fy)`` as returned by
                ``_slide_indices``.

        Returns:
            np.ndarray: (H_out, W_out) bilinearly-sampled window.
        """
        rows, rows_p1, cols, cols_p1, fx, fy = indices
        s0 = screen[rows]
        s1 = screen[rows_p1]
        a = s0[:, cols]
        b = s0[:, cols_p1]
        c = s1[:, cols]
        d = s1[:, cols_p1]
        return (
            (1 - fx) * (1 - fy) * a
            + fx * (1 - fy) * b
            + (1 - fx) * fy * c
            + fx * fy * d
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# TorchWarper — MPS-accelerated bilinear warping
# ---------------------------------------------------------------------------


def select_torch_device() -> str:
    """Choose the best available torch device.

    Returns ``"mps"`` on Apple Silicon when MPS is available, otherwise
    ``"cpu"``. CUDA is not selected: this project targets macOS.

    Returns:
        str: Device string accepted by ``torch.device``.
    """
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchWarper:
    """Bilinear image warper backed by ``torch.nn.functional.grid_sample``.

    The base frame stays resident on the chosen device between frames; only
    the (dx, dy) field is uploaded each call. On Apple Silicon MPS this
    delivers ~150 fps end-to-end at 1280x720.

    Attributes:
        device (str): Active torch device (``"mps"`` or ``"cpu"``).
    """

    def __init__(self, device: str):
        """Initialize without a base frame; call ``set_base_frame`` before warping.

        Args:
            device (str): Torch device string.
        """
        self.device = device
        self._img_t: t.Optional[torch.Tensor] = None
        self._sx: t.Optional[torch.Tensor] = None
        self._sy: t.Optional[torch.Tensor] = None
        self._h = 0
        self._w = 0

    def set_base_frame(self, frame: np.ndarray) -> None:
        """Upload a new base frame to the device.

        Call this whenever the underlying scene image changes (e.g. when the
        camera moves and the warp-preview pipeline regenerates the base
        frame). The grid coordinate templates are also (re)built when the
        size differs from the previous call.

        Args:
            frame (np.ndarray): (H, W, 3) uint8 RGB frame.
        """
        h, w, _ = frame.shape
        self._img_t = (
            torch.from_numpy(np.ascontiguousarray(frame))
            .to(self.device)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        if h != self._h or w != self._w:
            self._sx = torch.arange(w, device=self.device, dtype=torch.float32).unsqueeze(0)
            self._sy = torch.arange(h, device=self.device, dtype=torch.float32).unsqueeze(1)
            self._h = h
            self._w = w

    def warp(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Warp the loaded base frame by per-pixel ``(dx, dy)`` displacements.

        Args:
            dx (np.ndarray): (H, W) float32 horizontal displacement (pixels).
            dy (np.ndarray): (H, W) float32 vertical displacement (pixels).

        Returns:
            np.ndarray: (H, W, 3) uint8 warped RGB frame.

        Raises:
            RuntimeError: If ``set_base_frame`` has not been called.
        """
        if self._img_t is None or self._sx is None or self._sy is None:
            raise RuntimeError("TorchWarper.set_base_frame must be called before warp().")
        dx_t = torch.from_numpy(np.ascontiguousarray(dx)).to(self.device)
        dy_t = torch.from_numpy(np.ascontiguousarray(dy)).to(self.device)
        src_x = (self._sx + dx_t).clamp(0, self._w - 1)
        src_y = (self._sy + dy_t).clamp(0, self._h - 1)
        gx = (src_x / (self._w - 1)) * 2.0 - 1.0
        gy = (src_y / (self._h - 1)) * 2.0 - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        out = F.grid_sample(
            self._img_t, grid, mode="bilinear", padding_mode="border", align_corners=True,
        )
        return (
            (out.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )


# ---------------------------------------------------------------------------
# HeatHaze — top-level orchestrator
# ---------------------------------------------------------------------------

# All pixel-space config values (amplitude_px, wind_*_pxs, pixel_scale_m) are
# defined at this reference width.  HeatHaze scales them to the actual output
# width so that the shimmer looks identical regardless of render resolution.
_REFERENCE_WIDTH_PX = 1920


class HeatHaze:
    """Top-level heat haze pipeline used by preview and animate flows.

    Owns the multi-layer phase screens (one-time startup cost) and the
    GPU-resident base frame. Callers feed it a base frame and a depth map
    when those change (camera move), then call ``apply(t)`` each frame.

    Typical usage::

        hh = HeatHaze(width=1280, height=720, config=cfg.heat_haze)
        hh.set_base_frame(frame)              # whenever camera moves
        hh.set_depth_map(depth)               # whenever camera moves
        warped = hh.apply(elapsed_t)          # each frame
    """

    def __init__(
        self,
        width: int,
        height: int,
        config: HeatHazeConfig,
        device: t.Optional[str] = None,
    ):
        """Allocate phase screens and the torch warper.

        All pixel-space config values (``amplitude_px``, ``wind_*_pxs``,
        ``pixel_scale_m``) are defined at ``_REFERENCE_WIDTH_PX`` (1920 px).
        They are rescaled here so the shimmer looks identical at any output
        resolution:

        - ``amplitude_px`` and wind speeds scale *up* with resolution (same
          physical distortion spans more pixels at higher res).
        - ``pixel_scale_m`` scales *down* (each output pixel covers a smaller
          phase-plane area, so eddies span the same fraction of the image).

        Args:
            width (int): Output frame width in pixels.
            height (int): Output frame height in pixels.
            config (HeatHazeConfig): Heat haze configuration.
            device (str | None): Torch device. Defaults to MPS if available.
        """
        self._config = config
        self._width = int(width)
        self._height = int(height)

        res_scale = self._width / _REFERENCE_WIDTH_PX
        scaled_layers = [
            HeatHazeLayerConfig(
                r0=lc.r0,
                L0=lc.L0,
                pixel_scale_m=lc.pixel_scale_m / res_scale,
                weight=lc.weight,
                wind_x_pxs=lc.wind_x_pxs * res_scale,
                wind_y_pxs=lc.wind_y_pxs * res_scale,
            )
            for lc in config.layers
        ]
        self._field = HazeField(
            out_height=self._height,
            out_width=self._width,
            screen_height=int(config.screen_height_px),
            screen_width=int(config.screen_width_px),
            amplitude_px=config.amplitude_px * res_scale,
            layers=scaled_layers,
            seed=int(config.seed),
        )
        self._warper = TorchWarper(device=device or select_torch_device())
        self._distance_scale: t.Optional[np.ndarray] = None

    def set_base_frame(self, frame: np.ndarray) -> None:
        """Upload a new base frame to the GPU.

        Args:
            frame (np.ndarray): (H, W, 3) uint8 RGB frame.
        """
        self._warper.set_base_frame(frame)

    def set_depth_map(self, depth: np.ndarray) -> None:
        """Compute the per-pixel distance scale from a depth map.

        Args:
            depth (np.ndarray): (H, W) float32 depth in meters, as produced
                by ``compute_depth_map``.
        """
        self._distance_scale = build_distance_scale(
            depth, self._config.max_distance_m, self._config.distance_exponent,
        )

    def apply(self, t: float) -> np.ndarray:
        """Render one warped frame at time ``t``.

        Args:
            t (float): Elapsed seconds.

        Returns:
            np.ndarray: (H, W, 3) uint8 warped RGB frame.

        Raises:
            RuntimeError: If ``set_base_frame`` or ``set_depth_map`` has not
                been called.
        """
        if self._distance_scale is None:
            raise RuntimeError("HeatHaze.set_depth_map must be called before apply().")
        dx, dy = self._field.sample(t)
        dx *= self._distance_scale
        dy *= self._distance_scale
        return self._warper.warp(dx, dy)
