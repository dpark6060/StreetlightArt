"""Color-space conversions and tone mapping for the linear-light pipeline.

Light is additive in *linear radiance*, but authored colors and the display are
sRGB-encoded (roughly perceptual / logarithmic). The renderer therefore:

  1. Linearizes authored colors on input (``srgb_to_linear`` / ``hex_to_linear``).
  2. Accumulates all light contributions in linear space, unclamped (HDR).
  3. Tone-maps and re-encodes to sRGB once at the very end (``tonemap_aces`` +
     ``linear_to_srgb``).

All array functions operate on float arrays of any shape; sRGB values are in
[0, 1] and linear values are in [0, inf).
"""

import typing as t

import numpy as np

from ..utils.colors import hex_to_rgb

# sRGB transfer function breakpoints (IEC 61966-2-1).
_SRGB_LINEAR_THRESHOLD = 0.04045
_LINEAR_SRGB_THRESHOLD = 0.0031308


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB-encoded values in [0, 1] to linear radiance.

    Args:
        srgb (np.ndarray): Float array of sRGB values in [0, 1].

    Returns:
        np.ndarray: Float32 array of linear values, same shape as input.
    """
    s = np.asarray(srgb, dtype=np.float32)
    low = s / 12.92
    high = ((s + 0.055) / 1.055) ** 2.4
    return np.where(s <= _SRGB_LINEAR_THRESHOLD, low, high).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear radiance in [0, 1] to sRGB-encoded values.

    Input is clipped to [0, 1] before encoding, so this is the final display
    transform applied after tone mapping has mapped HDR values into range.

    Args:
        linear (np.ndarray): Float array of linear values.

    Returns:
        np.ndarray: Float32 array of sRGB values in [0, 1], same shape as input.
    """
    lin = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    low = lin * 12.92
    high = 1.055 * (lin ** (1.0 / 2.4)) - 0.055
    return np.where(lin <= _LINEAR_SRGB_THRESHOLD, low, high).astype(np.float32)


def hex_to_linear(hex_str: str) -> t.Tuple[float, float, float]:
    """Convert a '#RRGGBB' hex color to a linear RGB tuple in [0, 1].

    Args:
        hex_str (str): Hex color string, with or without leading '#'.

    Returns:
        tuple[float, float, float]: Linear (R, G, B) each in [0, 1].
    """
    rgb = np.array(hex_to_rgb(hex_str), dtype=np.float32) / 255.0
    lin = srgb_to_linear(rgb)
    return (float(lin[0]), float(lin[1]), float(lin[2]))


# Memo of 0-255 sRGB tuple -> linear RGB tuple. The light and illumination
# passes convert LightSource.color hundreds of thousands of times per frame,
# but only a few hundred distinct colors exist (finite authored palettes), so a
# dict lookup replaces almost all of the per-light float work. Bounded by the
# palette size; persists across frames (helps preview/devmode too).
_SRGB255_LINEAR_CACHE: t.Dict[
    t.Tuple[float, float, float], t.Tuple[float, float, float]
] = {}


def srgb255_to_linear(rgb: t.Tuple[int, int, int]) -> t.Tuple[float, float, float]:
    """Convert an 8-bit sRGB tuple (0-255) to a linear RGB tuple in [0, 1].

    Convenience for the many light models that already hold their color as a
    0-255 RGB tuple (``LightSource.color``). Results are memoized by color, so
    repeated conversions of the same authored color are a dict lookup.

    Args:
        rgb (tuple[int, int, int]): sRGB color, each channel 0-255.

    Returns:
        tuple[float, float, float]: Linear (R, G, B) each in [0, 1].
    """
    key = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    cached = _SRGB255_LINEAR_CACHE.get(key)
    if cached is not None:
        return cached

    arr = np.array(key, dtype=np.float32) / 255.0
    lin = srgb_to_linear(arr)
    result = (float(lin[0]), float(lin[1]), float(lin[2]))
    _SRGB255_LINEAR_CACHE[key] = result
    return result


def tonemap_aces(linear: np.ndarray) -> np.ndarray:
    """Apply the Narkowicz ACES filmic tone-mapping curve.

    Maps unbounded linear HDR radiance into [0, 1] with a smooth highlight
    shoulder, so bright cores roll off to white rather than hard-clipping.
    Monotonic, maps 0 -> 0.

    Args:
        linear (np.ndarray): Float array of linear radiance in [0, inf).

    Returns:
        np.ndarray: Float32 array tone-mapped into [0, 1], same shape as input.
    """
    x = np.asarray(linear, dtype=np.float32)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32)
