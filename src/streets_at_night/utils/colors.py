"""Shared color utility functions."""

import math
import typing as t

# Blackbody temperature range (Kelvin) supported by the approximation below.
_MIN_KELVIN = 1000.0
_MAX_KELVIN = 40000.0

# Rec. 709 luminance weights, used as the gray pivot for saturation.
_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def hex_to_rgb(hex_str: str) -> t.Tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple.

    Args:
        hex_str (str): Hex color string like "#FF0000" or "#ff0000".

    Returns:
        tuple[int, int, int]: (R, G, B) values 0-255.
    """
    stripped = hex_str.lstrip("#")
    r = int(stripped[0:2], 16)
    g = int(stripped[2:4], 16)
    b = int(stripped[4:6], 16)
    return r, g, b


def kelvin_to_rgb(kelvin: float, saturation: float = 1.0) -> t.Tuple[int, int, int]:
    """Convert a blackbody color temperature to an sRGB tuple.

    Uses the Tanner Helland piecewise approximation of the Planckian locus,
    then applies a saturation control around the color's own luminance. This
    is the authoring primitive for street lighting: warm sodium is a low
    temperature, cool LED retrofit a high one. Gas-discharge lamps (sodium,
    mercury) are not true blackbodies, so ``saturation`` lets an entry push
    past the blackbody chromaticity toward the real monochromatic amber.

    Args:
        kelvin (float): Color temperature in Kelvin. Clamped to [1000, 40000].
        saturation (float): Saturation multiplier around the luminance gray.
            1.0 is the unmodified blackbody color, 0.0 collapses to gray,
            and values above 1.0 increase chroma (e.g. ~1.3 for sodium amber).

    Returns:
        tuple[int, int, int]: sRGB (R, G, B) values 0-255.
    """
    temp = max(_MIN_KELVIN, min(_MAX_KELVIN, kelvin)) / 100.0

    if temp <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)

    if temp >= 66.0:
        blue = 255.0
    elif temp <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307

    rgb = [_clamp_255(red), _clamp_255(green), _clamp_255(blue)]
    return _apply_saturation(rgb, saturation)


def _apply_saturation(
    rgb: t.List[float], saturation: float
) -> t.Tuple[int, int, int]:
    """Scale chroma around the color's luminance and quantize to 0-255.

    Args:
        rgb (list[float]): Linear-position RGB values in [0, 255].
        saturation (float): Saturation multiplier; 1.0 leaves the color
            unchanged, 0.0 collapses to its luminance gray.

    Returns:
        tuple[int, int, int]: Saturation-adjusted sRGB (R, G, B) 0-255.
    """
    gray = _LUMA_R * rgb[0] + _LUMA_G * rgb[1] + _LUMA_B * rgb[2]
    adjusted = [gray + saturation * (channel - gray) for channel in rgb]
    return tuple(int(_clamp_255(channel) + 0.5) for channel in adjusted)


def _clamp_255(value: float) -> float:
    """Clamp a float to the [0, 255] range."""
    return max(0.0, min(255.0, value))
