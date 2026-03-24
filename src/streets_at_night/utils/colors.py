"""Shared color utility functions."""

import typing as t


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
