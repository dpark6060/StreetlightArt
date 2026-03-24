"""Shared 2D geometry utilities for road placement modules."""

import numpy as np


def perpendicular_2d(tangent: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to the tangent in the XY plane.

    Rotates the tangent 90 degrees counter-clockwise. The z component of
    the result is always zero.

    Args:
        tangent (np.ndarray): 3-element tangent vector (may not be unit length).

    Returns:
        np.ndarray: 3-element perpendicular unit vector with z=0. Returns
            [1, 0, 0] if the input is near-zero to avoid division by zero.
    """
    perp = np.array([-tangent[1], tangent[0], 0.0])
    length = np.linalg.norm(perp)
    if length < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    return perp / length
