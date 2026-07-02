"""Universal light source data model consumed by the renderer."""

import typing as t
from dataclasses import dataclass

import numpy as np


@dataclass
class LightSource:
    """A single light source in 3D ENU space.

    This is the sole light interface the renderer sees. Higher-level
    objects (streetlights, cars, house lights, storefront signs) are all
    reduced to LightSource instances before rendering. The renderer does
    not know what kind of object produced a given LightSource.

    Attributes:
        position: 3-element array of (x, y, z) coordinates in local ENU
            metres. Origin is the scene midpoint as set by the camera.
        color: Light emission colour as an (R, G, B) tuple with values
            in [0, 255].
        brightness: Relative intensity scalar. 1.0 is "normal"; values
            above 1.0 produce a brighter light. Renderer interprets the
            scale.
        size: Physical radius of the emitting surface in metres. Used to
            set the core glow disc size.
        glow_radius: Radius of the atmospheric glow halo around the
            light, in metres. Larger values spread the bloom further.
        illumination_radius: Ground illumination radius in metres — how
            far from the light the ground surface is visibly brightened.
        illumination_intensity: Strength of ground illumination, in
            [0, 1]. 0.0 means no ground brightening; 1.0 means full
            contribution from this light.
        illumination_falloff: Exponent controlling the cone falloff
            shape. Controls how fast intensity drops from the inner edge
            to the outer radius. Lower values (0.5) produce a gradual
            fade; higher values (4.0+) produce a tight spotlight cone.
        illumination_inner_fraction: Fraction of illumination_radius
            that is at full brightness before falloff begins, in [0, 1].
            0.0 means falloff starts immediately at the center; 1.0
            means the entire radius is fully lit (no falloff at all).
        beam_angle_deg: Full luminaire cone angle in degrees. When set (and
            the light has a positive z height), the illumination pass derives
            the ground pool physically from pole height and this angle and
            ignores illumination_radius / illumination_falloff /
            illumination_inner_fraction. None keeps the legacy two-zone cone,
            used by lights with no meaningful pole height (cars, signs, houses).
    """

    position: np.ndarray
    color: t.Tuple[int, int, int]
    brightness: float
    size: float
    glow_radius: float
    illumination_radius: float
    illumination_intensity: float
    illumination_falloff: float
    illumination_inner_fraction: float
    beam_angle_deg: t.Optional[float] = None
