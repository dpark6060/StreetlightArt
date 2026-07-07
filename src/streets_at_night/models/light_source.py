"""Universal light source data model consumed by the renderer."""

import typing as t
from dataclasses import dataclass

import numpy as np

from .illumination import IlluminationModel


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
        illumination_intensity: Strength of ground illumination, in
            [0, 1]. 0.0 means no ground brightening; 1.0 means full
            contribution from this light. Shared by both pool models.
        illumination: The ground-pool model — ``BeamIllumination`` for
            pole lights, ``ConeIllumination`` for authored cone pools, or
            ``None`` for lights that cast no ground pool (taillights, etc.).
    """

    position: np.ndarray
    color: t.Tuple[int, int, int]
    brightness: float
    size: float
    glow_radius: float
    illumination_intensity: float
    illumination: IlluminationModel = None
