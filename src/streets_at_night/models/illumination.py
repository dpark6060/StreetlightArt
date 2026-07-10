"""Ground-illumination pool models attached to a LightSource.

A ``LightSource`` describes how it pools light on the ground with exactly one
of these models (or ``None`` for lights that cast no ground pool at all, e.g.
taillights or window emitters). This replaces the older shape where a single
``LightSource`` carried both the cone fields and a ``beam_angle_deg`` with an
implicit "physical if beam set and z > 0" selector.
"""

import typing as t
from dataclasses import dataclass


@dataclass(frozen=True)
class ConeIllumination:
    """Authored two-zone cone ground pool.

    A flat inner disc (``inner_fraction`` of ``radius``) at full intensity,
    then a power-law fade to zero at ``radius``. Used by lights with no
    meaningful pole height — cars, signs, storefronts.

    Attributes:
        radius: Ground pool radius in meters.
        falloff: Exponent controlling the fade from the inner edge to the
            outer radius. Lower is gradual, higher is a tight spotlight.
        inner_fraction: Fraction of ``radius`` at full brightness before the
            falloff begins, in [0, 1].
    """

    radius: float
    falloff: float
    inner_fraction: float


@dataclass(frozen=True)
class BeamIllumination:
    """Physical ground pool derived from pole height + luminaire beam angle.

    The pool radius and intensity are computed at render time from the
    light's z height and this cone angle (inverse-square + cosine-incidence),
    so a taller pole gives a wider, flatter, dimmer pool for free. Used by
    streetlights, which have a real pole height.

    Attributes:
        beam_angle_deg: Full luminaire cone angle in degrees.
    """

    beam_angle_deg: float


# A light's ground-pool model: a physical beam, an authored cone, or no pool.
IlluminationModel = t.Union[ConeIllumination, BeamIllumination, None]
