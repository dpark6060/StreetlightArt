"""Ground-illumination pool models attached to a LightSource.

A ``LightSource`` describes how it pools light on the ground with exactly one
of these models (or ``None`` for lights that cast no ground pool at all, e.g.
taillights or window emitters). This replaces the older shape where a single
``LightSource`` carried both the cone fields and a ``beam_angle_deg`` with an
implicit "physical if beam set and z > 0" selector.

Available models:

- ``ConeIllumination`` — radially symmetric authored two-zone cone pool.
- ``BeamIllumination`` — physical pool derived from pole height + beam angle.
- ``DirectionalConeIllumination`` — oriented ground cone cast forward along a
  heading (car headlights).
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
    streetlights and parking lot lights, which have a real pole height.

    The pool has two zones. Inside ``bulb_fraction`` of the beam radius the
    bulb is in direct view. Past that, the housing hides the bulb and only the
    concave reflector lights the ground; less of it is visible the further out
    you go, so this band fades from the bulb-zone brightness down to zero at
    the housing cutoff, with no step at the boundary. Because both cutoffs are
    angles, both radii scale with pole height.

    Attributes:
        beam_angle_deg: Full housing cutoff angle in degrees. Sets the outer
            pool radius, ``height * tan(beam_angle_deg / 2)``.
        bulb_fraction: Bulb cutoff radius as a fraction of the beam radius, in
            [0, 1]. 1.0 means the bulb is visible across the whole pool.
        reflector_falloff: Fade exponent for the reflector band. 1.0 is a
            gentle smoothstep fade; higher values die off faster.
    """

    beam_angle_deg: float
    bulb_fraction: float = 1.0
    reflector_falloff: float = 1.0


@dataclass(frozen=True)
class DirectionalConeIllumination:
    """Oriented ground cone cast forward from the light along a heading.

    A wedge-shaped pool on the ground: full intensity near the apex fading
    to zero at ``length`` along the heading (same two-zone axial curve as
    ConeIllumination), feathered to zero at the cone's angular edge, and
    zero everywhere behind the apex. Used for car headlights — one cone per
    car, pointing in the travel direction.

    Attributes:
        direction: Unit (east, north) heading of the cone axis in ENU.
        length: Cone reach in meters along the heading.
        beam_angle_deg: Full horizontal cone angle in degrees.
        falloff: Axial fade exponent (same semantics as ConeIllumination).
        inner_fraction: Fraction of ``length`` at full brightness, in [0, 1].
    """

    direction: t.Tuple[float, float]
    length: float
    beam_angle_deg: float
    falloff: float
    inner_fraction: float


# A light's ground-pool model: a physical beam, an authored cone, an oriented
# headlight cone, or no pool.
IlluminationModel = t.Union[
    ConeIllumination, BeamIllumination, DirectionalConeIllumination, None
]
