# Streetlight tuning cheat sheet

Research-backed values for realistic streetlight ground pools (2026-07-10).
Full sources, IES photometric reasoning, and the unit-label code plan:
`investigation/plan-03-streetlight-pool-size.md`.

## Units and the radius formula

Every radius / height / spacing box in the devmode panel is world-space ENU
**meters**. Streetlight pools have no radius box — the pool size is derived:

```
pool_radius_m = pole_height_m * tan(beam_angle_deg / 2)
```

So `Height Min/Max` and `Beam Angle Deg` are the knobs. Target from IES
Type II/III half-max throw: pool radius ≈ 1.2–1.9× pole height.

## Pole heights — already realistic, don't touch

| Class | height_min–height_max |
|---|---|
| Residential | 5–7 m |
| Collector | 7–9 m |
| Major Road | 8–10 m |
| Highway | 12–15 m |

## Values to enter (same numbers in every pool of the class)

| Devmode section | Beam Angle Deg | Illumination Intensity | Resulting pool radius |
|---|---|---|---|
| Streetlight: Highway | **120** (was 147.7) | **0.9** (was 1.48) | 21–26 m |
| Streetlight: Major Road | **115** (was 147.0) | 0.75 (keep) | 12.6–15.7 m |
| Streetlight: Collector | **110** (was 179.0!) | **0.65** (was 2.13) | 10–13 m (was 802–1031 m) |
| Streetlight: Residential | **100** (was 136.0) | **0.5** (was 1.85) | 6–8.3 m |
| Streetlight: Service | **100** (was 138.3) | **0.4** (was 1.94) | 4.8–7.2 m |

Sanity check: residential pool diameter comes out 12–17 m ≈ one US house
frontage, matching real aerial night photos. The collector class at 179° was
the blowout — `tan(89.5°) = 114.6`, giving kilometer-wide pools per fixture.

## Secondary oversized radii (authored cone-model lights)

| Section | Field | Current | Suggested |
|---|---|---|---|
| Commercial: Parking | Illumination Radius | 71.7 | **20–25** (standout offender) |
| Cars: Headlights | Illumination Radius | 22.2 | **10–15** |
| Commercial: Signs | Illumination Radius | 13.3 | 8–12 |
| Commercial: Storefronts | Illumination Radius | 14.8 | 10–14 |

## If everything reads too dim afterward

Raise the global `Atmosphere → illumination_brightness` instead of
re-inflating per-fixture intensities — that preserves the relative pool sizes
you just fixed.
