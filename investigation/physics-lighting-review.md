# Physics / Lighting-Model Review

**Branch:** `heat_haze` · **Date:** 2026-06-14 · Point-in-time review of the lighting and atmosphere model.

The architecture is genuinely good — `LightSource` as the universal primitive, three separable contributions (core / screen-space glow / world-space ground pool), atmospheric perspective as its own layer. The heat-haze module is the strongest physics in the repo (Von Kármán PSD with the correct k^(-11/3) law, multi-layer frozen flow, resolution-invariant scaling — textbook-correct). This isn't a "you got it wrong" review. It's where the model is leaving realism and control on the table.

---

## The big one: everything composites in 8-bit gamma space and clamps at 255

`composite_additive` is `ImageChops.add` on uint8 (`rendering/pillow_backend.py:172`), and every pass — glow buffer, illumination map, horizon glow, haze — converts to uint8 and clamp-adds. Two problems, both physical:

1. **Summing gamma-encoded values.** Light is linear-additive in *radiance*, but sRGB 8-bit is perceptually encoded (~γ2.2). Adding two encoded half-values doesn't equal the encoding of the summed radiance. Overlapping halos and pooled illumination come out subtly wrong — usually muddy where they should be bright.
2. **Hard clip at 255.** Real night lights have enormous dynamic range; a bright core *should* blow out, but with a colored, rolled-off shoulder, not a flat-topped clip to pure white. Right now bright overlaps just smear to white with a hard edge. None of the highlight roll-off that makes night photography read as night photography.

**Fix (highest leverage in the whole review):** carry the whole pipeline as **float32 linear radiance**, composite additively in linear, and tone-map once at the very end (Reinhard / filmic / ACES) before encoding to 8-bit. The `_apply_color_grade` stub in `rendering/compositor.py:180` is the exact right home for the tone map — it's currently empty. This single change does more for realism than any per-fixture tuning, and it hands you the single most useful missing lever: global **exposure / EV**.

---

## The distance model conflates two different physical effects

`apply_brightness_falloff` is `brightness * exp(-d/falloff_distance)` (`rendering/render_utils.py:110`). That exponential is **Beer-Lambert atmospheric extinction** — real and legitimate. But:

- **No inverse-square term anywhere.** A point source's irradiance falls as 1/d². The core only shrinks angularly (`size/distance`); the glow peak alpha gets only extinction. For a stylized aerial piece you may *want* distant lights to stay readable — but right now you can't choose, because there's no knob for it.
- `falloff_distance`, `scatter_distance`, and `max_distance` are three loosely-coupled knobs all describing "the air," and color uses a *different* law than brightness: `_apply_haze_color` is a **linear** blend toward `haze_color` by `d/max_distance` (`rendering/light_pass.py:280`), while brightness is exponential. Inconsistent.

**Fix:** the standard airlight/fog model, one extinction coefficient driving both:

```
observed = direct·exp(-d/k)  +  airlight·(1 - exp(-d/k))
```

Direct light dims by extinction; haze color is *added* proportional to how much atmosphere is in the path. Collapses `falloff_distance`/`scatter_distance`/`max_distance`/`haze_color` into **one physical "haziness" coefficient + an airlight color**, and makes brightness and color consistent.

---

## Light color is authored as hex RGB instead of color temperature

The lever-design miss that matters most for this project. Street lighting is the canonical Kelvin problem: low-pressure sodium ~1700K (monochromatic amber), HPS ~2100K, mercury vapor ~4000K, modern LED 3000–5000K. The config hand-picks `#FFA040`, `#E0E8FF`, `#FFE080` and eyeballs the mix.

**Fix:** replace (or back) the color with `color_temp_k` → blackbody/CIE conversion. Then "shift the whole city from old sodium to new LED retrofit" is one slider instead of editing dozens of hex values across five road-type pools. The weighted pools themselves are great (mixed retrofits are realistic) — they're just authoring at the wrong primitive. Keep the pools; make each entry a temperature + saturation, not a hex string.

---

## The glow is a single screen-space gaussian — that's why bloom reads "soft" not "glary"

`_accumulate_glow` is one gaussian (`rendering/light_pass.py:396`). Real veiling glare / bloom around a bright point is a sharp core plus **wide power-law wings** (~1/θ² to 1/θ³) — that's what gives night lights their characteristic glare and faint starburst halo. A pure gaussian falls off too fast and looks dull.

**Fix (cheap, high impact):** two components — a tight gaussian + a wide power-law tail. The `glow_falloff_steepness` knob becomes a "wing exponent," which is both more intuitive and more physical.

---

## Ground illumination ignores pole height and incidence angle

`height` is configured per fixture but only sets the light's z-position (`models/streetlight.py:130` — which also hardcodes `size=0.3`; `size` isn't physical anywhere). The ground pool's `illumination_radius` is authored *independently* of height. Physically the illuminance under a lamp at height h goes as `h/(h²+r²)^{3/2}` (inverse-square + cosine incidence), so a taller pole gives a wider, flatter, dimmer pool — for free, from a number you already have.

Right now the two-zone smoothstep cone (`rendering/illumination_pass.py:185`) is a decent art approximation. Optional fix: *derive* radius/falloff from height + beam angle. Defensible as a stylization as-is, but the height data is sitting there unused.

---

## Smaller notes

- **Heat haze at night is physically marginal.** Thermal shimmer is a daytime-over-hot-surfaces effect; at night the real equivalent is convective plumes / waste heat, which is subtle. The *model* (`rendering/heat_haze.py`) is excellent and `amplitude_px=0.25` is appropriately conservative — just flagging that the phenomenon is an art choice more than a given. The depth-scaling (shimmer ∝ path length, far ground shimmers more) is the correct lever and well done.
- **Stars** at brightness 0.03–0.15 → 8–38 in 8-bit; near the horizon they'll get crushed by additive haze/horizon-glow and the vignette multiply. In a linear pipeline you'd set these in magnitude terms and they'd survive correctly.
- **No occlusion** — buildings don't block light pools. At this altitude mostly fine; storefront/sign spill across walls is a possible artifact, low priority.

---

## What I'd actually do, in order

1. **Linear float HDR pipeline + final tone-map** (`compositor._apply_color_grade`). Biggest realism gain, and it gives you a global exposure/EV knob that doesn't exist today. "Make it brighter" currently means editing dozens of per-fixture brightness ranges; it should be one exposure value.
2. **Airlight/extinction model** with one shared haziness coefficient driving both dimming and color. Collapses four confused knobs into one physical one.
3. **Color temperature (Kelvin)** as the color authoring primitive.
4. **Power-law glare PSF** instead of the lone gaussian.

Items 1 and 2 are where realism and lever quality both jump. Item 3 best fits the "art-direct a whole city" goal.
