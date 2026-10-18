# Plan 05 — GPU 3D tuning scene (`streets-at-night tune`)

A small, real-time, orbitable 3D scene for tuning light values against human-scale
references. First piece of the GPU renderer: the light math written here is the same
math the full GPU city renderer will use later.

Stack (decided): wgpu-py on Metal + rendercanvas (glfw) window + imgui-bundle panel.
Spike on 2026-10-18: wgpu 0.32.0, rendercanvas 2.7.2, imgui-bundle 1.92.9, adapter
"Metal / Apple M4 Pro", `wgpu.utils.imgui.ImguiRenderer` imports fine.

## 1. Scene (ENU meters, origin at the person)

```
        B  (8 x 6 m footprint, 6 m tall, 2 floors, at (-14, +16))

  ======================== road, E-W, centered y = +5 =========================
                C  (4.7 x 1.8 x 1.45 m car on the road, at (0, +3.5))
  L0  P                        L1                         L2         (south curb, y = 0)
```

- Road: one E-W strip, x in [-150, +250], width = 10 m (two lanes). Asphalt color from
  `config.roads` style, same as the 2D road pass.
- Building `B`: one `Building` box with the real window grid (`build_window_grid`), so
  window size, pitch, lit fraction, color temp, brightness come from `[buildings]`.
- Car `C`: an extruded box, 4.7 x 1.8 x 1.45 m (plan-04 length). Its headlight cone,
  head/taillight points come from a real `CarLightPair` (cars config), facing +x.
- Person `P`: 0.45 x 0.3 x 1.75 m box at (1.5, -0.5) next to the first pole.
- Poles: L0 at (0, 0); L1, L2 further east at the selected fixture's sampled
  `spacing` (spacing_min..max, real RNG). Pole drawn as a 0.2 m box to the fixture's
  sampled height; the light sits at the top.
- Ground plane: large dark quad, `default_ground` color.

## 2. What gets lit depends on the selected light kind

The selector reuses `devmode/light_catalog.build_light_catalog`:

| Selected kind | What is placed at L0..L2 |
|---|---|
| Streetlight: <class> (pool n) | that pool's profile via `StreetLight.create_from_profile` |
| Commercial: Parking | `create_parking_lot_light` (pole height from parking_height_min/max) |
| Commercial: Signs | one sign light on the building's road-facing wall, 5 m up |
| Commercial: Storefronts | storefront lights along the building's road-facing wall, 3 m up |
| Cars: Headlights / Taillights | (poles keep the last streetlight kind) — the car's own lights |
| Buildings: Windows | (poles keep the last streetlight kind) — the building's windows |

Two checkboxes: **Road lights** (poles + car lights) and **Building lights** (windows,
signs, storefronts). Panel has two sub-tabs: **Road / Commercial** and **Building**,
splitting the catalog the same way.

Lights are built with the real factory functions, so any field the 2D renderer reads is
what the 3D scene reads. Sliders reuse `devmode/field_meta.get_field_meta` ranges and
edit the live `SceneConfig` dataclasses; any edit rebuilds the tiny scene (<1 ms).

## 3. Rendering — parity with the 2D export pipeline

HDR float target (rgba16float), additive light, then one composite pass. Each item
below names the CPU function it mirrors; the WGSL is a line-for-line port.

| GPU pass | Mirrors | Notes |
|---|---|---|
| Opaque geometry (ground, road, boxes) | `ground_pass`, `road_pass`, `building_pass` face shade | flat sRGB→linear albedo, `face_shade_strength` per wall normal |
| Ground illumination | `illumination_pass._physical_falloff`, `_cone_falloff`, `_directional_cone_falloff`, 1/h² scaling, `_apply_soft_knee` | per-pixel loop over a storage buffer of lights (scene has < 64); evaluated on the ground plane only, exactly like 2D; knee applied to the summed map |
| Windows | `building_pass` window cells | emissive quads; `window_glow_px` approximated by the bloom below |
| Cores + glows | `light_pass._accumulate_core`, `_accumulate_glow` | camera-facing billboards sized in world meters (`size`, `glow_radius * glow_cutoff_mult`); same gaussian + power-law tail profile; extinction `exp(-d / falloff_distance)` |
| Composite | `compositor._apply_color_grade`, `_apply_vignette` | exposure (stops), Narkowicz ACES, sRGB encode; stars skipped |

Out of scope for the tuning scene: atmosphere airlight veil, horizon glow, heat haze,
stars. Extinction is kept (cheap, and it affects brightness slightly even at 100 m).

Projection: reuse `camera.Camera` for view/projection, then remap clip z from
OpenGL [-1, 1] to WebGPU [0, 1]. near = 0.1 m, far = 2000 m.

## 4. Camera controls

Orbit camera around a ground pivot (starts at the person, distance 25 m, pitch 30°):
- drag: yaw/pitch around the pivot (pitch clamped 2°..89°)
- shift + drag: pan the pivot on the ground plane (screen-relative, scaled by distance)
- scroll: zoom (distance x 1.1 per notch, clamped 2 m..1500 m)
- R: reset view
Mouse events go to imgui first; the camera only sees them when imgui doesn't want them.

## 5. Files

```
src/streets_at_night/gpu/
  __init__.py
  device.py            # adapter/device creation, offscreen target helper
  orbit_camera.py      # OrbitCamera state + input → camera.Camera
  tuning_scene.py      # build the scene: boxes, lights via real factories, pole layout
  gpu_scene.py         # pack boxes/lights into vertex + storage buffers
  renderer.py          # pipelines + per-frame passes
  shaders/
    geometry.wgsl      # opaque boxes, ground, road
    illumination.wgsl  # ground light pools (falloff ports)
    billboard.wgsl     # cores + glows
    composite.wgsl     # exposure, ACES, vignette, sRGB
  tune_app.py          # window, imgui panel, event loop, Save
main.py                # new `tune` subcommand → tune_app.run_tune(config, config_path)
```

Save button writes via `devmode/config_writer.write_config` to the loaded config path.
Nothing writes to disk otherwise.

Dependencies added with `uv add`: wgpu, rendercanvas, glfw, imgui-bundle.

## 6. Tests

- `tests/test_orbit_camera.py`: drag/pan/zoom math, clamps, reset (pure numpy).
- `tests/test_tuning_scene.py`: pole spacing comes from the selected profile, selected
  kind → right factory, toggles drop the right lights, car/person dimensions.
- `tests/test_gpu_parity.py`: render a single light's ground pool offscreen on the GPU
  and compare against `_paint_illumination_blob` sampled at the same ground points
  (physical, cone, directional), plus composite vs `tonemap_aces`/`linear_to_srgb`.
  Skipped when no wgpu adapter is available. This is the drift guard.

## 7. Known gaps vs. the 2D render

- Window glow is a bloom approximation, not the 2D box blur.
- Lights only illuminate the ground (as in 2D). Walls, car, and person are not lit by
  pools — a deliberate parity choice; lighting them is a later GPU-renderer feature.
