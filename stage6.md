# Stage 6: Polish, Export, & Artistic Tuning

## Goal
Refine the full render pipeline for print-quality output. Add final atmospheric effects,
tune parameters for the San Bernardino scene, and produce a finished image ready for
large-format printing.

## Architecture Reference
See `.claude/work/architecture-plan.md` for the full module map and design decisions.

## Atmospheric & Visual Effects

### Horizon Glow
- Near the horizon, there should be a faint warm glow (light pollution effect)
- Cities cast light upward that scatters in the atmosphere
- Render as a gradient band above the horizon: warm orange-brown fading to black
- Intensity based on the density of lights below that part of the horizon

### Distance Haze
- Beyond ~15-20km, lights blur together into a general glow
- Individual lights are no longer distinguishable at extreme distance
- Implement as a progressive blur + brightness reduction with distance

### Vignette (optional)
- Slight darkening at the edges of frame
- Common in aerial photography, adds focus to the center

### Stars (optional)
- A few faint stars in the sky portion above the horizon glow
- Very subtle — the light pollution washes most out

## High-Resolution Render Pipeline

### Render Passes (full export pipeline)
Orchestrated by `render_pipeline.py` using the Pillow backend:

1. **Ground pass** (`ground_pass.py`): render ground tiles at full resolution
2. **Road pass** (`road_pass.py`): render road surfaces
3. **Illumination pass** (`illumination_pass.py`): accumulate all light contributions onto
   the ground (rendered at reduced resolution, gaussian blur, upscale)
4. **Light pass** (`light_pass.py`): render bright point sources + glow halos on separate
   layer with additive blending
5. **Atmosphere pass** (`atmosphere_pass.py`): apply distance haze and horizon glow
6. **Composite** (`compositor.py`): combine all layers, apply vignette, color grading
7. **Post-process**: final adjustments

### Resolution & Memory
For the "high" quality preset (7680x4320):
- Ground tiles loaded on-demand (only visible tiles, ~128×128 px each)
- Illumination rendered at reduced resolution (configurable multiplier)
- Light/glow layer at full resolution
- Total memory: ~500MB-1GB for all layers
- Tiled ground textures prevent the OOM risk that a single 20000×20000 texture would cause

### Export Formats
- PNG for digital viewing (sRGB color space)
- TIFF for print (can embed ICC profile if needed)
- Include metadata: camera parameters, render settings, date, seed

### Render Progress
- Print progress to console: "Rendering pass 3/6: illumination... 45%"
- For a high-res render that takes minutes, this is essential feedback

## Parameter Tuning

### Scene-Specific Adjustments for San Bernardino
After the full pipeline works, tune via `scene_config.toml` for this specific view:
- I-15 and I-215 should be prominently lit (major highways through the area)
- The commercial areas along Baseline St and Highland Ave should have business lighting
- The residential grid pattern of the valley should be visible
- The mountains to the north (behind the camera) should be dark
- University/institutional areas may have their own lighting character

### Tuning Workflow
The pygame preview + "P" key export makes for a fast iteration loop:
1. Edit `scene_config.toml` to adjust parameters
2. Re-launch preview to check changes (fast, approximate)
3. Press "P" or run `render` subcommand for mid-res export to check lighting quality
4. Repeat until satisfied
5. Final high-res export

## New/Modified Files
```
src/streets_at_night/
    rendering/
        atmosphere_pass.py   # Horizon glow, distance haze, vignette
        compositor.py        # Multi-layer compositing + post-processing
```

Also update:
- `config/schema.py` — add AtmosphereConfig dataclass
- `config/defaults.py` — add atmosphere default parameters
- `scene_config.toml` — add atmosphere section

### CLI Usage
All rendering uses the `main.py` entry point:
```bash
# Quick preview render
uv run python -m streets_at_night render --quality low --output output/test.png

# Full quality render
uv run python -m streets_at_night render --quality high --output output/san_bernardino_final.tiff

# Custom resolution
uv run python -m streets_at_night render --width 5000 --height 3000 --output output/custom.png

# Interactive preview
uv run python -m streets_at_night preview
```

## Acceptance Criteria
- [ ] Horizon glow effect visible (warm light pollution band)
- [ ] Distance haze makes far lights blur together naturally
- [ ] High-res render completes without running out of memory
- [ ] Render progress is reported to console
- [ ] Output image at "high" quality is suitable for large-format print
- [ ] CLI supports quality presets and custom resolution
- [ ] Atmosphere parameters configurable via scene_config.toml
- [ ] The San Bernardino scene is visually recognizable (highway patterns, grid streets)
- [ ] The overall image looks like an aerial night photograph of a city

## Notes
- This stage is about polish. The heavy engineering is done. Spend time looking
  at actual aerial night photography for reference and matching the feel.
- The horizon glow is crucial for selling the "looking from altitude" perspective.
  Without it, the scene will feel like floating in space rather than looking from
  a mountain or aircraft.
- Consider offering a "debug" render mode that labels road types, shows light
  placement, etc. — useful for diagnosing visual issues.
