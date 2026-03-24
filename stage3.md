# Stage 3: Pygame Live Preview & Render Backend

## Goal
Create an interactive pygame window that renders the perspective view of roads and horizon
in real-time, with keyboard/mouse controls to adjust the camera. Also establish the
RenderBackend abstraction and Scene facade that all subsequent stages build on.

## Architecture Reference
See `.claude/work/architecture-plan.md` for the full module map and design decisions.

## Technical Approach

### RenderBackend Abstraction (Strategy Pattern)
Rendering logic (what to draw, in what order) is written once. Two backends implement
the drawing primitives differently:

**`rendering/backend.py`** — abstract base class:
```python
class RenderBackend(ABC):
    draw_line(p1, p2, color, width)
    draw_polygon(vertices, fill_color)
    draw_circle(center, radius, color)
    get_surface_size() -> tuple[int, int]
    clear(color)
    present()           # pygame: flip; Pillow: no-op
    get_image()         # returns PIL.Image (for export)
```

**`rendering/pygame_backend.py`** — implements drawing via `pygame.draw.*`
**`rendering/pillow_backend.py`** — implements drawing via `PIL.ImageDraw`

The Pillow backend also supports layer operations (create_layer, composite_layers,
apply_gaussian_blur) that the pygame backend does not. These are used by export-only
render passes in later stages.

### Scene Facade
Build the `Scene` class in `scene/scene.py`. At this stage it holds:
- Road segments (from stage 1 data, converted to local meters in stage 2)
- Camera
- Config

Later stages add zones, lights, and the spatial index.

```python
class Scene:
    road_segments: list[RoadSegment]
    camera: Camera
    config: SceneConfig
    # zones, lights, spatial_index added in stages 4-5
```

### Spatial Index
Build `scene/spatial_index.py` now — it's needed for preview performance with 100k+ road
segments.

Uniform grid, 500m × 500m cells:
```python
class SpatialIndex:
    __init__(bounds, cell_size=500)
    insert(item, position_or_bbox)
    query_rect(min_x, min_y, max_x, max_y) -> list
    query_radius(x, y, radius) -> list
```

Index all road segments on Scene construction. Frustum culling = "which grid cells overlap
the camera frustum?" → draw everything in those cells.

### Render Pipeline
**`rendering/render_pipeline.py`** orchestrates the multi-pass render sequence:
```python
def render_frame(scene, backend, passes):
    for render_pass in passes:
        render_pass(scene, scene.camera, backend)

def get_preview_passes():
    return [road_pass.render_roads]  # minimal for now

def get_export_passes():
    return [road_pass.render_roads]  # expanded in later stages
```

**`rendering/road_pass.py`** — road rendering logic (backend-agnostic):
- Query Scene for visible road segments (via spatial index + frustum culling)
- Sort back-to-front by distance
- Project endpoints, clip at near plane
- Draw via backend.draw_line() with color/width per road type + distance

### Preview Architecture
The preview uses the render pipeline with the pygame backend:
- Roads drawn as colored lines (no glow, no lights)
- Flat sky/ground colors split at horizon
- Target: 30+ FPS at low resolution (1280x720)
- This becomes the primary tool for framing the shot before doing a full render

### Pygame Window
- Default to 1280x720 for the preview (responsive, fast)
- Display current camera parameters in a HUD overlay (position, altitude, angle, FOV)
- Show FPS counter

### Rendering Loop (Preview)
Each frame:
1. Handle input events → update camera state
2. Call `render_pipeline.render_frame(scene, pygame_backend, preview_passes)`
   - Pipeline queries spatial index for visible roads
   - Projects and draws via pygame backend
3. Draw horizon line
4. Draw HUD overlay
5. `backend.present()` → flip pygame surface

### Performance Considerations
- Spatial index provides fast frustum culling (no need to project all 100k+ segments)
- Pre-compute: road geometry in local coordinates doesn't change — only the projection
  changes when the camera moves
- numpy-vectorized projection: project all visible points in one matrix multiply
- If still too slow, drop service/residential roads beyond a distance threshold

### Camera Controls
```
Keyboard:
  W/S         — Move camera forward/backward (along ground plane, toward/away from target)
  A/D         — Strafe camera left/right
  Q/E         — Raise/lower camera altitude
  Up/Down     — Tilt camera up/down (pitch)
  Left/Right  — Pan camera left/right (yaw)
  +/-         — Adjust FOV (zoom)
  1/2/3       — Switch quality preset (for export)
  R           — Reset camera to default position
  P           — Save current view as high-res image (using Pillow backend + export passes)
  ESC         — Quit

Mouse:
  Right-click drag — Rotate camera (yaw/pitch)
  Scroll wheel     — Adjust altitude
```

### HUD Display
Show in a semi-transparent overlay:
- Camera lat/lon and altitude (convert back from local coords via coordinates.py)
- Look direction (bearing and pitch angle)
- FOV
- Number of road segments being rendered
- FPS
- Current quality preset

### CLI Integration
The `preview` subcommand in `main.py` launches the pygame window:
```bash
uv run python -m streets_at_night preview
```

The `render` subcommand uses the Pillow backend:
```bash
uv run python -m streets_at_night render --quality low --output output/test.png
```

Both share the same init path: load config → load data → build Scene → render.

### New Files
```
src/streets_at_night/
    scene/
        __init__.py
        scene.py             # Scene facade
        spatial_index.py     # Uniform grid spatial index
    rendering/
        __init__.py
        render_pipeline.py   # Multi-pass orchestrator
        backend.py           # RenderBackend ABC
        pygame_backend.py    # Pygame implementation
        pillow_backend.py    # Pillow implementation
        road_pass.py         # Road rendering logic (backend-agnostic)
    camera/
        controls.py          # Keyboard/mouse input → camera state changes
```

## Dependencies (new)
- `pygame` — interactive preview window

## Road Drawing Details
Road types and their preview colors/widths:
```
Highway:     color=#FFD700 (amber/yellow), width=4px at base
Major Road:  color=#FF6B35 (orange),       width=3px at base
Collector:   color=#FFFFFF (white),         width=2px at base
Residential: color=#888888 (grey),          width=1px at base
Service:     color=#555555 (dark grey),     width=1px at base
```
Width scales with distance: `apparent_width = base_width * (reference_dist / actual_dist)`

## Acceptance Criteria
- [ ] RenderBackend ABC defined with both pygame and Pillow implementations
- [ ] Scene facade holds road segments, camera, config
- [ ] Spatial index built over road segments, frustum queries work
- [ ] Render pipeline runs road_pass through either backend
- [ ] Pygame window opens showing the San Bernardino road network in perspective
- [ ] Horizon line visible, sky above and ground below
- [ ] Roads colored by type with appropriate widths
- [ ] Camera moves smoothly with keyboard controls
- [ ] Mouse right-drag rotates the camera view
- [ ] HUD shows camera position and FPS
- [ ] "P" key exports a high-res still image via Pillow backend
- [ ] Maintains 30+ FPS with full road dataset at 1280x720
- [ ] ESC quits cleanly
- [ ] `preview` and `render` subcommands both work from main.py

## Notes
- Don't try to make the preview look like the final render. It's a framing tool.
  Keep it simple and fast.
- pygame draws lines with integer coordinates — we'll need to round projected
  positions. Anti-aliasing via `pygame.draw.aaline` is available but slower.
  Start without AA and add it if performance allows.
- The back-to-front sort is important: without it, distant highways will draw
  over nearby residential streets and it'll look wrong.
- The RenderBackend abstraction is intentionally thin. It provides drawing primitives
  only. Render *logic* (what to draw, in what order) lives in the pass modules.
  The Pillow backend has extra methods (layer compositing, blur) that are used by
  export-only passes — the pygame backend doesn't need these.
