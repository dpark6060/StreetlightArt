# Stage 2: Coordinate System & Camera Model

## Goal
Convert all geo data into a local meter-based coordinate system, implement a 3D camera with
perspective projection, and render a static test image of the projected roads (proving the
math works before adding interactivity).

## Architecture Reference
See `.claude/work/architecture-plan.md` for the full module map and design decisions.

## Technical Approach

### Coordinate Conversion
- Convert all lat/lon coordinates to a local East-North-Up (ENU) coordinate system in meters
- Origin: midpoint between camera and target (~34.17°N, 117.30°W)
- Use `pyproj` for the conversion (WGS84 → UTM Zone 11N for this area, then offset to local origin)
- X = East, Y = North, Z = Up (altitude in meters)
- Camera position: (x, y, 1581m)
- Target position: (x, y, 305m)
- Roads: z=0 (ground level, all at ~305m elevation — San Bernardino valley floor)

### Camera Model
Implement a standard pinhole camera in `camera/camera.py`:
- **Position**: 3D point in local coordinates
- **Look-at**: 3D target point
- **Up vector**: (0, 0, 1) — world up
- **FOV**: horizontal field of view (from `scene_config.toml`, default ~60°)
- **Aspect ratio**: based on output resolution (from quality preset in config)

From these, derive:
- Forward vector: normalize(target - position)
- Right vector: normalize(forward × up)
- Camera up: normalize(right × forward)
- View matrix: transforms world coords to camera space
- Projection matrix: perspective projection
- Frustum planes: for culling

### Perspective Projection Pipeline
```
World coords (meters) → View space (camera-relative) → Clip space → Screen space (pixels)
```

Implemented in `camera/projection.py` with batch operations:
- `project_points(points_3d, view_matrix, proj_matrix, screen_size)` — vectorized with numpy
- `clip_line_segment(p1, p2, near_plane)` — clips a line at the near plane
- `frustum_cull_points(points_3d, frustum_planes)` — returns mask of visible points

Key considerations:
- The camera is at 5186ft looking ~5° below horizontal — the horizon is near the top of frame
- Objects far from camera should get smaller (perspective foreshortening)
- Roads behind the camera should be culled
- Roads beyond a max render distance should be culled (dim/invisible anyway)
- Near plane: ~100m, Far plane: ~60000m (60km)

### Horizon Line
- The horizon is where the ground plane (z=0) meets the edge of visibility
- At 1581m altitude, the geometric horizon is ~142km away (earth curvature)
- We'll use a flat earth approximation (fine for <60km) with a far-plane cutoff
- Everything above the horizon line = sky color
- Everything below = ground (will be rendered in later stages)

### Test Render
Use the `render` subcommand in `main.py` to produce a static Pillow image:
- Black background (sky)
- Dark grey below horizon (ground)
- Roads drawn as lines, colored by type, with perspective projection applied
- Thicker lines for highways, thinner for residential
- Line thickness should also decrease with distance (perspective)
- Save to `output/stage2_perspective_test.png`

```bash
uv run python -m streets_at_night render --quality low --output output/stage2_perspective_test.png
```

### New Files
```
src/streets_at_night/
    camera/
        __init__.py
        coordinates.py     # WGS84 ↔ local ENU conversion
        camera.py          # Camera class (position, orientation, matrices, frustum)
        projection.py      # Batch 3D→2D projection, line clipping, frustum culling
```

Also update:
- `config/schema.py` — add CameraConfig, QualityPreset dataclasses
- `main.py` — add `render` subcommand

### Quality Presets
Defined in `scene_config.toml` and loaded into `QualityPreset` dataclass:
```python
@dataclass
class QualityPreset:
    width: int
    height: int
    illumination_resolution_multiplier: float = 1.0  # <1.0 for faster illumination
    ground_tile_resolution: float = 2.0               # meters per pixel
```

## Dependencies (new)
- `pyproj` — coordinate system transformations
- `Pillow` — image rendering (used from here on for static renders)

## Key Math
The perspective projection for a point P in world space:
1. Transform to camera space: P_cam = ViewMatrix × P_world
2. Check if behind camera (P_cam.z < near_plane → cull)
3. Project: screen_x = (P_cam.x / P_cam.z) * focal_length + cx
4. Project: screen_y = (P_cam.y / P_cam.z) * focal_length + cy
5. Lines that cross the near plane need to be clipped (not just culled)

For road line segments: project both endpoints, clip against the view frustum, draw.

## Acceptance Criteria
- [ ] Coordinate conversion produces sensible meter values for the San Bernardino area
- [ ] Camera positioned at correct altitude with correct look direction
- [ ] Camera parameters loaded from scene_config.toml
- [ ] Perspective test image shows roads converging toward a vanishing point
- [ ] Horizon line is visible near the top of frame (camera is looking slightly down)
- [ ] Roads get thinner/smaller with distance
- [ ] Different road types have different colors/thicknesses
- [ ] No visual artifacts from roads behind the camera
- [ ] Quality presets produce different resolution outputs

## Notes
- The projection math needs to handle line clipping carefully — a road segment
  can start behind the camera and end in front of it. We need to clip at the
  near plane, not just discard the whole segment.
- numpy will be heavily used here for matrix math. Keep the projection vectorized
  (project all points at once) for performance — this is critical for the preview.
- At this shallow camera angle, foreshortening will be extreme — roads near the
  camera will be large, roads 30km away will be tiny. This is correct and is
  part of the aesthetic.
