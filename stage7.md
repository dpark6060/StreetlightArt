# Stage 7: 3D Depth Illusion (Future)

## Goal
Move the light points into true 3D space such that from exactly one camera angle the image
looks like a perfect 2D rendering of the city at night, but from any other angle the lights
scatter into a volumetric point cloud — like stars or a galaxy.

## Concept
Every light point currently sits on the ground plane (z = 0 in local coords, or at ground
elevation). In this stage:
- Each light gets a unique, random Z offset (depth) — some are close, some are far
- The X and Y positions are adjusted so that from the "design camera" position, each light
  projects to exactly the same screen position as it did in the flat version
- From any other viewpoint, the lights appear scattered in 3D space

This is the same principle as an anamorphic illusion — it only looks "correct" from one
specific viewpoint.

## Technical Approach

### Depth Assignment
For each light at screen position (sx, sy), currently at world position (wx, wy, 0):
1. Assign a random depth value `d` (e.g., uniform between 0 and some max depth, or
   exponential distribution for more variation)
2. Compute the new 3D position (wx', wy', wz') that projects to the same (sx, sy) from
   the design camera: shoot a ray from the camera through (sx, sy) and find the point at
   depth `d` along that ray
3. This new position IS the light's 3D location

### Blender Pipeline
Blender is the natural choice for this stage:
- Export all light positions as a point cloud
- Each point gets an emissive material with the light's color and size
- Render from the design camera → should match the 2D image exactly
- Render from other angles → reveals the 3D structure
- Blender's volumetric rendering can add atmospheric effects
- Can output as a turntable animation showing the reveal

### Blender Python Script (bpy)
```python
import bpy

# For each light:
#   - Create a small emissive sphere (or point light) at (wx', wy', wz')
#   - Set material to emission shader with the light's color
#   - Set the sphere radius proportional to the light's size

# Set up the design camera at the exact position/orientation from our config
# Render → should match the Pillow output

# Create an orbit animation:
# - Camera orbits around the scene center
# - Frame 0: design camera position (2D illusion)
# - Frame 120: 45° offset (3D scatter revealed)
```

### Alternative: Three.js / WebGL
If the goal is an interactive viewer (rotate the scene in a browser):
- Export lights as a JSON point cloud
- Render with Three.js PointsMaterial
- User can orbit the camera
- Starts at the design angle, can freely rotate
- This might be more impactful than a video — the viewer discovers the 3D themselves

### Output Formats
- **Still image**: from the design angle (identical to stage 6 output)
- **Turntable video**: slowly orbiting around the point cloud
- **Interactive web viewer**: browser-based Three.js scene (optional)
- **Physical installation**: if this becomes a physical art piece, the point cloud
  could be rendered as LED positions in a 3D volume (but that's well beyond this project)

## Acceptance Criteria
- [ ] Each light has a unique 3D position
- [ ] From the design camera angle, the 3D render matches the 2D render
- [ ] From other angles, lights appear as a scattered point cloud
- [ ] Blender scene renders correctly
- [ ] Turntable animation shows the 2D → 3D transition

## Notes
- This stage is deliberately separate and future-looking. Stages 1-6 produce a
  complete, beautiful artwork. Stage 7 transforms it into something conceptually
  different — an anamorphic installation.
- The depth distribution matters artistically. Uniform random will look like noise.
  Consider clustering depths by zone (residential lights at one depth range,
  highway lights at another) or using the actual distance-from-camera to inform
  depth. Experiment.
- The "reveal" moment — rotating away from the design angle and seeing the flat
  image dissolve into a 3D cloud — is the whole point. The transition should feel
  magical, not just messy.
