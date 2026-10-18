// Shared per-frame uniforms. Layout must match gpu_scene.pack_frame_uniforms.
struct Frame {
    view_proj: mat4x4<f32>,
    camera: vec4<f32>,    // xyz = camera position (m), w = focal length (px)
    viewport: vec4<f32>,  // x = width (px), y = height (px)
    knee: vec4<f32>,      // x = knee_start, y = knee_max, z = pool count, w = window extinction
    glow: vec4<f32>,      // x = steepness, y = wing exponent, z = tail strength, w = cutoff mult
    grade: vec4<f32>,     // x = 2^exposure, y = vignette strength, z = vignette radius, w = encode sRGB
    grade2: vec4<f32>,    // x = 1 for ACES, 0 for clip
};

@group(0) @binding(0) var<uniform> frame: Frame;
