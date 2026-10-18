// Light cores and glow halos as screen-aligned quads. Ports of
// rendering/light_pass.py _accumulate_core and _accumulate_glow; sizes and
// peaks come from compute_light_sprite on the CPU.

struct SpriteOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) offset_px: vec2<f32>,
    @location(1) color_core: vec4<f32>,  // rgb = linear color, w = core radius (px)
    @location(2) params: vec4<f32>,      // x = core brightness, y = glow radius (px), z = glow peak
};

// Scene depth from the opaque pass. Only the light's center is depth-tested:
// a hidden source draws nothing, a visible one draws its whole glow over
// everything, like real veiling glare. The CPU pass never occludes lights.
@group(0) @binding(1) var scene_depth: texture_depth_2d;

// Lights sit exactly on pole tops and car faces; nudge them toward the camera
// so the occlusion test doesn't hide them behind their own fixture.
const DEPTH_NUDGE_M: f32 = 0.3;

fn hidden() -> vec4<f32> {
    return vec4<f32>(2.0, 2.0, 2.0, 1.0);  // degenerate quad off screen
}

fn glow_margin_px(glow_radius_px: f32) -> f32 {
    let margin = floor(glow_radius_px * frame.glow.w) + 2.0;
    return min(margin, floor(max(frame.viewport.x, frame.viewport.y) / 2.0));
}

@vertex
fn vs_sprite(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec4<f32>,
    @location(1) color_core: vec4<f32>,
    @location(2) params: vec4<f32>,
) -> SpriteOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var out: SpriteOut;
    out.color_core = color_core;
    out.params = params;

    let to_camera = frame.camera.xyz - position.xyz;
    let nudged = position.xyz + normalize(to_camera) * min(DEPTH_NUDGE_M, length(to_camera) * 0.5);
    let clip = frame.view_proj * vec4<f32>(nudged, 1.0);
    out.offset_px = vec2<f32>(0.0);
    if (clip.w <= 0.0) {
        out.clip = hidden();
        return out;
    }
    let ndc = clip.xyz / clip.w;
    // Like the CPU pass, lights whose center is off screen are skipped.
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) {
        out.clip = hidden();
        return out;
    }
    let center_px = vec2<i32>(
        i32((ndc.x + 1.0) * 0.5 * frame.viewport.x),
        i32((1.0 - ndc.y) * 0.5 * frame.viewport.y),
    );
    if (ndc.z > textureLoad(scene_depth, center_px, 0)) {
        out.clip = hidden();
        return out;
    }

    var extent = floor(color_core.w) + 2.0;
    if (params.y > 0.0) {
        extent = max(extent, glow_margin_px(params.y));
    }
    let offset = corners[vertex_index] * extent;
    let to_ndc = vec2<f32>(2.0 / frame.viewport.x, 2.0 / frame.viewport.y);
    out.clip = vec4<f32>(ndc.xy + offset * to_ndc, 0.0, 1.0);
    out.offset_px = offset;
    return out;
}

@fragment
fn fs_sprite(in: SpriteOut) -> @location(0) vec4<f32> {
    let d = length(in.offset_px);
    let core_radius = in.color_core.w;
    let core = clamp(core_radius - d + 0.5, 0.0, 1.0) * in.params.x;

    var glow = 0.0;
    let glow_radius = in.params.y;
    let margin = glow_margin_px(glow_radius);
    if (glow_radius > 0.0 && abs(in.offset_px.x) <= margin && abs(in.offset_px.y) <= margin) {
        let sigma = max(glow_radius / 2.0 * frame.glow.x, 0.5);
        let gaussian = exp(-d * d / (2.0 * sigma * sigma));
        let knee = max(glow_radius * 0.5, 1.0);
        let tail = 1.0 / (1.0 + pow(d / knee, frame.glow.y));
        let tail_strength = frame.glow.z;
        glow = in.params.z * ((1.0 - tail_strength) * gaussian + tail_strength * tail);
    }
    return vec4<f32>(in.color_core.rgb * (core + glow), 1.0);
}
