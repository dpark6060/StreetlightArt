// Final grade: vignette, exposure, tone map, sRGB encode. Port of
// rendering/compositor.py (_build_vignette_scalar, _apply_color_grade) and
// color_space.tonemap_aces / linear_to_srgb.

@group(0) @binding(1) var hdr: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

fn vignette(pixel: vec2<f32>) -> f32 {
    let center = frame.viewport.xy / 2.0;
    let half_diag = length(center);
    let inner = frame.grade.z * half_diag;
    let fade_range = max(half_diag - inner, 1.0);
    let t = clamp((length(pixel - center) - inner) / fade_range, 0.0, 1.0);
    return 1.0 - frame.grade.y * t;
}

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let lin = clamp(linear, vec3<f32>(0.0), vec3<f32>(1.0));
    let low = lin * 12.92;
    let high = 1.055 * pow(lin, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, lin <= vec3<f32>(0.0031308));
}

@fragment
fn fs_composite(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    // The CPU vignette measures from integer pixel indices, not pixel centers.
    let pixel = floor(frag.xy);
    let linear = textureLoad(hdr, vec2<i32>(pixel), 0).rgb * vignette(pixel);
    let scaled = linear * frame.grade.x;
    var mapped = clamp(scaled, vec3<f32>(0.0), vec3<f32>(1.0));
    if (frame.grade2.x > 0.5) {
        mapped = tonemap_aces(scaled);
    }
    if (frame.grade.w > 0.5) {
        mapped = linear_to_srgb(mapped);
    }
    return vec4<f32>(mapped, 1.0);
}
