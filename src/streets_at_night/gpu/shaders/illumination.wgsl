// Ground light pools. Line-for-line ports of rendering/illumination_pass.py:
// _physical_falloff, _cone_falloff, _directional_cone_falloff, _apply_soft_knee.
// Peak radiance (extinction, 1/h^2) is computed on the CPU by compute_pool_peak.

struct PoolLight {
    pos_kind: vec4<f32>,    // xyz = light position (m), w = model (1 physical, 2 cone, 3 directional)
    color_peak: vec4<f32>,  // rgb = linear color, w = peak radiance
    a: vec4<f32>,           // physical: radius, bulb_fraction, reflector_falloff, height
                            // cone: radius, inner_fraction, falloff
                            // directional: dir_x, dir_y, length, half_angle (rad)
    b: vec4<f32>,           // directional: falloff, inner_fraction
};

@group(0) @binding(1) var<storage, read> pools: array<PoolLight>;

const CUTOFF_EDGE_SOFTNESS: f32 = 0.15;
const CONE_EDGE_SOFTNESS: f32 = 0.3;

fn smoothstep_ramp(x: f32, edge0: f32, edge1: f32) -> f32 {
    let t = clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn physical_falloff(d: f32, height: f32, radius: f32, bulb_fraction: f32, reflector_falloff: f32) -> f32 {
    let r = d / height;
    let inverse_square = 1.0 / pow(1.0 + r * r, 1.5);
    let bulb_radius = clamp(bulb_fraction, 0.0, 1.0) * radius;
    let fade_start = min(bulb_radius, (1.0 - CUTOFF_EDGE_SOFTNESS) * radius);
    let fade = pow(1.0 - smoothstep_ramp(d, fade_start, radius), max(reflector_falloff, 1e-6));
    return inverse_square * fade;
}

fn cone_falloff(d: f32, radius: f32, inner_fraction: f32, exponent: f32) -> f32 {
    let t = clamp(d / radius, 0.0, 1.0);
    let denom = max(1.0 - inner_fraction, 1e-6);
    var fade_t = clamp((t - inner_fraction) / denom, 0.0, 1.0);
    fade_t = fade_t * fade_t * (3.0 - 2.0 * fade_t);
    return pow(1.0 - fade_t, exponent);
}

fn directional_falloff(dx: f32, dy: f32, light: PoolLight) -> f32 {
    let ux = light.a.x;
    let uy = light.a.y;
    let along = dx * ux + dy * uy;
    if (along <= 0.0) {
        return 0.0;
    }
    let lateral = -dx * uy + dy * ux;
    let axial = cone_falloff(along, light.a.z, light.b.y, light.b.x);
    let half_angle = light.a.w;
    let off_axis = atan2(abs(lateral), max(along, 1e-6));
    let edge_start = half_angle * (1.0 - CONE_EDGE_SOFTNESS);
    var edge = clamp((half_angle - off_axis) / max(half_angle - edge_start, 1e-6), 0.0, 1.0);
    edge = edge * edge * (3.0 - 2.0 * edge);
    return axial * edge;
}

fn pool_falloff(light: PoolLight, ground: vec2<f32>) -> f32 {
    let dx = ground.x - light.pos_kind.x;
    let dy = ground.y - light.pos_kind.y;
    let d = sqrt(dx * dx + dy * dy);
    let kind = light.pos_kind.w;
    if (kind == 1.0) {
        if (d > light.a.x) {
            return 0.0;
        }
        return physical_falloff(d, light.a.w, light.a.x, light.a.y, light.a.z);
    }
    if (kind == 2.0) {
        if (d > light.a.x) {
            return 0.0;
        }
        return cone_falloff(d, light.a.x, light.a.y, light.a.z);
    }
    if (kind == 3.0) {
        if (d > light.a.z) {
            return 0.0;
        }
        return directional_falloff(dx, dy, light);
    }
    return 0.0;
}

fn soft_knee(v: vec3<f32>) -> vec3<f32> {
    let knee_start = frame.knee.x;
    let knee_max = frame.knee.y;
    if (knee_max <= knee_start) {
        return v;
    }
    let span = knee_max - knee_start;
    let over = max(v - vec3<f32>(knee_start), vec3<f32>(0.0));
    let compressed = vec3<f32>(knee_start) + span * over / (over + vec3<f32>(span));
    return select(v, compressed, v > vec3<f32>(knee_start));
}

fn ground_illumination(ground: vec2<f32>) -> vec3<f32> {
    var total = vec3<f32>(0.0);
    let count = u32(frame.knee.z);
    for (var i = 0u; i < count; i = i + 1u) {
        let light = pools[i];
        total += light.color_peak.rgb * (light.color_peak.w * pool_falloff(light, ground));
    }
    return soft_knee(total);
}
