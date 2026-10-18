// Opaque surfaces (ground, road, building, boxes) and emissive windows.
// Ground-flagged surfaces receive the ground light pools, as in the 2D render;
// walls and roofs do not.

struct GeometryOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) ground_flag: f32,
};

@vertex
fn vs_geometry(
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) ground_flag: f32,
) -> GeometryOut {
    var out: GeometryOut;
    out.clip = frame.view_proj * vec4<f32>(position, 1.0);
    out.world = position;
    out.color = color;
    out.ground_flag = ground_flag;
    return out;
}

@fragment
fn fs_opaque(in: GeometryOut) -> @location(0) vec4<f32> {
    var color = in.color;
    if (in.ground_flag > 0.5) {
        color += ground_illumination(in.world.xy);
    }
    return vec4<f32>(color, 1.0);
}

@fragment
fn fs_window(in: GeometryOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color * frame.knee.w, 1.0);
}
