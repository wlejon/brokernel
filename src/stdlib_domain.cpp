#include "brokernel/module_loader.h"

namespace brokernel {

namespace {

constexpr const char* kStdDspSource = R"BK(
module dsp;

const DSP_TAU: f32 = 6.283185307179586;

pub struct BiquadCoeffs {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

pub struct BiquadResult {
    y: f32,
    z1: f32,
    z2: f32,
}

pub fn db_to_lin(db: f32) -> f32 {
    return pow(10.0, db * 0.05);
}

pub fn lin_to_db(lin: f32) -> f32 {
    return 20.0 * log(max(abs(lin), 1.0e-7)) / 2.302585092994046;
}

pub fn one_pole(prev: f32, target: f32, coeff: f32) -> f32 {
    return prev + (target - prev) * clamp(coeff, 0.0, 1.0);
}

pub fn soft_clip(x: f32, drive: f32 = 1.0) -> f32 {
    let d: f32 = x * drive;
    return d / (1.0 + abs(d));
}

pub fn tanh_sat(x: f32, drive: f32 = 1.0) -> f32 {
    return tanh(x * drive);
}

pub fn phasor_step(phase: f32, freq_hz: f32, sample_rate: f32) -> f32 {
    let next: f32 = phase + freq_hz / max(sample_rate, 1.0);
    return next - floor(next);
}

pub fn biquad_lowpass(cutoff_hz: f32, q: f32, sample_rate: f32) -> BiquadCoeffs {
    let sr: f32 = max(sample_rate, 1.0);
    let w0: f32 = DSP_TAU * clamp(cutoff_hz, 1.0, sr * 0.49) / sr;
    let cw: f32 = cos(w0);
    let alpha: f32 = sin(w0) / (2.0 * max(q, 0.05));
    let a0_inv: f32 = 1.0 / (1.0 + alpha);
    let one_m: f32 = (1.0 - cw) * 0.5;
    return BiquadCoeffs(
        one_m * a0_inv,
        (1.0 - cw) * a0_inv,
        one_m * a0_inv,
        (-2.0 * cw) * a0_inv,
        (1.0 - alpha) * a0_inv
    );
}

pub fn biquad_step(x: f32, z1: f32, z2: f32, c: BiquadCoeffs) -> BiquadResult {
    let y: f32 = c.b0 * x + z1;
    let nz1: f32 = c.b1 * x - c.a1 * y + z2;
    let nz2: f32 = c.b2 * x - c.a2 * y;
    return BiquadResult(y, nz1, nz2);
}

pub fn adsr_envelope(t: f32, attack: f32, decay: f32, sustain: f32, release: f32, gate_off_time: f32) -> f32 {
    if t < 0.0 {
        return 0.0;
    }
    let a: f32 = max(attack, 1.0e-4);
    let d: f32 = max(decay, 1.0e-4);
    let r: f32 = max(release, 1.0e-4);
    let sus: f32 = clamp(sustain, 0.0, 1.0);
    if t >= gate_off_time {
        let on_val: f32 = (gate_off_time < a)
            ? (gate_off_time / a)
            : ((gate_off_time < a + d)
                ? mix(1.0, sus, (gate_off_time - a) / d)
                : sus);
        return max(0.0, on_val * (1.0 - (t - gate_off_time) / r));
    }
    if t < a {
        return t / a;
    }
    if t < a + d {
        return mix(1.0, sus, (t - a) / d);
    }
    return sus;
}
)BK";

constexpr const char* kStdImageSource = R"BK(
module image;

pub fn srgb_to_linear(c: f32) -> f32 {
    let s: f32 = max(c, 0.0);
    return (s <= 0.04045) ? (s / 12.92) : pow((s + 0.055) / 1.055, 2.4);
}

pub fn linear_to_srgb(c: f32) -> f32 {
    let s: f32 = max(c, 0.0);
    return (s <= 0.0031308) ? (s * 12.92) : (1.055 * pow(s, 1.0 / 2.4) - 0.055);
}

pub fn srgb_to_linear3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}

pub fn linear_to_srgb3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

pub fn luma709(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

pub fn premultiply(rgba: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(rgba.x * rgba.w, rgba.y * rgba.w, rgba.z * rgba.w, rgba.w);
}

pub fn unpremultiply(rgba: vec4<f32>) -> vec4<f32> {
    if rgba.w <= 1.0e-6 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let inv_a: f32 = 1.0 / rgba.w;
    return vec4<f32>(rgba.x * inv_a, rgba.y * inv_a, rgba.z * inv_a, rgba.w);
}

pub fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let v: vec3<f32> = max(x, vec3<f32>(0.0, 0.0, 0.0));
    let num: vec3<f32> = v * (2.51 * v + vec3<f32>(0.03, 0.03, 0.03));
    let den: vec3<f32> = v * (2.43 * v + vec3<f32>(0.59, 0.59, 0.59)) + vec3<f32>(0.14, 0.14, 0.14);
    return clamp(num / den, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 1.0));
}

pub fn box_blur_3x3(src: stencil<f32, 2, 1>) -> f32 {
    var acc: f32 = 0.0;
    acc += src[-1, -1] + src[0, -1] + src[1, -1];
    acc += src[-1, 0] + src[0, 0] + src[1, 0];
    acc += src[-1, 1] + src[0, 1] + src[1, 1];
    return acc * (1.0 / 9.0);
}

pub fn sobel_mag_3x3(src: stencil<f32, 2, 1>) -> f32 {
    let tl: f32 = src[-1, -1];
    let tc: f32 = src[0, -1];
    let tr: f32 = src[1, -1];
    let ml: f32 = src[-1, 0];
    let mr: f32 = src[1, 0];
    let bl: f32 = src[-1, 1];
    let bc: f32 = src[0, 1];
    let br: f32 = src[1, 1];
    let gx: f32 = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    let gy: f32 = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    return sqrt(gx * gx + gy * gy);
}
)BK";

constexpr const char* kStdSimSource = R"BK(
module sim;

pub struct BodyState {
    pos: vec3<f32>,
    vel: vec3<f32>,
}

pub fn body_state(pos: vec3<f32>, vel: vec3<f32>) -> BodyState {
    return BodyState(pos, vel);
}

pub fn symplectic_euler_step(s: BodyState, accel: vec3<f32>, dt: f32, damping: f32 = 0.0) -> BodyState {
    let damp: f32 = max(0.0, 1.0 - damping * dt);
    let nv: vec3<f32> = (s.vel + accel * dt) * damp;
    let np: vec3<f32> = s.pos + nv * dt;
    return BodyState(np, nv);
}

pub fn verlet_step(pos: vec3<f32>, prev_pos: vec3<f32>, accel: vec3<f32>, dt: f32, damping: f32 = 0.0) -> vec3<f32> {
    let damp: f32 = max(0.0, 1.0 - damping);
    return pos + (pos - prev_pos) * damp + accel * (dt * dt);
}

pub fn spring_force(delta: vec3<f32>, rest_len: f32, stiffness: f32) -> vec3<f32> {
    let dist: f32 = length(delta);
    if dist <= 1.0e-6 {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let dir: vec3<f32> = delta / dist;
    return dir * ((dist - rest_len) * (-stiffness));
}

pub fn sdf_circle(p: vec2<f32>, center: vec2<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

pub fn sdf_aabb(p: vec2<f32>, center: vec2<f32>, half_extents: vec2<f32>) -> f32 {
    let q: vec2<f32> = abs(p - center) - half_extents;
    let outside: f32 = length(max(q, vec2<f32>(0.0, 0.0)));
    let inside: f32 = min(max(q.x, q.y), 0.0);
    return outside + inside;
}

pub fn collide_circle_boundary(s: BodyState, center: vec2<f32>, radius: f32, restitution: f32 = 0.5) -> BodyState {
    let p2: vec2<f32> = vec2<f32>(s.pos.x, s.pos.y) - center;
    let dist: f32 = length(p2);
    if dist <= radius || dist <= 1.0e-6 {
        return s;
    }
    let n: vec2<f32> = p2 / dist;
    let clamped_xy: vec2<f32> = center + n * radius;
    let vn: f32 = s.vel.x * n.x + s.vel.y * n.y;
    var nv: vec3<f32> = s.vel;
    if vn > 0.0 {
        let impulse: f32 = (1.0 + clamp(restitution, 0.0, 1.0)) * vn;
        nv = vec3<f32>(s.vel.x - n.x * impulse, s.vel.y - n.y * impulse, s.vel.z);
    }
    return BodyState(vec3<f32>(clamped_xy.x, clamped_xy.y, s.pos.z), nv);
}
)BK";

} // namespace

void ModuleRegistry::register_domain_builtin_modules() {
    register_module("dsp", kStdDspSource, "<builtin:dsp.bk>");
    aliases_["std::dsp"] = "dsp";

    register_module("image", kStdImageSource, "<builtin:image.bk>");
    aliases_["std::image"] = "image";

    register_module("sim", kStdSimSource, "<builtin:sim.bk>");
    aliases_["std::sim"] = "sim";
}

} // namespace brokernel
