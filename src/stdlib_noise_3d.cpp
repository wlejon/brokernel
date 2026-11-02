#include "stdlib_noise_parts.h"

namespace brokernel {
namespace {

constexpr const char* kStdNoisePart4b = R"BK(
pub fn supersimplex4d_v3<T>(
    seed: i32,
    sm: bool,
    xp: i32, yp: i32, zp: i32, wp: i32,
    dx_b: T, dy_b: T, dz_b: T, dw_b: T,
    off_x: T, off_y: T, off_z: T, off_w: T,
    off_mul: T, coord_sum: T, stem: T, sub_axis: T
) -> T {
    let sgn: T = sm ? -1.0 : 1.0;
    let g: T = grad_dot_perlin4d<T>(
        hash_primes4d(seed, xp, yp, zp, wp),
        dx_b - off_x * sgn, dy_b - off_y * sgn, dz_b - off_z * sgn, dw_b - off_w * sgn);
    let fb: T = max(((off_mul * sgn) * coord_sum + stem) + sub_axis * sgn, 0.0);
    return ((fb * fb) * (fb * fb)) * g;
}

pub fn supersimplex4d<T>(p: vec4<T>, seed: i32 = 0) -> T {
    let k_skew: T = 0.30901699437494742410229341718282;
    let k_unskew: T = -0.13819660112501051517954131656344;
    let k_twice_unskew: T = -0.2;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var so2: i32 = 0;
    var so3: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    var z_skewed: T = 0.0;
    var w_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64) + (p.z as f64) + (p.w as f64);
        let s4: f64 = (0.30901699437494742410229341718282 as f32) as f64;
        let sx: f64 = (p.x as f64) + s4 * sum;
        let sy: f64 = (p.y as f64) + s4 * sum;
        let sz: f64 = (p.z as f64) + s4 * sum;
        let sw: f64 = (p.w as f64) + s4 * sum;
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        let cz: f64 = floor(sz);
        let cw: f64 = floor(sw);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        so2 = (cz as i64) as i32;
        so3 = (cw as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
        z_skewed = (sz - cz) as T;
        w_skewed = (sw - cw) as T;
    } else {
        let s: T = k_skew * (p.x + p.y + p.z + p.w);
        x_skewed = p.x + s;
        y_skewed = p.y + s;
        z_skewed = p.z + s;
        w_skewed = p.w + s;
    }
    var x_base: T = floor(x_skewed);
    var y_base: T = floor(y_skewed);
    var z_base: T = floor(z_skewed);
    var w_base: T = floor(w_skewed);
    var dx_skewed: T = x_skewed - x_base;
    var dy_skewed: T = y_skewed - y_base;
    var dz_skewed: T = z_skewed - z_base;
    var dw_skewed: T = w_skewed - w_base;

    let tu0: T = k_twice_unskew * (dx_skewed + dy_skewed + dz_skewed + dw_skewed);
    let xn0: T = dx_skewed + tu0;
    let yn0: T = dy_skewed + tu0;
    let zn0: T = dz_skewed + tu0;
    let wn0: T = dw_skewed + tu0;
    let xyzwn0: T = -tu0;

    var max_score: T = 0.6 - xyzwn0;
    var move_bits: i32 = (0.2 > max_score) ? -1 : 0;
    max_score = max(max_score, 0.2);
    if (-wn0) > max_score { move_bits = 7; max_score = -wn0; }
    if (-zn0) > max_score { move_bits = 11; max_score = -zn0; }
    if (-yn0) > max_score { move_bits = 13; max_score = -yn0; }
    if (-xn0) > max_score { move_bits = 14; max_score = -xn0; }
    max_score += xyzwn0 - 0.2;
    if xn0 > max_score { move_bits = 1; max_score = xn0; }
    if yn0 > max_score { move_bits = 2; max_score = yn0; }
    if zn0 > max_score { move_bits = 4; max_score = zn0; }
    if wn0 > max_score { move_bits = 8; max_score = wn0; }
    max_score += 0.2 - xn0;
    if yn0 > max_score { move_bits = 3; max_score = yn0; }
    if zn0 > max_score { move_bits = 5; max_score = zn0; }
    if wn0 > max_score { move_bits = 9; max_score = wn0; }
    max_score += xn0;
    if (yn0 + zn0) > max_score { move_bits = 6; max_score = yn0 + zn0; }
    max_score -= wn0;
    if yn0 > max_score { move_bits = 10; max_score = yn0; }
    if zn0 > max_score { move_bits = 12; max_score = zn0; }

    if (move_bits & 1) != 0 { x_base += 1.0; dx_skewed -= 1.0; }
    if (move_bits & 2) != 0 { y_base += 1.0; dy_skewed -= 1.0; }
    if (move_bits & 4) != 0 { z_base += 1.0; dz_skewed -= 1.0; }
    if (move_bits & 8) != 0 { w_base += 1.0; dw_skewed -= 1.0; }

    let x_p: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_p: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let z_p: i32 = ((z_base as i32) + so2) * PRIME_Z;
    let w_p: i32 = ((w_base as i32) + so3) * PRIME_W;

    let sk_sum: T = dx_skewed + dy_skewed + dz_skewed + dw_skewed;
    let tu: T = k_twice_unskew * sk_sum;
    let xn: T = dx_skewed + tu;
    let yn: T = dy_skewed + tu;
    let zn: T = dz_skewed + tu;
    let wn: T = dw_skewed + tu;
    let xyzwn: T = -tu;

    let unskew_d: T = k_unskew * sk_sum;
    let dx_b: T = dx_skewed + unskew_d;
    let dy_b: T = dy_skewed + unskew_d;
    let dz_b: T = dz_skewed + unskew_d;
    let dw_b: T = dw_skewed + unskew_d;
    let coord_sum: T = (1.0 + 4.0 * k_unskew) * sk_sum;

    let g0: T = grad_dot_perlin4d<T>(hash_primes4d(seed, x_p, y_p, z_p, w_p), dx_b, dy_b, dz_b, dw_b);
    let fb0: T = ((((0.8 - dx_b * dx_b) - dy_b * dy_b) - dz_b * dz_b) - dw_b * dw_b) * 0.5;
    let stem_a: T = fb0 - 0.4;
    let stem_b: T = fb0 - 0.6;
    var value: T = ((fb0 * fb0) * (fb0 * fb0)) * g0;

    let u1_0: T = k_unskew;
    let u1_1: T = k_unskew + 1.0;
    let u2_0: T = 2.0 * k_unskew;
    let u2_1: T = 2.0 * k_unskew + 1.0;
    let u3_0: T = 3.0 * k_unskew;
    let u3_1: T = 3.0 * k_unskew + 1.0;
    let u4_1: T = 4.0 * k_unskew + 1.0;

    let sm1: bool = xyzwn < 0.0;
    value += supersimplex4d_v3<T>(seed, sm1, x_p + (sm1 ? -PRIME_X : PRIME_X), y_p + (sm1 ? -PRIME_Y : PRIME_Y), z_p + (sm1 ? -PRIME_Z : PRIME_Z), w_p + (sm1 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u4_1, u4_1, u4_1, u4_1, u4_1, coord_sum, stem_a, 0.0);

    let sm2: bool = xyzwn < wn;
    value += supersimplex4d_v3<T>(seed, sm2, x_p + (sm2 ? -PRIME_X : PRIME_X), y_p + (sm2 ? -PRIME_Y : PRIME_Y), z_p + (sm2 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u3_1, u3_1, u3_1, u3_0, u3_1, coord_sum, stem_b, -dw_b);

    let sm3: bool = xyzwn < zn;
    value += supersimplex4d_v3<T>(seed, sm3, x_p + (sm3 ? -PRIME_X : PRIME_X), y_p + (sm3 ? -PRIME_Y : PRIME_Y), z_p, w_p + (sm3 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_1, u3_1, u3_0, u3_1, u3_1, coord_sum, stem_b, -dz_b);

    let sm4: bool = xyzwn < yn;
    value += supersimplex4d_v3<T>(seed, sm4, x_p + (sm4 ? -PRIME_X : PRIME_X), y_p, z_p + (sm4 ? -PRIME_Z : PRIME_Z), w_p + (sm4 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_1, u3_0, u3_1, u3_1, u3_1, coord_sum, stem_b, -dy_b);

    let sm5: bool = xyzwn < xn;
    value += supersimplex4d_v3<T>(seed, sm5, x_p, y_p + (sm5 ? -PRIME_Y : PRIME_Y), z_p + (sm5 ? -PRIME_Z : PRIME_Z), w_p + (sm5 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_0, u3_1, u3_1, u3_1, u3_1, coord_sum, stem_b, -dx_b);

    let sm6: bool = xn < 0.0;
    value += supersimplex4d_v3<T>(seed, sm6, x_p + (sm6 ? -PRIME_X : PRIME_X), y_p, z_p, w_p, dx_b, dy_b, dz_b, dw_b, u1_1, u1_0, u1_0, u1_0, u1_0, coord_sum, stem_a, dx_b);

    let sm7: bool = xn < (-yn);
    value += supersimplex4d_v3<T>(seed, sm7, x_p + (sm7 ? -PRIME_X : PRIME_X), y_p + (sm7 ? -PRIME_Y : PRIME_Y), z_p, w_p, dx_b, dy_b, dz_b, dw_b, u2_1, u2_1, u2_0, u2_0, u2_0, coord_sum, stem_b, dx_b + dy_b);

    let sm8: bool = xn < (-zn);
    value += supersimplex4d_v3<T>(seed, sm8, x_p + (sm8 ? -PRIME_X : PRIME_X), y_p, z_p + (sm8 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u2_1, u2_0, u2_1, u2_0, u2_0, coord_sum, stem_b, dx_b + dz_b);

    let sm9: bool = xn < (-wn);
    value += supersimplex4d_v3<T>(seed, sm9, x_p + (sm9 ? -PRIME_X : PRIME_X), y_p, z_p, w_p + (sm9 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_1, u2_0, u2_0, u2_1, u2_0, coord_sum, stem_b, dx_b + dw_b);

    let sm10: bool = yn < 0.0;
    value += supersimplex4d_v3<T>(seed, sm10, x_p, y_p + (sm10 ? -PRIME_Y : PRIME_Y), z_p, w_p, dx_b, dy_b, dz_b, dw_b, u1_0, u1_1, u1_0, u1_0, u1_0, coord_sum, stem_a, dy_b);

    let sm11: bool = yn < (-zn);
    value += supersimplex4d_v3<T>(seed, sm11, x_p, y_p + (sm11 ? -PRIME_Y : PRIME_Y), z_p + (sm11 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u2_0, u2_1, u2_1, u2_0, u2_0, coord_sum, stem_b, dy_b + dz_b);

    let sm12: bool = yn < (-wn);
    value += supersimplex4d_v3<T>(seed, sm12, x_p, y_p + (sm12 ? -PRIME_Y : PRIME_Y), z_p, w_p + (sm12 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_0, u2_1, u2_0, u2_1, u2_0, coord_sum, stem_b, dy_b + dw_b);

    let sm13: bool = zn < 0.0;
    value += supersimplex4d_v3<T>(seed, sm13, x_p, y_p, z_p + (sm13 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u1_0, u1_0, u1_1, u1_0, u1_0, coord_sum, stem_a, dz_b);

    let sm14: bool = zn < (-wn);
    value += supersimplex4d_v3<T>(seed, sm14, x_p, y_p, z_p + (sm14 ? -PRIME_Z : PRIME_Z), w_p + (sm14 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_0, u2_0, u2_1, u2_1, u2_0, coord_sum, stem_b, dz_b + dw_b);

    let sm15: bool = wn < 0.0;
    value += supersimplex4d_v3<T>(seed, sm15, x_p, y_p, z_p, w_p + (sm15 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u1_0, u1_0, u1_0, u1_1, u1_0, coord_sum, stem_a, dw_b);

    let k_bound: T = 0.011310013942420482635498046875;
    return scale_output<T>(value, -k_bound, k_bound);
}

pub fn supersimplex<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return supersimplex2d<T>(p, seed);
    } else if D == 3 {
        return supersimplex3d<T>(p, seed);
    } else {
        return supersimplex4d<T>(p, seed);
    }
}
)BK";

constexpr const char* kStdNoisePart5 = R"BK(
// BasicGenerators (from FastNoise2 BasicGenerators.inl)
pub fn constant_noise<T>(val: T) -> T {
    return val;
}

pub fn white_coord<T>(x: T, prime: i32) -> i32 {
    let xf: f32 = x as f32;
    let bits: i32 = bitcast<i32>(xf);
    return (bits ^ (bits >> 16)) * prime;
}

pub fn white2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let x0: i32 = white_coord<T>(p.x, PRIME_X);
    let y0: i32 = white_coord<T>(p.y, PRIME_Y);
    return scale_output<T>(get_value_coord2d<T>(seed, x0, y0), -2147483648.0, 2147483648.0);
}

pub fn white3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let x0: i32 = white_coord<T>(p.x, PRIME_X);
    let y0: i32 = white_coord<T>(p.y, PRIME_Y);
    let z0: i32 = white_coord<T>(p.z, PRIME_Z);
    return scale_output<T>(get_value_coord3d<T>(seed, x0, y0, z0), -2147483648.0, 2147483648.0);
}

pub fn white4d<T>(p: vec4<T>, seed: i32 = 0) -> T {
    let x0: i32 = white_coord<T>(p.x, PRIME_X);
    let y0: i32 = white_coord<T>(p.y, PRIME_Y);
    let z0: i32 = white_coord<T>(p.z, PRIME_Z);
    let w0: i32 = white_coord<T>(p.w, PRIME_W);
    return scale_output<T>(get_value_coord4d<T>(seed, x0, y0, z0, w0), -2147483648.0, 2147483648.0);
}

pub fn white<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 2 {
        return white2d<T>(p, seed);
    } else if D == 3 {
        return white3d<T>(p, seed);
    } else {
        return white4d<T>(p, seed);
    }
}

pub fn checkerboard_axis<T>(x: T) -> i32 {
    let xc: i32 = rebase_cell<T>(x);
    let xf: T = rebase_frac<T>(x);
    return (round(xf) as i32) + xc;
}

pub fn checkerboard2d<T>(p: vec2<T>) -> T {
    let v: i32 = checkerboard_axis<T>(p.x) ^ checkerboard_axis<T>(p.y);
    return ((v & 1) != 0) ? 1.0 : -1.0;
}

pub fn checkerboard3d<T>(p: vec3<T>) -> T {
    let v: i32 = checkerboard_axis<T>(p.x) ^ checkerboard_axis<T>(p.y) ^ checkerboard_axis<T>(p.z);
    return ((v & 1) != 0) ? 1.0 : -1.0;
}

pub fn checkerboard4d<T>(p: vec4<T>) -> T {
    let v: i32 = checkerboard_axis<T>(p.x) ^ checkerboard_axis<T>(p.y) ^ checkerboard_axis<T>(p.z) ^ checkerboard_axis<T>(p.w);
    return ((v & 1) != 0) ? 1.0 : -1.0;
}

pub fn checkerboard<T, const D: i32>(p: vec<T, D>) -> T {
    if D == 2 {
        return checkerboard2d<T>(p);
    } else if D == 3 {
        return checkerboard3d<T>(p);
    } else {
        return checkerboard4d<T>(p);
    }
}

pub fn sinewave_axis<T>(x: T) -> T {
    let tau: T = 6.283185307179586;
    if is_f64<T>() {
        let w: f64 = (x as f64) - floor((x as f64) / 6.283185307179586) * 6.283185307179586;
        return sin(w as T);
    } else {
        return sin(x);
    }
}

pub fn sinewave2d<T>(p: vec2<T>) -> T {
    return sinewave_axis<T>(p.x) * sinewave_axis<T>(p.y);
}

pub fn sinewave3d<T>(p: vec3<T>) -> T {
    return sinewave_axis<T>(p.x) * sinewave_axis<T>(p.y) * sinewave_axis<T>(p.z);
}

pub fn sinewave4d<T>(p: vec4<T>) -> T {
    return sinewave_axis<T>(p.x) * sinewave_axis<T>(p.y) * sinewave_axis<T>(p.z) * sinewave_axis<T>(p.w);
}

pub fn sinewave<T, const D: i32>(p: vec<T, D>) -> T {
    if D == 2 {
        return sinewave2d<T>(p);
    } else if D == 3 {
        return sinewave3d<T>(p);
    } else {
        return sinewave4d<T>(p);
    }
}

pub fn position_output<T, const D: i32>(p: vec<T, D>, mult: vec<T, D>, offset: vec<T, D>) -> T {
    return dot(p + offset, mult);
}

pub fn calc_distance2d<T>(dist_func: i32, dx: T, dy: T, minkowski_p: T = 1.5) -> T {
    if dist_func == 0 {
        return sqrt(dx * dx + dy * dy);
    } else if dist_func == 1 {
        return dx * dx + dy * dy;
    } else if dist_func == 2 {
        return abs(dx) + abs(dy);
    } else if dist_func == 3 {
        return abs(dx) + abs(dy) + (dx * dx + dy * dy);
    } else if dist_func == 4 {
        return max(abs(dx), abs(dy));
    } else {
        return pow(pow(abs(dx), minkowski_p) + pow(abs(dy), minkowski_p), 1.0 / minkowski_p);
    }
}

pub fn calc_distance3d<T>(dist_func: i32, dx: T, dy: T, dz: T, minkowski_p: T = 1.5) -> T {
    if dist_func == 0 {
        return sqrt(dx * dx + dy * dy + dz * dz);
    } else if dist_func == 1 {
        return dx * dx + dy * dy + dz * dz;
    } else if dist_func == 2 {
        return abs(dx) + abs(dy) + abs(dz);
    } else if dist_func == 3 {
        return (abs(dx) + abs(dy) + abs(dz)) + (dx * dx + dy * dy + dz * dz);
    } else if dist_func == 4 {
        return max(max(abs(dx), abs(dy)), abs(dz));
    } else {
        return pow(pow(abs(dx), minkowski_p) + pow(abs(dy), minkowski_p) + pow(abs(dz), minkowski_p), 1.0 / minkowski_p);
    }
}

pub fn calc_distance4d<T>(dist_func: i32, dx: T, dy: T, dz: T, dw: T, minkowski_p: T = 1.5) -> T {
    if dist_func == 0 {
        return sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
    } else if dist_func == 1 {
        return dx * dx + dy * dy + dz * dz + dw * dw;
    } else if dist_func == 2 {
        return abs(dx) + abs(dy) + abs(dz) + abs(dw);
    } else if dist_func == 3 {
        return (abs(dx) + abs(dy) + abs(dz) + abs(dw)) + (dx * dx + dy * dy + dz * dz + dw * dw);
    } else if dist_func == 4 {
        return max(max(abs(dx), abs(dy)), max(abs(dz), abs(dw)));
    } else {
        return pow(pow(abs(dx), minkowski_p) + pow(abs(dy), minkowski_p) + pow(abs(dz), minkowski_p) + pow(abs(dw), minkowski_p), 1.0 / minkowski_p);
    }
}
)BK";

constexpr const char* kStdNoisePart6 = R"BK(
// Fractals and Modifiers (from FastNoise2 Fractal.inl, Modifiers.inl, Blends.inl)
pub fn ping_pong_fn<T>(val: T, strength: T = 1.0) -> T {
    let t0: T = val * strength;
    let t: T = t0 - floor(t0 * 0.5) * 2.0;
    return (t < 1.0) ? t : (2.0 - t);
}

pub fn fbm_perlin<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = perlin<T, D>(freq_p, s);
    var sum: T = noise * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, (noise + 1.0) * 0.5, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = perlin<T, D>(freq_p, s);
        sum += noise * amp;
    }
    return sum;
}

pub fn fbm_simplex<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = simplex<T, D>(freq_p, s);
    var sum: T = noise * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, (noise + 1.0) * 0.5, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = simplex<T, D>(freq_p, s);
        sum += noise * amp;
    }
    return sum;
}

pub fn fbm_supersimplex<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = supersimplex<T, D>(freq_p, s);
    var sum: T = noise * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, (noise + 1.0) * 0.5, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = supersimplex<T, D>(freq_p, s);
        sum += noise * amp;
    }
    return sum;
}

pub fn fbm_value<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = value<T, D>(freq_p, s);
    var sum: T = noise * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, (noise + 1.0) * 0.5, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = value<T, D>(freq_p, s);
        sum += noise * amp;
    }
    return sum;
}

pub fn fbm<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    seed: i32 = 0
) -> T {
    return fbm_perlin<T, D>(p, octaves, lacunarity, gain, 0.0, seed);
}

pub fn ridged_perlin<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = abs(perlin<T, D>(freq_p, s));
    var sum: T = (noise * -2.0 + 1.0) * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, 1.0 - noise, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = abs(perlin<T, D>(freq_p, s));
        sum += (noise * -2.0 + 1.0) * amp;
    }
    return sum;
}

pub fn ridged_simplex<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = abs(simplex<T, D>(freq_p, s));
    var sum: T = (noise * -2.0 + 1.0) * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, 1.0 - noise, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = abs(simplex<T, D>(freq_p, s));
        sum += (noise * -2.0 + 1.0) * amp;
    }
    return sum;
}

pub fn ridged_supersimplex<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = abs(supersimplex<T, D>(freq_p, s));
    var sum: T = (noise * -2.0 + 1.0) * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, 1.0 - noise, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = abs(supersimplex<T, D>(freq_p, s));
        sum += (noise * -2.0 + 1.0) * amp;
    }
    return sum;
}

pub fn ridged_value<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = abs(value<T, D>(freq_p, s));
    var sum: T = (noise * -2.0 + 1.0) * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, 1.0 - noise, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = abs(value<T, D>(freq_p, s));
        sum += (noise * -2.0 + 1.0) * amp;
    }
    return sum;
}

pub fn ridged<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    seed: i32 = 0
) -> T {
    return ridged_perlin<T, D>(p, octaves, lacunarity, gain, 0.0, seed);
}

pub fn pingpong_perlin<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    ping_pong_strength: T = 2.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = ping_pong_fn<T>(perlin<T, D>(freq_p, s), ping_pong_strength);
    var sum: T = noise * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, (noise + 1.0) * 0.5, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = ping_pong_fn<T>(perlin<T, D>(freq_p, s), ping_pong_strength);
        sum += noise * amp;
    }
    return sum;
}

pub fn pingpong_simplex<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    ping_pong_strength: T = 2.0,
    seed: i32 = 0
) -> T {
    var amp: T = 1.0;
    var freq_p: vec<T, D> = p;
    var s: i32 = seed;
    var noise: T = ping_pong_fn<T>(simplex<T, D>(freq_p, s), ping_pong_strength);
    var sum: T = noise * amp;
    for (var oct: i32 = 1; oct < octaves; oct += 1) {
        s += 1;
        amp *= fn_lerp<T>(1.0, (noise + 1.0) * 0.5, weighted_strength);
        amp *= gain;
        freq_p = freq_p * lacunarity;
        noise = ping_pong_fn<T>(simplex<T, D>(freq_p, s), ping_pong_strength);
        sum += noise * amp;
    }
    return sum;
}

// Modifiers and Blends
pub fn signed_sqrt<T>(x: T) -> T {
    return (x < 0.0) ? -sqrt(-x) : sqrt(x);
}

pub fn min_smooth<T>(a: T, b: T, smoothness: T) -> T {
    let sm: T = max(abs(smoothness), 1.1754943508e-38);
    let h: T = max(sm - abs(a - b), 0.0) / sm;
    return min(a, b) - 0.16666667163372039794921875 * h * h * h * sm;
}

pub fn max_smooth<T>(a: T, b: T, smoothness: T) -> T {
    return -min_smooth<T>(-a, -b, smoothness);
}

pub fn pow_int<T>(val: T, p: i32) -> T {
    var res: T = val * val;
    for (var i: i32 = 2; i < p; i += 1) {
        res *= val;
    }
    return res;
}

pub fn pow_float<T>(val: T, p: T) -> T {
    return pow(abs(val), p);
}

pub fn terrace<T>(val: T, step_count: T, smoothness: T = 0.0) -> T {
    let s: T = val * step_count;
    var r: T = round(s);
    if smoothness != 0.0 {
        var diff: T = r - s;
        let diff_sign: bool = diff < 0.0;
        diff = 0.5 - abs(diff);
        diff *= (1.0 + 1.0 / smoothness);
        diff = min(diff, 0.5);
        diff = diff_sign ? (0.5 - diff) : (diff - 0.5);
        r += diff;
    }
    return r * (1.0 / step_count);
}
)BK";

constexpr const char* kStdNoisePart7 = R"BK(
// Cellular Noise (from FastNoise2 Cellular.inl)
pub fn cellular_return<T>(ret_type: i32, d0: T, d1: T, max_dist: T) -> T {
    if ret_type == 0 {
        return scale_output<T>(d0, 0.0, max_dist);
    } else if ret_type == 1 {
        return scale_output<T>(d0 + d1, 0.0, max_dist * 2.0);
    } else if ret_type == 2 {
        return scale_output<T>(abs(d0 - d1), 0.0, max_dist);
    } else if ret_type == 3 {
        return scale_output<T>(d0 * d1, 0.0, max_dist * max_dist);
    } else {
        return scale_output<T>(d0 / max(d1, 0.000001), 0.0, max_dist);
    }
}

pub fn cellular_value2d<T>(p: vec2<T>, seed: i32 = 0, dist_func: i32 = 1, jitter_mod: T = 1.0) -> T {
    let k_jitter: T = 0.437015950679779052734375 * jitter_mod;
    let x_off: i32 = rebase_cell<T>(p.x);
    let y_off: i32 = rebase_cell<T>(p.y);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let xc_base: i32 = ((round(x) as i32) - 1 + x_off) * PRIME_X;
    let yc_base: i32 = ((round(y) as i32) - 1 + y_off) * PRIME_Y;
    let xcf_base: T = (round(x) - 1.0) - x;
    let ycf_base: T = (round(y) - 1.0) - y;

    var best_dist: T = 1.0e10;
    var best_hash: i32 = 0;
    var xc: i32 = xc_base;
    var xcf: T = xcf_base;
    @unroll for ix in 0..3 {
        var yc: i32 = yc_base;
        var ycf: T = ycf_base;
        @unroll for iy in 0..3 {
            let h: i32 = hash_primes_hb2d(seed, xc, yc);
            let xd: T = ((h & 0x7ff) as T) - 1023.5;
            let yd: T = (((h >> 21) & 0x7ff) as T) - 1023.5;
            let inv_mag: T = k_jitter / sqrt(xd * xd + yd * yd);
            let px_d: T = xd * inv_mag + xcf;
            let py_d: T = yd * inv_mag + ycf;
            let d: T = calc_distance2d<T>((dist_func == 0) ? 1 : dist_func, px_d, py_d, 1.5);
            if d < best_dist {
                best_dist = d;
                best_hash = h;
            }
            yc += PRIME_Y;
            ycf += 1.0;
        }
        xc += PRIME_X;
        xcf += 1.0;
    }
    return scale_output<T>(best_hash as T, -2147483648.0, 2147483648.0);
}

pub fn cellular_value3d<T>(p: vec3<T>, seed: i32 = 0, dist_func: i32 = 1, jitter_mod: T = 1.0) -> T {
    let k_jitter: T = 0.3961435258388519287109375 * jitter_mod;
    let x_off: i32 = rebase_cell<T>(p.x);
    let y_off: i32 = rebase_cell<T>(p.y);
    let z_off: i32 = rebase_cell<T>(p.z);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let xc_base: i32 = ((round(x) as i32) - 1 + x_off) * PRIME_X;
    let yc_base: i32 = ((round(y) as i32) - 1 + y_off) * PRIME_Y;
    let zc_base: i32 = ((round(z) as i32) - 1 + z_off) * PRIME_Z;
    let xcf_base: T = (round(x) - 1.0) - x;
    let ycf_base: T = (round(y) - 1.0) - y;
    let zcf_base: T = (round(z) - 1.0) - z;

    var best_dist: T = 1.0e10;
    var best_hash: i32 = 0;
    var xc: i32 = xc_base;
    var xcf: T = xcf_base;
    @unroll for ix in 0..3 {
        var yc: i32 = yc_base;
        var ycf: T = ycf_base;
        @unroll for iy in 0..3 {
            var zc: i32 = zc_base;
            var zcf: T = zcf_base;
            @unroll for iz in 0..3 {
                let h: i32 = hash_primes_hb3d(seed, xc, yc, zc);
                let xd: T = ((h & 0x3ff) as T) - 511.5;
                let yd: T = (((h >> 11) & 0x3ff) as T) - 511.5;
                let zd: T = (((h >> 22) & 0x3ff) as T) - 511.5;
                let inv_mag: T = k_jitter / sqrt(xd * xd + yd * yd + zd * zd);
                let px_d: T = xd * inv_mag + xcf;
                let py_d: T = yd * inv_mag + ycf;
                let pz_d: T = zd * inv_mag + zcf;
                let d: T = calc_distance3d<T>((dist_func == 0) ? 1 : dist_func, px_d, py_d, pz_d, 1.5);
                if d < best_dist {
                    best_dist = d;
                    best_hash = h;
                }
                zc += PRIME_Z;
                zcf += 1.0;
            }
            yc += PRIME_Y;
            ycf += 1.0;
        }
        xc += PRIME_X;
        xcf += 1.0;
    }
    return scale_output<T>(best_hash as T, -2147483648.0, 2147483648.0);
}

pub fn cellular_value4d<T>(p: vec4<T>, seed: i32 = 0, dist_func: i32 = 1, jitter_mod: T = 1.0) -> T {
    let k_jitter: T = 0.3660254180431365966796875 * jitter_mod;
    let x_off: i32 = rebase_cell<T>(p.x);
    let y_off: i32 = rebase_cell<T>(p.y);
    let z_off: i32 = rebase_cell<T>(p.z);
    let w_off: i32 = rebase_cell<T>(p.w);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let w: T = rebase_frac<T>(p.w);
    let xc_base: i32 = ((round(x) as i32) - 1 + x_off) * PRIME_X;
    let yc_base: i32 = ((round(y) as i32) - 1 + y_off) * PRIME_Y;
    let zc_base: i32 = ((round(z) as i32) - 1 + z_off) * PRIME_Z;
    let wc_base: i32 = ((round(w) as i32) - 1 + w_off) * PRIME_W;
    let xcf_base: T = (round(x) - 1.0) - x;
    let ycf_base: T = (round(y) - 1.0) - y;
    let zcf_base: T = (round(z) - 1.0) - z;
    let wcf_base: T = (round(w) - 1.0) - w;

    var best_dist: T = 1.0e10;
    var best_hash: i32 = 0;
    for (var ix: i32 = 0; ix < 3; ix += 1) {
        let xc: i32 = xc_base + ix * PRIME_X;
        let xcf: T = xcf_base + (ix as T);
        for (var iy: i32 = 0; iy < 3; iy += 1) {
            let yc: i32 = yc_base + iy * PRIME_Y;
            let ycf: T = ycf_base + (iy as T);
            for (var iz: i32 = 0; iz < 3; iz += 1) {
                let zc: i32 = zc_base + iz * PRIME_Z;
                let zcf: T = zcf_base + (iz as T);
                for (var iw: i32 = 0; iw < 3; iw += 1) {
                    let wc: i32 = wc_base + iw * PRIME_W;
                    let wcf: T = wcf_base + (iw as T);
                    let h: i32 = hash_primes_hb4d(seed, xc, yc, zc, wc);
                    let xd: T = ((h & 0xff) as T) - 127.5;
                    let yd: T = (((h >> 8) & 0xff) as T) - 127.5;
                    let zd: T = (((h >> 16) & 0xff) as T) - 127.5;
                    let wd: T = (((h >> 24) & 0xff) as T) - 127.5;
                    let inv_mag: T = k_jitter / sqrt(xd * xd + yd * yd + zd * zd + wd * wd);
                    let px_d: T = xd * inv_mag + xcf;
                    let py_d: T = yd * inv_mag + ycf;
                    let pz_d: T = zd * inv_mag + zcf;
                    let pw_d: T = wd * inv_mag + wcf;
                    let d: T = calc_distance4d<T>((dist_func == 0) ? 1 : dist_func, px_d, py_d, pz_d, pw_d, 1.5);
                    if d < best_dist {
                        best_dist = d;
                        best_hash = h;
                    }
                }
            }
        }
    }
    return scale_output<T>(best_hash as T, -2147483648.0, 2147483648.0);
}

pub fn cellular_value<T, const D: i32>(p: vec<T, D>, seed: i32 = 0, dist_func: i32 = 1, jitter_mod: T = 1.0) -> T {
    if D == 2 {
        return cellular_value2d<T>(p, seed, dist_func, jitter_mod);
    } else if D == 3 {
        return cellular_value3d<T>(p, seed, dist_func, jitter_mod);
    } else {
        return cellular_value4d<T>(p, seed, dist_func, jitter_mod);
    }
}
)BK";

} // namespace

void append_std_noise_3d(std::string& out) {
    out += kStdNoisePart4b;
    out += kStdNoisePart5;
    out += kStdNoisePart6;
    out += kStdNoisePart7;
}

} // namespace brokernel
