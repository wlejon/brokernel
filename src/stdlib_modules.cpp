#include "brokernel/module_loader.h"

#include <fstream>
#include <sstream>

namespace brokernel {

namespace {

constexpr const char* kStdMathSource = R"BK(
module math;

pub const PI: f64 = 3.14159265358979323846;
pub const TAU: f64 = 6.28318530717958647692;
pub const HALF_PI: f64 = 1.57079632679489661923;
pub const E: f64 = 2.71828182845904523536;
pub const SQRT2: f64 = 1.41421356237309504880;
pub const LN2: f64 = 0.69314718055994530942;
pub const LOG2E: f64 = 1.44269504088896340736;

pub fn square<T>(x: T) -> T {
    return x * x;
}

pub fn cube<T>(x: T) -> T {
    return x * x * x;
}

pub fn saturate<T>(x: T) -> T {
    return clamp(x, 0.0, 1.0);
}

pub fn lerp<T>(a: T, b: T, t: T) -> T {
    return a + (b - a) * t;
}

pub fn smootherstep<T>(edge0: T, edge1: T, x: T) -> T {
    let t: T = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

pub fn remap<T>(x: T, in_lo: T, in_hi: T, out_lo: T, out_hi: T) -> T {
    let t: T = (x - in_lo) / (in_hi - in_lo);
    return out_lo + (out_hi - out_lo) * t;
}

pub fn radians<T>(deg: T) -> T {
    return deg * 0.01745329251994329577;
}

pub fn degrees<T>(rad: T) -> T {
    return rad * 57.2957795130823208768;
}
)BK";

constexpr const char* kStdNoisePart1 = R"BK(
module noise;

// FastNoise2 Primes and Hash Multiplier (from FastNoise2 Utils.inl)
pub const PRIME_X: i32 = -141048377;
pub const PRIME_Y: i32 = 1812335753;
pub const PRIME_Z: i32 = 1180684367;
pub const PRIME_W: i32 = -140107143;
pub const HASH_A: i32 = -1210014219;

pub const ROOT2: f64 = 1.41421353816986083984375;
pub const ROOT3: f64 = 1.7320508075688772935274463415059;
pub const ROOT5: f64 = 2.2360679774997896964091736687313;
pub const SKEW2: f64 = 0.3660254180431365966796875;
pub const UNSKEW2: f64 = -0.211324870586395263671875;
pub const SKEW3: f64 = 0.3333333432674407958984375;
pub const REFLECT_UNSKEW3: f64 = -0.5;
pub const TWICE_UNSKEW3: f64 = -0.25;
pub const SKEW4: f64 = 0.30901699437494742410229341718282;
pub const UNSKEW4: f64 = -0.13819660112501051517954131656344;
pub const TWICE_UNSKEW4: f64 = -0.2;

// FastNoise2's Lerp, a + t * (b - a).
pub fn fn_lerp<T>(a: T, b: T, t: T) -> T {
    return a + t * (b - a);
}

pub fn interp_hermite<T>(t: T) -> T {
    return t * t * (3.0 - 2.0 * t);
}

pub fn interp_quintic<T>(t: T) -> T {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

pub fn scale_output<T>(val: T, min_b: T, max_b: T) -> T {
    return (1.0 / (max_b - min_b)) * 2.0 * (val - min_b) - 1.0;
}

pub fn hash_primes2d(seed: i32, x: i32, y: i32) -> i32 {
    let h: i32 = (seed ^ x ^ y) * HASH_A;
    return (h >> 15) ^ h;
}

pub fn hash_primes3d(seed: i32, x: i32, y: i32, z: i32) -> i32 {
    let h: i32 = (seed ^ x ^ y ^ z) * HASH_A;
    return (h >> 15) ^ h;
}

pub fn hash_primes4d(seed: i32, x: i32, y: i32, z: i32, w: i32) -> i32 {
    let h: i32 = (seed ^ x ^ y ^ z ^ w) * HASH_A;
    return (h >> 15) ^ h;
}

pub fn hash_primes_hb2d(seed: i32, x: i32, y: i32) -> i32 {
    return (seed ^ x ^ y) * HASH_A;
}

pub fn hash_primes_hb3d(seed: i32, x: i32, y: i32, z: i32) -> i32 {
    return (seed ^ x ^ y ^ z) * HASH_A;
}

pub fn hash_primes_hb4d(seed: i32, x: i32, y: i32, z: i32, w: i32) -> i32 {
    return (seed ^ x ^ y ^ z ^ w) * HASH_A;
}

pub fn get_value_coord2d<T>(seed: i32, x: i32, y: i32) -> T {
    let h: i32 = seed ^ x ^ y;
    let h2: i32 = h * (h * HASH_A);
    return h2 as T;
}

pub fn get_value_coord3d<T>(seed: i32, x: i32, y: i32, z: i32) -> T {
    let h: i32 = seed ^ x ^ y ^ z;
    let h2: i32 = h * (h * HASH_A);
    return h2 as T;
}

pub fn get_value_coord4d<T>(seed: i32, x: i32, y: i32, z: i32, w: i32) -> T {
    let h: i32 = seed ^ x ^ y ^ z ^ w;
    let h2: i32 = h * (h * HASH_A);
    return h2 as T;
}

pub fn xor_sign_t<T>(val: T, sign_bit: i32) -> T {
    let sign_mask: i32 = -2147483647 - 1;
    if is_f64<T>() {
        return ((sign_bit & sign_mask) != 0) ? -val : val;
    } else {
        return bitcast<f32>(bitcast<i32>(val as f32) ^ (sign_bit & sign_mask)) as T;
    }
}

pub fn grad_dot_perlin2d<T>(hash: i32, fx: T, fy: T) -> T {
    let sx: T = xor_sign_t<T>(fx, hash << 31);
    let sy: T = xor_sign_t<T>(fy, (hash >> 1) << 31);
    let sel: bool = (hash & 4) != 0;
    let u: T = sel ? sy : sx;
    let v: T = sel ? sx : sy;
    return 2.414213657379150390625 * u + v;
}

pub fn grad_dot_common3d<T>(hash: i32, fx: T, fy: T, fz: T) -> T {
    let hasha13: i32 = hash & 13;
    let u: T = (hasha13 > 7) ? fy : fx;
    var v: T = (hasha13 == 12) ? fx : fz;
    v = (hasha13 < 2) ? fy : v;
    return xor_sign_t<T>(u, hash << 31) + xor_sign_t<T>(v, (hash >> 1) << 31);
}

pub fn grad_dot_perlin4d<T>(hash: i32, fx: T, fy: T, fz: T, fw: T) -> T {
    let p: i32 = hash & 24;
    let a: T = (p > 0) ? fx : fy;
    let b: T = ((hash & 16) != 0) ? fy : fz;
    let c: T = (p > 16) ? fz : fw;
    return xor_sign_t<T>(a, hash << 31) + xor_sign_t<T>(b, (hash >> 1) << 31) + xor_sign_t<T>(c, (hash >> 2) << 31);
}

pub fn rebase_cell<T>(x: T) -> i32 {
    if is_f64<T>() {
        let c: T = floor(x);
        return (c as i64) as i32;
    } else {
        return 0;
    }
}

pub fn rebase_frac<T>(x: T) -> T {
    if is_f64<T>() {
        return x - floor(x);
    } else {
        return x;
    }
}
)BK";

constexpr const char* kStdNoisePart2 = R"BK(
pub fn value1d<T>(x: T, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(x);
    let xr: T = rebase_frac<T>(x);
    let xs: T = floor(xr);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let tx: T = interp_hermite<T>(xr - xs);
    let v0: T = get_value_coord2d<T>(seed, x0, 0);
    let v1: T = get_value_coord2d<T>(seed, x0 + PRIME_X, 0);
    return scale_output<T>(fn_lerp<T>(v0, v1, tx), -2147483648.0, 2147483648.0);
}

pub fn value2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let xf0: T = interp_hermite<T>(x - xs);
    let yf0: T = interp_hermite<T>(y - ys);
    let v0: T = fn_lerp<T>(get_value_coord2d<T>(seed, x0, y0), get_value_coord2d<T>(seed, x1, y0), xf0);
    let v1: T = fn_lerp<T>(get_value_coord2d<T>(seed, x0, y1), get_value_coord2d<T>(seed, x1, y1), xf0);
    return scale_output<T>(fn_lerp<T>(v0, v1, yf0), -2147483648.0, 2147483648.0);
}

pub fn value3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let zc: i32 = rebase_cell<T>(p.z);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let zs: T = floor(z);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let z0: i32 = ((zs as i32) + zc) * PRIME_Z;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let z1: i32 = z0 + PRIME_Z;
    let xf0: T = interp_hermite<T>(x - xs);
    let yf0: T = interp_hermite<T>(y - ys);
    let zf0: T = interp_hermite<T>(z - zs);
    let vy0: T = fn_lerp<T>(
        fn_lerp<T>(get_value_coord3d<T>(seed, x0, y0, z0), get_value_coord3d<T>(seed, x1, y0, z0), xf0),
        fn_lerp<T>(get_value_coord3d<T>(seed, x0, y1, z0), get_value_coord3d<T>(seed, x1, y1, z0), xf0),
        yf0);
    let vy1: T = fn_lerp<T>(
        fn_lerp<T>(get_value_coord3d<T>(seed, x0, y0, z1), get_value_coord3d<T>(seed, x1, y0, z1), xf0),
        fn_lerp<T>(get_value_coord3d<T>(seed, x0, y1, z1), get_value_coord3d<T>(seed, x1, y1, z1), xf0),
        yf0);
    return scale_output<T>(fn_lerp<T>(vy0, vy1, zf0), -2147483648.0, 2147483648.0);
}

pub fn value4d<T>(p: vec4<T>, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let zc: i32 = rebase_cell<T>(p.z);
    let wc: i32 = rebase_cell<T>(p.w);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let w: T = rebase_frac<T>(p.w);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let zs: T = floor(z);
    let ws: T = floor(w);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let z0: i32 = ((zs as i32) + zc) * PRIME_Z;
    let w0: i32 = ((ws as i32) + wc) * PRIME_W;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let z1: i32 = z0 + PRIME_Z;
    let w1: i32 = w0 + PRIME_W;
    let xf0: T = interp_hermite<T>(x - xs);
    let yf0: T = interp_hermite<T>(y - ys);
    let zf0: T = interp_hermite<T>(z - zs);
    let wf0: T = interp_hermite<T>(w - ws);
    let vz00: T = fn_lerp<T>(
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y0, z0, w0), get_value_coord4d<T>(seed, x1, y0, z0, w0), xf0),
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y1, z0, w0), get_value_coord4d<T>(seed, x1, y1, z0, w0), xf0),
        yf0);
    let vz10: T = fn_lerp<T>(
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y0, z1, w0), get_value_coord4d<T>(seed, x1, y0, z1, w0), xf0),
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y1, z1, w0), get_value_coord4d<T>(seed, x1, y1, z1, w0), xf0),
        yf0);
    let vw0: T = fn_lerp<T>(vz00, vz10, zf0);
    let vz01: T = fn_lerp<T>(
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y0, z0, w1), get_value_coord4d<T>(seed, x1, y0, z0, w1), xf0),
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y1, z0, w1), get_value_coord4d<T>(seed, x1, y1, z0, w1), xf0),
        yf0);
    let vz11: T = fn_lerp<T>(
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y0, z1, w1), get_value_coord4d<T>(seed, x1, y0, z1, w1), xf0),
        fn_lerp<T>(get_value_coord4d<T>(seed, x0, y1, z1, w1), get_value_coord4d<T>(seed, x1, y1, z1, w1), xf0),
        yf0);
    let vw1: T = fn_lerp<T>(vz01, vz11, zf0);
    return scale_output<T>(fn_lerp<T>(vw0, vw1, wf0), -2147483648.0, 2147483648.0);
}

pub fn value<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return value2d<T>(p, seed);
    } else if D == 3 {
        return value3d<T>(p, seed);
    } else {
        return value4d<T>(p, seed);
    }
}

pub fn perlin2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let xf0: T = x - xs;
    let yf0: T = y - ys;
    let xf1: T = xf0 - 1.0;
    let yf1: T = yf0 - 1.0;
    let u: T = interp_quintic<T>(xf0);
    let v: T = interp_quintic<T>(yf0);
    let g0: T = fn_lerp<T>(
        grad_dot_perlin2d<T>(hash_primes2d(seed, x0, y0), xf0, yf0),
        grad_dot_perlin2d<T>(hash_primes2d(seed, x1, y0), xf1, yf0),
        u);
    let g1: T = fn_lerp<T>(
        grad_dot_perlin2d<T>(hash_primes2d(seed, x0, y1), xf0, yf1),
        grad_dot_perlin2d<T>(hash_primes2d(seed, x1, y1), xf1, yf1),
        u);
    let k_bound: T = 1.726796627044677734375;
    return scale_output<T>(fn_lerp<T>(g0, g1, v), -k_bound, k_bound);
}

pub fn perlin3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let zc: i32 = rebase_cell<T>(p.z);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let zs: T = floor(z);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let z0: i32 = ((zs as i32) + zc) * PRIME_Z;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let z1: i32 = z0 + PRIME_Z;
    let xf0: T = x - xs;
    let yf0: T = y - ys;
    let zf0: T = z - zs;
    let xf1: T = xf0 - 1.0;
    let yf1: T = yf0 - 1.0;
    let zf1: T = zf0 - 1.0;
    let u: T = interp_quintic<T>(xf0);
    let v: T = interp_quintic<T>(yf0);
    let w: T = interp_quintic<T>(zf0);
    let gy0: T = fn_lerp<T>(
        fn_lerp<T>(grad_dot_common3d<T>(hash_primes3d(seed, x0, y0, z0), xf0, yf0, zf0),
                   grad_dot_common3d<T>(hash_primes3d(seed, x1, y0, z0), xf1, yf0, zf0), u),
        fn_lerp<T>(grad_dot_common3d<T>(hash_primes3d(seed, x0, y1, z0), xf0, yf1, zf0),
                   grad_dot_common3d<T>(hash_primes3d(seed, x1, y1, z0), xf1, yf1, zf0), u),
        v);
    let gy1: T = fn_lerp<T>(
        fn_lerp<T>(grad_dot_common3d<T>(hash_primes3d(seed, x0, y0, z1), xf0, yf0, zf1),
                   grad_dot_common3d<T>(hash_primes3d(seed, x1, y0, z1), xf1, yf0, zf1), u),
        fn_lerp<T>(grad_dot_common3d<T>(hash_primes3d(seed, x0, y1, z1), xf0, yf1, zf1),
                   grad_dot_common3d<T>(hash_primes3d(seed, x1, y1, z1), xf1, yf1, zf1), u),
        v);
    let k_bound: T = 1.0363423824310302734375;
    return scale_output<T>(fn_lerp<T>(gy0, gy1, w), -k_bound, k_bound);
}

pub fn perlin4d<T>(p: vec4<T>, seed: i32 = 0) -> T {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let zc: i32 = rebase_cell<T>(p.z);
    let wc: i32 = rebase_cell<T>(p.w);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let w: T = rebase_frac<T>(p.w);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let zs: T = floor(z);
    let ws: T = floor(w);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let z0: i32 = ((zs as i32) + zc) * PRIME_Z;
    let w0: i32 = ((ws as i32) + wc) * PRIME_W;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let z1: i32 = z0 + PRIME_Z;
    let w1: i32 = w0 + PRIME_W;
    let xf0: T = x - xs;
    let yf0: T = y - ys;
    let zf0: T = z - zs;
    let wf0: T = w - ws;
    let xf1: T = xf0 - 1.0;
    let yf1: T = yf0 - 1.0;
    let zf1: T = zf0 - 1.0;
    let wf1: T = wf0 - 1.0;
    let u: T = interp_quintic<T>(xf0);
    let v: T = interp_quintic<T>(yf0);
    let zt: T = interp_quintic<T>(zf0);
    let wt: T = interp_quintic<T>(wf0);
    let gz00: T = fn_lerp<T>(
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y0, z0, w0), xf0, yf0, zf0, wf0),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y0, z0, w0), xf1, yf0, zf0, wf0), u),
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y1, z0, w0), xf0, yf1, zf0, wf0),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y1, z0, w0), xf1, yf1, zf0, wf0), u),
        v);
    let gz10: T = fn_lerp<T>(
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y0, z1, w0), xf0, yf0, zf1, wf0),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y0, z1, w0), xf1, yf0, zf1, wf0), u),
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y1, z1, w0), xf0, yf1, zf1, wf0),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y1, z1, w0), xf1, yf1, zf1, wf0), u),
        v);
    let gw0: T = fn_lerp<T>(gz00, gz10, zt);
    let gz01: T = fn_lerp<T>(
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y0, z0, w1), xf0, yf0, zf0, wf1),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y0, z0, w1), xf1, yf0, zf0, wf1), u),
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y1, z0, w1), xf0, yf1, zf0, wf1),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y1, z0, w1), xf1, yf1, zf0, wf1), u),
        v);
    let gz11: T = fn_lerp<T>(
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y0, z1, w1), xf0, yf0, zf1, wf1),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y0, z1, w1), xf1, yf0, zf1, wf1), u),
        fn_lerp<T>(grad_dot_perlin4d<T>(hash_primes4d(seed, x0, y1, z1, w1), xf0, yf1, zf1, wf1),
                   grad_dot_perlin4d<T>(hash_primes4d(seed, x1, y1, z1, w1), xf1, yf1, zf1, wf1), u),
        v);
    let gw1: T = fn_lerp<T>(gz01, gz11, zt);
    let k_bound: T = 1.33858621120452880859375;
    return scale_output<T>(fn_lerp<T>(gw0, gw1, wt), -k_bound, k_bound);
}

pub fn perlin<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return perlin2d<T>(p, seed);
    } else if D == 3 {
        return perlin3d<T>(p, seed);
    } else {
        return perlin4d<T>(p, seed);
    }
}
)BK";

constexpr const char* kStdNoisePart3 = R"BK(
pub fn simplex2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let k_skew: T = 0.3660254180431365966796875;
    let k_unskew: T = -0.211324870586395263671875;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64);
        let sx: f64 = (p.x as f64) + (0.3660254180431365966796875 * sum);
        let sy: f64 = (p.y as f64) + (0.3660254180431365966796875 * sum);
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
    } else {
        let skew_delta: T = k_skew * (p.x + p.y);
        x_skewed = p.x + skew_delta;
        y_skewed = p.y + skew_delta;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let x_ge_y: bool = dx_skewed >= dy_skewed;
    let unskew_delta: T = k_unskew * (dx_skewed + dy_skewed);
    let dx0: T = dx_skewed + unskew_delta;
    let dy0: T = dy_skewed + unskew_delta;
    let dx1: T = (x_ge_y ? dx0 : (dx0 + 1.0)) - (-0.211324870586395263671875 + 1.0);
    let dy1: T = (x_ge_y ? (dy0 + 1.0) : dy0) - (-0.211324870586395263671875 + 1.0);
    let dx2: T = dx0 - (-0.211324870586395263671875 * 2.0 + 1.0);
    let dy2: T = dy0 - (-0.211324870586395263671875 * 2.0 + 1.0);

    var falloff0: T = 0.5 - dy0 * dy0 - dx0 * dx0;
    var falloff1: T = 0.5 - dy1 * dy1 - dx1 * dx1;
    var falloff2: T = falloff0 + (unskew_delta * -3.1547005176544189453125 - 0.666666686534881591796875);
    falloff0 = max(falloff0, 0.0);
    falloff1 = max(falloff1, 0.0);
    falloff2 = max(falloff2, 0.0);
    falloff0 = (falloff0 * falloff0) * (falloff0 * falloff0);
    falloff1 = (falloff1 * falloff1) * (falloff1 * falloff1);
    falloff2 = (falloff2 * falloff2) * (falloff2 * falloff2);

    let g0: T = grad_dot_perlin2d<T>(hash_primes2d(seed, x_primed, y_primed), dx0, dy0);
    let g1: T = grad_dot_perlin2d<T>(
        hash_primes2d(seed, x_ge_y ? (x_primed + PRIME_X) : x_primed, x_ge_y ? y_primed : (y_primed + PRIME_Y)),
        dx1, dy1);
    let g2: T = grad_dot_perlin2d<T>(hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y), dx2, dy2);
    let val: T = g0 * falloff0 + (g1 * falloff1 + g2 * falloff2);
    let k_bound: T = 0.0261208079755306243896484375;
    return scale_output<T>(val, -k_bound, k_bound);
}

pub fn simplex3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let k_skew: T = 0.3333333432674407958984375;
    let k_unskew: T = -0.5;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var so2: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    var z_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64) + (p.z as f64);
        let sx: f64 = (p.x as f64) + (0.3333333432674407958984375 * sum);
        let sy: f64 = (p.y as f64) + (0.3333333432674407958984375 * sum);
        let sz: f64 = (p.z as f64) + (0.3333333432674407958984375 * sum);
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        let cz: f64 = floor(sz);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        so2 = (cz as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
        z_skewed = (sz - cz) as T;
    } else {
        let skew_delta: T = k_skew * (p.x + p.y + p.z);
        x_skewed = p.x + skew_delta;
        y_skewed = p.y + skew_delta;
        z_skewed = p.z + skew_delta;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let z_base: T = floor(z_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let dz_skewed: T = z_skewed - z_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let z_primed: i32 = ((z_base as i32) + so2) * PRIME_Z;

    let x_ge_y: bool = dx_skewed >= dy_skewed;
    let y_ge_z: bool = dy_skewed >= dz_skewed;
    let x_ge_z: bool = dx_skewed >= dz_skewed;

    let unskew_delta: T = k_unskew * (dx_skewed + dy_skewed + dz_skewed);
    let dx0: T = dx_skewed + unskew_delta;
    let dy0: T = dy_skewed + unskew_delta;
    let dz0: T = dz_skewed + unskew_delta;

    let mask_x1: bool = x_ge_y && x_ge_z;
    let mask_y1: bool = (!x_ge_y) && y_ge_z;
    let mask_z1: bool = x_ge_z || y_ge_z;

    let nmask_x2: bool = x_ge_y || x_ge_z;
    let nmask_y2: bool = (!y_ge_z) && x_ge_y;
    let nmask_z2: bool = x_ge_z && y_ge_z;

    let dx3: T = dx0 - (-0.5);
    let dy3: T = dy0 - (-0.5);
    let dz3: T = dz0 - (-0.5);
    let dx1: T = mask_x1 ? (dx3 - 1.0) : dx3;
    let dy1: T = mask_y1 ? (dy3 - 1.0) : dy3;
    let dz1: T = (!mask_z1) ? (dz3 - 1.0) : dz3;
    let dx2: T = (!nmask_x2) ? (dx0 + 1.0) : dx0;
    let dy2: T = nmask_y2 ? (dy0 + 1.0) : dy0;
    let dz2: T = nmask_z2 ? (dz0 + 1.0) : dz0;

    var falloff0: T = (0.6 - dx0 * dx0 - dy0 * dy0) - dz0 * dz0;
    var falloff1: T = (0.6 - dx1 * dx1 - dy1 * dy1) - dz1 * dz1;
    var falloff2: T = (0.6 - dx2 * dx2 - dy2 * dy2) - dz2 * dz2;
    var falloff3: T = falloff0 - (unskew_delta + 0.75);

    falloff0 = max(falloff0, 0.0);
    falloff1 = max(falloff1, 0.0);
    falloff2 = max(falloff2, 0.0);
    falloff3 = max(falloff3, 0.0);
    falloff0 = (falloff0 * falloff0) * (falloff0 * falloff0);
    falloff1 = (falloff1 * falloff1) * (falloff1 * falloff1);
    falloff2 = (falloff2 * falloff2) * (falloff2 * falloff2);
    falloff3 = (falloff3 * falloff3) * (falloff3 * falloff3);

    let g0: T = grad_dot_common3d<T>(hash_primes3d(seed, x_primed, y_primed, z_primed), dx0, dy0, dz0);
    let g1: T = grad_dot_common3d<T>(
        hash_primes3d(seed,
            mask_x1 ? (x_primed + PRIME_X) : x_primed,
            mask_y1 ? (y_primed + PRIME_Y) : y_primed,
            (!mask_z1) ? (z_primed + PRIME_Z) : z_primed),
        dx1, dy1, dz1);
    let g2: T = grad_dot_common3d<T>(
        hash_primes3d(seed,
            nmask_x2 ? (x_primed + PRIME_X) : x_primed,
            (!nmask_y2) ? (y_primed + PRIME_Y) : y_primed,
            (!nmask_z2) ? (z_primed + PRIME_Z) : z_primed),
        dx2, dy2, dz2);
    let g3: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z),
        dx3, dy3, dz3);

    let val: T = g3 * falloff3 + (g2 * falloff2 + (g1 * falloff1 + g0 * falloff0));
    let k_bound: T = 0.030586399137973785400390625;
    return scale_output<T>(val, -k_bound, k_bound);
}

pub fn simplex4d<T>(p: vec4<T>, seed: i32 = 0) -> T {
    let k_skew: T = 0.30901699437494742410229341718282;
    let k_unskew: T = -0.13819660112501051517954131656344;
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
        let skew_delta: T = k_skew * (p.x + p.y + p.z + p.w);
        x_skewed = p.x + skew_delta;
        y_skewed = p.y + skew_delta;
        z_skewed = p.z + skew_delta;
        w_skewed = p.w + skew_delta;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let z_base: T = floor(z_skewed);
    let w_base: T = floor(w_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let dz_skewed: T = z_skewed - z_base;
    let dw_skewed: T = w_skewed - w_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let z_primed: i32 = ((z_base as i32) + so2) * PRIME_Z;
    let w_primed: i32 = ((w_base as i32) + so3) * PRIME_W;

    let unskew_delta: T = k_unskew * (dx_skewed + dy_skewed + dz_skewed + dw_skewed);
    let dx0: T = dx_skewed + unskew_delta;
    let dy0: T = dy_skewed + unskew_delta;
    let dz0: T = dz_skewed + unskew_delta;
    let dw0: T = dw_skewed + unskew_delta;

    var rx: i32 = 0;
    var ry: i32 = 0;
    var rz: i32 = 0;
    var rw: i32 = 0;
    if dx0 >= dy0 { rx += 1; } else { ry += 1; }
    if dx0 >= dz0 { rx += 1; } else { rz += 1; }
    if dx0 >= dw0 { rx += 1; } else { rw += 1; }
    if dy0 >= dz0 { ry += 1; } else { rz += 1; }
    if dy0 >= dw0 { ry += 1; } else { rw += 1; }
    if dz0 >= dw0 { rz += 1; } else { rw += 1; }

    let mx1: bool = rx > 2; let my1: bool = ry > 2; let mz1: bool = rz > 2; let mw1: bool = rw > 2;
    let mx2: bool = rx > 1; let my2: bool = ry > 1; let mz2: bool = rz > 1; let mw2: bool = rw > 1;
    let mx3: bool = rx > 0; let my3: bool = ry > 0; let mz3: bool = rz > 0; let mw3: bool = rw > 0;

    let dx1: T = (mx1 ? (dx0 - 1.0) : dx0) - k_unskew;
    let dy1: T = (my1 ? (dy0 - 1.0) : dy0) - k_unskew;
    let dz1: T = (mz1 ? (dz0 - 1.0) : dz0) - k_unskew;
    let dw1: T = (mw1 ? (dw0 - 1.0) : dw0) - k_unskew;
    let dx2: T = (mx2 ? (dx0 - 1.0) : dx0) - (k_unskew * 2.0);
    let dy2: T = (my2 ? (dy0 - 1.0) : dy0) - (k_unskew * 2.0);
    let dz2: T = (mz2 ? (dz0 - 1.0) : dz0) - (k_unskew * 2.0);
    let dw2: T = (mw2 ? (dw0 - 1.0) : dw0) - (k_unskew * 2.0);
    let dx3: T = (mx3 ? (dx0 - 1.0) : dx0) - (k_unskew * 3.0);
    let dy3: T = (my3 ? (dy0 - 1.0) : dy0) - (k_unskew * 3.0);
    let dz3: T = (mz3 ? (dz0 - 1.0) : dz0) - (k_unskew * 3.0);
    let dw3: T = (mw3 ? (dw0 - 1.0) : dw0) - (k_unskew * 3.0);
    let dx4: T = dx0 - (k_unskew * 4.0 + 1.0);
    let dy4: T = dy0 - (k_unskew * 4.0 + 1.0);
    let dz4: T = dz0 - (k_unskew * 4.0 + 1.0);
    let dw4: T = dw0 - (k_unskew * 4.0 + 1.0);

    var falloff0: T = (((0.6 - dx0 * dx0) - dy0 * dy0) - dz0 * dz0) - dw0 * dw0;
    var falloff1: T = (((0.6 - dx1 * dx1) - dy1 * dy1) - dz1 * dz1) - dw1 * dw1;
    var falloff2: T = (((0.6 - dx2 * dx2) - dy2 * dy2) - dz2 * dz2) - dw2 * dw2;
    var falloff3: T = (((0.6 - dx3 * dx3) - dy3 * dy3) - dz3 * dz3) - dw3 * dw3;
    var falloff4: T = falloff0 + (unskew_delta * -2.89442729949951171875 - 0.800000011920928955078125);

    falloff0 = max(falloff0, 0.0);
    falloff1 = max(falloff1, 0.0);
    falloff2 = max(falloff2, 0.0);
    falloff3 = max(falloff3, 0.0);
    falloff4 = max(falloff4, 0.0);
    falloff0 = (falloff0 * falloff0) * (falloff0 * falloff0);
    falloff1 = (falloff1 * falloff1) * (falloff1 * falloff1);
    falloff2 = (falloff2 * falloff2) * (falloff2 * falloff2);
    falloff3 = (falloff3 * falloff3) * (falloff3 * falloff3);
    falloff4 = (falloff4 * falloff4) * (falloff4 * falloff4);

    let g0: T = grad_dot_perlin4d<T>(hash_primes4d(seed, x_primed, y_primed, z_primed, w_primed), dx0, dy0, dz0, dw0);
    let g1: T = grad_dot_perlin4d<T>(hash_primes4d(seed,
        mx1 ? (x_primed + PRIME_X) : x_primed,
        my1 ? (y_primed + PRIME_Y) : y_primed,
        mz1 ? (z_primed + PRIME_Z) : z_primed,
        mw1 ? (w_primed + PRIME_W) : w_primed), dx1, dy1, dz1, dw1);
    let g2: T = grad_dot_perlin4d<T>(hash_primes4d(seed,
        mx2 ? (x_primed + PRIME_X) : x_primed,
        my2 ? (y_primed + PRIME_Y) : y_primed,
        mz2 ? (z_primed + PRIME_Z) : z_primed,
        mw2 ? (w_primed + PRIME_W) : w_primed), dx2, dy2, dz2, dw2);
    let g3: T = grad_dot_perlin4d<T>(hash_primes4d(seed,
        mx3 ? (x_primed + PRIME_X) : x_primed,
        my3 ? (y_primed + PRIME_Y) : y_primed,
        mz3 ? (z_primed + PRIME_Z) : z_primed,
        mw3 ? (w_primed + PRIME_W) : w_primed), dx3, dy3, dz3, dw3);
    let g4: T = grad_dot_perlin4d<T>(hash_primes4d(seed,
        x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z, w_primed + PRIME_W), dx4, dy4, dz4, dw4);

    let val: T = g0 * falloff0 + (g1 * falloff1 + (g2 * falloff2 + (g3 * falloff3 + g4 * falloff4)));
    let k_bound: T = 0.036730043590068817138671875;
    return scale_output<T>(val, -k_bound, k_bound);
}

pub fn simplex<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return simplex2d<T>(p, seed);
    } else if D == 3 {
        return simplex3d<T>(p, seed);
    } else {
        return simplex4d<T>(p, seed);
    }
}
)BK";

constexpr const char* kStdNoisePart4a = R"BK(
pub fn supersimplex2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let k_skew: T = 0.3660254180431365966796875;
    let k_unskew: T = -0.211324870586395263671875;
    let k_rad2: T = 0.666666686534881591796875;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64);
        let s2: f64 = (0.3660254180431365966796875 as f32) as f64;
        let sx: f64 = (p.x as f64) + s2 * sum;
        let sy: f64 = (p.y as f64) + s2 * sum;
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
    } else {
        let s: T = k_skew * (p.x + p.y);
        x_skewed = p.x + s;
        y_skewed = p.y + s;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;

    let fwd_xy: bool = (dx_skewed + dy_skewed) > 1.0;
    let bound_xy: T = fwd_xy ? -1.0 : 0.0;
    let fwd_x: bool = (dx_skewed * -2.0 + dy_skewed) < bound_xy;
    let fwd_y: bool = (dy_skewed * -2.0 + dx_skewed) < bound_xy;

    let unskew_d: T = k_unskew * (dx_skewed + dy_skewed);
    var dx_base: T = dx_skewed + unskew_d;
    var dy_base: T = dy_skewed + unskew_d;

    let g0: T = grad_dot_perlin2d<T>(hash_primes2d(seed, x_primed, y_primed), dx_base, dy_base);
    let fb0: T = (k_rad2 - dy_base * dy_base) - dx_base * dx_base;
    let f0: T = (fb0 * fb0) * (fb0 * fb0);
    var value: T = f0 * g0;

    let g1: T = grad_dot_perlin2d<T>(
        hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y),
        dx_base - (2.0 * k_unskew + 1.0),
        dy_base - (2.0 * k_unskew + 1.0));
    let fb1: T = unskew_d * -3.1547005176544189453125 + (fb0 - k_rad2);
    let f1: T = (fb1 * fb1) * (fb1 * fb1);
    value += f1 * g1;

    let xy_delta: T = fwd_xy ? (k_unskew + 1.0) : (-k_unskew);
    dx_base -= xy_delta;
    dy_base -= xy_delta;

    let xp2_tmp: i32 = fwd_x ? (x_primed + (PRIME_X * 2)) : x_primed;
    let xp2: i32 = (!fwd_xy) ? (xp2_tmp - PRIME_X) : xp2_tmp;
    let yp2: i32 = fwd_xy ? (y_primed + PRIME_Y) : y_primed;
    let dx2: T = dx_base - (fwd_x ? (1.0 + 2.0 * k_unskew) : -1.0);
    let dy2: T = fwd_x ? (dy_base - 2.0 * k_unskew) : dy_base;
    let g2: T = grad_dot_perlin2d<T>(hash_primes2d(seed, xp2, yp2), dx2, dy2);
    let fb2: T = max((k_rad2 - dy2 * dy2) - dx2 * dx2, 0.0);
    value += ((fb2 * fb2) * (fb2 * fb2)) * g2;

    let xp3: i32 = fwd_xy ? (x_primed + PRIME_X) : x_primed;
    let yp3_tmp: i32 = fwd_y ? (y_primed + (PRIME_Y * 2)) : y_primed;
    let yp3: i32 = (!fwd_xy) ? (yp3_tmp - PRIME_Y) : yp3_tmp;
    let dx3: T = fwd_y ? (dx_base - 2.0 * k_unskew) : dx_base;
    let dy3: T = dy_base - (fwd_y ? (1.0 + 2.0 * k_unskew) : -1.0);
    let g3: T = grad_dot_perlin2d<T>(hash_primes2d(seed, xp3, yp3), dx3, dy3);
    let fb3: T = max((k_rad2 - dy3 * dy3) - dx3 * dx3, 0.0);
    value += ((fb3 * fb3) * (fb3 * fb3)) * g3;

    let k_bound: T = 0.14084912836551666259765625;
    return scale_output<T>(value, -k_bound, k_bound);
}

pub fn supersimplex3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let k_skew: T = 0.3333333432674407958984375;
    let k_unskew: T = -0.5;
    let k_twice_unskew: T = -0.25;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var so2: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    var z_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64) + (p.z as f64);
        let s3: f64 = (0.3333333432674407958984375 as f32) as f64;
        let sx: f64 = (p.x as f64) + s3 * sum;
        let sy: f64 = (p.y as f64) + s3 * sum;
        let sz: f64 = (p.z as f64) + s3 * sum;
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        let cz: f64 = floor(sz);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        so2 = (cz as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
        z_skewed = (sz - cz) as T;
    } else {
        let s: T = k_skew * (p.x + p.y + p.z);
        x_skewed = p.x + s;
        y_skewed = p.y + s;
        z_skewed = p.z + s;
    }
    var x_base: T = floor(x_skewed);
    var y_base: T = floor(y_skewed);
    var z_base: T = floor(z_skewed);
    var dx_skewed: T = x_skewed - x_base;
    var dy_skewed: T = y_skewed - y_base;
    var dz_skewed: T = z_skewed - z_base;

    let tu0: T = k_twice_unskew * (dx_skewed + dy_skewed + dz_skewed);
    let xn0: T = dx_skewed + tu0;
    let yn0: T = dy_skewed + tu0;
    let zn0: T = dz_skewed + tu0;
    let xyzn0: T = -tu0;

    var max_score: T = 0.375;
    var move_bits: i32 = (xyzn0 > max_score) ? -1 : 0;
    max_score = max(max_score, xyzn0);
    if xn0 > max_score { move_bits = 1; max_score = xn0; }
    if yn0 > max_score { move_bits = 2; max_score = yn0; }
    if zn0 > max_score { move_bits = 4; max_score = zn0; }
    max_score += 0.125 - xyzn0;
    if (-zn0) > max_score { move_bits = 3; max_score = -zn0; }
    if (-yn0) > max_score { move_bits = 5; max_score = -yn0; }
    if (-xn0) > max_score { move_bits = 6; max_score = -xn0; }

    if (move_bits & 1) != 0 { x_base += 1.0; dx_skewed -= 1.0; }
    if (move_bits & 2) != 0 { y_base += 1.0; dy_skewed -= 1.0; }
    if (move_bits & 4) != 0 { z_base += 1.0; dz_skewed -= 1.0; }

    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let z_primed: i32 = ((z_base as i32) + so2) * PRIME_Z;

    let sk_sum: T = dx_skewed + dy_skewed + dz_skewed;
    let tu: T = k_twice_unskew * sk_sum;
    let xn: T = dx_skewed + tu;
    let yn: T = dy_skewed + tu;
    let zn: T = dz_skewed + tu;
    let xyzn: T = -tu;

    let unskew_d: T = k_unskew * sk_sum;
    let dx_b: T = dx_skewed + unskew_d;
    let dy_b: T = dy_skewed + unskew_d;
    let dz_b: T = dz_skewed + unskew_d;
    let coord_sum: T = -0.5 * sk_sum;

    let g0: T = grad_dot_common3d<T>(hash_primes3d(seed, x_primed, y_primed, z_primed), dx_b, dy_b, dz_b);
    let fb0: T = (((0.75 - dx_b * dx_b) - dy_b * dy_b) - dz_b * dz_b) * 0.5;
    let stem_a: T = fb0 - 0.375;
    let stem_b: T = fb0 - 0.5;
    var value: T = ((fb0 * fb0) * (fb0 * fb0)) * g0;

    // <1,1,1> or <-1,-1,-1>
    let sm1: bool = xyzn < 0.0;
    let off1: T = sm1 ? 0.5 : -0.5;
    let g1: T = grad_dot_common3d<T>(
        hash_primes3d(seed,
            x_primed + (sm1 ? -PRIME_X : PRIME_X),
            y_primed + (sm1 ? -PRIME_Y : PRIME_Y),
            z_primed + (sm1 ? -PRIME_Z : PRIME_Z)),
        dx_b - off1, dy_b - off1, dz_b - off1);
    let fb1: T = max(off1 * coord_sum + stem_a, 0.0);
    value += ((fb1 * fb1) * (fb1 * fb1)) * g1;

    // <1,1,0> or <-1,-1,0>
    let sm2: bool = xyzn < zn;
    let off2: T = sm2 ? 1.0 : -1.0;
    let g2: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed + (sm2 ? -PRIME_X : PRIME_X), y_primed + (sm2 ? -PRIME_Y : PRIME_Y), z_primed),
        dx_b, dy_b, dz_b - off2);
    let fb2: T = min((sm2 ? -dz_b : dz_b) - stem_b, 0.0);
    value += ((fb2 * fb2) * (fb2 * fb2)) * g2;

    // <1,0,1> or <-1,0,-1>
    let sm3: bool = xyzn < yn;
    let off3: T = sm3 ? 1.0 : -1.0;
    let g3: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed + (sm3 ? -PRIME_X : PRIME_X), y_primed, z_primed + (sm3 ? -PRIME_Z : PRIME_Z)),
        dx_b, dy_b - off3, dz_b);
    let fb3: T = min((sm3 ? -dy_b : dy_b) - stem_b, 0.0);
    value += ((fb3 * fb3) * (fb3 * fb3)) * g3;

    // <0,1,1> or <0,-1,-1>
    let sm4: bool = xyzn < xn;
    let off4: T = sm4 ? 1.0 : -1.0;
    let g4: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed, y_primed + (sm4 ? -PRIME_Y : PRIME_Y), z_primed + (sm4 ? -PRIME_Z : PRIME_Z)),
        dx_b - off4, dy_b, dz_b);
    let fb4: T = min((sm4 ? -dx_b : dx_b) - stem_b, 0.0);
    value += ((fb4 * fb4) * (fb4 * fb4)) * g4;

    // <1,0,0> or <-1,0,0>
    let sm5: bool = xn < 0.0;
    let off5: T = sm5 ? 0.5 : -0.5;
    let g5: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed + (sm5 ? -PRIME_X : PRIME_X), y_primed, z_primed),
        dx_b + off5, dy_b - off5, dz_b - off5);
    let fb5: T = max((off5 * coord_sum + stem_a) + (sm5 ? -dx_b : dx_b), 0.0);
    value += ((fb5 * fb5) * (fb5 * fb5)) * g5;

    // <0,1,0> or <0,-1,0>
    let sm6: bool = yn < 0.0;
    let off6: T = sm6 ? 0.5 : -0.5;
    let g6: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed, y_primed + (sm6 ? -PRIME_Y : PRIME_Y), z_primed),
        dx_b - off6, dy_b + off6, dz_b - off6);
    let fb6: T = max((off6 * coord_sum + stem_a) + (sm6 ? -dy_b : dy_b), 0.0);
    value += ((fb6 * fb6) * (fb6 * fb6)) * g6;

    // <0,0,1> or <0,0,-1>
    let sm7: bool = zn < 0.0;
    let off7: T = sm7 ? 0.5 : -0.5;
    let g7: T = grad_dot_common3d<T>(
        hash_primes3d(seed, x_primed, y_primed, z_primed + (sm7 ? -PRIME_Z : PRIME_Z)),
        dx_b - off7, dy_b - off7, dz_b + off7);
    let fb7: T = max((off7 * coord_sum + stem_a) + (sm7 ? -dz_b : dz_b), 0.0);
    value += ((fb7 * fb7) * (fb7 * fb7)) * g7;

    let k_bound: T = 0.0069091119803488254547119140625;
    return scale_output<T>(value, -k_bound, k_bound);
}
)BK";

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

constexpr const char* kStdNoisePart8 = R"BK(
pub fn cellular_distance2d<T>(
    p: vec2<T>,
    seed: i32 = 0,
    dist_func: i32 = 0,
    ret_type: i32 = 0,
    jitter_mod: T = 1.0
) -> T {
    let k_jitter: T = 0.437015950679779052734375 * jitter_mod;
    let x_off: i32 = rebase_cell<T>(p.x);
    let y_off: i32 = rebase_cell<T>(p.y);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let xc_base: i32 = ((round(x) as i32) - 1 + x_off) * PRIME_X;
    let yc_base: i32 = ((round(y) as i32) - 1 + y_off) * PRIME_Y;
    let xcf_base: T = (round(x) - 1.0) - x;
    let ycf_base: T = (round(y) - 1.0) - y;

    var d0: T = 1.0e10;
    var d1: T = 1.0e10;
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
            d1 = max(min(d1, d), d0);
            d0 = min(d0, d);
            yc += PRIME_Y;
            ycf += 1.0;
        }
        xc += PRIME_X;
        xcf += 1.0;
    }
    if dist_func == 0 {
        d0 = sqrt(d0);
        d1 = sqrt(d1);
    }
    let max_dist: T = (1.0 + 0.437015950679779052734375) * (1.0 + 0.437015950679779052734375);
    return cellular_return<T>(ret_type, d0, d1, max_dist);
}

pub fn cellular_distance3d<T>(
    p: vec3<T>,
    seed: i32 = 0,
    dist_func: i32 = 0,
    ret_type: i32 = 0,
    jitter_mod: T = 1.0
) -> T {
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

    var d0: T = 1.0e10;
    var d1: T = 1.0e10;
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
                d1 = max(min(d1, d), d0);
                d0 = min(d0, d);
                zc += PRIME_Z;
                zcf += 1.0;
            }
            yc += PRIME_Y;
            ycf += 1.0;
        }
        xc += PRIME_X;
        xcf += 1.0;
    }
    if dist_func == 0 {
        d0 = sqrt(d0);
        d1 = sqrt(d1);
    }
    let max_dist: T = (1.0 + 0.3961435258388519287109375) * (1.0 + 0.3961435258388519287109375);
    return cellular_return<T>(ret_type, d0, d1, max_dist);
}

pub fn cellular_distance4d<T>(
    p: vec4<T>,
    seed: i32 = 0,
    dist_func: i32 = 0,
    ret_type: i32 = 0,
    jitter_mod: T = 1.0
) -> T {
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

    var d0: T = 1.0e10;
    var d1: T = 1.0e10;
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
                    d1 = max(min(d1, d), d0);
                    d0 = min(d0, d);
                }
            }
        }
    }
    if dist_func == 0 {
        d0 = sqrt(d0);
        d1 = sqrt(d1);
    }
    let max_dist: T = (1.0 + 0.3660254180431365966796875) * (1.0 + 0.3660254180431365966796875);
    return cellular_return<T>(ret_type, d0, d1, max_dist);
}

pub fn cellular_distance<T, const D: i32>(
    p: vec<T, D>,
    seed: i32 = 0,
    dist_func: i32 = 0,
    ret_type: i32 = 0,
    jitter_mod: T = 1.0
) -> T {
    if D == 2 {
        return cellular_distance2d<T>(p, seed, dist_func, ret_type, jitter_mod);
    } else if D == 3 {
        return cellular_distance3d<T>(p, seed, dist_func, ret_type, jitter_mod);
    } else {
        return cellular_distance4d<T>(p, seed, dist_func, ret_type, jitter_mod);
    }
}

pub fn cellular_lookup_perlin2d<T>(p: vec2<T>, seed: i32 = 0, lookup_freq: T = 0.2, jitter_mod: T = 1.0) -> T {
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
    var best_x: T = 0.0;
    var best_y: T = 0.0;
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
            let d: T = px_d * px_d + py_d * py_d;
            if d < best_dist {
                best_dist = d;
                best_x = px_d + p.x;
                best_y = py_d + p.y;
            }
            yc += PRIME_Y;
            ycf += 1.0;
        }
        xc += PRIME_X;
        xcf += 1.0;
    }
    return perlin2d<T>(vec2<T>(best_x * lookup_freq, best_y * lookup_freq), seed + 1);
}

pub fn cellular_lookup_perlin3d<T>(p: vec3<T>, seed: i32 = 0, lookup_freq: T = 0.2, jitter_mod: T = 1.0) -> T {
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
    var best_x: T = 0.0;
    var best_y: T = 0.0;
    var best_z: T = 0.0;
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
                let d: T = px_d * px_d + py_d * py_d + pz_d * pz_d;
                if d < best_dist {
                    best_dist = d;
                    best_x = px_d + p.x;
                    best_y = py_d + p.y;
                    best_z = pz_d + p.z;
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
    return perlin3d<T>(vec3<T>(best_x * lookup_freq, best_y * lookup_freq, best_z * lookup_freq), seed + 1);
}

pub fn cellular_lookup_perlin4d<T>(p: vec4<T>, seed: i32 = 0, lookup_freq: T = 0.2, jitter_mod: T = 1.0) -> T {
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
    var best_x: T = 0.0;
    var best_y: T = 0.0;
    var best_z: T = 0.0;
    var best_w: T = 0.0;
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
                    let d: T = px_d * px_d + py_d * py_d + pz_d * pz_d + pw_d * pw_d;
                    if d < best_dist {
                        best_dist = d;
                        best_x = px_d + p.x;
                        best_y = py_d + p.y;
                        best_z = pz_d + p.z;
                        best_w = pw_d + p.w;
                    }
                }
            }
        }
    }
    return perlin4d<T>(vec4<T>(best_x * lookup_freq, best_y * lookup_freq, best_z * lookup_freq, best_w * lookup_freq), seed + 1);
}

pub fn cellular_lookup_perlin<T, const D: i32>(p: vec<T, D>, seed: i32 = 0, lookup_freq: T = 0.2, jitter_mod: T = 1.0) -> T {
    if D == 2 {
        return cellular_lookup_perlin2d<T>(p, seed, lookup_freq, jitter_mod);
    } else if D == 3 {
        return cellular_lookup_perlin3d<T>(p, seed, lookup_freq, jitter_mod);
    } else {
        return cellular_lookup_perlin4d<T>(p, seed, lookup_freq, jitter_mod);
    }
}
)BK";

constexpr const char* kStdNoisePart9 = R"BK(
// Domain Warp (from FastNoise2 DomainWarp.inl, DomainWarpSimplex.inl, DomainWarpFractal.inl)
pub fn domain_warp_gradient_delta2d<T>(p: vec2<T>, seed: i32 = 0) -> vec2<T> {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let xs1: T = interp_hermite<T>(x - xs);
    let ys1: T = interp_hermite<T>(y - ys);
    let xs0: T = 1.0 - xs1;
    let ys0: T = 1.0 - ys1;

    let h00: i32 = hash_primes_hb2d(seed, x0, y0);
    let h10: i32 = hash_primes_hb2d(seed, x1, y0);
    let h01: i32 = hash_primes_hb2d(seed, x0, y1);
    let h11: i32 = hash_primes_hb2d(seed, x1, y1);

    let c00: T = xs0 * ys0;
    let c10: T = xs1 * ys0;
    let c01: T = xs0 * ys1;
    let c11: T = xs1 * ys1;

    let xw: T = c00 * ((h00 & 0xffff) as T) + c10 * ((h10 & 0xffff) as T) + c01 * ((h01 & 0xffff) as T) + c11 * ((h11 & 0xffff) as T);
    let yw: T = c00 * (((h00 >> 16) & 0xffff) as T) + c10 * (((h10 >> 16) & 0xffff) as T) + c01 * (((h01 >> 16) & 0xffff) as T) + c11 * (((h11 >> 16) & 0xffff) as T);

    let norm: T = 1.0 / (65535.0 / 2.0);
    return vec2<T>(xw * norm - 1.0, yw * norm - 1.0);
}

pub fn domain_warp_gradient2d<T>(p: vec2<T>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec2<T> {
    let w: vec2<T> = domain_warp_gradient_delta2d<T>(p * warp_freq, seed);
    return p + w * warp_amp;
}

pub fn domain_warp_gradient_delta3d<T>(p: vec3<T>, seed: i32 = 0) -> vec3<T> {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let zc: i32 = rebase_cell<T>(p.z);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let zs: T = floor(z);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let z0: i32 = ((zs as i32) + zc) * PRIME_Z;
    let x1: i32 = x0 + PRIME_X;
    let y1: i32 = y0 + PRIME_Y;
    let z1: i32 = z0 + PRIME_Z;
    let xs1: T = interp_hermite<T>(x - xs);
    let ys1: T = interp_hermite<T>(y - ys);
    let zs1: T = interp_hermite<T>(z - zs);
    let xs0: T = 1.0 - xs1;
    let ys0: T = 1.0 - ys1;
    let zs0: T = 1.0 - zs1;

    let h000: i32 = hash_primes_hb3d(seed, x0, y0, z0);
    let h100: i32 = hash_primes_hb3d(seed, x1, y0, z0);
    let h010: i32 = hash_primes_hb3d(seed, x0, y1, z0);
    let h110: i32 = hash_primes_hb3d(seed, x1, y1, z0);
    let h001: i32 = hash_primes_hb3d(seed, x0, y0, z1);
    let h101: i32 = hash_primes_hb3d(seed, x1, y0, z1);
    let h011: i32 = hash_primes_hb3d(seed, x0, y1, z1);
    let h111: i32 = hash_primes_hb3d(seed, x1, y1, z1);

    let c000: T = xs0 * ys0 * zs0;
    let c100: T = xs1 * ys0 * zs0;
    let c010: T = xs0 * ys1 * zs0;
    let c110: T = xs1 * ys1 * zs0;
    let c001: T = xs0 * ys0 * zs1;
    let c101: T = xs1 * ys0 * zs1;
    let c011: T = xs0 * ys1 * zs1;
    let c111: T = xs1 * ys1 * zs1;

    let xw: T = c000 * ((h000 & 0x3ff) as T) + c100 * ((h100 & 0x3ff) as T) + c010 * ((h010 & 0x3ff) as T) + c110 * ((h110 & 0x3ff) as T) +
                c001 * ((h001 & 0x3ff) as T) + c101 * ((h101 & 0x3ff) as T) + c011 * ((h011 & 0x3ff) as T) + c111 * ((h111 & 0x3ff) as T);
    let yw: T = c000 * (((h000 >> 11) & 0x3ff) as T) + c100 * (((h100 >> 11) & 0x3ff) as T) + c010 * (((h010 >> 11) & 0x3ff) as T) + c110 * (((h110 >> 11) & 0x3ff) as T) +
                c001 * (((h001 >> 11) & 0x3ff) as T) + c101 * (((h101 >> 11) & 0x3ff) as T) + c011 * (((h011 >> 11) & 0x3ff) as T) + c111 * (((h111 >> 11) & 0x3ff) as T);
    let zw: T = c000 * (((h000 >> 22) & 0x3ff) as T) + c100 * (((h100 >> 22) & 0x3ff) as T) + c010 * (((h010 >> 22) & 0x3ff) as T) + c110 * (((h110 >> 22) & 0x3ff) as T) +
                c001 * (((h001 >> 22) & 0x3ff) as T) + c101 * (((h101 >> 22) & 0x3ff) as T) + c011 * (((h011 >> 22) & 0x3ff) as T) + c111 * (((h111 >> 22) & 0x3ff) as T);

    let norm: T = 1.0 / (1023.0 / 2.0);
    return vec3<T>(xw * norm - 1.0, yw * norm - 1.0, zw * norm - 1.0);
}

pub fn domain_warp_gradient3d<T>(p: vec3<T>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec3<T> {
    let w: vec3<T> = domain_warp_gradient_delta3d<T>(p * warp_freq, seed);
    return p + w * warp_amp;
}

pub fn domain_warp_gradient_delta4d<T>(p: vec4<T>, seed: i32 = 0) -> vec4<T> {
    let xc: i32 = rebase_cell<T>(p.x);
    let yc: i32 = rebase_cell<T>(p.y);
    let zc: i32 = rebase_cell<T>(p.z);
    let wc: i32 = rebase_cell<T>(p.w);
    let x: T = rebase_frac<T>(p.x);
    let y: T = rebase_frac<T>(p.y);
    let z: T = rebase_frac<T>(p.z);
    let w: T = rebase_frac<T>(p.w);
    let xs: T = floor(x);
    let ys: T = floor(y);
    let zs: T = floor(z);
    let ws: T = floor(w);
    let x0: i32 = ((xs as i32) + xc) * PRIME_X;
    let y0: i32 = ((ys as i32) + yc) * PRIME_Y;
    let z0: i32 = ((zs as i32) + zc) * PRIME_Z;
    let w0: i32 = ((ws as i32) + wc) * PRIME_W;
    let xf: T = interp_hermite<T>(x - xs);
    let yf: T = interp_hermite<T>(y - ys);
    let zf: T = interp_hermite<T>(z - zs);
    let wf: T = interp_hermite<T>(w - ws);

    var xw: T = 0.0;
    var yw: T = 0.0;
    var zw: T = 0.0;
    var ww: T = 0.0;
    for (var iw: i32 = 0; iw < 2; iw += 1) {
        let wt: T = (iw == 0) ? (1.0 - wf) : wf;
        for (var iz: i32 = 0; iz < 2; iz += 1) {
            let zt: T = wt * ((iz == 0) ? (1.0 - zf) : zf);
            for (var iy: i32 = 0; iy < 2; iy += 1) {
                let yt: T = zt * ((iy == 0) ? (1.0 - yf) : yf);
                for (var ix: i32 = 0; ix < 2; ix += 1) {
                    let xt: T = yt * ((ix == 0) ? (1.0 - xf) : xf);
                    let h: i32 = hash_primes_hb4d(seed, x0 + ix * PRIME_X, y0 + iy * PRIME_Y, z0 + iz * PRIME_Z, w0 + iw * PRIME_W);
                    xw += xt * ((h & 0xff) as T);
                    yw += xt * (((h >> 8) & 0xff) as T);
                    zw += xt * (((h >> 16) & 0xff) as T);
                    ww += xt * (((h >> 24) & 0xff) as T);
                }
            }
        }
    }
    let norm: T = 1.0 / (255.0 / 2.0);
    return vec4<T>(xw * norm - 1.0, yw * norm - 1.0, zw * norm - 1.0, ww * norm - 1.0);
}

pub fn domain_warp_gradient4d<T>(p: vec4<T>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec4<T> {
    let w: vec4<T> = domain_warp_gradient_delta4d<T>(p * warp_freq, seed);
    return p + w * warp_amp;
}

pub fn domain_warp_gradient_delta<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> vec<T, D> {
    if D == 2 {
        return domain_warp_gradient_delta2d<T>(p, seed);
    } else if D == 3 {
        return domain_warp_gradient_delta3d<T>(p, seed);
    } else {
        return domain_warp_gradient_delta4d<T>(p, seed);
    }
}

pub fn domain_warp_gradient<T, const D: i32>(p: vec<T, D>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec<T, D> {
    if D == 2 {
        return domain_warp_gradient2d<T>(p, seed, warp_amp, warp_freq);
    } else if D == 3 {
        return domain_warp_gradient3d<T>(p, seed, warp_amp, warp_freq);
    } else {
        return domain_warp_gradient4d<T>(p, seed, warp_amp, warp_freq);
    }
}

pub fn domain_warp_fractal_progressive<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    warp_amp: T = 1.0,
    warp_freq: T = 0.5,
    seed: i32 = 0
) -> vec<T, D> {
    var wp: vec<T, D> = p;
    var amp: T = warp_amp;
    var freq: T = warp_freq;
    var s: i32 = seed;
    var w: vec<T, D> = domain_warp_gradient_delta<T, D>(wp * freq, s);
    wp = wp + w * amp;
    var strength: T = dot(w, w);
    for (var i: i32 = 1; i < octaves; i += 1) {
        strength = sqrt(strength);
        s += 1;
        freq *= lacunarity;
        amp *= fn_lerp<T>(1.0, 1.0 - strength, weighted_strength);
        amp *= gain;
        w = domain_warp_gradient_delta<T, D>(wp * freq, s);
        wp = wp + w * amp;
        strength = dot(w, w);
    }
    return wp;
}

pub fn domain_warp_fractal_independent<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    weighted_strength: T = 0.0,
    warp_amp: T = 1.0,
    warp_freq: T = 0.5,
    seed: i32 = 0
) -> vec<T, D> {
    var wp: vec<T, D> = p;
    var amp: T = warp_amp;
    var freq: T = warp_freq;
    var s: i32 = seed;
    var w: vec<T, D> = domain_warp_gradient_delta<T, D>(p * freq, s);
    wp = wp + w * amp;
    var strength: T = dot(w, w);
    for (var i: i32 = 1; i < octaves; i += 1) {
        strength = sqrt(strength);
        s += 1;
        freq *= lacunarity;
        amp *= fn_lerp<T>(1.0, 1.0 - strength, weighted_strength);
        amp *= gain;
        w = domain_warp_gradient_delta<T, D>(p * freq, s);
        wp = wp + w * amp;
        strength = dot(w, w);
    }
    return wp;
}
)BK";

constexpr const char* kStdNoisePart10 = R"BK(
pub fn warp_vec_contrib_simplex2d<T>(hash31: i32, fx: T, fy: T, mult: T) -> vec2<T> {
    let idx: i32 = (((hash31 as u32) >> 1u) as i32) * 3;
    let of_three: i32 = ((idx as u32) >> 30u) as i32;
    let sign_x: i32 = (idx >> 28) << 31;
    let sign_y: i32 = (idx >> 29) << 31;
    let masked: T = (idx >= 0) ? 0.3660254180431365966796875 : 0.0;
    let gx: T = xor_sign_t<T>((of_three != 0) ? (masked + 1.0) : masked, sign_x);
    let gy: T = xor_sign_t<T>((of_three != 1) ? (masked + 1.0) : masked, sign_y);
    let vx: T = mult * (fy * gy + fx * gx);
    let vy: T = xor_sign_t<T>(mult, hash31 << 31) * (fy * gx - fx * gy);
    return vec2<T>(vx, vy);
}

pub fn warp_vec_contrib_common3d<T>(hash31: i32, fx: T, fy: T, fz: T, mult: T) -> vec3<T> {
    let k_a: T = 2.224744871391589;
    let k_b: T = -0.224744871391589;
    let k_c: T = -1.0;
    let k_de: T = 1.0;
    let k_f: T = 2.0;

    let hs: i32 = ((hash31 as u32) >> 1u) as i32;
    let idx_fb: i32 = hs * (-1431655765);
    let idx_p2_hb: i32 = idx_fb << 29;
    let idx_p3: i32 = (((hs * 3) as u32) >> 30u) as i32;
    let final_sign: i32 = hash31 << 31;
    let sign0: i32 = idx_fb << 31;
    let sign1: i32 = (idx_fb << 30) & (-2147483647 - 1);

    let not_yz_neg: bool = idx_fb < 0;
    let not_xy_neg: bool = (idx_fb << 1) < 0;
    let not_or_neg: bool = (idx_fb | (idx_fb << 1)) < 0;

    let va_gx: T = not_yz_neg ? k_c : xor_sign_t<T>(k_a, sign0);
    let va_gy: T = not_or_neg ? (not_xy_neg ? xor_sign_t<T>(k_b, sign1) : xor_sign_t<T>(k_a, sign0)) : k_c;
    let va_gz: T = not_xy_neg ? k_c : xor_sign_t<T>(k_b, sign1);
    let val_a: T = va_gz * fz + (fy * va_gy + fx * va_gx);

    let vb_gx: T = not_yz_neg ? k_c : xor_sign_t<T>(k_b, sign0);
    let vb_gy: T = not_or_neg ? (not_xy_neg ? xor_sign_t<T>(k_a, sign1) : xor_sign_t<T>(k_b, sign0)) : k_c;
    let vb_gz: T = not_xy_neg ? k_c : xor_sign_t<T>(k_a, sign1);
    let val_b: T = vb_gz * fz + (fy * vb_gy + fx * vb_gx);

    let vc_gx: T = not_yz_neg ? k_f : xor_sign_t<T>(k_de, sign0);
    let vc_gy: T = not_or_neg ? (not_xy_neg ? xor_sign_t<T>(k_de, sign1) : xor_sign_t<T>(k_de, sign0)) : k_f;
    let vc_gz: T = not_xy_neg ? k_f : xor_sign_t<T>(k_de, sign1);
    let val_c: T = vc_gz * fz + (fy * vc_gy + fx * vc_gx);

    let p2_neg: bool = idx_p2_hb < 0;
    let val_ab: T = p2_neg ? val_b : val_a;
    let val_ba: T = p2_neg ? val_a : val_b;

    let m_signed: T = xor_sign_t<T>(mult, final_sign);
    let vx: T = m_signed * ((idx_p3 == 0) ? val_c : val_ab);
    let vy: T = m_signed * ((idx_p3 == 1) ? val_c : ((idx_p3 == 2) ? val_ba : val_ab));
    let vz: T = m_signed * ((idx_p3 == 2) ? val_c : val_ba);
    return vec3<T>(vx, vy, vz);
}

pub fn domain_warp_simplex_delta2d<T>(p: vec2<T>, seed: i32 = 0) -> vec2<T> {
    let k_skew: T = 0.3660254180431365966796875;
    let k_unskew: T = -0.211324870586395263671875;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64);
        let s2: f64 = (0.3660254180431365966796875 as f32) as f64;
        let sx: f64 = (p.x as f64) + s2 * sum;
        let sy: f64 = (p.y as f64) + s2 * sum;
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
    } else {
        let skew_delta: T = k_skew * (p.x + p.y);
        x_skewed = p.x + skew_delta;
        y_skewed = p.y + skew_delta;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let x_ge_y: bool = dx_skewed >= dy_skewed;
    let unskew_delta: T = k_unskew * (dx_skewed + dy_skewed);
    let dx0: T = dx_skewed + unskew_delta;
    let dy0: T = dy_skewed + unskew_delta;
    let dx1: T = (x_ge_y ? dx0 : (dx0 + 1.0)) - (k_unskew + 1.0);
    let dy1: T = (x_ge_y ? (dy0 + 1.0) : dy0) - (k_unskew + 1.0);
    let dx2: T = dx0 - (k_unskew * 2.0 + 1.0);
    let dy2: T = dy0 - (k_unskew * 2.0 + 1.0);

    var f0: T = max((0.5 - dy0 * dy0) - dx0 * dx0, 0.0);
    var f1: T = max((0.5 - dy1 * dy1) - dx1 * dx1, 0.0);
    var f2: T = max(((0.5 - dy0 * dy0) - dx0 * dx0) + (unskew_delta * -3.1547005176544189453125 - 0.666666686534881591796875), 0.0);
    f0 = (f0 * f0) * (f0 * f0);
    f1 = (f1 * f1) * (f1 * f1);
    f2 = (f2 * f2) * (f2 * f2);

    var v: vec2<T> = warp_vec_contrib_simplex2d<T>(hash_primes2d(seed, x_primed, y_primed), dx0, dy0, f0);
    v = v + warp_vec_contrib_simplex2d<T>(
        hash_primes2d(seed, x_ge_y ? (x_primed + PRIME_X) : x_primed, x_ge_y ? y_primed : (y_primed + PRIME_Y)),
        dx1, dy1, f1);
    v = v + warp_vec_contrib_simplex2d<T>(hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y), dx2, dy2, f2);
    return v * 70.1480560302734375;
}

pub fn domain_warp_simplex2d<T>(p: vec2<T>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec2<T> {
    return p + domain_warp_simplex_delta2d<T>(p * warp_freq, seed) * warp_amp;
}

pub fn domain_warp_simplex_delta3d<T>(p: vec3<T>, seed: i32 = 0) -> vec3<T> {
    let k_skew: T = 0.3333333432674407958984375;
    let k_unskew: T = -0.5;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var so2: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    var z_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64) + (p.z as f64);
        let s3: f64 = (0.3333333432674407958984375 as f32) as f64;
        let sx: f64 = (p.x as f64) + s3 * sum;
        let sy: f64 = (p.y as f64) + s3 * sum;
        let sz: f64 = (p.z as f64) + s3 * sum;
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        let cz: f64 = floor(sz);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        so2 = (cz as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
        z_skewed = (sz - cz) as T;
    } else {
        let skew_delta: T = k_skew * (p.x + p.y + p.z);
        x_skewed = p.x + skew_delta;
        y_skewed = p.y + skew_delta;
        z_skewed = p.z + skew_delta;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let z_base: T = floor(z_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let dz_skewed: T = z_skewed - z_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;
    let z_primed: i32 = ((z_base as i32) + so2) * PRIME_Z;

    let x_ge_y: bool = dx_skewed >= dy_skewed;
    let y_ge_z: bool = dy_skewed >= dz_skewed;
    let x_ge_z: bool = dx_skewed >= dz_skewed;
    let unskew_delta: T = k_unskew * (dx_skewed + dy_skewed + dz_skewed);
    let dx0: T = dx_skewed + unskew_delta;
    let dy0: T = dy_skewed + unskew_delta;
    let dz0: T = dz_skewed + unskew_delta;

    let mask_x1: bool = x_ge_y && x_ge_z;
    let mask_y1: bool = (!x_ge_y) && y_ge_z;
    let mask_z1: bool = x_ge_z || y_ge_z;
    let nmask_x2: bool = x_ge_y || x_ge_z;
    let nmask_y2: bool = (!y_ge_z) && x_ge_y;
    let nmask_z2: bool = x_ge_z && y_ge_z;

    let dx3: T = dx0 + 0.5;
    let dy3: T = dy0 + 0.5;
    let dz3: T = dz0 + 0.5;
    let dx1: T = mask_x1 ? (dx3 - 1.0) : dx3;
    let dy1: T = mask_y1 ? (dy3 - 1.0) : dy3;
    let dz1: T = (!mask_z1) ? (dz3 - 1.0) : dz3;
    let dx2: T = (!nmask_x2) ? (dx0 + 1.0) : dx0;
    let dy2: T = nmask_y2 ? (dy0 + 1.0) : dy0;
    let dz2: T = nmask_z2 ? (dz0 + 1.0) : dz0;

    let fb0_raw: T = ((0.6 - dx0 * dx0) - dy0 * dy0) - dz0 * dz0;
    var f0: T = max(fb0_raw, 0.0);
    var f1: T = max(((0.6 - dx1 * dx1) - dy1 * dy1) - dz1 * dz1, 0.0);
    var f2: T = max(((0.6 - dx2 * dx2) - dy2 * dy2) - dz2 * dz2, 0.0);
    var f3: T = max(fb0_raw - (unskew_delta + 0.75), 0.0);
    f0 = (f0 * f0) * (f0 * f0);
    f1 = (f1 * f1) * (f1 * f1);
    f2 = (f2 * f2) * (f2 * f2);
    f3 = (f3 * f3) * (f3 * f3);

    var v: vec3<T> = warp_vec_contrib_common3d<T>(hash_primes3d(seed, x_primed, y_primed, z_primed), dx0, dy0, dz0, f0);
    v = v + warp_vec_contrib_common3d<T>(
        hash_primes3d(seed,
            mask_x1 ? (x_primed + PRIME_X) : x_primed,
            mask_y1 ? (y_primed + PRIME_Y) : y_primed,
            (!mask_z1) ? (z_primed + PRIME_Z) : z_primed),
        dx1, dy1, dz1, f1);
    v = v + warp_vec_contrib_common3d<T>(
        hash_primes3d(seed,
            nmask_x2 ? (x_primed + PRIME_X) : x_primed,
            (!nmask_y2) ? (y_primed + PRIME_Y) : y_primed,
            (!nmask_z2) ? (z_primed + PRIME_Z) : z_primed),
        dx2, dy2, dz2, f2);
    v = v + warp_vec_contrib_common3d<T>(
        hash_primes3d(seed, x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z),
        dx3, dy3, dz3, f3);
    return v * 16.2816314697265625;
}

pub fn domain_warp_simplex3d<T>(p: vec3<T>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec3<T> {
    return p + domain_warp_simplex_delta3d<T>(p * warp_freq, seed) * warp_amp;
}

pub fn domain_warp_supersimplex_delta2d<T>(p: vec2<T>, seed: i32 = 0) -> vec2<T> {
    let k_skew: T = 0.3660254180431365966796875;
    let k_unskew: T = -0.211324870586395263671875;
    let k_rad2: T = 0.666666686534881591796875;
    var so0: i32 = 0;
    var so1: i32 = 0;
    var x_skewed: T = 0.0;
    var y_skewed: T = 0.0;
    if is_f64<T>() {
        let sum: f64 = (p.x as f64) + (p.y as f64);
        let s2: f64 = (0.3660254180431365966796875 as f32) as f64;
        let sx: f64 = (p.x as f64) + s2 * sum;
        let sy: f64 = (p.y as f64) + s2 * sum;
        let cx: f64 = floor(sx);
        let cy: f64 = floor(sy);
        so0 = (cx as i64) as i32;
        so1 = (cy as i64) as i32;
        x_skewed = (sx - cx) as T;
        y_skewed = (sy - cy) as T;
    } else {
        let s: T = k_skew * (p.x + p.y);
        x_skewed = p.x + s;
        y_skewed = p.y + s;
    }
    let x_base: T = floor(x_skewed);
    let y_base: T = floor(y_skewed);
    let dx_skewed: T = x_skewed - x_base;
    let dy_skewed: T = y_skewed - y_base;
    let x_primed: i32 = ((x_base as i32) + so0) * PRIME_X;
    let y_primed: i32 = ((y_base as i32) + so1) * PRIME_Y;

    let fwd_xy: bool = (dx_skewed + dy_skewed) > 1.0;
    let bound_xy: T = fwd_xy ? -1.0 : 0.0;
    let fwd_x: bool = (dx_skewed * -2.0 + dy_skewed) < bound_xy;
    let fwd_y: bool = (dy_skewed * -2.0 + dx_skewed) < bound_xy;

    let unskew_d: T = k_unskew * (dx_skewed + dy_skewed);
    var dx_base: T = dx_skewed + unskew_d;
    var dy_base: T = dy_skewed + unskew_d;

    let fb0: T = (k_rad2 - dy_base * dy_base) - dx_base * dx_base;
    let f0: T = (fb0 * fb0) * (fb0 * fb0);
    var v: vec2<T> = warp_vec_contrib_simplex2d<T>(hash_primes2d(seed, x_primed, y_primed), dx_base, dy_base, f0);

    let fb1: T = unskew_d * -3.1547005176544189453125 + (fb0 - k_rad2);
    let f1: T = (fb1 * fb1) * (fb1 * fb1);
    v = v + warp_vec_contrib_simplex2d<T>(
        hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y),
        dx_base - (2.0 * k_unskew + 1.0),
        dy_base - (2.0 * k_unskew + 1.0),
        f1);

    let xy_delta: T = fwd_xy ? (k_unskew + 1.0) : (-k_unskew);
    dx_base -= xy_delta;
    dy_base -= xy_delta;

    let xp2_tmp: i32 = fwd_x ? (x_primed + (PRIME_X * 2)) : x_primed;
    let xp2: i32 = (!fwd_xy) ? (xp2_tmp - PRIME_X) : xp2_tmp;
    let yp2: i32 = fwd_xy ? (y_primed + PRIME_Y) : y_primed;
    let dx2: T = dx_base - (fwd_x ? (1.0 + 2.0 * k_unskew) : -1.0);
    let dy2: T = fwd_x ? (dy_base - 2.0 * k_unskew) : dy_base;
    let fb2: T = max((k_rad2 - dy2 * dy2) - dx2 * dx2, 0.0);
    v = v + warp_vec_contrib_simplex2d<T>(hash_primes2d(seed, xp2, yp2), dx2, dy2, (fb2 * fb2) * (fb2 * fb2));

    let xp3: i32 = fwd_xy ? (x_primed + PRIME_X) : x_primed;
    let yp3_tmp: i32 = fwd_y ? (y_primed + (PRIME_Y * 2)) : y_primed;
    let yp3: i32 = (!fwd_xy) ? (yp3_tmp - PRIME_Y) : yp3_tmp;
    let dx3: T = fwd_y ? (dx_base - 2.0 * k_unskew) : dx_base;
    let dy3: T = dy_base - (fwd_y ? (1.0 + 2.0 * k_unskew) : -1.0);
    let fb3: T = max((k_rad2 - dy3 * dy3) - dx3 * dx3, 0.0);
    v = v + warp_vec_contrib_simplex2d<T>(hash_primes2d(seed, xp3, yp3), dx3, dy3, (fb3 * fb3) * (fb3 * fb3));

    return v * 12.814453125;
}

pub fn domain_warp_supersimplex2d<T>(p: vec2<T>, seed: i32 = 0, warp_amp: T = 1.0, warp_freq: T = 0.5) -> vec2<T> {
    return p + domain_warp_supersimplex_delta2d<T>(p * warp_freq, seed) * warp_amp;
}
)BK";

constexpr const char* kStdFilterSource = R"BK(
module filter;

pub fn laplacian1d<T>(s: stencil<T, 1, 1>) -> T {
    return s.at(-1) + s.at(1) - 2.0 * s.at(0);
}

pub fn laplacian2d<T>(s: stencil<T, 2, 1>) -> T {
    return s.at(-1, 0) + s.at(1, 0) + s.at(0, -1) + s.at(0, 1) - 4.0 * s.at(0, 0);
}

pub fn laplacian3d<T>(s: stencil<T, 3, 1>) -> T {
    return s.at(-1, 0, 0) + s.at(1, 0, 0) +
           s.at(0, -1, 0) + s.at(0, 1, 0) +
           s.at(0, 0, -1) + s.at(0, 0, 1) -
           6.0 * s.at(0, 0, 0);
}

pub fn laplacian<T, const D: i32>(s: stencil<T, D, 1>) -> T {
    if D == 1 {
        return laplacian1d<T>(s);
    } else if D == 2 {
        return laplacian2d<T>(s);
    } else {
        return laplacian3d<T>(s);
    }
}

pub fn gradient2d<T>(s: stencil<T, 2, 1>, inv_step: T) -> vec2<T> {
    let gx: T = (s.at(1, 0) - s.at(-1, 0)) * (0.5 * inv_step);
    let gy: T = (s.at(0, 1) - s.at(0, -1)) * (0.5 * inv_step);
    return vec2<T>(gx, gy);
}

pub fn gradient<T, const D: i32>(s: stencil<T, D, 1>, inv_step: T) -> vec<T, D> {
    if D == 1 {
        return (s.at(1) - s.at(-1)) * (0.5 * inv_step);
    } else if D == 2 {
        return gradient2d<T>(s, inv_step);
    } else {
        let gx: T = (s.at(1, 0, 0) - s.at(-1, 0, 0)) * (0.5 * inv_step);
        let gy: T = (s.at(0, 1, 0) - s.at(0, -1, 0)) * (0.5 * inv_step);
        let gz: T = (s.at(0, 0, 1) - s.at(0, 0, -1)) * (0.5 * inv_step);
        return vec3<T>(gx, gy, gz);
    }
}

pub fn normal2d<T>(s: stencil<T, 2, 1>, cell_size: T) -> vec3<T> {
    let inv_two_step: T = 0.5 / cell_size;
    let dhdx: T = (s.at(1, 0) - s.at(-1, 0)) * inv_two_step;
    let dhdz: T = (s.at(0, 1) - s.at(0, -1)) * inv_two_step;
    return normalize(vec3<T>(-dhdx, 1.0, -dhdz));
}

pub fn tent2d<T>(s: stencil<T, 2, 1>) -> T {
    let corners: T = s.at(-1, -1) + s.at(1, -1) + s.at(-1, 1) + s.at(1, 1);
    let edges: T = (s.at(0, -1) + s.at(-1, 0) + s.at(1, 0) + s.at(0, 1)) * 2.0;
    let center: T = s.at(0, 0) * 4.0;
    return (corners + edges + center) * 0.0625;
}

pub fn sobel2d<T>(s: stencil<T, 2, 1>) -> vec2<T> {
    let gx: T = (s.at(1, -1) + 2.0 * s.at(1, 0) + s.at(1, 1)) -
                (s.at(-1, -1) + 2.0 * s.at(-1, 0) + s.at(-1, 1));
    let gy: T = (s.at(-1, 1) + 2.0 * s.at(0, 1) + s.at(1, 1)) -
                (s.at(-1, -1) + 2.0 * s.at(0, -1) + s.at(1, -1));
    return vec2<T>(gx, gy);
}

pub fn diffuse<T, const D: i32>(s: stencil<T, D, 1>, rate: T) -> T {
    if D == 1 {
        return s.at(0) + rate * laplacian1d<T>(s);
    } else if D == 2 {
        return s.at(0, 0) + rate * laplacian2d<T>(s);
    } else {
        return s.at(0, 0, 0) + rate * laplacian3d<T>(s);
    }
}

pub fn box_blur<T, const D: i32, const R: i32>(s: stencil<T, D, R>) -> T {
    var sum: T = 0.0;
    let span: i32 = 2 * R + 1;
    if D == 1 {
        @unroll for ix in -R..=R {
            sum += s.at(ix);
        }
        return sum / (span as T);
    } else if D == 2 {
        @unroll for iy in -R..=R {
            @unroll for ix in -R..=R {
                sum += s.at(ix, iy);
            }
        }
        let count: i32 = span * span;
        return sum / (count as T);
    } else {
        @unroll for iz in -R..=R {
            @unroll for iy in -R..=R {
                @unroll for ix in -R..=R {
                    sum += s.at(ix, iy, iz);
                }
            }
        }
        let count: i32 = span * span * span;
        return sum / (count as T);
    }
}

pub fn box_blur1d<T, const R: i32>(s: stencil<T, 1, R>) -> T {
    return box_blur<T, 1, R>(s);
}

pub fn box_blur2d<T, const R: i32>(s: stencil<T, 2, R>) -> T {
    return box_blur<T, 2, R>(s);
}

pub fn box_blur3d<T, const R: i32>(s: stencil<T, 3, R>) -> T {
    return box_blur<T, 3, R>(s);
}
)BK";

constexpr const char* kStdSphereSource = R"BK(
module sphere;

// Maps a cube face (0..5: +X, -X, +Y, -Y, +Z, -Z) and [-1, 1] UV coordinates
// to a unit direction vector on the sphere, using Everitt's tangent-warp reduction.
pub fn cube_to_unit<T>(face: i32, u: T, v: T) -> vec3<T> {
    var dir: vec3<T> = vec3<T>(0.0, 0.0, 1.0);
    if face == 0 {
        dir = vec3<T>(1.0, v, -u);
    } else if face == 1 {
        dir = vec3<T>(-1.0, v, u);
    } else if face == 2 {
        dir = vec3<T>(u, 1.0, -v);
    } else if face == 3 {
        dir = vec3<T>(u, -1.0, v);
    } else if face == 4 {
        dir = vec3<T>(u, v, 1.0);
    } else {
        dir = vec3<T>(-u, v, -1.0);
    }
    return normalize(dir);
}

pub fn cube_to_sphere<T>(face: i32, u: T, v: T) -> vec3<T> {
    return cube_to_unit<T>(face, u, v);
}

// Computes camera-relative or cell-anchor-relative f32 offset from f64 planetary positions.
pub fn relative_pos(world_pos: vec3<f64>, anchor_origin: vec3<f64>) -> vec3<f32> {
    let delta: vec3<f64> = world_pos - anchor_origin;
    return vec3<f32>(delta.x as f32, delta.y as f32, delta.z as f32);
}

// Converts latitude [-pi/2, pi/2] and longitude [-pi, pi] in radians to a unit sphere direction.
pub fn latlon_to_unit<T>(lat: T, lon: T) -> vec3<T> {
    let clat: T = cos(lat);
    return vec3<T>(clat * cos(lon), sin(lat), clat * sin(lon));
}
)BK";

constexpr const char* kStdColorSource = R"BK(
module color;

pub fn luminance<T>(rgb: vec3<T>) -> T {
    return rgb.x * 0.2126 + rgb.y * 0.7152 + rgb.z * 0.0722;
}

pub fn aces_tonemap<T>(x: vec3<T>) -> vec3<T> {
    let a: T = 2.51;
    let b: T = 0.03;
    let c: T = 2.43;
    let d: T = 0.59;
    let e: T = 0.14;
    let num: vec3<T> = x * (x * a + vec3<T>(b));
    let den: vec3<T> = x * (x * c + vec3<T>(d)) + vec3<T>(e);
    return clamp(num / den, vec3<T>(0.0), vec3<T>(1.0));
}
)BK";

} // namespace

ModuleRegistry::ModuleRegistry() {
    register_builtin_modules();
}

void ModuleRegistry::register_builtin_modules() {
    register_module("math", kStdMathSource, "<builtin:math.bk>");
    aliases_["std::math"] = "math";

    std::string noise_src;
    noise_src.reserve(64000);
    noise_src += kStdNoisePart1;
    noise_src += kStdNoisePart2;
    noise_src += kStdNoisePart3;
    noise_src += kStdNoisePart4a;
    noise_src += kStdNoisePart4b;
    noise_src += kStdNoisePart5;
    noise_src += kStdNoisePart6;
    noise_src += kStdNoisePart7;
    noise_src += kStdNoisePart8;
    noise_src += kStdNoisePart9;
    noise_src += kStdNoisePart10;

    register_module("noise", std::move(noise_src), "<builtin:noise.bk>");
    aliases_["std::noise"] = "noise";
    aliases_["fastnoise"] = "noise";
    aliases_["std::fastnoise"] = "noise";

    register_module("filter", kStdFilterSource, "<builtin:filter.bk>");
    aliases_["std::filter"] = "filter";

    register_module("sphere", kStdSphereSource, "<builtin:sphere.bk>");
    aliases_["std::sphere"] = "sphere";

    register_module("color", kStdColorSource, "<builtin:color.bk>");
    aliases_["std::color"] = "color";
}

void ModuleRegistry::register_module(std::string name, std::string source, std::string virtual_path) {
    if (virtual_path.empty()) {
        virtual_path = name + ".bk";
    }
    Entry e;
    e.name = name;
    e.path = std::move(virtual_path);
    e.source = std::move(source);
    modules_[name] = std::move(e);
}

void ModuleRegistry::add_search_path(std::filesystem::path dir) {
    search_paths_.push_back(std::move(dir));
}

std::string ModuleRegistry::format_module_path(const std::vector<std::string>& path) {
    std::string out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i > 0) out += "::";
        out += path[i];
    }
    return out;
}

ModuleRegistry::Entry* ModuleRegistry::find_or_discover(
    std::string_view key,
    const std::filesystem::path& importing_dir) {
    std::string k(key);
    auto alias_it = aliases_.find(k);
    if (alias_it != aliases_.end()) {
        k = alias_it->second;
    }
    auto it = modules_.find(k);
    if (it != modules_.end()) {
        return &it->second;
    }

    // Convert `foo::bar` to `foo/bar.bk` when searching on disk
    std::string rel_path = k;
    for (size_t pos = 0; (pos = rel_path.find("::", pos)) != std::string::npos;) {
        rel_path.replace(pos, 2, "/");
        pos += 1;
    }
    if (!rel_path.ends_with(".bk")) {
        rel_path += ".bk";
    }

    std::vector<std::filesystem::path> dirs;
    if (!importing_dir.empty()) dirs.push_back(importing_dir);
    for (const auto& sp : search_paths_) dirs.push_back(sp);

    for (const auto& dir : dirs) {
        std::filesystem::path candidate = dir / rel_path;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && std::filesystem::is_regular_file(candidate, ec)) {
            std::ifstream ifs(candidate, std::ios::binary);
            if (ifs) {
                std::ostringstream oss;
                oss << ifs.rdbuf();
                std::string canon = candidate.stem().string();
                register_module(k, oss.str(), candidate.generic_string());
                return &modules_[k];
            }
        }
    }
    return nullptr;
}

bool ModuleRegistry::has_module(std::string_view name) const {
    std::string k(name);
    if (aliases_.count(k)) k = aliases_.at(k);
    return modules_.count(k) > 0;
}

const ModuleAst* ModuleRegistry::load_module(
    std::string_view name_or_path,
    const SourceSpan& import_site,
    DiagnosticBag& diags,
    const std::filesystem::path& importing_file_dir) {
    Entry* entry = find_or_discover(name_or_path, importing_file_dir);
    if (!entry) {
        diags.error(import_site, "module '" + std::string(name_or_path) + "' not found");
        return nullptr;
    }
    if (entry->loading) {
        diags.error(import_site, "circular module import involving '" + entry->name + "'");
        return nullptr;
    }
    if (entry->failed) {
        return nullptr;
    }
    if (entry->ast) {
        return entry->ast.get();
    }

    entry->loading = true;
    DiagnosticBag mod_diags;
    entry->ast = parse_source(entry->source, entry->path, mod_diags);
    entry->loading = false;

    if (mod_diags.has_errors() || !entry->ast) {
        entry->failed = true;
        diags.merge_from(mod_diags);
        return nullptr;
    }
    if (entry->ast->module_name.empty()) {
        entry->ast->module_name = entry->name;
    }
    return entry->ast.get();
}

} // namespace brokernel
