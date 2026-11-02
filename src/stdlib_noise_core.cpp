#include "stdlib_noise_parts.h"

namespace brokernel {
namespace {

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

} // namespace

void append_std_noise_core(std::string& out) {
    out += kStdNoisePart1;
    out += kStdNoisePart2;
    out += kStdNoisePart3;
    out += kStdNoisePart4a;
}

} // namespace brokernel
