#include "stdlib_noise_parts.h"

namespace brokernel {
namespace {

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

} // namespace

void append_std_noise_fractal_warp(std::string& out) {
    out += kStdNoisePart8;
    out += kStdNoisePart9;
    out += kStdNoisePart10;
}

} // namespace brokernel
