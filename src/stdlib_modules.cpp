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

pub fn saturate_val<T>(x: T) -> T {
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

constexpr const char* kStdNoiseSource = R"BK(
module noise;

// Deterministic 24-bit integer hash in [0, 16777215], bit-identical on CPU and GPU.
pub fn hash_bits1(x: i32, seed: i32) -> i32 {
    var h: i32 = x * 374761393 + seed * 668265263 + 1013904223;
    h = (h ^ ((h >> 13) & 524287)) * 1274126177;
    return (h ^ ((h >> 16) & 65535)) & 16777215;
}

pub fn hash_bits2(x: i32, y: i32, seed: i32) -> i32 {
    var h: i32 = x * 374761393 + y * 668265263 + seed * 1442695041 + 1013904223;
    h = (h ^ ((h >> 13) & 524287)) * 1274126177;
    return (h ^ ((h >> 16) & 65535)) & 16777215;
}

pub fn hash_bits3(x: i32, y: i32, z: i32, seed: i32) -> i32 {
    var h: i32 = x * 374761393 + y * 668265263 + z * 2147483647 + seed * 1442695041 + 1013904223;
    h = (h ^ ((h >> 13) & 524287)) * 1274126177;
    return (h ^ ((h >> 16) & 65535)) & 16777215;
}

// Signed unit hash in [-1, 1] for lattice coordinates.
pub fn lattice1d<T>(ix: i32, seed: i32) -> T {
    let bits: i32 = hash_bits1(ix, seed);
    return (bits as T) * (2.0 / 16777215.0) - 1.0;
}

pub fn lattice2d<T>(ix: i32, iy: i32, seed: i32) -> T {
    let bits: i32 = hash_bits2(ix, iy, seed);
    return (bits as T) * (2.0 / 16777215.0) - 1.0;
}

pub fn lattice3d<T>(ix: i32, iy: i32, iz: i32, seed: i32) -> T {
    let bits: i32 = hash_bits3(ix, iy, iz, seed);
    return (bits as T) * (2.0 / 16777215.0) - 1.0;
}

pub fn fade<T>(t: T) -> T {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

pub fn value1d<T>(x: T, seed: i32 = 0) -> T {
    let fx: T = floor(x);
    let ix: i32 = fx as i32;
    let tx: T = fade<T>(x - fx);
    let v0: T = lattice1d<T>(ix, seed);
    let v1: T = lattice1d<T>(ix + 1, seed);
    return mix(v0, v1, tx);
}

pub fn value2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let fp: vec2<T> = floor(p);
    let ix: i32 = fp.x as i32;
    let iy: i32 = fp.y as i32;
    let f: vec2<T> = p - fp;
    let ux: T = fade<T>(f.x);
    let uy: T = fade<T>(f.y);

    let v00: T = lattice2d<T>(ix, iy, seed);
    let v10: T = lattice2d<T>(ix + 1, iy, seed);
    let v01: T = lattice2d<T>(ix, iy + 1, seed);
    let v11: T = lattice2d<T>(ix + 1, iy + 1, seed);

    let vx0: T = mix(v00, v10, ux);
    let vx1: T = mix(v01, v11, ux);
    return mix(vx0, vx1, uy);
}

pub fn value3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let fp: vec3<T> = floor(p);
    let ix: i32 = fp.x as i32;
    let iy: i32 = fp.y as i32;
    let iz: i32 = fp.z as i32;
    let f: vec3<T> = p - fp;
    let ux: T = fade<T>(f.x);
    let uy: T = fade<T>(f.y);
    let uz: T = fade<T>(f.z);

    let v000: T = lattice3d<T>(ix, iy, iz, seed);
    let v100: T = lattice3d<T>(ix + 1, iy, iz, seed);
    let v010: T = lattice3d<T>(ix, iy + 1, iz, seed);
    let v110: T = lattice3d<T>(ix + 1, iy + 1, iz, seed);
    let v001: T = lattice3d<T>(ix, iy, iz + 1, seed);
    let v101: T = lattice3d<T>(ix + 1, iy, iz + 1, seed);
    let v011: T = lattice3d<T>(ix, iy + 1, iz + 1, seed);
    let v111: T = lattice3d<T>(ix + 1, iy + 1, iz + 1, seed);

    let vx00: T = mix(v000, v100, ux);
    let vx10: T = mix(v010, v110, ux);
    let vx01: T = mix(v001, v101, ux);
    let vx11: T = mix(v011, v111, ux);

    let vxy0: T = mix(vx00, vx10, uy);
    let vxy1: T = mix(vx01, vx11, uy);
    return mix(vxy0, vxy1, uz);
}

// Generic value noise over scalar type T and dimension D (1, 2, or 3).
pub fn value<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return value2d<T>(p, seed);
    } else {
        return value3d<T>(p, seed);
    }
}

pub fn grad_dot2d<T>(ix: i32, iy: i32, dx: T, dy: T, seed: i32) -> T {
    let h: i32 = hash_bits2(ix, iy, seed) & 7;
    let u: T = (h < 4) ? dx : dy;
    let v: T = (h < 4) ? dy : dx;
    let su: T = ((h & 1) == 0) ? u : -u;
    let sv: T = ((h & 2) == 0) ? v : -v;
    return (su + sv) * 0.70710678118654752440;
}

pub fn grad_dot3d<T>(ix: i32, iy: i32, iz: i32, dx: T, dy: T, dz: T, seed: i32) -> T {
    let h: i32 = hash_bits3(ix, iy, iz, seed) & 15;
    let u: T = (h < 8) ? dx : dy;
    let v: T = (h < 4) ? dy : ((h == 12 || h == 14) ? dx : dz);
    let su: T = ((h & 1) == 0) ? u : -u;
    let sv: T = ((h & 2) == 0) ? v : -v;
    return (su + sv) * 0.70710678118654752440;
}

pub fn perlin2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let fp: vec2<T> = floor(p);
    let ix: i32 = fp.x as i32;
    let iy: i32 = fp.y as i32;
    let fx: T = p.x - fp.x;
    let fy: T = p.y - fp.y;
    let ux: T = fade<T>(fx);
    let uy: T = fade<T>(fy);

    let g00: T = grad_dot2d<T>(ix, iy, fx, fy, seed);
    let g10: T = grad_dot2d<T>(ix + 1, iy, fx - 1.0, fy, seed);
    let g01: T = grad_dot2d<T>(ix, iy + 1, fx, fy - 1.0, seed);
    let g11: T = grad_dot2d<T>(ix + 1, iy + 1, fx - 1.0, fy - 1.0, seed);

    return mix(mix(g00, g10, ux), mix(g01, g11, ux), uy);
}

pub fn perlin3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let fp: vec3<T> = floor(p);
    let ix: i32 = fp.x as i32;
    let iy: i32 = fp.y as i32;
    let iz: i32 = fp.z as i32;
    let fx: T = p.x - fp.x;
    let fy: T = p.y - fp.y;
    let fz: T = p.z - fp.z;
    let ux: T = fade<T>(fx);
    let uy: T = fade<T>(fy);
    let uz: T = fade<T>(fz);

    let g000: T = grad_dot3d<T>(ix, iy, iz, fx, fy, fz, seed);
    let g100: T = grad_dot3d<T>(ix + 1, iy, iz, fx - 1.0, fy, fz, seed);
    let g010: T = grad_dot3d<T>(ix, iy + 1, iz, fx, fy - 1.0, fz, seed);
    let g110: T = grad_dot3d<T>(ix + 1, iy + 1, iz, fx - 1.0, fy - 1.0, fz, seed);
    let g001: T = grad_dot3d<T>(ix, iy, iz + 1, fx, fy, fz - 1.0, seed);
    let g101: T = grad_dot3d<T>(ix + 1, iy, iz + 1, fx - 1.0, fy, fz - 1.0, seed);
    let g011: T = grad_dot3d<T>(ix, iy + 1, iz + 1, fx, fy - 1.0, fz - 1.0, seed);
    let g111: T = grad_dot3d<T>(ix + 1, iy + 1, iz + 1, fx - 1.0, fy - 1.0, fz - 1.0, seed);

    let gx00: T = mix(g000, g100, ux);
    let gx10: T = mix(g010, g110, ux);
    let gx01: T = mix(g001, g101, ux);
    let gx11: T = mix(g011, g111, ux);
    return mix(mix(gx00, gx10, uy), mix(gx01, gx11, uy), uz);
}

pub fn perlin<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return perlin2d<T>(p, seed);
    } else {
        return perlin3d<T>(p, seed);
    }
}

pub fn simplex2d<T>(p: vec2<T>, seed: i32 = 0) -> T {
    let F2: T = 0.36602540378443864676; // 0.5 * (sqrt(3) - 1)
    let G2: T = 0.21132486540518711775; // (3 - sqrt(3)) / 6
    let s: T = (p.x + p.y) * F2;
    let i: T = floor(p.x + s);
    let j: T = floor(p.y + s);
    let t: T = (i + j) * G2;
    let x0: T = p.x - (i - t);
    let y0: T = p.y - (j - t);

    let i1: i32 = (x0 > y0) ? 1 : 0;
    let j1: i32 = (x0 > y0) ? 0 : 1;

    let x1: T = x0 - (i1 as T) + G2;
    let y1: T = y0 - (j1 as T) + G2;
    let x2: T = x0 - 1.0 + 2.0 * G2;
    let y2: T = y0 - 1.0 + 2.0 * G2;

    let ii: i32 = i as i32;
    let jj: i32 = j as i32;

    let t0: T = max(0.5 - x0 * x0 - y0 * y0, 0.0);
    let t1: T = max(0.5 - x1 * x1 - y1 * y1, 0.0);
    let t2: T = max(0.5 - x2 * x2 - y2 * y2, 0.0);

    let n0: T = (t0 * t0) * (t0 * t0) * grad_dot2d<T>(ii, jj, x0, y0, seed);
    let n1: T = (t1 * t1) * (t1 * t1) * grad_dot2d<T>(ii + i1, jj + j1, x1, y1, seed);
    let n2: T = (t2 * t2) * (t2 * t2) * grad_dot2d<T>(ii + 1, jj + 1, x2, y2, seed);

    return 45.0 * (n0 + n1 + n2);
}

pub fn simplex3d<T>(p: vec3<T>, seed: i32 = 0) -> T {
    let F3: T = 0.33333333333333333333; // 1 / 3
    let G3: T = 0.16666666666666666667; // 1 / 6
    let s: T = (p.x + p.y + p.z) * F3;
    let i: T = floor(p.x + s);
    let j: T = floor(p.y + s);
    let k: T = floor(p.z + s);
    let t: T = (i + j + k) * G3;
    let x0: T = p.x - (i - t);
    let y0: T = p.y - (j - t);
    let z0: T = p.z - (k - t);

    // Which of the six tetrahedra of the skewed cube: the order of x0, y0, z0.
    var i1: i32 = 0;
    var j1: i32 = 0;
    var k1: i32 = 0;
    var i2: i32 = 0;
    var j2: i32 = 0;
    var k2: i32 = 0;
    if x0 >= y0 {
        if y0 >= z0 {
            i1 = 1; i2 = 1; j2 = 1;
        } else if x0 >= z0 {
            i1 = 1; i2 = 1; k2 = 1;
        } else {
            k1 = 1; i2 = 1; k2 = 1;
        }
    } else {
        if y0 < z0 {
            k1 = 1; j2 = 1; k2 = 1;
        } else if x0 < z0 {
            j1 = 1; j2 = 1; k2 = 1;
        } else {
            j1 = 1; i2 = 1; j2 = 1;
        }
    }

    let x1: T = x0 - (i1 as T) + G3;
    let y1: T = y0 - (j1 as T) + G3;
    let z1: T = z0 - (k1 as T) + G3;
    let x2: T = x0 - (i2 as T) + 2.0 * G3;
    let y2: T = y0 - (j2 as T) + 2.0 * G3;
    let z2: T = z0 - (k2 as T) + 2.0 * G3;
    let x3: T = x0 - 1.0 + 3.0 * G3;
    let y3: T = y0 - 1.0 + 3.0 * G3;
    let z3: T = z0 - 1.0 + 3.0 * G3;

    let ii: i32 = i as i32;
    let jj: i32 = j as i32;
    let kk: i32 = k as i32;

    let t0: T = max(0.6 - x0 * x0 - y0 * y0 - z0 * z0, 0.0);
    let t1: T = max(0.6 - x1 * x1 - y1 * y1 - z1 * z1, 0.0);
    let t2: T = max(0.6 - x2 * x2 - y2 * y2 - z2 * z2, 0.0);
    let t3: T = max(0.6 - x3 * x3 - y3 * y3 - z3 * z3, 0.0);

    let n0: T = (t0 * t0) * (t0 * t0) * grad_dot3d<T>(ii, jj, kk, x0, y0, z0, seed);
    let n1: T = (t1 * t1) * (t1 * t1) * grad_dot3d<T>(ii + i1, jj + j1, kk + k1, x1, y1, z1, seed);
    let n2: T = (t2 * t2) * (t2 * t2) * grad_dot3d<T>(ii + i2, jj + j2, kk + k2, x2, y2, z2, seed);
    let n3: T = (t3 * t3) * (t3 * t3) * grad_dot3d<T>(ii + 1, jj + 1, kk + 1, x3, y3, z3, seed);

    // Gustavson's 32 is for gradients of length sqrt(2); grad_dot3d's are unit.
    return 45.254833995939045 * (n0 + n1 + n2 + n3);
}

pub fn simplex<T, const D: i32>(p: vec<T, D>, seed: i32 = 0) -> T {
    if D == 1 {
        return value1d<T>(p, seed);
    } else if D == 2 {
        return simplex2d<T>(p, seed);
    } else {
        return simplex3d<T>(p, seed);
    }
}

pub fn fbm<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    seed: i32 = 0
) -> T {
    var sum: T = 0.0;
    var amp: T = 0.5;
    var freq_p: vec<T, D> = p;
    for (var oct: i32 = 0; oct < octaves; oct += 1) {
        sum += amp * perlin<T, D>(freq_p, seed + oct * 101);
        freq_p = freq_p * lacunarity;
        amp *= gain;
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
    var sum: T = 0.0;
    var amp: T = 0.5;
    var weight: T = 1.0;
    var freq_p: vec<T, D> = p;
    for (var oct: i32 = 0; oct < octaves; oct += 1) {
        let n: T = 1.0 - abs(perlin<T, D>(freq_p, seed + oct * 131));
        let signal: T = n * n * weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        sum += signal * amp;
        freq_p = freq_p * lacunarity;
        amp *= gain;
    }
    return sum;
}

pub fn ridged_fbm<T, const D: i32>(
    p: vec<T, D>,
    octaves: i32 = 4,
    lacunarity: T = 2.0,
    gain: T = 0.5,
    seed: i32 = 0
) -> T {
    return ridged<T, D>(p, octaves, lacunarity, gain, seed);
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

pub fn tonemap_aces<T>(x: vec3<T>) -> vec3<T> {
    let a: T = 2.51;
    let b: T = 0.03;
    let c: T = 2.43;
    let d: T = 0.59;
    let e: T = 0.14;
    let num: vec3<T> = x * (x * a + vec3<T>(b));
    let den: vec3<T> = x * (x * c + vec3<T>(d)) + vec3<T>(e);
    return clamp(num / den, vec3<T>(0.0), vec3<T>(1.0));
}

pub fn aces_tonemap<T>(x: vec3<T>) -> vec3<T> {
    return tonemap_aces<T>(x);
}
)BK";

} // namespace

ModuleRegistry::ModuleRegistry() {
    register_builtin_modules();
}

void ModuleRegistry::register_builtin_modules() {
    register_module("math", kStdMathSource, "<builtin:math.bk>");
    aliases_["std::math"] = "math";

    register_module("noise", kStdNoiseSource, "<builtin:noise.bk>");
    aliases_["std::noise"] = "noise";

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
