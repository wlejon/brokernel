#include "bench_hillclimb.h"

// The GLSL baselines: FastNoise2's nodes in f32, in std::noise's order of operations
// (Value, Perlin, the fractals, Simplex, SuperSimplex, White, PingPong, Cellular, the
// gradient warps). Split into literals under MSVC's 16 KB limit.
const char* kGlslNoiseHeader = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf {
    float v[];
};

const int PRIME_X = -141048377;
const int PRIME_Y = 1812335753;
const int PRIME_Z = 1180684367;
const int PRIME_W = -140107143;
const int HASH_A  = -1210014219;

int hash_primes2d(int seed, int x, int y) {
    int h = (seed ^ x ^ y) * HASH_A;
    return (h >> 15) ^ h;
}
int hash_primes3d(int seed, int x, int y, int z) {
    int h = (seed ^ x ^ y ^ z) * HASH_A;
    return (h >> 15) ^ h;
}
int hash_primes4d(int seed, int x, int y, int z, int w) {
    int h = (seed ^ x ^ y ^ z ^ w) * HASH_A;
    return (h >> 15) ^ h;
}

float get_value2d(int seed, int x, int y) {
    int h = seed ^ x ^ y;
    return float(h * (h * HASH_A));
}
float get_value3d(int seed, int x, int y, int z) {
    int h = seed ^ x ^ y ^ z;
    return float(h * (h * HASH_A));
}
float get_value4d(int seed, int x, int y, int z, int w) {
    int h = seed ^ x ^ y ^ z ^ w;
    return float(h * (h * HASH_A));
}

float interp_hermite(float t) { return t * t * (3.0 - 2.0 * t); }
float interp_quintic(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
float scale_output(float val, float min_b, float max_b) {
    return (1.0 / (max_b - min_b)) * 2.0 * (val - min_b) - 1.0;
}
float xor_sign(float val, int sign_bit) {
    return intBitsToFloat(floatBitsToInt(val) ^ (sign_bit & 0x80000000));
}
float grad_dot_perlin2d(int hash, float fx, float fy) {
    float sx = xor_sign(fx, hash << 31);
    float sy = xor_sign(fy, (hash >> 1) << 31);
    bool sel = (hash & 4) != 0;
    float u = sel ? sy : sx;
    float v = sel ? sx : sy;
    return 2.414213657379150390625 * u + v;
}
float grad_dot_common3d(int hash, float fx, float fy, float fz) {
    int h13 = hash & 13;
    float u = (h13 > 7) ? fy : fx;
    float v = (h13 == 12) ? fx : fz;
    v = (h13 < 2) ? fy : v;
    return xor_sign(u, hash << 31) + xor_sign(v, (hash >> 1) << 31);
}
float grad_dot_perlin4d(int hash, float fx, float fy, float fz, float fw) {
    int p = hash & 24;
    float a = (p > 0) ? fx : fy;
    float b = ((hash & 16) != 0) ? fy : fz;
    float c = (p > 16) ? fz : fw;
    return xor_sign(a, hash << 31) + xor_sign(b, (hash >> 1) << 31) + xor_sign(c, (hash >> 2) << 31);
}

float fn_value2d(vec2 p, int seed) {
    vec2 s = floor(p);
    ivec2 i0 = ivec2(s) * ivec2(PRIME_X, PRIME_Y);
    ivec2 i1 = i0 + ivec2(PRIME_X, PRIME_Y);
    vec2 f = vec2(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y));
    float v0 = mix(get_value2d(seed, i0.x, i0.y), get_value2d(seed, i1.x, i0.y), f.x);
    float v1 = mix(get_value2d(seed, i0.x, i1.y), get_value2d(seed, i1.x, i1.y), f.x);
    return scale_output(mix(v0, v1, f.y), -2147483648.0, 2147483648.0);
}
float fn_value3d(vec3 p, int seed) {
    vec3 s = floor(p);
    ivec3 i0 = ivec3(s) * ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    ivec3 i1 = i0 + ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    vec3 f = vec3(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z));
    float vy0 = mix(mix(get_value3d(seed, i0.x, i0.y, i0.z), get_value3d(seed, i1.x, i0.y, i0.z), f.x),
                    mix(get_value3d(seed, i0.x, i1.y, i0.z), get_value3d(seed, i1.x, i1.y, i0.z), f.x), f.y);
    float vy1 = mix(mix(get_value3d(seed, i0.x, i0.y, i1.z), get_value3d(seed, i1.x, i0.y, i1.z), f.x),
                    mix(get_value3d(seed, i0.x, i1.y, i1.z), get_value3d(seed, i1.x, i1.y, i1.z), f.x), f.y);
    return scale_output(mix(vy0, vy1, f.z), -2147483648.0, 2147483648.0);
}
float fn_value4d(vec4 p, int seed) {
    vec4 s = floor(p);
    ivec4 i0 = ivec4(s) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    ivec4 i1 = i0 + ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 f = vec4(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z), interp_hermite(p.w - s.w));
    float vz00 = mix(mix(get_value4d(seed, i0.x, i0.y, i0.z, i0.w), get_value4d(seed, i1.x, i0.y, i0.z, i0.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i0.z, i0.w), get_value4d(seed, i1.x, i1.y, i0.z, i0.w), f.x), f.y);
    float vz10 = mix(mix(get_value4d(seed, i0.x, i0.y, i1.z, i0.w), get_value4d(seed, i1.x, i0.y, i1.z, i0.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i1.z, i0.w), get_value4d(seed, i1.x, i1.y, i1.z, i0.w), f.x), f.y);
    float vw0 = mix(vz00, vz10, f.z);
    float vz01 = mix(mix(get_value4d(seed, i0.x, i0.y, i0.z, i1.w), get_value4d(seed, i1.x, i0.y, i0.z, i1.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i0.z, i1.w), get_value4d(seed, i1.x, i1.y, i0.z, i1.w), f.x), f.y);
    float vz11 = mix(mix(get_value4d(seed, i0.x, i0.y, i1.z, i1.w), get_value4d(seed, i1.x, i0.y, i1.z, i1.w), f.x),
                     mix(get_value4d(seed, i0.x, i1.y, i1.z, i1.w), get_value4d(seed, i1.x, i1.y, i1.z, i1.w), f.x), f.y);
    float vw1 = mix(vz01, vz11, f.z);
    return scale_output(mix(vw0, vw1, f.w), -2147483648.0, 2147483648.0);
}

float fn_perlin2d(vec2 p, int seed) {
    vec2 s = floor(p);
    ivec2 i0 = ivec2(s) * ivec2(PRIME_X, PRIME_Y);
    ivec2 i1 = i0 + ivec2(PRIME_X, PRIME_Y);
    vec2 f0 = p - s;
    vec2 f1 = f0 - 1.0;
    vec2 u = vec2(interp_quintic(f0.x), interp_quintic(f0.y));
    float g0 = mix(grad_dot_perlin2d(hash_primes2d(seed, i0.x, i0.y), f0.x, f0.y),
                   grad_dot_perlin2d(hash_primes2d(seed, i1.x, i0.y), f1.x, f0.y), u.x);
    float g1 = mix(grad_dot_perlin2d(hash_primes2d(seed, i0.x, i1.y), f0.x, f1.y),
                   grad_dot_perlin2d(hash_primes2d(seed, i1.x, i1.y), f1.x, f1.y), u.x);
    return scale_output(mix(g0, g1, u.y), -1.726796627044677734375, 1.726796627044677734375);
}
float fn_perlin3d(vec3 p, int seed) {
    vec3 s = floor(p);
    ivec3 i0 = ivec3(s) * ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    ivec3 i1 = i0 + ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    vec3 f0 = p - s;
    vec3 f1 = f0 - 1.0;
    vec3 u = vec3(interp_quintic(f0.x), interp_quintic(f0.y), interp_quintic(f0.z));
    float gy0 = mix(mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i0.y, i0.z), f0.x, f0.y, f0.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i0.y, i0.z), f1.x, f0.y, f0.z), u.x),
                    mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i1.y, i0.z), f0.x, f1.y, f0.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i1.y, i0.z), f1.x, f1.y, f0.z), u.x), u.y);
    float gy1 = mix(mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i0.y, i1.z), f0.x, f0.y, f1.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i0.y, i1.z), f1.x, f0.y, f1.z), u.x),
                    mix(grad_dot_common3d(hash_primes3d(seed, i0.x, i1.y, i1.z), f0.x, f1.y, f1.z),
                        grad_dot_common3d(hash_primes3d(seed, i1.x, i1.y, i1.z), f1.x, f1.y, f1.z), u.x), u.y);
    return scale_output(mix(gy0, gy1, u.z), -1.0363423824310302734375, 1.0363423824310302734375);
}
float fn_perlin4d(vec4 p, int seed) {
    vec4 s = floor(p);
    ivec4 i0 = ivec4(s) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    ivec4 i1 = i0 + ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 f0 = p - s;
    vec4 f1 = f0 - 1.0;
    vec4 u = vec4(interp_quintic(f0.x), interp_quintic(f0.y), interp_quintic(f0.z), interp_quintic(f0.w));
    float gz00 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i0.z, i0.w), f0.x, f0.y, f0.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i0.z, i0.w), f1.x, f0.y, f0.z, f0.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i0.z, i0.w), f0.x, f1.y, f0.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i0.z, i0.w), f1.x, f1.y, f0.z, f0.w), u.x), u.y);
    float gz10 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i1.z, i0.w), f0.x, f0.y, f1.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i1.z, i0.w), f1.x, f0.y, f1.z, f0.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i1.z, i0.w), f0.x, f1.y, f1.z, f0.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i1.z, i0.w), f1.x, f1.y, f1.z, f0.w), u.x), u.y);
    float gw0 = mix(gz00, gz10, u.z);
    float gz01 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i0.z, i1.w), f0.x, f0.y, f0.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i0.z, i1.w), f1.x, f0.y, f0.z, f1.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i0.z, i1.w), f0.x, f1.y, f0.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i0.z, i1.w), f1.x, f1.y, f0.z, f1.w), u.x), u.y);
    float gz11 = mix(mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i0.y, i1.z, i1.w), f0.x, f0.y, f1.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i0.y, i1.z, i1.w), f1.x, f0.y, f1.z, f1.w), u.x),
                     mix(grad_dot_perlin4d(hash_primes4d(seed, i0.x, i1.y, i1.z, i1.w), f0.x, f1.y, f1.z, f1.w),
                         grad_dot_perlin4d(hash_primes4d(seed, i1.x, i1.y, i1.z, i1.w), f1.x, f1.y, f1.z, f1.w), u.x), u.y);
    float gw1 = mix(gz01, gz11, u.z);
    return scale_output(mix(gw0, gw1, u.w), -1.33858621120452880859375, 1.33858621120452880859375);
}

float fn_perlin(vec2 p, int seed) { return fn_perlin2d(p, seed); }
float fn_perlin(vec3 p, int seed) { return fn_perlin3d(p, seed); }
float fn_perlin(vec4 p, int seed) { return fn_perlin4d(p, seed); }

// FastNoise2's FractalFBm and FractalRidged over Perlin (seed + 1 an octave,
// the weighted strength's lerp of the amplitude), one per dimension.
#define FN_FRACTALS(VT) \
float fn_fbm(VT p, int seed, int octaves, float lac, float gain, float ws) { \
    float amp = 1.0; \
    float noise = fn_perlin(p, seed); \
    float sum = noise * amp; \
    for (int i = 1; i < octaves; ++i) { \
        seed += 1; \
        amp *= 1.0 + ws * ((noise + 1.0) * 0.5 - 1.0); \
        amp *= gain; \
        p *= lac; \
        noise = fn_perlin(p, seed); \
        sum += noise * amp; \
    } \
    return sum; \
} \
float fn_ridged(VT p, int seed, int octaves, float lac, float gain, float ws) { \
    float amp = 1.0; \
    float noise = abs(fn_perlin(p, seed)); \
    float sum = (noise * -2.0 + 1.0) * amp; \
    for (int i = 1; i < octaves; ++i) { \
        seed += 1; \
        amp *= 1.0 + ws * ((1.0 - noise) - 1.0); \
        amp *= gain; \
        p *= lac; \
        noise = abs(fn_perlin(p, seed)); \
        sum += (noise * -2.0 + 1.0) * amp; \
    } \
    return sum; \
}
FN_FRACTALS(vec2)
FN_FRACTALS(vec3)
FN_FRACTALS(vec4)
)GLSL" R"GLSL(
float fn_simplex2d(vec2 p, int seed) {
    const float k_unskew = -0.211324870586395263671875;
    float skew_delta = 0.3660254180431365966796875 * (p.x + p.y);
    float x_skewed = p.x + skew_delta;
    float y_skewed = p.y + skew_delta;
    float x_base = floor(x_skewed);
    float y_base = floor(y_skewed);
    float dx_skewed = x_skewed - x_base;
    float dy_skewed = y_skewed - y_base;
    int x_primed = int(x_base) * PRIME_X;
    int y_primed = int(y_base) * PRIME_Y;
    bool x_ge_y = dx_skewed >= dy_skewed;
    float unskew_delta = k_unskew * (dx_skewed + dy_skewed);
    float dx0 = dx_skewed + unskew_delta;
    float dy0 = dy_skewed + unskew_delta;
    float dx1 = (x_ge_y ? dx0 : (dx0 + 1.0)) - (k_unskew + 1.0);
    float dy1 = (x_ge_y ? (dy0 + 1.0) : dy0) - (k_unskew + 1.0);
    float dx2 = dx0 - (k_unskew * 2.0 + 1.0);
    float dy2 = dy0 - (k_unskew * 2.0 + 1.0);
    float f0 = 0.5 - dy0 * dy0 - dx0 * dx0;
    float f1 = 0.5 - dy1 * dy1 - dx1 * dx1;
    float f2 = f0 + (unskew_delta * -3.1547005176544189453125 - 0.666666686534881591796875);
    f0 = max(f0, 0.0); f1 = max(f1, 0.0); f2 = max(f2, 0.0);
    f0 = (f0 * f0) * (f0 * f0); f1 = (f1 * f1) * (f1 * f1); f2 = (f2 * f2) * (f2 * f2);
    float g0 = grad_dot_perlin2d(hash_primes2d(seed, x_primed, y_primed), dx0, dy0);
    float g1 = grad_dot_perlin2d(hash_primes2d(seed, x_ge_y ? (x_primed + PRIME_X) : x_primed,
                                               x_ge_y ? y_primed : (y_primed + PRIME_Y)), dx1, dy1);
    float g2 = grad_dot_perlin2d(hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y), dx2, dy2);
    float val = g0 * f0 + (g1 * f1 + g2 * f2);
    return scale_output(val, -0.0261208079755306243896484375, 0.0261208079755306243896484375);
}
float fn_simplex3d(vec3 p, int seed) {
    float skew_delta = 0.3333333432674407958984375 * (p.x + p.y + p.z);
    vec3 skewed = p + skew_delta;
    vec3 base = floor(skewed);
    vec3 d = skewed - base;
    int x_primed = int(base.x) * PRIME_X;
    int y_primed = int(base.y) * PRIME_Y;
    int z_primed = int(base.z) * PRIME_Z;
    bool x_ge_y = d.x >= d.y;
    bool y_ge_z = d.y >= d.z;
    bool x_ge_z = d.x >= d.z;
    float unskew_delta = -0.5 * (d.x + d.y + d.z);
    float dx0 = d.x + unskew_delta;
    float dy0 = d.y + unskew_delta;
    float dz0 = d.z + unskew_delta;
    bool mask_x1 = x_ge_y && x_ge_z;
    bool mask_y1 = (!x_ge_y) && y_ge_z;
    bool mask_z1 = x_ge_z || y_ge_z;
    bool nmask_x2 = x_ge_y || x_ge_z;
    bool nmask_y2 = (!y_ge_z) && x_ge_y;
    bool nmask_z2 = x_ge_z && y_ge_z;
    float dx3 = dx0 + 0.5;
    float dy3 = dy0 + 0.5;
    float dz3 = dz0 + 0.5;
    float dx1 = mask_x1 ? (dx3 - 1.0) : dx3;
    float dy1 = mask_y1 ? (dy3 - 1.0) : dy3;
    float dz1 = (!mask_z1) ? (dz3 - 1.0) : dz3;
    float dx2 = (!nmask_x2) ? (dx0 + 1.0) : dx0;
    float dy2 = nmask_y2 ? (dy0 + 1.0) : dy0;
    float dz2 = nmask_z2 ? (dz0 + 1.0) : dz0;
    float f0 = (0.6 - dx0 * dx0 - dy0 * dy0) - dz0 * dz0;
    float f1 = (0.6 - dx1 * dx1 - dy1 * dy1) - dz1 * dz1;
    float f2 = (0.6 - dx2 * dx2 - dy2 * dy2) - dz2 * dz2;
    float f3 = f0 - (unskew_delta + 0.75);
    f0 = max(f0, 0.0); f1 = max(f1, 0.0); f2 = max(f2, 0.0); f3 = max(f3, 0.0);
    f0 = (f0 * f0) * (f0 * f0); f1 = (f1 * f1) * (f1 * f1);
    f2 = (f2 * f2) * (f2 * f2); f3 = (f3 * f3) * (f3 * f3);
    float g0 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed, z_primed), dx0, dy0, dz0);
    float g1 = grad_dot_common3d(hash_primes3d(seed,
        mask_x1 ? (x_primed + PRIME_X) : x_primed,
        mask_y1 ? (y_primed + PRIME_Y) : y_primed,
        (!mask_z1) ? (z_primed + PRIME_Z) : z_primed), dx1, dy1, dz1);
    float g2 = grad_dot_common3d(hash_primes3d(seed,
        nmask_x2 ? (x_primed + PRIME_X) : x_primed,
        (!nmask_y2) ? (y_primed + PRIME_Y) : y_primed,
        (!nmask_z2) ? (z_primed + PRIME_Z) : z_primed), dx2, dy2, dz2);
    float g3 = grad_dot_common3d(hash_primes3d(seed, x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z), dx3, dy3, dz3);
    float val = g3 * f3 + (g2 * f2 + (g1 * f1 + g0 * f0));
    return scale_output(val, -0.030586399137973785400390625, 0.030586399137973785400390625);
}
float fn_simplex4d(vec4 p, int seed) {
    const float k_unskew = -0.13819660112501051517954131656344;
    float skew_delta = 0.30901699437494742410229341718282 * (p.x + p.y + p.z + p.w);
    vec4 skewed = p + skew_delta;
    vec4 base = floor(skewed);
    vec4 d = skewed - base;
    int x_primed = int(base.x) * PRIME_X;
    int y_primed = int(base.y) * PRIME_Y;
    int z_primed = int(base.z) * PRIME_Z;
    int w_primed = int(base.w) * PRIME_W;
    float unskew_delta = k_unskew * (d.x + d.y + d.z + d.w);
    float dx0 = d.x + unskew_delta;
    float dy0 = d.y + unskew_delta;
    float dz0 = d.z + unskew_delta;
    float dw0 = d.w + unskew_delta;
    int rx = 0, ry = 0, rz = 0, rw = 0;
    if (dx0 >= dy0) rx += 1; else ry += 1;
    if (dx0 >= dz0) rx += 1; else rz += 1;
    if (dx0 >= dw0) rx += 1; else rw += 1;
    if (dy0 >= dz0) ry += 1; else rz += 1;
    if (dy0 >= dw0) ry += 1; else rw += 1;
    if (dz0 >= dw0) rz += 1; else rw += 1;
    float dx1 = (rx > 2 ? (dx0 - 1.0) : dx0) - k_unskew;
    float dy1 = (ry > 2 ? (dy0 - 1.0) : dy0) - k_unskew;
    float dz1 = (rz > 2 ? (dz0 - 1.0) : dz0) - k_unskew;
    float dw1 = (rw > 2 ? (dw0 - 1.0) : dw0) - k_unskew;
    float dx2 = (rx > 1 ? (dx0 - 1.0) : dx0) - (k_unskew * 2.0);
    float dy2 = (ry > 1 ? (dy0 - 1.0) : dy0) - (k_unskew * 2.0);
    float dz2 = (rz > 1 ? (dz0 - 1.0) : dz0) - (k_unskew * 2.0);
    float dw2 = (rw > 1 ? (dw0 - 1.0) : dw0) - (k_unskew * 2.0);
    float dx3 = (rx > 0 ? (dx0 - 1.0) : dx0) - (k_unskew * 3.0);
    float dy3 = (ry > 0 ? (dy0 - 1.0) : dy0) - (k_unskew * 3.0);
    float dz3 = (rz > 0 ? (dz0 - 1.0) : dz0) - (k_unskew * 3.0);
    float dw3 = (rw > 0 ? (dw0 - 1.0) : dw0) - (k_unskew * 3.0);
    float dx4 = dx0 - (k_unskew * 4.0 + 1.0);
    float dy4 = dy0 - (k_unskew * 4.0 + 1.0);
    float dz4 = dz0 - (k_unskew * 4.0 + 1.0);
    float dw4 = dw0 - (k_unskew * 4.0 + 1.0);
    float f0 = (((0.6 - dx0 * dx0) - dy0 * dy0) - dz0 * dz0) - dw0 * dw0;
    float f1 = (((0.6 - dx1 * dx1) - dy1 * dy1) - dz1 * dz1) - dw1 * dw1;
    float f2 = (((0.6 - dx2 * dx2) - dy2 * dy2) - dz2 * dz2) - dw2 * dw2;
    float f3 = (((0.6 - dx3 * dx3) - dy3 * dy3) - dz3 * dz3) - dw3 * dw3;
    float f4 = f0 + (unskew_delta * -2.89442729949951171875 - 0.800000011920928955078125);
    f0 = max(f0, 0.0); f1 = max(f1, 0.0); f2 = max(f2, 0.0); f3 = max(f3, 0.0); f4 = max(f4, 0.0);
    f0 = (f0 * f0) * (f0 * f0); f1 = (f1 * f1) * (f1 * f1); f2 = (f2 * f2) * (f2 * f2);
    f3 = (f3 * f3) * (f3 * f3); f4 = (f4 * f4) * (f4 * f4);
    float g0 = grad_dot_perlin4d(hash_primes4d(seed, x_primed, y_primed, z_primed, w_primed), dx0, dy0, dz0, dw0);
    float g1 = grad_dot_perlin4d(hash_primes4d(seed,
        rx > 2 ? (x_primed + PRIME_X) : x_primed, ry > 2 ? (y_primed + PRIME_Y) : y_primed,
        rz > 2 ? (z_primed + PRIME_Z) : z_primed, rw > 2 ? (w_primed + PRIME_W) : w_primed), dx1, dy1, dz1, dw1);
    float g2 = grad_dot_perlin4d(hash_primes4d(seed,
        rx > 1 ? (x_primed + PRIME_X) : x_primed, ry > 1 ? (y_primed + PRIME_Y) : y_primed,
        rz > 1 ? (z_primed + PRIME_Z) : z_primed, rw > 1 ? (w_primed + PRIME_W) : w_primed), dx2, dy2, dz2, dw2);
    float g3 = grad_dot_perlin4d(hash_primes4d(seed,
        rx > 0 ? (x_primed + PRIME_X) : x_primed, ry > 0 ? (y_primed + PRIME_Y) : y_primed,
        rz > 0 ? (z_primed + PRIME_Z) : z_primed, rw > 0 ? (w_primed + PRIME_W) : w_primed), dx3, dy3, dz3, dw3);
    float g4 = grad_dot_perlin4d(hash_primes4d(seed,
        x_primed + PRIME_X, y_primed + PRIME_Y, z_primed + PRIME_Z, w_primed + PRIME_W), dx4, dy4, dz4, dw4);
    float val = g0 * f0 + (g1 * f1 + (g2 * f2 + (g3 * f3 + g4 * f4)));
    return scale_output(val, -0.036730043590068817138671875, 0.036730043590068817138671875);
}
)GLSL" R"GLSL(
float fn_supersimplex2d(vec2 p, int seed) {
    const float k_unskew = -0.211324870586395263671875;
    const float k_rad2 = 0.666666686534881591796875;
    float s = 0.3660254180431365966796875 * (p.x + p.y);
    float x_skewed = p.x + s;
    float y_skewed = p.y + s;
    float x_base = floor(x_skewed);
    float y_base = floor(y_skewed);
    float dx_skewed = x_skewed - x_base;
    float dy_skewed = y_skewed - y_base;
    int x_primed = int(x_base) * PRIME_X;
    int y_primed = int(y_base) * PRIME_Y;
    bool fwd_xy = (dx_skewed + dy_skewed) > 1.0;
    float bound_xy = fwd_xy ? -1.0 : 0.0;
    bool fwd_x = (dx_skewed * -2.0 + dy_skewed) < bound_xy;
    bool fwd_y = (dy_skewed * -2.0 + dx_skewed) < bound_xy;
    float unskew_d = k_unskew * (dx_skewed + dy_skewed);
    float dx_base = dx_skewed + unskew_d;
    float dy_base = dy_skewed + unskew_d;
    float g0 = grad_dot_perlin2d(hash_primes2d(seed, x_primed, y_primed), dx_base, dy_base);
    float fb0 = (k_rad2 - dy_base * dy_base) - dx_base * dx_base;
    float value = ((fb0 * fb0) * (fb0 * fb0)) * g0;
    float g1 = grad_dot_perlin2d(hash_primes2d(seed, x_primed + PRIME_X, y_primed + PRIME_Y),
                                 dx_base - (2.0 * k_unskew + 1.0), dy_base - (2.0 * k_unskew + 1.0));
    float fb1 = unskew_d * -3.1547005176544189453125 + (fb0 - k_rad2);
    value += ((fb1 * fb1) * (fb1 * fb1)) * g1;
    float xy_delta = fwd_xy ? (k_unskew + 1.0) : (-k_unskew);
    dx_base -= xy_delta;
    dy_base -= xy_delta;
    int xp2_tmp = fwd_x ? (x_primed + (PRIME_X * 2)) : x_primed;
    int xp2 = (!fwd_xy) ? (xp2_tmp - PRIME_X) : xp2_tmp;
    int yp2 = fwd_xy ? (y_primed + PRIME_Y) : y_primed;
    float dx2 = dx_base - (fwd_x ? (1.0 + 2.0 * k_unskew) : -1.0);
    float dy2 = fwd_x ? (dy_base - 2.0 * k_unskew) : dy_base;
    float g2 = grad_dot_perlin2d(hash_primes2d(seed, xp2, yp2), dx2, dy2);
    float fb2 = max((k_rad2 - dy2 * dy2) - dx2 * dx2, 0.0);
    value += ((fb2 * fb2) * (fb2 * fb2)) * g2;
    int xp3 = fwd_xy ? (x_primed + PRIME_X) : x_primed;
    int yp3_tmp = fwd_y ? (y_primed + (PRIME_Y * 2)) : y_primed;
    int yp3 = (!fwd_xy) ? (yp3_tmp - PRIME_Y) : yp3_tmp;
    float dx3 = fwd_y ? (dx_base - 2.0 * k_unskew) : dx_base;
    float dy3 = dy_base - (fwd_y ? (1.0 + 2.0 * k_unskew) : -1.0);
    float g3 = grad_dot_perlin2d(hash_primes2d(seed, xp3, yp3), dx3, dy3);
    float fb3 = max((k_rad2 - dy3 * dy3) - dx3 * dx3, 0.0);
    value += ((fb3 * fb3) * (fb3 * fb3)) * g3;
    return scale_output(value, -0.14084912836551666259765625, 0.14084912836551666259765625);
}
float fn_supersimplex3d(vec3 p, int seed) {
    const float k_twice_unskew = -0.25;
    float s = 0.3333333432674407958984375 * (p.x + p.y + p.z);
    float x_base = floor(p.x + s);
    float y_base = floor(p.y + s);
    float z_base = floor(p.z + s);
    float dx_skewed = (p.x + s) - x_base;
    float dy_skewed = (p.y + s) - y_base;
    float dz_skewed = (p.z + s) - z_base;
    float tu0 = k_twice_unskew * (dx_skewed + dy_skewed + dz_skewed);
    float xn0 = dx_skewed + tu0;
    float yn0 = dy_skewed + tu0;
    float zn0 = dz_skewed + tu0;
    float xyzn0 = -tu0;
    float max_score = 0.375;
    int move_bits = (xyzn0 > max_score) ? -1 : 0;
    max_score = max(max_score, xyzn0);
    if (xn0 > max_score) { move_bits = 1; max_score = xn0; }
    if (yn0 > max_score) { move_bits = 2; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 4; max_score = zn0; }
    max_score += 0.125 - xyzn0;
    if (-zn0 > max_score) { move_bits = 3; max_score = -zn0; }
    if (-yn0 > max_score) { move_bits = 5; max_score = -yn0; }
    if (-xn0 > max_score) { move_bits = 6; max_score = -xn0; }
    if ((move_bits & 1) != 0) { x_base += 1.0; dx_skewed -= 1.0; }
    if ((move_bits & 2) != 0) { y_base += 1.0; dy_skewed -= 1.0; }
    if ((move_bits & 4) != 0) { z_base += 1.0; dz_skewed -= 1.0; }
    int x_primed = int(x_base) * PRIME_X;
    int y_primed = int(y_base) * PRIME_Y;
    int z_primed = int(z_base) * PRIME_Z;
    float sk_sum = dx_skewed + dy_skewed + dz_skewed;
    float tu = k_twice_unskew * sk_sum;
    float xn = dx_skewed + tu;
    float yn = dy_skewed + tu;
    float zn = dz_skewed + tu;
    float xyzn = -tu;
    float unskew_d = -0.5 * sk_sum;
    float dx_b = dx_skewed + unskew_d;
    float dy_b = dy_skewed + unskew_d;
    float dz_b = dz_skewed + unskew_d;
    float coord_sum = -0.5 * sk_sum;
    float g0 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed, z_primed), dx_b, dy_b, dz_b);
    float fb0 = (((0.75 - dx_b * dx_b) - dy_b * dy_b) - dz_b * dz_b) * 0.5;
    float stem_a = fb0 - 0.375;
    float stem_b = fb0 - 0.5;
    float value = ((fb0 * fb0) * (fb0 * fb0)) * g0;
    bool sm1 = xyzn < 0.0;
    float off1 = sm1 ? 0.5 : -0.5;
    float g1 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm1 ? -PRIME_X : PRIME_X),
        y_primed + (sm1 ? -PRIME_Y : PRIME_Y), z_primed + (sm1 ? -PRIME_Z : PRIME_Z)),
        dx_b - off1, dy_b - off1, dz_b - off1);
    float fb1 = max(off1 * coord_sum + stem_a, 0.0);
    value += ((fb1 * fb1) * (fb1 * fb1)) * g1;
    bool sm2 = xyzn < zn;
    float off2 = sm2 ? 1.0 : -1.0;
    float g2 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm2 ? -PRIME_X : PRIME_X), y_primed + (sm2 ? -PRIME_Y : PRIME_Y), z_primed),
        dx_b, dy_b, dz_b - off2);
    float fb2 = min((sm2 ? -dz_b : dz_b) - stem_b, 0.0);
    value += ((fb2 * fb2) * (fb2 * fb2)) * g2;
    bool sm3 = xyzn < yn;
    float off3 = sm3 ? 1.0 : -1.0;
    float g3 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm3 ? -PRIME_X : PRIME_X), y_primed, z_primed + (sm3 ? -PRIME_Z : PRIME_Z)),
        dx_b, dy_b - off3, dz_b);
    float fb3 = min((sm3 ? -dy_b : dy_b) - stem_b, 0.0);
    value += ((fb3 * fb3) * (fb3 * fb3)) * g3;
    bool sm4 = xyzn < xn;
    float off4 = sm4 ? 1.0 : -1.0;
    float g4 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed + (sm4 ? -PRIME_Y : PRIME_Y), z_primed + (sm4 ? -PRIME_Z : PRIME_Z)),
        dx_b - off4, dy_b, dz_b);
    float fb4 = min((sm4 ? -dx_b : dx_b) - stem_b, 0.0);
    value += ((fb4 * fb4) * (fb4 * fb4)) * g4;
    bool sm5 = xn < 0.0;
    float off5 = sm5 ? 0.5 : -0.5;
    float g5 = grad_dot_common3d(hash_primes3d(seed, x_primed + (sm5 ? -PRIME_X : PRIME_X), y_primed, z_primed),
        dx_b + off5, dy_b - off5, dz_b - off5);
    float fb5 = max((off5 * coord_sum + stem_a) + (sm5 ? -dx_b : dx_b), 0.0);
    value += ((fb5 * fb5) * (fb5 * fb5)) * g5;
    bool sm6 = yn < 0.0;
    float off6 = sm6 ? 0.5 : -0.5;
    float g6 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed + (sm6 ? -PRIME_Y : PRIME_Y), z_primed),
        dx_b - off6, dy_b + off6, dz_b - off6);
    float fb6 = max((off6 * coord_sum + stem_a) + (sm6 ? -dy_b : dy_b), 0.0);
    value += ((fb6 * fb6) * (fb6 * fb6)) * g6;
    bool sm7 = zn < 0.0;
    float off7 = sm7 ? 0.5 : -0.5;
    float g7 = grad_dot_common3d(hash_primes3d(seed, x_primed, y_primed, z_primed + (sm7 ? -PRIME_Z : PRIME_Z)),
        dx_b - off7, dy_b - off7, dz_b + off7);
    float fb7 = max((off7 * coord_sum + stem_a) + (sm7 ? -dz_b : dz_b), 0.0);
    value += ((fb7 * fb7) * (fb7 * fb7)) * g7;
    return scale_output(value, -0.0069091119803488254547119140625, 0.0069091119803488254547119140625);
}
)GLSL" R"GLSL(
float supersimplex4d_v3(int seed, bool sm, int xp, int yp, int zp, int wp,
                        float dx_b, float dy_b, float dz_b, float dw_b,
                        float off_x, float off_y, float off_z, float off_w,
                        float off_mul, float coord_sum, float stem, float sub_axis) {
    float sgn = sm ? -1.0 : 1.0;
    float g = grad_dot_perlin4d(hash_primes4d(seed, xp, yp, zp, wp),
        dx_b - off_x * sgn, dy_b - off_y * sgn, dz_b - off_z * sgn, dw_b - off_w * sgn);
    float fb = max(((off_mul * sgn) * coord_sum + stem) + sub_axis * sgn, 0.0);
    return ((fb * fb) * (fb * fb)) * g;
}
float fn_supersimplex4d(vec4 p, int seed) {
    const float k_unskew = -0.13819660112501051517954131656344;
    const float k_twice_unskew = -0.2;
    float s = 0.30901699437494742410229341718282 * (p.x + p.y + p.z + p.w);
    vec4 base = floor(p + s);
    vec4 dsk = (p + s) - base;
    float x_base = base.x, y_base = base.y, z_base = base.z, w_base = base.w;
    float dx_skewed = dsk.x, dy_skewed = dsk.y, dz_skewed = dsk.z, dw_skewed = dsk.w;
    float tu0 = k_twice_unskew * (dx_skewed + dy_skewed + dz_skewed + dw_skewed);
    float xn0 = dx_skewed + tu0;
    float yn0 = dy_skewed + tu0;
    float zn0 = dz_skewed + tu0;
    float wn0 = dw_skewed + tu0;
    float xyzwn0 = -tu0;
    float max_score = 0.6 - xyzwn0;
    int move_bits = (0.2 > max_score) ? -1 : 0;
    max_score = max(max_score, 0.2);
    if (-wn0 > max_score) { move_bits = 7; max_score = -wn0; }
    if (-zn0 > max_score) { move_bits = 11; max_score = -zn0; }
    if (-yn0 > max_score) { move_bits = 13; max_score = -yn0; }
    if (-xn0 > max_score) { move_bits = 14; max_score = -xn0; }
    max_score += xyzwn0 - 0.2;
    if (xn0 > max_score) { move_bits = 1; max_score = xn0; }
    if (yn0 > max_score) { move_bits = 2; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 4; max_score = zn0; }
    if (wn0 > max_score) { move_bits = 8; max_score = wn0; }
    max_score += 0.2 - xn0;
    if (yn0 > max_score) { move_bits = 3; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 5; max_score = zn0; }
    if (wn0 > max_score) { move_bits = 9; max_score = wn0; }
    max_score += xn0;
    if (yn0 + zn0 > max_score) { move_bits = 6; max_score = yn0 + zn0; }
    max_score -= wn0;
    if (yn0 > max_score) { move_bits = 10; max_score = yn0; }
    if (zn0 > max_score) { move_bits = 12; max_score = zn0; }
    if ((move_bits & 1) != 0) { x_base += 1.0; dx_skewed -= 1.0; }
    if ((move_bits & 2) != 0) { y_base += 1.0; dy_skewed -= 1.0; }
    if ((move_bits & 4) != 0) { z_base += 1.0; dz_skewed -= 1.0; }
    if ((move_bits & 8) != 0) { w_base += 1.0; dw_skewed -= 1.0; }
    int x_p = int(x_base) * PRIME_X;
    int y_p = int(y_base) * PRIME_Y;
    int z_p = int(z_base) * PRIME_Z;
    int w_p = int(w_base) * PRIME_W;
    float sk_sum = dx_skewed + dy_skewed + dz_skewed + dw_skewed;
    float tu = k_twice_unskew * sk_sum;
    float xn = dx_skewed + tu;
    float yn = dy_skewed + tu;
    float zn = dz_skewed + tu;
    float wn = dw_skewed + tu;
    float xyzwn = -tu;
    float unskew_d = k_unskew * sk_sum;
    float dx_b = dx_skewed + unskew_d;
    float dy_b = dy_skewed + unskew_d;
    float dz_b = dz_skewed + unskew_d;
    float dw_b = dw_skewed + unskew_d;
    float coord_sum = (1.0 + 4.0 * k_unskew) * sk_sum;
    float g0 = grad_dot_perlin4d(hash_primes4d(seed, x_p, y_p, z_p, w_p), dx_b, dy_b, dz_b, dw_b);
    float fb0 = ((((0.8 - dx_b * dx_b) - dy_b * dy_b) - dz_b * dz_b) - dw_b * dw_b) * 0.5;
    float stem_a = fb0 - 0.4;
    float stem_b = fb0 - 0.6;
    float value = ((fb0 * fb0) * (fb0 * fb0)) * g0;
    float u1_0 = k_unskew, u1_1 = k_unskew + 1.0;
    float u2_0 = 2.0 * k_unskew, u2_1 = 2.0 * k_unskew + 1.0;
    float u3_0 = 3.0 * k_unskew, u3_1 = 3.0 * k_unskew + 1.0;
    float u4_1 = 4.0 * k_unskew + 1.0;
    bool sm1 = xyzwn < 0.0;
    value += supersimplex4d_v3(seed, sm1, x_p + (sm1 ? -PRIME_X : PRIME_X), y_p + (sm1 ? -PRIME_Y : PRIME_Y), z_p + (sm1 ? -PRIME_Z : PRIME_Z), w_p + (sm1 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u4_1, u4_1, u4_1, u4_1, u4_1, coord_sum, stem_a, 0.0);
    bool sm2 = xyzwn < wn;
    value += supersimplex4d_v3(seed, sm2, x_p + (sm2 ? -PRIME_X : PRIME_X), y_p + (sm2 ? -PRIME_Y : PRIME_Y), z_p + (sm2 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u3_1, u3_1, u3_1, u3_0, u3_1, coord_sum, stem_b, -dw_b);
    bool sm3 = xyzwn < zn;
    value += supersimplex4d_v3(seed, sm3, x_p + (sm3 ? -PRIME_X : PRIME_X), y_p + (sm3 ? -PRIME_Y : PRIME_Y), z_p, w_p + (sm3 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_1, u3_1, u3_0, u3_1, u3_1, coord_sum, stem_b, -dz_b);
    bool sm4 = xyzwn < yn;
    value += supersimplex4d_v3(seed, sm4, x_p + (sm4 ? -PRIME_X : PRIME_X), y_p, z_p + (sm4 ? -PRIME_Z : PRIME_Z), w_p + (sm4 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_1, u3_0, u3_1, u3_1, u3_1, coord_sum, stem_b, -dy_b);
    bool sm5 = xyzwn < xn;
    value += supersimplex4d_v3(seed, sm5, x_p, y_p + (sm5 ? -PRIME_Y : PRIME_Y), z_p + (sm5 ? -PRIME_Z : PRIME_Z), w_p + (sm5 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u3_0, u3_1, u3_1, u3_1, u3_1, coord_sum, stem_b, -dx_b);
    bool sm6 = xn < 0.0;
    value += supersimplex4d_v3(seed, sm6, x_p + (sm6 ? -PRIME_X : PRIME_X), y_p, z_p, w_p, dx_b, dy_b, dz_b, dw_b, u1_1, u1_0, u1_0, u1_0, u1_0, coord_sum, stem_a, dx_b);
    bool sm7 = xn < -yn;
    value += supersimplex4d_v3(seed, sm7, x_p + (sm7 ? -PRIME_X : PRIME_X), y_p + (sm7 ? -PRIME_Y : PRIME_Y), z_p, w_p, dx_b, dy_b, dz_b, dw_b, u2_1, u2_1, u2_0, u2_0, u2_0, coord_sum, stem_b, dx_b + dy_b);
    bool sm8 = xn < -zn;
    value += supersimplex4d_v3(seed, sm8, x_p + (sm8 ? -PRIME_X : PRIME_X), y_p, z_p + (sm8 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u2_1, u2_0, u2_1, u2_0, u2_0, coord_sum, stem_b, dx_b + dz_b);
    bool sm9 = xn < -wn;
    value += supersimplex4d_v3(seed, sm9, x_p + (sm9 ? -PRIME_X : PRIME_X), y_p, z_p, w_p + (sm9 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_1, u2_0, u2_0, u2_1, u2_0, coord_sum, stem_b, dx_b + dw_b);
    bool sm10 = yn < 0.0;
    value += supersimplex4d_v3(seed, sm10, x_p, y_p + (sm10 ? -PRIME_Y : PRIME_Y), z_p, w_p, dx_b, dy_b, dz_b, dw_b, u1_0, u1_1, u1_0, u1_0, u1_0, coord_sum, stem_a, dy_b);
    bool sm11 = yn < -zn;
    value += supersimplex4d_v3(seed, sm11, x_p, y_p + (sm11 ? -PRIME_Y : PRIME_Y), z_p + (sm11 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u2_0, u2_1, u2_1, u2_0, u2_0, coord_sum, stem_b, dy_b + dz_b);
    bool sm12 = yn < -wn;
    value += supersimplex4d_v3(seed, sm12, x_p, y_p + (sm12 ? -PRIME_Y : PRIME_Y), z_p, w_p + (sm12 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_0, u2_1, u2_0, u2_1, u2_0, coord_sum, stem_b, dy_b + dw_b);
    bool sm13 = zn < 0.0;
    value += supersimplex4d_v3(seed, sm13, x_p, y_p, z_p + (sm13 ? -PRIME_Z : PRIME_Z), w_p, dx_b, dy_b, dz_b, dw_b, u1_0, u1_0, u1_1, u1_0, u1_0, coord_sum, stem_a, dz_b);
    bool sm14 = zn < -wn;
    value += supersimplex4d_v3(seed, sm14, x_p, y_p, z_p + (sm14 ? -PRIME_Z : PRIME_Z), w_p + (sm14 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u2_0, u2_0, u2_1, u2_1, u2_0, coord_sum, stem_b, dz_b + dw_b);
    bool sm15 = wn < 0.0;
    value += supersimplex4d_v3(seed, sm15, x_p, y_p, z_p, w_p + (sm15 ? -PRIME_W : PRIME_W), dx_b, dy_b, dz_b, dw_b, u1_0, u1_0, u1_0, u1_1, u1_0, coord_sum, stem_a, dw_b);
    return scale_output(value, -0.011310013942420482635498046875, 0.011310013942420482635498046875);
}
)GLSL" R"GLSL(
int white_coord(float x, int prime) {
    int bits = floatBitsToInt(x);
    return (bits ^ (bits >> 16)) * prime;
}
float fn_white2d(vec2 p, int seed) {
    return scale_output(get_value2d(seed, white_coord(p.x, PRIME_X), white_coord(p.y, PRIME_Y)), -2147483648.0, 2147483648.0);
}
float fn_white3d(vec3 p, int seed) {
    return scale_output(get_value3d(seed, white_coord(p.x, PRIME_X), white_coord(p.y, PRIME_Y), white_coord(p.z, PRIME_Z)),
                        -2147483648.0, 2147483648.0);
}
float fn_white4d(vec4 p, int seed) {
    return scale_output(get_value4d(seed, white_coord(p.x, PRIME_X), white_coord(p.y, PRIME_Y), white_coord(p.z, PRIME_Z),
                                    white_coord(p.w, PRIME_W)), -2147483648.0, 2147483648.0);
}

float fn_ping_pong(float val, float strength) {
    float t0 = val * strength;
    float t = t0 - floor(t0 * 0.5) * 2.0;
    return (t < 1.0) ? t : (2.0 - t);
}

// FastNoise2's cellular nodes with the Euclidean-squared distance (the
// bench's), a 3^D neighbourhood of jittered points: the nearest point's hash
// (value), the nearest distance (distance, rooted), or Perlin at the nearest
// point (lookup, seed + 1).
const float kJitter2 = 0.437015950679779052734375;
const float kJitter3 = 0.3961435258388519287109375;
const float kJitter4 = 0.3660254180431365966796875;
void cell_search2d(vec2 p, int seed, out float d0, out int best_hash, out vec2 best_p) {
    vec2 r = roundEven(p);
    int xc = (int(r.x) - 1) * PRIME_X;
    float xcf = (r.x - 1.0) - p.x;
    d0 = 1.0e10; best_hash = 0; best_p = vec2(0.0);
    for (int ix = 0; ix < 3; ++ix) {
        int yc = (int(r.y) - 1) * PRIME_Y;
        float ycf = (r.y - 1.0) - p.y;
        for (int iy = 0; iy < 3; ++iy) {
            int h = (seed ^ xc ^ yc) * HASH_A;
            float xd = float(h & 0x7ff) - 1023.5;
            float yd = float((h >> 21) & 0x7ff) - 1023.5;
            float inv_mag = kJitter2 / sqrt(xd * xd + yd * yd);
            float px_d = xd * inv_mag + xcf;
            float py_d = yd * inv_mag + ycf;
            float d = px_d * px_d + py_d * py_d;
            if (d < d0) { d0 = d; best_hash = h; best_p = vec2(px_d, py_d) + p; }
            yc += PRIME_Y; ycf += 1.0;
        }
        xc += PRIME_X; xcf += 1.0;
    }
}
void cell_search3d(vec3 p, int seed, out float d0, out int best_hash, out vec3 best_p) {
    vec3 r = roundEven(p);
    int xc = (int(r.x) - 1) * PRIME_X;
    float xcf = (r.x - 1.0) - p.x;
    d0 = 1.0e10; best_hash = 0; best_p = vec3(0.0);
    for (int ix = 0; ix < 3; ++ix) {
        int yc = (int(r.y) - 1) * PRIME_Y;
        float ycf = (r.y - 1.0) - p.y;
        for (int iy = 0; iy < 3; ++iy) {
            int zc = (int(r.z) - 1) * PRIME_Z;
            float zcf = (r.z - 1.0) - p.z;
            for (int iz = 0; iz < 3; ++iz) {
                int h = (seed ^ xc ^ yc ^ zc) * HASH_A;
                float xd = float(h & 0x3ff) - 511.5;
                float yd = float((h >> 11) & 0x3ff) - 511.5;
                float zd = float((h >> 22) & 0x3ff) - 511.5;
                float inv_mag = kJitter3 / sqrt(xd * xd + yd * yd + zd * zd);
                vec3 pd = vec3(xd, yd, zd) * inv_mag + vec3(xcf, ycf, zcf);
                float d = pd.x * pd.x + pd.y * pd.y + pd.z * pd.z;
                if (d < d0) { d0 = d; best_hash = h; best_p = pd + p; }
                zc += PRIME_Z; zcf += 1.0;
            }
            yc += PRIME_Y; ycf += 1.0;
        }
        xc += PRIME_X; xcf += 1.0;
    }
}
void cell_search4d(vec4 p, int seed, out float d0, out int best_hash, out vec4 best_p) {
    vec4 r = roundEven(p);
    ivec4 cb = (ivec4(r) - 1) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 cf = (r - 1.0) - p;
    d0 = 1.0e10; best_hash = 0; best_p = vec4(0.0);
    for (int ix = 0; ix < 3; ++ix)
    for (int iy = 0; iy < 3; ++iy)
    for (int iz = 0; iz < 3; ++iz)
    for (int iw = 0; iw < 3; ++iw) {
        int h = (seed ^ (cb.x + ix * PRIME_X) ^ (cb.y + iy * PRIME_Y) ^ (cb.z + iz * PRIME_Z) ^ (cb.w + iw * PRIME_W)) * HASH_A;
        vec4 dv = vec4(float(h & 0xff), float((h >> 8) & 0xff), float((h >> 16) & 0xff), float((h >> 24) & 0xff)) - 127.5;
        float inv_mag = kJitter4 / sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z + dv.w * dv.w);
        vec4 pd = dv * inv_mag + (cf + vec4(float(ix), float(iy), float(iz), float(iw)));
        float d = pd.x * pd.x + pd.y * pd.y + pd.z * pd.z + pd.w * pd.w;
        if (d < d0) { d0 = d; best_hash = h; best_p = pd + p; }
    }
}
// The second-nearest distance is not tracked: the bench's distance rows
// return d0 (FastNoise2's Index0).
#define FN_CELLULAR(VT, D, JIT) \
float fn_cell_value##D##d(VT p, int seed) { \
    float d0; int h; VT bp; cell_search##D##d(p, seed, d0, h, bp); \
    return scale_output(float(h), -2147483648.0, 2147483648.0); \
} \
float fn_cell_distance##D##d(VT p, int seed) { \
    float d0; int h; VT bp; cell_search##D##d(p, seed, d0, h, bp); \
    return scale_output(sqrt(d0), 0.0, (1.0 + JIT) * (1.0 + JIT)); \
} \
float fn_cell_lookup##D##d(VT p, int seed) { \
    float d0; int h; VT bp; cell_search##D##d(p, seed, d0, h, bp); \
    return fn_perlin(bp, seed + 1); \
}
FN_CELLULAR(vec2, 2, kJitter2)
FN_CELLULAR(vec3, 3, kJitter3)
FN_CELLULAR(vec4, 4, kJitter4)
)GLSL" R"GLSL(
// FastNoise2's gradient domain warp (hermite-weighted lattice hashes, its
// bits as the offset), alone and as the progressive / independent fractals.
vec2 warp_delta(vec2 p, int seed) {
    vec2 s = floor(p);
    ivec2 i0 = ivec2(s) * ivec2(PRIME_X, PRIME_Y);
    ivec2 i1 = i0 + ivec2(PRIME_X, PRIME_Y);
    float xs1 = interp_hermite(p.x - s.x), ys1 = interp_hermite(p.y - s.y);
    float xs0 = 1.0 - xs1, ys0 = 1.0 - ys1;
    int h00 = (seed ^ i0.x ^ i0.y) * HASH_A;
    int h10 = (seed ^ i1.x ^ i0.y) * HASH_A;
    int h01 = (seed ^ i0.x ^ i1.y) * HASH_A;
    int h11 = (seed ^ i1.x ^ i1.y) * HASH_A;
    float c00 = xs0 * ys0, c10 = xs1 * ys0, c01 = xs0 * ys1, c11 = xs1 * ys1;
    float xw = c00 * float(h00 & 0xffff) + c10 * float(h10 & 0xffff) + c01 * float(h01 & 0xffff) + c11 * float(h11 & 0xffff);
    float yw = c00 * float((h00 >> 16) & 0xffff) + c10 * float((h10 >> 16) & 0xffff) + c01 * float((h01 >> 16) & 0xffff) + c11 * float((h11 >> 16) & 0xffff);
    float norm = 1.0 / (65535.0 / 2.0);
    return vec2(xw * norm - 1.0, yw * norm - 1.0);
}
vec3 warp_delta(vec3 p, int seed) {
    vec3 s = floor(p);
    ivec3 i0 = ivec3(s) * ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    ivec3 i1 = i0 + ivec3(PRIME_X, PRIME_Y, PRIME_Z);
    vec3 f1 = vec3(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z));
    vec3 f0 = 1.0 - f1;
    vec3 acc = vec3(0.0);
    for (int iz = 0; iz < 2; ++iz)
    for (int iy = 0; iy < 2; ++iy)
    for (int ix = 0; ix < 2; ++ix) {
        float c = (ix == 0 ? f0.x : f1.x) * (iy == 0 ? f0.y : f1.y) * (iz == 0 ? f0.z : f1.z);
        int h = (seed ^ (ix == 0 ? i0.x : i1.x) ^ (iy == 0 ? i0.y : i1.y) ^ (iz == 0 ? i0.z : i1.z)) * HASH_A;
        acc += c * vec3(float(h & 0x3ff), float((h >> 11) & 0x3ff), float((h >> 22) & 0x3ff));
    }
    return acc * (1.0 / (1023.0 / 2.0)) - 1.0;
}
vec4 warp_delta(vec4 p, int seed) {
    vec4 s = floor(p);
    ivec4 i0 = ivec4(s) * ivec4(PRIME_X, PRIME_Y, PRIME_Z, PRIME_W);
    vec4 f = vec4(interp_hermite(p.x - s.x), interp_hermite(p.y - s.y), interp_hermite(p.z - s.z), interp_hermite(p.w - s.w));
    vec4 acc = vec4(0.0);
    for (int iw = 0; iw < 2; ++iw) {
        float wt = (iw == 0) ? (1.0 - f.w) : f.w;
        for (int iz = 0; iz < 2; ++iz) {
            float zt = wt * ((iz == 0) ? (1.0 - f.z) : f.z);
            for (int iy = 0; iy < 2; ++iy) {
                float yt = zt * ((iy == 0) ? (1.0 - f.y) : f.y);
                for (int ix = 0; ix < 2; ++ix) {
                    float xt = yt * ((ix == 0) ? (1.0 - f.x) : f.x);
                    int h = (seed ^ (i0.x + ix * PRIME_X) ^ (i0.y + iy * PRIME_Y) ^ (i0.z + iz * PRIME_Z) ^ (i0.w + iw * PRIME_W)) * HASH_A;
                    acc += xt * vec4(float(h & 0xff), float((h >> 8) & 0xff), float((h >> 16) & 0xff), float((h >> 24) & 0xff));
                }
            }
        }
    }
    return acc * (1.0 / (255.0 / 2.0)) - 1.0;
}
#define FN_WARPS(VT) \
VT fn_warp_grad(VT p, int seed, float amp, float freq) { return p + warp_delta(p * freq, seed) * amp; } \
VT fn_warp_fractal(VT p, int seed, int octaves, float lac, float gain, float ws, float amp, float freq, bool progressive) { \
    VT wp = p; \
    VT w = warp_delta(p * freq, seed); \
    wp = wp + w * amp; \
    float strength = dot(w, w); \
    for (int i = 1; i < octaves; ++i) { \
        strength = sqrt(strength); \
        seed += 1; \
        freq *= lac; \
        amp *= 1.0 + ws * ((1.0 - strength) - 1.0); \
        amp *= gain; \
        w = warp_delta((progressive ? wp : p) * freq, seed); \
        wp = wp + w * amp; \
        strength = dot(w, w); \
    } \
    return wp; \
}
FN_WARPS(vec2)
FN_WARPS(vec3)
FN_WARPS(vec4)
)GLSL";

std::string make_glsl_noise_shader(int dim, const std::string& expr) {
    std::ostringstream ss;
    ss << kGlslNoiseHeader;
    if (dim == 2) {
        ss << "layout(push_constant) uniform PC { uint64_t px; uint64_t py; uint64_t out_p; int seed; int n; } pc;\n"
           << "void main() {\n"
           << "    uint i = gl_GlobalInvocationID.x;\n"
           << "    if (i >= uint(pc.n)) return;\n"
           << "    vec2 p = vec2(FloatBuf(pc.px).v[i], FloatBuf(pc.py).v[i]);\n"
           << "    int seed = pc.seed;\n"
           << "    FloatBuf(pc.out_p).v[i] = " << expr << ";\n"
           << "}\n";
    } else if (dim == 3) {
        ss << "layout(push_constant) uniform PC { uint64_t px; uint64_t py; uint64_t pz; uint64_t out_p; int seed; int n; } pc;\n"
           << "void main() {\n"
           << "    uint i = gl_GlobalInvocationID.x;\n"
           << "    if (i >= uint(pc.n)) return;\n"
           << "    vec3 p = vec3(FloatBuf(pc.px).v[i], FloatBuf(pc.py).v[i], FloatBuf(pc.pz).v[i]);\n"
           << "    int seed = pc.seed;\n"
           << "    FloatBuf(pc.out_p).v[i] = " << expr << ";\n"
           << "}\n";
    } else {
        ss << "layout(push_constant) uniform PC { uint64_t px; uint64_t py; uint64_t pz; uint64_t pw; uint64_t out_p; int seed; int n; } pc;\n"
           << "void main() {\n"
           << "    uint i = gl_GlobalInvocationID.x;\n"
           << "    if (i >= uint(pc.n)) return;\n"
           << "    vec4 p = vec4(FloatBuf(pc.px).v[i], FloatBuf(pc.py).v[i], FloatBuf(pc.pz).v[i], FloatBuf(pc.pw).v[i]);\n"
           << "    int seed = pc.seed;\n"
           << "    FloatBuf(pc.out_p).v[i] = " << expr << ";\n"
           << "}\n";
    }
    return ss.str();
}
