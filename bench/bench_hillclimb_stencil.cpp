#include "bench_hillclimb.h"

namespace {
// ============================================================================
// Tier 2: Stencil & Grid Kernels
// ============================================================================
// The planet's flow state: per cell the ground h, drainage a, the drainage
// handed on a unit of weight z, the steepest slope down w, the water level wl.
struct FlowState {
    std::vector<float> h, a, z, w, wl;
    explicit FlowState(size_t n) : h(n), a(n), z(n), w(n), wl(n) {}
};

constexpr int kPlanetFlowIters = 8;
constexpr float kPlanetFlowCell = 19500.0f;   // the global grid's cells, metres

// The start procInit leaves: the ground in metres, the water high over the
// land (it drains from the coasts), nothing routed yet.
FlowState planet_flow_start(const std::vector<float>& field) {
    FlowState s(field.size());
    for (size_t i = 0; i < field.size(); ++i) {
        s.h[i] = field[i] * 100.0f;
        s.a[i] = 1.0f;
        s.wl[i] = s.h[i] < 0.0f ? s.h[i] : 1e5f;
    }
    return s;
}

// k_planet_flow in C++, the same operations in the same order.
void planet_flow_ref(const FlowState& in, FlowState& out, int W, int H, float cell) {
    const int nbx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int nby[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const float nd[8] = {1.41421356f, 1.0f, 1.41421356f, 1.0f, 1.0f, 1.41421356f, 1.0f, 1.41421356f};
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const size_t c = static_cast<size_t>(y) * W + x;
            float h = in.h[c];
            const float w = in.wl[c];
            const bool sea = h < 0.0f;
            float wn[8], a = 1.0f, rest = 1e30f, mean4 = 0.0f;
            for (int k = 0; k < 8; ++k) {
                const size_t n = static_cast<size_t>(std::clamp(y + nby[k], 0, H - 1)) * W + std::clamp(x + nbx[k], 0, W - 1);
                const float dist = nd[k] * cell;
                wn[k] = in.wl[n];
                if (wn[k] > w && in.w[n] > 0.0f) {
                    float t = std::min((wn[k] - w) / dist / in.w[n], 1.0f);
                    a += in.z[n] * ((t * t) * (t * t));
                }
                rest = std::min(rest, wn[k] + 2e-4f * dist);
                if (nd[k] == 1.0f) mean4 += 0.25f * in.h[n];
            }
            if (!sea) {
                if (w - h < 1.0f) {
                    float drop = 0.0f;
                    for (int k = 0; k < 8; ++k) drop = std::max(drop, (h - wn[k]) / nd[k]);
                    float e = 2.5e-4f * std::sqrt(a);
                    float hc = (h - drop * e / (1.0f + e)) + 0.0015f * (mean4 - h);
                    h = std::max(hc, std::min(h, 1.0f));
                } else {
                    h = h + 0.01f * (w - h);
                }
            }
            const float wl = sea ? h : std::max(h, rest);
            float smax = 0.0f, s = 0.0f;
            for (int k = 0; k < 8; ++k) smax = std::max(smax, (wl - wn[k]) / (nd[k] * cell));
            if (!sea && smax > 0.0f) {
                for (int k = 0; k < 8; ++k) {
                    float down = (wl - wn[k]) / (nd[k] * cell);
                    if (down > 0.0f) { float t = down / smax; s += (t * t) * (t * t); }
                }
            }
            out.h[c] = h;
            out.a[c] = a;
            out.z[c] = s > 0.0f ? a / s : 0.0f;
            out.w[c] = sea ? 0.0f : smax;
            out.wl[c] = wl;
        }
    }
}

const char* kGlslPlanetFlow = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC {
    uint64_t h; uint64_t z; uint64_t w; uint64_t wl;
    uint64_t h_out; uint64_t a_out; uint64_t z_out; uint64_t w_out; uint64_t wl_out;
    float cell; int width; int height;
} pc;
const ivec2 NB[8] = ivec2[8](ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0), ivec2(1, 0),
                             ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));
const float ND[8] = float[8](1.41421356, 1.0, 1.41421356, 1.0, 1.0, 1.41421356, 1.0, 1.41421356);
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= pc.width || p.y >= pc.height) return;
    int c = p.y * pc.width + p.x;
    float L = pc.cell;
    float h = FloatBuf(pc.h).v[c];
    float W = FloatBuf(pc.wl).v[c];
    bool sea = h < 0.0;
    float Wn[8];
    float A = 1.0, rest = 1e30, mean4 = 0.0;
    for (int k = 0; k < 8; ++k) {
        ivec2 q = clamp(p + NB[k], ivec2(0), ivec2(pc.width - 1, pc.height - 1));
        int n = q.y * pc.width + q.x;
        float dist = ND[k] * L;
        Wn[k] = FloatBuf(pc.wl).v[n];
        float nw = FloatBuf(pc.w).v[n];
        if (Wn[k] > W && nw > 0.0) {
            float t = min((Wn[k] - W) / dist / nw, 1.0);
            A += FloatBuf(pc.z).v[n] * ((t * t) * (t * t));
        }
        rest = min(rest, Wn[k] + 2e-4 * dist);
        if (ND[k] == 1.0) mean4 += 0.25 * FloatBuf(pc.h).v[n];
    }
    if (!sea) {
        if (W - h < 1.0) {
            float drop = 0.0;
            for (int k = 0; k < 8; ++k) drop = max(drop, (h - Wn[k]) / ND[k]);
            float e = 2.5e-4 * sqrt(A);
            float hc = (h - drop * e / (1.0 + e)) + 0.0015 * (mean4 - h);
            h = max(hc, min(h, 1.0));
        } else {
            h += 0.01 * (W - h);
        }
    }
    W = sea ? h : max(h, rest);
    float smax = 0.0, S = 0.0;
    for (int k = 0; k < 8; ++k) smax = max(smax, (W - Wn[k]) / (ND[k] * L));
    if (!sea && smax > 0.0)
        for (int k = 0; k < 8; ++k) {
            float down = (W - Wn[k]) / (ND[k] * L);
            if (down > 0.0) { float t = down / smax; S += (t * t) * (t * t); }
        }
    FloatBuf(pc.h_out).v[c] = h;
    FloatBuf(pc.a_out).v[c] = A;
    FloatBuf(pc.z_out).v[c] = S > 0.0 ? A / S : 0.0;
    FloatBuf(pc.w_out).v[c] = sea ? 0.0 : smax;
    FloatBuf(pc.wl_out).v[c] = W;
}
)GLSL";

BenchRow run_planet_flow_row(const CompileResult& res, uint32_t W, uint32_t H, const std::vector<float>& field,
                             uint32_t TW, uint32_t TH, const std::vector<float>& field_time) {
    BenchRow row;
    row.tier = "Stencil";
    row.name = "PlanetFlow_2D";
    row.dim = 2;
    const size_t N = static_cast<size_t>(W) * H;
    row.points = N;
    const GridShape shape(W, H);

    FlowState ref = planet_flow_start(field), ref_next(N);
    double ref_ms = best_cpu_ms([&] {
        FlowState a = planet_flow_start(field), b(N);
        for (int it = 0; it < kPlanetFlowIters; ++it) {
            planet_flow_ref(a, b, W, H, kPlanetFlowCell);
            std::swap(a, b);
        }
        ref = std::move(a);
    });
    row.cpu_ref_mpts = mpts(N * kPlanetFlowIters, ref_ms);

    const auto* k = res.find_kernel("k_planet_flow");
    auto args = [&](FlowState& in, FlowState& out) {
        return std::vector<KernelArg>{KernelArg::store(in.h), KernelArg::store(in.z), KernelArg::store(in.w),
                                      KernelArg::store(in.wl), KernelArg::store(out.h), KernelArg::store(out.a),
                                      KernelArg::store(out.z), KernelArg::store(out.w), KernelArg::store(out.wl),
                                      KernelArg(kPlanetFlowCell)};
    };
    FlowState cpu = planet_flow_start(field);
    double bk_cpu_ms = best_cpu_ms([&] {
        FlowState a = planet_flow_start(field), b(N);
        for (int it = 0; it < kPlanetFlowIters; ++it) {
            k->dispatch_cpu(shape, args(a, b));
            std::swap(a, b);
        }
        cpu = std::move(a);
    });
    row.cpu_bk_mpts = mpts(N * kPlanetFlowIters, bk_cpu_ms);

    FlowState gpu = planet_flow_start(field), gpu_b(N);
    for (int it = 0; it < kPlanetFlowIters; ++it) {
        k->dispatch_gpu(shape, args(gpu, gpu_b));
        std::swap(gpu, gpu_b);
    }

    // Heights to 5 mm (they are hundreds of metres; the kernel's fused
    // multiply-adds drift from the reference's rounding over the
    // iterations), the drainage to 1e-3 of a cell, the water's level as the
    // heights.
    auto check = [&](const FlowState& got, bool& pass, float& max_abs, uint32_t& max_ulp) {
        pass = true;
        max_abs = 0.0f;
        max_ulp = 0;
        for (const auto& [g, r, tol] : {std::tuple{&got.h, &ref.h, 5e-3f}, std::tuple{&got.a, &ref.a, 1e-3f},
                                        std::tuple{&got.wl, &ref.wl, 5e-3f}}) {
            auto s = compare_buffers(*g, *r, tol);
            pass = pass && s.pass;
            max_abs = std::max(max_abs, s.max_abs);
            max_ulp = std::max(max_ulp, s.max_ulp);
        }
    };
    check(cpu, row.cpu_pass, row.cpu_max_abs, row.cpu_max_ulp);
    check(gpu, row.gpu_pass, row.gpu_max_abs, row.gpu_max_ulp);

    // The GPU timed an iteration on the TW x TH field, from its start.
    const size_t NT = static_cast<size_t>(TW) * TH;
    FlowState t0 = planet_flow_start(field_time);
    std::vector<brass::gpu::VulkanBuffer> bufs;
    for (int i = 0; i < 9; ++i) bufs.push_back(brass::gpu::VulkanBuffer::alloc(NT * sizeof(float)));
    bufs[0].upload(t0.h.data(), NT * sizeof(float));
    bufs[1].upload(t0.z.data(), NT * sizeof(float));
    bufs[2].upload(t0.w.data(), NT * sizeof(float));
    bufs[3].upload(t0.wl.data(), NT * sizeof(float));
    std::vector<GpuArg> vk_args;
    for (auto& b : bufs) vk_args.emplace_back(b);
    vk_args.emplace_back(kPlanetFlowCell);
    double bk_gpu_ms = time_bk_gpu_ms(k, GridShape(TW, TH), vk_args);
    row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

    auto spv = compile_glsl_compute(kGlslPlanetFlow, "stencil_planet_flow");
    std::vector<brass::Type> layout(9, brass::Type::ptr());
    layout.push_back(brass::Type::f32());
    layout.push_back(brass::Type::i32());
    layout.push_back(brass::Type::i32());
    auto vmod = load_glsl_vulkan_module(spv, layout, 8, 8, 1);
    vk_args.emplace_back(static_cast<int32_t>(TW));
    vk_args.emplace_back(static_cast<int32_t>(TH));
    double glsl_ms = time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, glsl_args(vk_args));
    row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));

    const bool pass = row.cpu_pass && row.gpu_pass;
    row.cpu_ratio = pass ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
    row.has_gpu_baseline = true;
    row.gpu_ratio = pass ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
    return row;
}
} // namespace

std::vector<BenchRow> run_stencil_tier() {
    std::vector<BenchRow> rows;
    const uint32_t W = 256, H = 256;
    const size_t N = static_cast<size_t>(W) * H;
    const GridShape shape2d(W, H);

    std::vector<float> h_in(N);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float fx = static_cast<float>(x) * 0.05f;
            float fy = static_cast<float>(y) * 0.05f;
            h_in[y * W + x] = std::sin(fx) * std::cos(fy) * 10.0f + std::sin(fx * 2.3f + fy * 1.7f) * 3.0f;
        }
    }

    const char* bk_stencils_src = R"BK(
        @kernel @grid(2)
        fn k_blur_h(g: grid<2>, src: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            dst[x, y] = src[-2, 0] * 0.0625 + src[-1, 0] * 0.25 + src[0, 0] * 0.375 + src[1, 0] * 0.25 + src[2, 0] * 0.0625;
        }

        @kernel @grid(2)
        fn k_blur_v(g: grid<2>, src: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            dst[x, y] = src[0, -2] * 0.0625 + src[0, -1] * 0.25 + src[0, 0] * 0.375 + src[0, 1] * 0.25 + src[0, 2] * 0.0625;
        }

        @kernel @grid(2)
        fn k_grad_normals(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let dx = 0.5 * (h[1, 0] - h[-1, 0]);
            let dy = 0.5 * (h[0, 1] - h[0, -1]);
            dst[x, y] = 1.0 / sqrt(dx * dx + dy * dy + 1.0);
        }

        fn thermal_flux(d: f32, talus: f32) -> f32 {
            if d > talus { return d - talus; }
            if d < -talus { return d + talus; }
            return 0.0;
        }

        @kernel @grid(2)
        fn k_thermal_step(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let hc = h[0, 0];
            let talus = 0.1;
            let f = thermal_flux(h[-1, 0] - hc, talus)
                  + thermal_flux(h[ 1, 0] - hc, talus)
                  + thermal_flux(h[0, -1] - hc, talus)
                  + thermal_flux(h[0,  1] - hc, talus);
            dst[x, y] = hc + 0.125 * f;
        }

        @kernel @grid(2)
        fn k_hydraulic_step(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let hc = h[0, 0];
            let hL = h[-1, 0];
            let hR = h[ 1, 0];
            let hD = h[0, -1];
            let hU = h[0,  1];
            let dx = hR - hL;
            let dy = hU - hD;
            let slope = 0.5 * sqrt(dx * dx + dy * dy);
            let lap = 0.25 * (hL + hR + hD + hU) - hc;
            let capacity = slope * 0.15;
            dst[x, y] = hc - 0.05 * capacity + 0.1 * lap;
        }

        @kernel @grid(2)
        fn k_flow_accum(g: grid<2>, h: stencil<f32, 2>, dst: write_store<f32, 2>) {
            let x = g.x();
            let y = g.y();
            let hc = h[0, 0];
            let card = max(0.0, h[-1, 0] - hc)
                     + max(0.0, h[ 1, 0] - hc)
                     + max(0.0, h[0, -1] - hc)
                     + max(0.0, h[0,  1] - hc);
            let diag = max(0.0, h[-1, -1] - hc)
                     + max(0.0, h[ 1, -1] - hc)
                     + max(0.0, h[-1,  1] - hc)
                     + max(0.0, h[ 1,  1] - hc);
            dst[x, y] = 1.0 + card + 0.70710678 * diag;
        }

        // The planet's rivers (broworkshop 1fbd228^:demos/planet/terrain.js,
        // flow() as erode ran it on the global grid: no region edge, seedA
        // and ks 0, so its bed and slide clamps do nothing; FLOW_P 4 and M
        // 0.5 as squares and a root). Water level W drains from the sea
        // inward (Planchon-Darboux), drainage A is the cell and what its
        // higher neighbours hand it by slope^4, and land out of the water is
        // cut by implicit stream power and crept by diffusion.
        fn flow_in(wn: f32, nz: f32, nw: f32, w: f32, dist: f32) -> f32 {
            if wn > w && nw > 0.0 {
                let t = min((wn - w) / dist / nw, 1.0);
                return nz * ((t * t) * (t * t));
            }
            return 0.0;
        }
        fn flow_share(w: f32, wn: f32, dist: f32, smax: f32) -> f32 {
            let down = (w - wn) / dist;
            if down > 0.0 {
                let t = down / smax;
                return (t * t) * (t * t);
            }
            return 0.0;
        }

        @kernel @grid(2)
        fn k_planet_flow(g: grid<2>, hs: stencil<f32, 2>, zs: stencil<f32, 2>, ws: stencil<f32, 2>, wl: stencil<f32, 2>,
                         h_out: write_store<f32, 2>, a_out: write_store<f32, 2>, z_out: write_store<f32, 2>,
                         w_out: write_store<f32, 2>, wl_out: write_store<f32, 2>, cell: f32) {
            let x = g.x();
            let y = g.y();
            let fill = 2e-4;
            let k_rate = 2.5e-4;
            let creep = 0.0015;
            let silt = 0.01;
            var h = hs[0, 0];
            let w = wl[0, 0];
            let sea = h < 0.0;
            let d1 = cell;
            let d2 = 1.41421356 * cell;
            let w0 = wl[-1, -1]; let w1 = wl[0, -1]; let w2 = wl[1, -1]; let w3 = wl[-1, 0];
            let w4 = wl[1, 0]; let w5 = wl[-1, 1]; let w6 = wl[0, 1]; let w7 = wl[1, 1];
            var a = 1.0;
            a += flow_in(w0, zs[-1, -1], ws[-1, -1], w, d2);
            a += flow_in(w1, zs[0, -1], ws[0, -1], w, d1);
            a += flow_in(w2, zs[1, -1], ws[1, -1], w, d2);
            a += flow_in(w3, zs[-1, 0], ws[-1, 0], w, d1);
            a += flow_in(w4, zs[1, 0], ws[1, 0], w, d1);
            a += flow_in(w5, zs[-1, 1], ws[-1, 1], w, d2);
            a += flow_in(w6, zs[0, 1], ws[0, 1], w, d1);
            a += flow_in(w7, zs[1, 1], ws[1, 1], w, d2);
            let rest = min(min(min(w0 + fill * d2, w1 + fill * d1), min(w2 + fill * d2, w3 + fill * d1)),
                           min(min(w4 + fill * d1, w5 + fill * d2), min(w6 + fill * d1, w7 + fill * d2)));
            let mean4 = ((0.25 * hs[0, -1] + 0.25 * hs[-1, 0]) + 0.25 * hs[1, 0]) + 0.25 * hs[0, 1];
            if !sea {
                if w - h < 1.0 {
                    let drop = max(max(max(max(0.0, (h - w0) / 1.41421356), h - w1), max((h - w2) / 1.41421356, h - w3)),
                                   max(max(h - w4, (h - w5) / 1.41421356), max(h - w6, (h - w7) / 1.41421356)));
                    let e = k_rate * sqrt(a);
                    let hc = (h - drop * e / (1.0 + e)) + creep * (mean4 - h);
                    h = max(hc, min(h, 1.0));
                } else {
                    h = h + silt * (w - h);
                }
            }
            let wn = sea ? h : max(h, rest);
            let smax = max(max(max(max(0.0, (wn - w0) / d2), (wn - w1) / d1), max((wn - w2) / d2, (wn - w3) / d1)),
                           max(max((wn - w4) / d1, (wn - w5) / d2), max((wn - w6) / d1, (wn - w7) / d2)));
            var s = 0.0;
            if !sea && smax > 0.0 {
                s = flow_share(wn, w0, d2, smax) + flow_share(wn, w1, d1, smax) + flow_share(wn, w2, d2, smax)
                  + flow_share(wn, w3, d1, smax) + flow_share(wn, w4, d1, smax) + flow_share(wn, w5, d2, smax)
                  + flow_share(wn, w6, d1, smax) + flow_share(wn, w7, d2, smax);
            }
            h_out[x, y] = h;
            a_out[x, y] = a;
            z_out[x, y] = s > 0.0 ? a / s : 0.0;
            w_out[x, y] = sea ? 0.0 : smax;
            wl_out[x, y] = wn;
        }
    )BK";

    auto res = compile_source(bk_stencils_src, "stencils_bench.bk");
    if (!res.ok) {
        std::cerr << "Stencil compile error:\n" << res.error_message() << "\n";
        return rows;
    }

    auto clamp_i = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    // The GPU is timed on a kGpuTimeSide grid of the same field.
    const uint32_t TW = kGpuTimeSide, TH = kGpuTimeSide;
    const size_t NT = static_cast<size_t>(TW) * TH;
    const GridShape shape_t(TW, TH);
    std::vector<float> h_time(NT);
    for (uint32_t y = 0; y < TH; ++y) {
        for (uint32_t x = 0; x < TW; ++x) {
            float fx = static_cast<float>(x) * 0.05f;
            float fy = static_cast<float>(y) * 0.05f;
            h_time[y * TW + x] = std::sin(fx) * std::cos(fy) * 10.0f + std::sin(fx * 2.3f + fy * 1.7f) * 3.0f;
        }
    }
    brass::gpu::VulkanBuffer d_in = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_tmp = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    brass::gpu::VulkanBuffer d_out = brass::gpu::VulkanBuffer::alloc(NT * sizeof(float));
    d_in.upload(h_time.data(), NT * sizeof(float));

    // 1. Separable Blur (Horizontal + Vertical)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "SeparableBlur_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> ref_tmp(N), ref_out(N);
        double ref_ms = best_cpu_ms([&] {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    ref_tmp[y * W + x] = h_in[y * W + clamp_i(x - 2, 0, W - 1)] * 0.0625f
                                       + h_in[y * W + clamp_i(x - 1, 0, W - 1)] * 0.25f
                                       + h_in[y * W + x] * 0.375f
                                       + h_in[y * W + clamp_i(x + 1, 0, W - 1)] * 0.25f
                                       + h_in[y * W + clamp_i(x + 2, 0, W - 1)] * 0.0625f;
                }
            }
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    ref_out[y * W + x] = ref_tmp[clamp_i(y - 2, 0, H - 1) * W + x] * 0.0625f
                                       + ref_tmp[clamp_i(y - 1, 0, H - 1) * W + x] * 0.25f
                                       + ref_tmp[y * W + x] * 0.375f
                                       + ref_tmp[clamp_i(y + 1, 0, H - 1) * W + x] * 0.25f
                                       + ref_tmp[clamp_i(y + 2, 0, H - 1) * W + x] * 0.0625f;
                }
            }
        });
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* kh = res.find_kernel("k_blur_h");
        const auto* kv = res.find_kernel("k_blur_v");
        std::vector<float> cpu_tmp(N), cpu_out(N), gpu_tmp(N), gpu_out(N);

        double bk_cpu_ms = best_cpu_ms([&] {
            kh->dispatch_cpu(shape2d, {KernelArg::store(h_in), KernelArg::store(cpu_tmp)});
            kv->dispatch_cpu(shape2d, {KernelArg::store(cpu_tmp), KernelArg::store(cpu_out)});
        });
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        kh->dispatch_gpu(shape2d, {KernelArg::store(h_in), KernelArg::store(gpu_tmp)});
        kv->dispatch_gpu(shape2d, {KernelArg::store(gpu_tmp), KernelArg::store(gpu_out)});

        double bk_gpu_ms = time_bk_gpu_ms(kh, shape_t, {d_in, d_tmp}) +
                           time_bk_gpu_ms(kv, shape_t, {d_tmp, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref_out, 1e-4f);
        auto gs = compare_buffers(gpu_out, ref_out, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int clx(int x) { return clamp(x, 0, pc.w - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    FloatBuf(pc.dst).v[y * pc.w + x] =
        s.v[y * pc.w + clx(x - 2)] * 0.0625 + s.v[y * pc.w + clx(x - 1)] * 0.25 +
        s.v[y * pc.w + x] * 0.375 + s.v[y * pc.w + clx(x + 1)] * 0.25 + s.v[y * pc.w + clx(x + 2)] * 0.0625;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_blur");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = 2.0 * time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 2. Gradient & Normals from Height Grid
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "GradientNormals_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> ref_out(N);
        double ref_ms = best_cpu_ms([&] {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    float dx = 0.5f * (h_in[y * W + clamp_i(x + 1, 0, W - 1)] - h_in[y * W + clamp_i(x - 1, 0, W - 1)]);
                    float dy = 0.5f * (h_in[clamp_i(y + 1, 0, H - 1) * W + x] - h_in[clamp_i(y - 1, 0, H - 1) * W + x]);
                    ref_out[y * W + x] = 1.0f / std::sqrt(dx * dx + dy * dy + 1.0f);
                }
            }
        });
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_grad_normals");
        std::vector<float> cpu_out(N), gpu_out(N);
        double bk_cpu_ms = best_cpu_ms([&] {
            k->dispatch_cpu(shape2d, {KernelArg::store(h_in), KernelArg::store(cpu_out)});
        });
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        k->dispatch_gpu(shape2d, {KernelArg::store(h_in), KernelArg::store(gpu_out)});
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref_out, 1e-4f);
        auto gs = compare_buffers(gpu_out, ref_out, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float dx = 0.5 * (s.v[y * pc.w + cl(x + 1, pc.w)] - s.v[y * pc.w + cl(x - 1, pc.w)]);
    float dy = 0.5 * (s.v[cl(y + 1, pc.h) * pc.w + x] - s.v[cl(y - 1, pc.h) * pc.w + x]);
    FloatBuf(pc.dst).v[y * pc.w + x] = 1.0 / sqrt(dx * dx + dy * dy + 1.0);
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_grad_normals");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 3. Thermal Erosion (3 iterations)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "ThermalErosion_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> cur, nxt(N);
        double ref_ms = best_cpu_ms([&] {
            cur = h_in;
            for (int iter = 0; iter < 3; ++iter) {
                for (int y = 0; y < (int)H; ++y) {
                    for (int x = 0; x < (int)W; ++x) {
                        float hc = cur[y * W + x];
                        auto flux = [&](float hn) {
                            float d = hn - hc;
                            if (d > 0.1f) return d - 0.1f;
                            if (d < -0.1f) return d + 0.1f;
                            return 0.0f;
                        };
                        float f = flux(cur[y * W + clamp_i(x - 1, 0, W - 1)])
                                + flux(cur[y * W + clamp_i(x + 1, 0, W - 1)])
                                + flux(cur[clamp_i(y - 1, 0, H - 1) * W + x])
                                + flux(cur[clamp_i(y + 1, 0, H - 1) * W + x]);
                        nxt[y * W + x] = hc + 0.125f * f;
                    }
                }
                cur.swap(nxt);
            }
        });
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_thermal_step");
        std::vector<float> cpu_a = h_in, cpu_b(N), gpu_a = h_in, gpu_b(N);
        double bk_cpu_ms = best_cpu_ms([&] {
            cpu_a = h_in;
            for (int iter = 0; iter < 3; ++iter) {
                k->dispatch_cpu(shape2d, {KernelArg::store(cpu_a), KernelArg::store(cpu_b)});
                cpu_a.swap(cpu_b);
            }
        });
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        for (int iter = 0; iter < 3; ++iter) {
            k->dispatch_gpu(shape2d, {KernelArg::store(gpu_a), KernelArg::store(gpu_b)});
            gpu_a.swap(gpu_b);
        }
        double bk_gpu_ms = 3.0 * time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_a, cur, 1e-4f);
        auto gs = compare_buffers(gpu_a, cur, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
float flux(float d) {
    if (d > 0.1) return d - 0.1;
    if (d < -0.1) return d + 0.1;
    return 0.0;
}
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float hc = s.v[y * pc.w + x];
    float f = flux(s.v[y * pc.w + cl(x - 1, pc.w)] - hc)
            + flux(s.v[y * pc.w + cl(x + 1, pc.w)] - hc)
            + flux(s.v[cl(y - 1, pc.h) * pc.w + x] - hc)
            + flux(s.v[cl(y + 1, pc.h) * pc.w + x] - hc);
    FloatBuf(pc.dst).v[y * pc.w + x] = hc + 0.125 * f;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_thermal");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = 3.0 * time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 4. Hydraulic Erosion (3 iterations)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "HydraulicErosion_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> cur, nxt(N);
        double ref_ms = best_cpu_ms([&] {
            cur = h_in;
            for (int iter = 0; iter < 3; ++iter) {
                for (int y = 0; y < (int)H; ++y) {
                    for (int x = 0; x < (int)W; ++x) {
                        float hc = cur[y * W + x];
                        float hL = cur[y * W + clamp_i(x - 1, 0, W - 1)];
                        float hR = cur[y * W + clamp_i(x + 1, 0, W - 1)];
                        float hD = cur[clamp_i(y - 1, 0, H - 1) * W + x];
                        float hU = cur[clamp_i(y + 1, 0, H - 1) * W + x];
                        float dx = hR - hL;
                        float dy = hU - hD;
                        float slope = 0.5f * std::sqrt(dx * dx + dy * dy);
                        float lap = 0.25f * (hL + hR + hD + hU) - hc;
                        nxt[y * W + x] = hc - 0.05f * (slope * 0.15f) + 0.1f * lap;
                    }
                }
                cur.swap(nxt);
            }
        });
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_hydraulic_step");
        std::vector<float> cpu_a = h_in, cpu_b(N), gpu_a = h_in, gpu_b(N);
        double bk_cpu_ms = best_cpu_ms([&] {
            cpu_a = h_in;
            for (int iter = 0; iter < 3; ++iter) {
                k->dispatch_cpu(shape2d, {KernelArg::store(cpu_a), KernelArg::store(cpu_b)});
                cpu_a.swap(cpu_b);
            }
        });
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        for (int iter = 0; iter < 3; ++iter) {
            k->dispatch_gpu(shape2d, {KernelArg::store(gpu_a), KernelArg::store(gpu_b)});
            gpu_a.swap(gpu_b);
        }
        double bk_gpu_ms = 3.0 * time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_a, cur, 1e-4f);
        auto gs = compare_buffers(gpu_a, cur, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float hc = s.v[y * pc.w + x];
    float hL = s.v[y * pc.w + cl(x - 1, pc.w)];
    float hR = s.v[y * pc.w + cl(x + 1, pc.w)];
    float hD = s.v[cl(y - 1, pc.h) * pc.w + x];
    float hU = s.v[cl(y + 1, pc.h) * pc.w + x];
    float dx = hR - hL, dy = hU - hD;
    float slope = 0.5 * sqrt(dx * dx + dy * dy);
    float lap = 0.25 * (hL + hR + hD + hU) - hc;
    FloatBuf(pc.dst).v[y * pc.w + x] = hc - 0.05 * (slope * 0.15) + 0.1 * lap;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_hydraulic");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = 3.0 * time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 5. Flow Accumulation (8-neighbour upslope inflow)
    {
        BenchRow row;
        row.tier = "Stencil";
        row.name = "FlowAccumulation_2D";
        row.dim = 2;
        row.points = N;

        std::vector<float> ref_out(N);
        double ref_ms = best_cpu_ms([&] {
            for (int y = 0; y < (int)H; ++y) {
                for (int x = 0; x < (int)W; ++x) {
                    float hc = h_in[y * W + x];
                    int xm = clamp_i(x - 1, 0, W - 1), xp = clamp_i(x + 1, 0, W - 1);
                    int ym = clamp_i(y - 1, 0, H - 1), yp = clamp_i(y + 1, 0, H - 1);
                    float card = std::max(0.0f, h_in[y * W + xm] - hc)
                               + std::max(0.0f, h_in[y * W + xp] - hc)
                               + std::max(0.0f, h_in[ym * W + x] - hc)
                               + std::max(0.0f, h_in[yp * W + x] - hc);
                    float diag = std::max(0.0f, h_in[ym * W + xm] - hc)
                               + std::max(0.0f, h_in[ym * W + xp] - hc)
                               + std::max(0.0f, h_in[yp * W + xm] - hc)
                               + std::max(0.0f, h_in[yp * W + xp] - hc);
                    ref_out[y * W + x] = 1.0f + card + 0.70710678f * diag;
                }
            }
        });
        row.cpu_ref_mpts = (N / 1e6) / (std::max(ref_ms, 1e-4) / 1000.0);

        const auto* k = res.find_kernel("k_flow_accum");
        std::vector<float> cpu_out(N), gpu_out(N);
        double bk_cpu_ms = best_cpu_ms([&] {
            k->dispatch_cpu(shape2d, {KernelArg::store(h_in), KernelArg::store(cpu_out)});
        });
        row.cpu_bk_mpts = (N / 1e6) / (std::max(bk_cpu_ms, 1e-4) / 1000.0);

        k->dispatch_gpu(shape2d, {KernelArg::store(h_in), KernelArg::store(gpu_out)});
        double bk_gpu_ms = time_bk_gpu_ms(k, shape_t, {d_in, d_out});
        row.gpu_bk_mpts = mpts(NT, std::max(bk_gpu_ms, 1e-4));

        auto cs = compare_buffers(cpu_out, ref_out, 1e-4f);
        auto gs = compare_buffers(gpu_out, ref_out, 1e-4f);
        row.cpu_pass = cs.pass; row.cpu_max_abs = cs.max_abs; row.cpu_max_ulp = cs.max_ulp;
        row.gpu_pass = gs.pass; row.gpu_max_abs = gs.max_abs; row.gpu_max_ulp = gs.max_ulp;

        std::string glsl = R"GLSL(
#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_ARB_gpu_shader_int64 : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(push_constant) uniform PC { uint64_t src; uint64_t dst; int w; int h; } pc;
int cl(int v, int hi) { return clamp(v, 0, hi - 1); }
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= pc.w || y >= pc.h) return;
    FloatBuf s = FloatBuf(pc.src);
    float hc = s.v[y * pc.w + x];
    int xm = cl(x - 1, pc.w), xp = cl(x + 1, pc.w);
    int ym = cl(y - 1, pc.h), yp = cl(y + 1, pc.h);
    float card = max(0.0, s.v[y * pc.w + xm] - hc) + max(0.0, s.v[y * pc.w + xp] - hc)
               + max(0.0, s.v[ym * pc.w + x] - hc) + max(0.0, s.v[yp * pc.w + x] - hc);
    float diag = max(0.0, s.v[ym * pc.w + xm] - hc) + max(0.0, s.v[ym * pc.w + xp] - hc)
               + max(0.0, s.v[yp * pc.w + xm] - hc) + max(0.0, s.v[yp * pc.w + xp] - hc);
    FloatBuf(pc.dst).v[y * pc.w + x] = 1.0 + card + 0.70710678 * diag;
}
)GLSL";
        auto spv = compile_glsl_compute(glsl, "stencil_flow_accum");
        auto vmod = load_glsl_vulkan_module(spv, {brass::Type::ptr(), brass::Type::ptr(), brass::Type::i32(), brass::Type::i32()}, 8, 8, 1);
        double glsl_ms = time_glsl_gpu_ms(vmod, TW / 8, TH / 8, 1, 8, 8, 1, {d_in, d_out, (int32_t)TW, (int32_t)TH});
        row.gpu_glsl_mpts = mpts(NT, std::max(glsl_ms, 1e-4));
        row.cpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.cpu_bk_mpts / row.cpu_ref_mpts) : 0.0;
        row.has_gpu_baseline = true;
        row.gpu_ratio = (row.cpu_pass && row.gpu_pass) ? (row.gpu_bk_mpts / row.gpu_glsl_mpts) : 0.0;
        rows.push_back(row);
    }

    // 6. The planet's rivers (k_planet_flow), kPlanetFlowIters iterations
    // from the sea-drained start, over the field in metres (half of it sea).
    rows.push_back(run_planet_flow_row(res, W, H, h_in, TW, TH, h_time));

    return rows;
}
