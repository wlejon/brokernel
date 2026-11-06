#include "bench_hillclimb.h"

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

    return rows;
}
