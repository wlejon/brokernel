#include "brokernel/brokernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace brokernel;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK FAILED (%s:%d): %s\n", __FILE__, __LINE__, #cond);         \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

ConstValue make_vec3(float x, float y, float z) {
    ConstValue c;
    c.type = Type::vec(ScalarKind::F32, 3);
    c.elements = {ConstValue::from_f32(x), ConstValue::from_f32(y), ConstValue::from_f32(z)};
    return c;
}

void test_grid_gradient_optimization_cpu_and_gpu() {
    // Fit y_hat(t) = A * cos(t) + B.x + B.y * sin(t) + B.z * cos(2*t) over N=64 orthogonal samples.
    const char* src = R"(
        @knob const A: f32 = 0.0;
        @knob const B: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

        pub fn cell_loss(g: grid<1>, target: read_store<f32, 1>, inv_n: f32) -> f32 {
            let i: i32 = g.x();
            let t: f32 = f32(i) * 0.09817477;
            let pred: f32 = A * cos(t) + B.x + B.y * sin(t) + B.z * cos(2.0 * t);
            let diff: f32 = pred - target[i];
            return 0.5 * diff * diff * inv_n;
        }

        @kernel @grid(1)
        pub fn eval_loss_grid(g: grid<1>, target: read_store<f32, 1>, inv_n: f32, out_loss: rw_store<f32, 1>) {
            out_loss.atomic_add(0, cell_loss(g, target, inv_n));
        }
    )";

    CompileOptions opts;
    opts.runtime_knobs = true;
    auto res = compile_source(src, "grid_opt.bk", opts);
    if (!res.ok) {
        std::fprintf(stderr, "compile failed: %s\n", res.error_message().c_str());
    }
    CHECK(res.ok);

    const CompiledKernel* k_loss = res.find_kernel("eval_loss_grid");
    CHECK(k_loss != nullptr);

    auto grad_res = res.compile_knob_grid_grads("cell_loss", {"A", "B"}, 1);
    if (!grad_res.ok) {
        std::fprintf(stderr, "compile_knob_grid_grads failed: %s\n", grad_res.error_message().c_str());
    }
    CHECK(grad_res.ok);
    const CompiledKernel* k_grad = grad_res.find_kernel("grad_grid_cell_loss");
    CHECK(k_grad != nullptr);
    CHECK(k_grad->grid_dim == 1);
    CHECK(k_grad->runtime_knobs.size() == 2);

    constexpr size_t N = 64;
    const float inv_n = 1.0f / static_cast<float>(N);
    const float true_a = 1.75f;
    const float true_bx = -0.6f;
    const float true_by = 0.35f;
    const float true_bz = 0.8f;

    std::vector<float> target(N);
    for (size_t i = 0; i < N; ++i) {
        const float t = static_cast<float>(i) * 0.09817477f;
        target[i] = true_a * std::cos(t) + true_bx + true_by * std::sin(t) + true_bz * std::cos(2.0f * t);
    }

    auto compute_loss_cpu = [&](float a, float bx, float by, float bz) -> float {
        float acc = 0.0f;
        CHECK(k_loss->set_knobs({{"A", ConstValue::from_f32(a)}, {"B", make_vec3(bx, by, bz)}}));
        KernelArg args[] = {KernelArg::store(target), KernelArg(inv_n), KernelArg::buffer(&acc, sizeof(acc))};
        std::string err;
        CHECK(k_loss->dispatch_cpu(GridShape(N), args, nullptr, &err));
        return acc;
    };

    // 1. Finite-difference verification against compiled grad_grid kernel on CPU.
    {
        const float a0 = 0.5f, bx0 = 0.2f, by0 = -0.1f, bz0 = 0.3f;
        CHECK(k_grad->set_knobs({{"A", ConstValue::from_f32(a0)}, {"B", make_vec3(bx0, by0, bz0)}}));
        float grads[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        KernelArg gargs[] = {KernelArg::store(target), KernelArg(inv_n), KernelArg::buffer(grads, sizeof(grads))};
        std::string err;
        CHECK(k_grad->dispatch_cpu(GridShape(N), gargs, nullptr, &err));

        const float eps = 1e-3f;
        const float fd_a = (compute_loss_cpu(a0 + eps, bx0, by0, bz0) - compute_loss_cpu(a0 - eps, bx0, by0, bz0)) / (2.0f * eps);
        const float fd_bx = (compute_loss_cpu(a0, bx0 + eps, by0, bz0) - compute_loss_cpu(a0, bx0 - eps, by0, bz0)) / (2.0f * eps);
        const float fd_by = (compute_loss_cpu(a0, bx0, by0 + eps, bz0) - compute_loss_cpu(a0, bx0, by0 - eps, bz0)) / (2.0f * eps);
        const float fd_bz = (compute_loss_cpu(a0, bx0, by0, bz0 + eps) - compute_loss_cpu(a0, bx0, by0, bz0 - eps)) / (2.0f * eps);
        CHECK(std::fabs(grads[0] - fd_a) < 5e-3f);
        CHECK(std::fabs(grads[1] - fd_bx) < 5e-3f);
        CHECK(std::fabs(grads[2] - fd_by) < 5e-3f);
        CHECK(std::fabs(grads[3] - fd_bz) < 5e-3f);
    }

    // 2. Multi-step Adam optimization loop with zero recompilation (code_hash invariant).
    const uint64_t grad_hash0 = k_grad->code_hash;
    float p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float loss0 = compute_loss_cpu(p[0], p[1], p[2], p[3]);
    CHECK(loss0 > 1.0f);

    const float lr = 0.08f, b1 = 0.9f, b2 = 0.999f, adam_eps = 1e-8f;
    for (int step = 1; step <= 160; ++step) {
        CHECK(k_grad->set_knobs({{"A", ConstValue::from_f32(p[0])},
                                 {"B", make_vec3(p[1], p[2], p[3])}}));
        float grads[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        KernelArg gargs[] = {KernelArg::store(target), KernelArg(inv_n), KernelArg::buffer(grads, sizeof(grads))};
        std::string err;
        CHECK(k_grad->dispatch_cpu(GridShape(N), gargs, nullptr, &err));
        const float bc1 = 1.0f - std::pow(b1, static_cast<float>(step));
        const float bc2 = 1.0f - std::pow(b2, static_cast<float>(step));
        for (int j = 0; j < 4; ++j) {
            m[j] = b1 * m[j] + (1.0f - b1) * grads[j];
            v[j] = b2 * v[j] + (1.0f - b2) * grads[j] * grads[j];
            p[j] -= lr * (m[j] / bc1) / (std::sqrt(v[j] / bc2) + adam_eps);
        }
    }
    CHECK(k_grad->code_hash == grad_hash0);
    const float loss_final = compute_loss_cpu(p[0], p[1], p[2], p[3]);
    std::printf("  adam_800: loss0=%.4f loss_final=%.6f p=(%.3f, %.3f, %.3f, %.3f)\n",
                loss0, loss_final, p[0], p[1], p[2], p[3]);
    CHECK(loss_final * 100.0f < loss0);
    CHECK(loss_final < 1e-4f);
    CHECK(std::fabs(p[0] - true_a) < 5e-2f);
    CHECK(std::fabs(p[1] - true_bx) < 5e-2f);
    CHECK(std::fabs(p[2] - true_by) < 5e-2f);
    CHECK(std::fabs(p[3] - true_bz) < 5e-2f);

    // 3. GPU atomic_add reduction parity check (if Vulkan device supports it).
    if (brass::gpu::vulkan_available()) {
        CHECK(k_grad->set_knobs({{"A", ConstValue::from_f32(0.5f)},
                                 {"B", make_vec3(0.2f, -0.1f, 0.3f)}}));
        float cpu_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float gpu_g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        KernelArg cargs[] = {KernelArg::store(target), KernelArg(inv_n), KernelArg::buffer(cpu_g, sizeof(cpu_g))};
        KernelArg gargs[] = {KernelArg::store(target), KernelArg(inv_n), KernelArg::buffer(gpu_g, sizeof(gpu_g))};
        std::string err;
        CHECK(k_grad->dispatch_cpu(GridShape(N), cargs, nullptr, &err));
        if (k_grad->dispatch_gpu(GridShape(N), gargs, nullptr, &err)) {
            for (int j = 0; j < 4; ++j) {
                CHECK(std::fabs(gpu_g[j] - cpu_g[j]) < 1e-3f);
            }
            std::printf("test_knob_grid_opt GPU atomic_add parity OK\n");
        } else {
            std::printf("test_knob_grid_opt GPU skipped (%s)\n", err.c_str());
        }
    }
    std::printf("test_grid_gradient_optimization OK (loss0=%.4f -> loss_final=%.6f)\n", loss0, loss_final);
}

} // namespace

int main() {
    test_grid_gradient_optimization_cpu_and_gpu();
    return 0;
}
