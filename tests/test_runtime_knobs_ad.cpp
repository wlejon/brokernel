#include "brokernel/brokernel.h"

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

static ConstValue make_vec3(float x, float y, float z) {
    ConstValue c;
    c.type = Type::vec(ScalarKind::F32, 3);
    c.elements = {ConstValue::from_f32(x), ConstValue::from_f32(y), ConstValue::from_f32(z)};
    return c;
}

static void test_runtime_knobs_cpu_and_gpu() {
    const char* src = R"(
        @knob const GAIN: f32 = 2.0;
        @knob const OFFSET: vec3<f32> = vec3<f32>(1.0, -2.0, 0.5);

        @kernel fn eval_gain(g: grid1d, out: write_store<f32, 1>) {
            let x: f32 = f32(g.x());
            out[g.x()] = x * GAIN + OFFSET.x + OFFSET.y + OFFSET.z;
        }
    )";

    CompileOptions opts;
    opts.runtime_knobs = true;
    auto res = compile_source(src, "rt_knobs.bk", opts);
    if (!res.ok) std::fprintf(stderr, "compile err: %s\n", res.error_message().c_str());
    CHECK(res.ok);
    const CompiledKernel* k = res.find_kernel("eval_gain");
    CHECK(k != nullptr);
    CHECK(k->runtime_knobs.size() == 2);
    const uint64_t hash0 = k->code_hash;

    const size_t n = 16;
    std::vector<float> out(n, 0.0f);
    KernelArg args[] = {KernelArg::store(out)};
    std::string err;
    CHECK(k->dispatch_cpu(GridShape(n), args, nullptr, &err));
    // Default: 2.0 * i + (1 - 2 + 0.5) = 2*i - 0.5
    for (size_t i = 0; i < n; ++i) {
        CHECK(std::fabs(out[i] - (2.0f * static_cast<float>(i) - 0.5f)) < 1e-5f);
    }

    // Mutate runtime knobs in-place with ZERO recompilation
    CHECK(k->set_knobs({
        {"GAIN", ConstValue::from_f32(5.0f)},
        {"OFFSET", make_vec3(10.0f, 20.0f, 30.0f)},
    }));
    CHECK(k->code_hash == hash0);
    CHECK(k->runtime_knob_generation() >= 2);
    CHECK(k->dispatch_cpu(GridShape(n), args, nullptr, &err));
    for (size_t i = 0; i < n; ++i) {
        const float expected = 5.0f * static_cast<float>(i) + 60.0f;
        if (std::fabs(out[i] - expected) >= 1e-4f) {
            std::fprintf(stderr, "i=%zu got=%f expected=%f (gen=%llu)\n",
                         i, out[i], expected, static_cast<unsigned long long>(k->runtime_knob_generation()));
            CHECK(false);
        }
    }

    // Reject unknown knob name cleanly
    CHECK(!k->set_knobs({{"NO_SUCH_KNOB", ConstValue::from_f32(1.0f)}}));

    // Also test GPU dispatch + parity if Vulkan device available
    if (brass::gpu::vulkan_available()) {
        std::vector<float> gpu_host(n, 0.0f);
        KernelArg gpu_args[] = {KernelArg::store(gpu_host)};
        CHECK(k->dispatch_gpu(GridShape(n), gpu_args, nullptr, &err));
        for (size_t i = 0; i < n; ++i) {
            CHECK(std::fabs(gpu_host[i] - out[i]) < 1e-4f);
        }

        // Update knobs again and re-run GPU without recompiling
        CHECK(k->set_knobs({{"GAIN", ConstValue::from_f32(3.0f)}}));
        CHECK(k->dispatch_gpu(GridShape(n), gpu_args, nullptr, &err));
        for (size_t i = 0; i < n; ++i) {
            CHECK(std::fabs(gpu_host[i] - (3.0f * static_cast<float>(i) + 60.0f)) < 1e-4f);
        }
    }
    std::printf("test_runtime_knobs_cpu_and_gpu OK\n");
}

static void test_grad_knob_and_compile_knob_grads() {
    const char* src = R"(
        @knob const AMP: f32 = 1.5;
        @knob const FREQ: f32 = 2.0;
        @knob const WEIGHTS: vec3<f32> = vec3<f32>(0.5, -1.0, 2.0);

        pub fn helper(t: f32) -> f32 {
            return sin(FREQ * t) * AMP;
        }

        pub fn loss(x: f32, y: f32) -> f32 {
            let h = helper(x);
            return h * h + WEIGHTS.x * x + WEIGHTS.y * y + WEIGHTS.z * x * y;
        }

        @kernel fn grad_inline(x: f32, y: f32, out: write_store<vec2<f32>, 1>) {
            let da = grad_knob(loss, AMP, x, y);
            let df = grad_knob(loss, FREQ, x, y);
            out[0] = vec2<f32>(da, df);
        }
    )";

    // Test both compile-time and runtime knob AD!
    for (bool rt : {false, true}) {
        CompileOptions opts;
        opts.runtime_knobs = rt;
        auto res = compile_source(src, "knob_ad.bk", opts);
        if (!res.ok) {
            std::fprintf(stderr, "compile failed (rt=%d): %s\n", rt ? 1 : 0, res.error_message().c_str());
        }
        CHECK(res.ok);

        const float x = 0.7f, y = -0.3f;
        const CompiledKernel* gi = res.find_kernel("grad_inline");
        CHECK(gi != nullptr);
        float inline_out[2] = {0.0f, 0.0f};
        KernelArg gi_args[] = {
            KernelArg(x),
            KernelArg(y),
            KernelArg::buffer(inline_out, sizeof(inline_out)),
        };
        std::string err;
        CHECK(gi->dispatch_cpu(GridShape(1), gi_args, nullptr, &err));

        const float s = std::sin(2.0f * x);
        const float c = std::cos(2.0f * x);
        const float exp_da = 2.0f * 1.5f * s * s;
        const float exp_df = 2.0f * 1.5f * 1.5f * s * c * x;
        CHECK(std::fabs(inline_out[0] - exp_da) < 1e-4f);
        CHECK(std::fabs(inline_out[1] - exp_df) < 1e-4f);

        auto grads = res.compile_knob_grads("loss", {"AMP", "FREQ", "WEIGHTS"});
        if (!grads.ok) {
            std::fprintf(stderr, "compile_knob_grads failed: %s\n", grads.error_message().c_str());
        }
        CHECK(grads.ok);
        CHECK(grads.kernels.size() == 3);
        KernelArg l_args[] = {KernelArg(x), KernelArg(y)};
        CHECK(std::fabs(grads.kernels[0].eval_f32(l_args) - exp_da) < 1e-4f);
        CHECK(std::fabs(grads.kernels[1].eval_f32(l_args) - exp_df) < 1e-4f);
        float wgrad[3] = {0.0f, 0.0f, 0.0f};
        CHECK(grads.kernels[2].eval_cpu(l_args, wgrad, &err));
        CHECK(std::fabs(wgrad[0] - x) < 1e-4f);
        CHECK(std::fabs(wgrad[1] - y) < 1e-4f);
        CHECK(std::fabs(wgrad[2] - x * y) < 1e-4f);
    }
    std::printf("test_grad_knob_and_compile_knob_grads OK\n");
}

int main() {
    test_runtime_knobs_cpu_and_gpu();
    test_grad_knob_and_compile_knob_grads();
    return 0;
}
