// Tests std::dsp, std::image, and std::sim domain stdlib modules on CPU and GPU.
#include "brokernel/brokernel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

void test_dsp_module() {
    const char* src = R"bk(
import std::dsp;

@kernel
fn k_db(db: f32) -> f32 {
    return dsp::lin_to_db(dsp::db_to_lin(db));
}

@kernel
fn k_clip(x: f32) -> f32 {
    return dsp::soft_clip(x, 2.0) + dsp::tanh_sat(x, 0.5) + dsp::one_pole(0.0, 1.0, 0.25);
}

@kernel @grid(1)
fn k_biquad_impulse(g: grid<1>, out: write_store<f32, 1>) {
    let c = dsp::biquad_lowpass(1000.0, 0.707, 48000.0);
    var z1: f32 = 0.0;
    var z2: f32 = 0.0;
    for i in 0..8 {
        let x: f32 = (i == 0) ? 1.0 : 0.0;
        let r = dsp::biquad_step(x, z1, z2, c);
        z1 = r.z1;
        z2 = r.z2;
        if g.x() == i {
            out[i] = r.y;
        }
    }
}
)bk";

    brokernel::Compiler comp;
    auto res = comp.compile_source(src, "dsp_test.bk");
    if (!res.ok) std::fprintf(stderr, "%s\n", res.error_message().c_str());
    CHECK(res.ok);

    const auto* kdb = res.find_kernel("k_db");
    CHECK(kdb != nullptr);
    CHECK(approx(kdb->eval_f32({brokernel::KernelArg(-6.0f)}), -6.0f, 1e-3f));

    const auto* kclip = res.find_kernel("k_clip");
    CHECK(kclip != nullptr);
    const float expected = (1.0f / 2.0f) + std::tanh(0.25f) + 0.25f;
    CHECK(approx(kclip->eval_f32({brokernel::KernelArg(0.5f)}), expected, 1e-4f));

    const auto* kbq = res.find_kernel("k_biquad_impulse");
    CHECK(kbq != nullptr && !kbq->spirv.words.empty());
    std::vector<float> imp(8, 0.0f);
    CHECK(kbq->dispatch_cpu(brokernel::GridShape(8), {brokernel::KernelArg::store(imp)}, nullptr, nullptr));
    CHECK(imp[0] > 0.0f && imp[1] > imp[0]);
}

void test_image_module() {
    const char* src = R"bk(
import std::image;

@kernel
fn k_color(c: f32) -> f32 {
    let lin = image::srgb_to_linear(c);
    let rt = image::linear_to_srgb(lin);
    let tm = image::aces_tonemap(vec3<f32>(rt, rt * 0.5, rt * 0.25));
    return image::luma709(tm);
}

@kernel @grid(2)
fn k_edge(g: grid<2>, img: stencil<f32, 2, 1>, blur_out: write_store<f32, 2>, edge_out: write_store<f32, 2>) {
    blur_out[g.x(), g.y()] = image::box_blur_3x3(img);
    edge_out[g.x(), g.y()] = image::sobel_mag_3x3(img);
}
)bk";

    brokernel::Compiler comp;
    auto res = comp.compile_source(src, "image_test.bk");
    if (!res.ok) std::fprintf(stderr, "%s\n", res.error_message().c_str());
    CHECK(res.ok);

    const auto* kc = res.find_kernel("k_color");
    CHECK(kc != nullptr && !kc->spirv.words.empty());
    const float luma = kc->eval_f32({brokernel::KernelArg(0.5f)});
    CHECK(luma > 0.1f && luma < 0.8f);

    const auto* ke = res.find_kernel("k_edge");
    CHECK(ke != nullptr && !ke->spirv.words.empty());
    std::vector<float> img(16, 0.0f);
    for (int y = 0; y < 4; ++y) {
        for (int x = 2; x < 4; ++x) img[y * 4 + x] = 1.0f;
    }
    std::vector<float> blur(16, 0.0f);
    std::vector<float> edge(16, 0.0f);
    CHECK(ke->dispatch_cpu(
        brokernel::GridShape(4, 4),
        {brokernel::KernelArg::store(img), brokernel::KernelArg::store(blur), brokernel::KernelArg::store(edge)},
        nullptr, nullptr));
    CHECK(edge[1 * 4 + 1] > 1.0f && edge[1 * 4 + 2] > 1.0f);
}

void test_sim_module() {
    const char* src = R"bk(
import std::sim;

@kernel
fn k_step(px: f32, py: f32) -> f32 {
    let s0 = sim::body_state(vec3<f32>(px, py, 0.0), vec3<f32>(2.0, 0.0, 0.0));
    let f = sim::spring_force(vec3<f32>(2.0, 0.0, 0.0), 1.0, 10.0);
    let s1 = sim::symplectic_euler_step(s0, f, 0.1, 0.0);
    let s2 = sim::collide_circle_boundary(s1, vec2<f32>(0.0, 0.0), 1.0, 0.5);
    return sim::sdf_circle(vec2<f32>(s2.pos.x, s2.pos.y), vec2<f32>(0.0, 0.0), 1.0);
}
)bk";

    brokernel::Compiler comp;
    auto res = comp.compile_source(src, "sim_test.bk");
    if (!res.ok) std::fprintf(stderr, "%s\n", res.error_message().c_str());
    CHECK(res.ok);

    const auto* ks = res.find_kernel("k_step");
    CHECK(ks != nullptr && !ks->spirv.words.empty());
    // Boundary collision clamps particle back onto circle boundary (sdf ~ 0).
    CHECK(approx(ks->eval_f32({brokernel::KernelArg(1.5f), brokernel::KernelArg(0.0f)}), 0.0f, 1e-4f));
}

} // namespace

int main() {
    test_dsp_module();
    test_image_module();
    test_sim_module();
    return 0;
}
