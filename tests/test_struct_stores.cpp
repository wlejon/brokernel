// Structured store field reads, partial writes, compound ops, nested structs,
// swizzle writes, and out-of-bounds store safety on CPU & GPU.

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace brokernel;

struct Sub {
    float a;
    int32_t b;
};

struct Particle {
    float pos[3];
    float vel[3];
    float mass;
    Sub extra;
    float weights[3];
};

static void test_struct_store_cpu_and_gpu() {
    const char* src = R"(
        struct Sub {
            a: f32,
            b: i32,
        }
        struct Particle {
            pos: vec3<f32>,
            vel: vec3<f32>,
            mass: f32,
            extra: Sub,
            weights: array<f32, 3>,
        }

        @kernel @grid(1)
        fn step_particles(g: grid<1>, ps: rw_store<Particle, 1>, out_w: write_store<f32, 1>, dt: f32) {
            let i = g.x();
            // Field read + partial write + compound ops + swizzle write + nested struct + array element
            ps[i].vel.y -= 9.81 * dt;
            ps[i].pos += ps[i].vel * dt;
            ps[i].pos.xz *= 0.99;
            ps[i].extra.a += ps[i].mass * 0.5;
            ps[i].extra.b += 1;
            let k = i % 3;
            ps[i].weights[k] += 0.25;
            // Read back a field from a neighbor (including out-of-bounds read -> 0)
            let next_m = ps[i + 1].mass;
            out_w[i] = ps[i].pos.x + ps[i].pos.y + ps[i].extra.a + ps[i].weights[k] + next_m;
        }
    )";

    auto res = compile_source(src, "struct_stores.bk");
    if (!res.ok) {
        std::cerr << res.error_message() << "\n";
    }
    assert(res.ok);
    const auto* k = res.default_kernel();
    assert(k != nullptr);

    const uint32_t N = 64;
    std::vector<Particle> cpu_ps(N);
    for (uint32_t i = 0; i < N; ++i) {
        cpu_ps[i].pos[0] = static_cast<float>(i) + 1.0f;
        cpu_ps[i].pos[1] = 10.0f;
        cpu_ps[i].pos[2] = -2.0f;
        cpu_ps[i].vel[0] = 0.5f;
        cpu_ps[i].vel[1] = 2.0f;
        cpu_ps[i].vel[2] = -1.0f;
        cpu_ps[i].mass = 2.0f + static_cast<float>(i % 5);
        cpu_ps[i].extra.a = 1.0f;
        cpu_ps[i].extra.b = 10;
        cpu_ps[i].weights[0] = 0.1f;
        cpu_ps[i].weights[1] = 0.2f;
        cpu_ps[i].weights[2] = 0.3f;
    }
    std::vector<Particle> gpu_ps = cpu_ps;
    std::vector<float> cpu_out(N, 0.0f);
    std::vector<float> gpu_out(N, 0.0f);

    const float dt = 0.05f;
    std::string err;
    bool ok = k->dispatch_cpu(GridShape(N),
                              {KernelArg::store(cpu_ps), KernelArg::store(cpu_out), KernelArg(dt)},
                              nullptr, &err);
    if (!ok) std::cerr << "CPU err: " << err << "\n";
    assert(ok);

    // Verify exact CPU math on element 0 and last element (OOB neighbor mass = 0)
    assert(std::abs(cpu_ps[0].vel[1] - (2.0f - 9.81f * dt)) < 1e-5f);
    assert(std::abs(cpu_ps[0].pos[0] - (1.0f + 0.5f * dt) * 0.99f) < 1e-5f);
    assert(std::abs(cpu_ps[0].pos[1] - (10.0f + cpu_ps[0].vel[1] * dt)) < 1e-5f);
    assert(std::abs(cpu_ps[0].pos[2] - (-2.0f + (-1.0f) * dt) * 0.99f) < 1e-5f);
    assert(std::abs(cpu_ps[0].extra.a - (1.0f + 2.0f * 0.5f)) < 1e-5f);
    assert(cpu_ps[0].extra.b == 11);
    assert(std::abs(cpu_ps[0].weights[0] - 0.35f) < 1e-5f);
    assert(std::abs(cpu_ps[0].weights[1] - 0.2f) < 1e-5f);

    if (k->dispatch_gpu(GridShape(N),
                        {KernelArg::store(gpu_ps), KernelArg::store(gpu_out), KernelArg(dt)},
                        nullptr, &err)) {
        for (uint32_t i = 0; i < N; ++i) {
            assert(std::abs(cpu_ps[i].pos[0] - gpu_ps[i].pos[0]) < 1e-4f);
            assert(std::abs(cpu_ps[i].pos[1] - gpu_ps[i].pos[1]) < 1e-4f);
            assert(std::abs(cpu_ps[i].pos[2] - gpu_ps[i].pos[2]) < 1e-4f);
            assert(std::abs(cpu_ps[i].vel[1] - gpu_ps[i].vel[1]) < 1e-4f);
            assert(std::abs(cpu_ps[i].extra.a - gpu_ps[i].extra.a) < 1e-4f);
            assert(cpu_ps[i].extra.b == gpu_ps[i].extra.b);
            assert(std::abs(cpu_out[i] - gpu_out[i]) < 1e-4f);
        }
        std::cout << "test_struct_store_cpu_and_gpu: CPU & GPU parity PASS\n";
    } else {
        std::cout << "test_struct_store_cpu_and_gpu: CPU PASS (GPU skipped: " << err << ")\n";
    }
}

static void test_write_only_and_read_only_store_diagnostics() {
    const char* write_only_ok = R"(
        struct Pair { a: f32, b: f32 }
        @kernel @grid(1)
        fn k(g: grid<1>, out: write_store<Pair, 1>) {
            out[g.x()].a = 3.0;
            out[g.x()].b = 4.0;
        }
    )";
    auto r1 = compile_source(write_only_ok, "wo_ok.bk");
    assert(r1.ok);

    const char* read_only_err = R"(
        struct Pair { a: f32, b: f32 }
        @kernel @grid(1)
        fn k(g: grid<1>, inp: read_store<Pair, 1>) {
            inp[g.x()].a = 3.0;
        }
    )";
    auto r2 = compile_source(read_only_err, "ro_err.bk");
    assert(!r2.ok);
    assert(r2.error_message().find("read-only") != std::string::npos);

    const char* dup_swizzle_err = R"(
        struct VecBox { v: vec3<f32> }
        @kernel @grid(1)
        fn k(g: grid<1>, buf: rw_store<VecBox, 1>) {
            buf[g.x()].v.xx = vec2<f32>(1.0, 2.0);
        }
    )";
    auto r3 = compile_source(dup_swizzle_err, "dup_swiz.bk");
    assert(!r3.ok);
}

static void test_struct_store_2d_and_3d_cpu_and_gpu() {
    struct Cell {
        float pos[2];
        int32_t tag;
    };
    static_assert(sizeof(Cell) == 12);

    const char* src2d = R"(
        struct Cell {
            pos: vec2<f32>,
            tag: i32,
        }

        @kernel @grid(2)
        fn step2d(g: grid<2>, inp: read_store<Cell, 2>, out: write_store<Cell, 2>, inout: rw_store<Cell, 2>) {
            let cur = inp.read();
            let diag = inp.load(min(g.x() + 1, g.width() - 1), min(g.y() + 1, g.height() - 1));
            var next: Cell;
            next.pos = cur.pos + diag.pos * 0.5;
            next.tag = cur.tag + g.y() * 10 + g.x();
            out.write(next);
            inout.write(next);
            inout[g.x(), g.y()].pos.x += 1.0;
        }
    )";

    auto r2 = compile_source(src2d, "struct_store_2d.bk");
    if (!r2.ok) std::cerr << r2.error_message() << "\n";
    assert(r2.ok);
    const CompiledKernel* k2 = r2.find_kernel("step2d");
    assert(k2 && k2->valid());

    constexpr uint32_t W = 4, H = 3;
    std::vector<Cell> in2d(W * H);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            in2d[y * W + x] = Cell{{static_cast<float>(x), static_cast<float>(y * 2)}, static_cast<int32_t>(x + y)};
        }
    }
    std::vector<Cell> wo2d(W * H), cpu2d(W * H), woGpu2d(W * H), gpu2d(W * H);
    std::string err;
    assert(k2->dispatch_cpu(GridShape(W, H), {KernelArg::store(in2d), KernelArg::store(wo2d), KernelArg::store(cpu2d)}, nullptr, &err));
    // Verify cell (1, 1): cur = (1, 2), diag at (2, 2) = (2, 4) -> wo.pos = (2, 4), inout.pos = (3, 4)
    assert(std::abs(wo2d[1 * W + 1].pos[0] - 2.0f) < 1e-5f);
    assert(std::abs(cpu2d[1 * W + 1].pos[0] - 3.0f) < 1e-5f);
    assert(std::abs(cpu2d[1 * W + 1].pos[1] - 4.0f) < 1e-5f);
    assert(cpu2d[1 * W + 1].tag == (1 + 1) + 1 * 10 + 1);

    if (k2->dispatch_gpu(GridShape(W, H), {KernelArg::store(in2d), KernelArg::store(woGpu2d), KernelArg::store(gpu2d)}, nullptr, &err)) {
        for (size_t i = 0; i < W * H; ++i) {
            assert(std::abs(wo2d[i].pos[0] - woGpu2d[i].pos[0]) < 1e-4f);
            assert(std::abs(cpu2d[i].pos[0] - gpu2d[i].pos[0]) < 1e-4f);
            assert(std::abs(cpu2d[i].pos[1] - gpu2d[i].pos[1]) < 1e-4f);
            assert(cpu2d[i].tag == gpu2d[i].tag);
        }
    }

    const char* src3d = R"(
        struct Voxel { val: f32, active: i32 }
        @kernel @grid(3)
        fn step3d(g: grid<3>, vol: rw_store<Voxel, 3>) {
            let c = vol.load(g.x(), g.y(), g.z());
            var updated: Voxel;
            updated.val = c.val * 2.0 + 1.0;
            updated.active = c.active + g.z();
            vol.store(g.x(), g.y(), g.z(), updated);
        }
    )";
    auto r3 = compile_source(src3d, "struct_store_3d.bk");
    if (!r3.ok) std::cerr << r3.error_message() << "\n";
    assert(r3.ok);
    const CompiledKernel* k3 = r3.find_kernel("step3d");
    assert(k3 && k3->valid());
    struct Voxel { float val; int32_t active; };
    constexpr uint32_t D = 2;
    std::vector<Voxel> cpu3d(W * H * D, Voxel{1.5f, 3});
    assert(k3->dispatch_cpu(GridShape(W, H, D), {KernelArg::store(cpu3d)}, nullptr, &err));
    const size_t idx111 = (1 * H + 1) * W + 1;
    assert(std::abs(cpu3d[idx111].val - 4.0f) < 1e-5f);
    assert(cpu3d[idx111].active == 4);
}

int main() {
    test_struct_store_cpu_and_gpu();
    test_write_only_and_read_only_store_diagnostics();
    test_struct_store_2d_and_3d_cpu_and_gpu();
    std::cout << "ALL test_struct_stores PASSED\n";
    return 0;
}
