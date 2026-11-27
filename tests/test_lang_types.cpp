// Phase 1 language type system tests:
// 1. `enum` declarations (auto/explicit/negative/const/u32/cross-module/reflection/CPU+GPU)
// 2. Zero-cost nominal `newtype` declarations (safety, .0 unwrap, `as` cast, CPU+GPU stores, f64 tracking)
// 3. Compile-time `sizeof`, `offsetof`, `std430_sizeof`, `std140_sizeof`, `std430_offsetof`, `std140_offsetof`

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace brokernel;

static void test_enums_basic_and_reflection() {
    const char* src = R"(
        const BASE: i32 = 10;

        pub enum Mode {
            Zero,
            One,
            Neg = -3,
            NegNext,
            FromConst = BASE + 2,
            FromEarlier = FromConst + 5,
            Post = E_AFTER,
        }

        const E_AFTER: i32 = 42;
        pub const MODE_VAL: i32 = Mode::FromEarlier as i32;

        pub enum Flags : u32 {
            None = 0u,
            Read = 1u,
            Write = 2u,
            ReadWrite = Flags::Read | Flags::Write,
        }

        struct Packet {
            mode: Mode,
            flags: Flags,
        }

        @kernel fn eval_enum(m: Mode, f: Flags) -> i32 {
            let p = Packet { mode: m, flags: f };
            var acc: i32 = (p.mode as i32) + (p.flags as i32);
            if p.mode == Mode::FromEarlier {
                acc += 100;
            }
            if p.mode < Mode::Post {
                acc += 1000;
            }
            return acc + (Mode::NegNext as i32);
        }

        @kernel @grid(1)
        fn grid_enum(g: grid<1>, out: rw_store<i32, 1>, m: Mode) {
            if m == Mode::One {
                out[g.x()] = (m as i32) + g.x() + (Mode::FromConst as i32);
            } else {
                out[g.x()] = -1;
            }
        }
    )";

    auto res = compile_source(src, "enums_basic.bk");
    if (!res.ok) {
        std::cerr << "test_enums_basic_and_reflection failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);

    // Reflection: CompileResult::enums
    assert(res.enums.size() == 2);
    assert(res.enums[0].name == "Mode");
    assert(res.enums[0].underlying == ScalarKind::I32);
    assert(res.enums[0].variants.size() == 7);
    assert(res.enums[0].variants[0].first == "Zero" && res.enums[0].variants[0].second == 0);
    assert(res.enums[0].variants[1].first == "One" && res.enums[0].variants[1].second == 1);
    assert(res.enums[0].variants[2].first == "Neg" && res.enums[0].variants[2].second == -3);
    assert(res.enums[0].variants[3].first == "NegNext" && res.enums[0].variants[3].second == -2);
    assert(res.enums[0].variants[4].first == "FromConst" && res.enums[0].variants[4].second == 12);
    assert(res.enums[0].variants[5].first == "FromEarlier" && res.enums[0].variants[5].second == 17);
    assert(res.enums[0].variants[6].first == "Post" && res.enums[0].variants[6].second == 42);

    assert(res.enums[1].name == "Flags");
    assert(res.enums[1].underlying == ScalarKind::U32);
    assert(res.enums[1].variants[3].first == "ReadWrite" && res.enums[1].variants[3].second == 3);

    const auto* k = res.find_kernel("eval_enum");
    assert(k != nullptr);
    // m = 17 (FromEarlier), f = 3 (ReadWrite): 17 + 3 + 100 + 1000 + (-2) = 1118
    int32_t got = k->eval_i32({KernelArg(int32_t(17)), KernelArg(uint32_t(3))});
    assert(got == 1118);

    const auto* kg = res.find_kernel("grid_enum");
    assert(kg != nullptr);
    std::vector<int32_t> out_cpu(8, 0);
    assert(kg->dispatch_cpu(GridShape(8), {KernelArg::store(out_cpu), KernelArg(int32_t(1))}));
    for (int i = 0; i < 8; ++i) {
        assert(out_cpu[i] == 1 + i + 12);
    }

    if (brass::gpu::vulkan_available()) {
        std::vector<int32_t> out_gpu(8, 0);
        assert(kg->dispatch_gpu(GridShape(8), {KernelArg::store(out_gpu), KernelArg(int32_t(1))}));
        for (int i = 0; i < 8; ++i) {
            assert(out_gpu[i] == out_cpu[i]);
        }
    }
}

static void test_enums_cross_module_and_errors() {
    Compiler compiler;
    compiler.register_module("world::enums_leaf", R"(
        module world::enums_leaf;
        pub enum Biome : u32 {
            Ocean = 0u,
            Plains = 1u,
            Mountain = 2u,
        }
    )");
    compiler.register_module("world::enums_mid", R"(
        module world::enums_mid;
        pub use world::enums_leaf::Biome;
    )");

    const char* src = R"(
        pub use world::enums_mid::Biome;
        import world::enums_leaf as el;

        const MOUNTAIN_ID: i32 = Biome::Mountain as i32;
        const ARR_LEN: i32 = (el::Biome::Mountain as i32) + 1;

        @kernel fn pick_biome(b: Biome) -> u32 {
            let arr: [u32; ARR_LEN] = [10u, 20u, 30u];
            if b == el::Biome::Ocean {
                return arr[0];
            }
            return (b as u32) + (MOUNTAIN_ID as u32);
        }
    )";

    auto res = compiler.compile_source(src, "enums_cross.bk");
    if (!res.ok) {
        std::cerr << "test_enums_cross_module_and_errors failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);
    assert(res.enums.size() == 1);
    assert(res.enums[0].name == "Biome");
    assert(res.enums[0].module == "world::enums_leaf");
    assert(res.enums[0].underlying == ScalarKind::U32);

    const auto* k = res.find_kernel("pick_biome");
    assert(k != nullptr);
    uint32_t out0 = 0;
    assert(k->eval_cpu({KernelArg(uint32_t(0))}, &out0));
    assert(out0 == 10u);
    uint32_t out1 = 0;
    assert(k->eval_cpu({KernelArg(uint32_t(1))}, &out1));
    assert(out1 == 3u); // 1 + 2

    // Type errors: mixing distinct enums or raw integer assignment
    const char* bad_src = R"(
        enum A { X = 0 }
        enum B { Y = 0 }
        @kernel fn bad(a: A, b: B) -> bool {
            let c: A = 0;
            return a == b;
        }
    )";
    auto bad_res = compile_source(bad_src, "enums_bad.bk");
    assert(!bad_res.ok);
    std::string err = bad_res.error_message();
    assert(err.find("A") != std::string::npos);
}

static void test_newtypes_nominal_safety_and_parity() {
    Compiler compiler;
    compiler.register_module("geo::units", R"(
        module geo::units;
        pub newtype Meters = f32;
        pub newtype Radians = f32;
        pub newtype WorldPos = vec3<f64>;
        pub newtype LocalPos = vec3<f32>;
    )");

    const char* src = R"(
        use geo::units::*;

        const ONE_M: Meters = Meters(1.0);

        struct Particle {
            dist: Meters,
            angle: Radians,
        }

        fn add_dist(a: Meters, b: Meters) -> Meters {
            return a + b;
        }

        @kernel @grid(1)
        fn step_meters(g: grid<1>, buf: rw_store<Meters, 1>, delta: Meters) {
            var m = buf[g.x()];
            let neg = -m;
            let doubled = add_dist(m, m) + ONE_M - neg;
            m.0 += 0.5;
            let via_cast = (delta.0) as Meters;
            buf[g.x()] = doubled + via_cast + (m - buf[g.x()]);
        }

        @cpu @kernel fn world_offset(a: WorldPos, b: WorldPos) -> f64 {
            let diff: WorldPos = a - b;
            let neg: WorldPos = -diff;
            let sum: WorldPos = a + (-neg);
            let u: vec3<f64> = sum as vec3<f64>;
            let rewrapped: WorldPos = u as WorldPos;
            return rewrapped.0.x + rewrapped.0.y + rewrapped.0.z;
        }

        @kernel fn cmp_meters(a: Meters, b: Meters) -> i32 {
            let p = Particle { dist: a, angle: Radians(0.5) };
            let r: f32 = p.angle as f32;
            if a < b && a != b {
                return 1;
            }
            if p.dist >= b {
                return (r > 0.0) ? 2 : 0;
            }
            return 0;
        }
    )";

    auto res = compiler.compile_source(src, "newtypes_ok.bk");
    if (!res.ok) {
        std::cerr << "test_newtypes_nominal_safety_and_parity failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);

    // CPU + GPU parity for rw_store<Meters, 1>
    const auto* k_step = res.find_kernel("step_meters");
    assert(k_step != nullptr);
    std::vector<float> cpu_buf = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gpu_buf = cpu_buf;
    // m = v, neg = -v, doubled = 2v + 1 - (-v) = 3v + 1; m.0 += 0.5 -> (m - buf[i]) = 0.5;
    // result = 3v + 1 + 0.25 + 0.5 = 3v + 1.75
    assert(k_step->dispatch_cpu(GridShape(4), {KernelArg::store(cpu_buf), KernelArg(0.25f)}));
    for (int i = 0; i < 4; ++i) {
        float v = static_cast<float>(i + 1);
        assert(std::abs(cpu_buf[i] - (3.0f * v + 1.75f)) < 1e-5f);
    }
    if (brass::gpu::vulkan_available()) {
        assert(k_step->dispatch_gpu(GridShape(4), {KernelArg::store(gpu_buf), KernelArg(0.25f)}));
        for (int i = 0; i < 4; ++i) {
            assert(std::abs(gpu_buf[i] - cpu_buf[i]) < 1e-5f);
        }
    }

    // WorldPos = vec3<f64> on @cpu kernel
    const auto* k_world = res.find_kernel("world_offset");
    assert(k_world != nullptr);
    // a + (a - b) = 2a - b: (10,20,30)*2 - (1,2,3) = (19,38,57) -> sum = 114
    double wsum = k_world->eval_f64({KernelArg::vec3d(10.0, 20.0, 30.0), KernelArg::vec3d(1.0, 2.0, 3.0)});
    assert(std::abs(wsum - 114.0) < 1e-9);

    const auto* k_cmp = res.find_kernel("cmp_meters");
    assert(k_cmp != nullptr);
    assert(k_cmp->eval_i32({KernelArg(1.0f), KernelArg(2.0f)}) == 1);
    assert(k_cmp->eval_i32({KernelArg(2.0f), KernelArg(2.0f)}) == 2);
}

static void test_newtypes_rejections() {
    auto expect_err = [](const char* src, const char* must1, const char* must2) {
        auto res = compile_source(src, "newtype_err.bk");
        assert(!res.ok);
        std::string msg = res.error_message();
        assert(msg.find(must1) != std::string::npos);
        assert(msg.find(must2) != std::string::npos);
    };

    // 1. Mixed Meters + Radians
    expect_err(R"(
        newtype Meters = f32;
        newtype Radians = f32;
        @kernel fn k(m: Meters, r: Radians) -> Meters { return m + r; }
    )", "Meters", "Radians");

    // 2. Mixed Meters + f32
    expect_err(R"(
        newtype Meters = f32;
        @kernel fn k(m: Meters) -> Meters { return m + 1.0; }
    )", "Meters", "f32");

    // 3. Assigning f32 to Meters
    expect_err(R"(
        newtype Meters = f32;
        @kernel fn k() -> Meters {
            let m: Meters = 1.0;
            return m;
        }
    )", "Meters", "f32");

    // 4. Passing Meters where Radians is expected
    expect_err(R"(
        newtype Meters = f32;
        newtype Radians = f32;
        fn take_rad(r: Radians) -> f32 { return r.0; }
        @kernel fn k(m: Meters) -> f32 { return take_rad(m); }
    )", "Radians", "Meters");

    // 5. Passing Meters where f32 is expected
    expect_err(R"(
        newtype Meters = f32;
        fn take_f32(x: f32) -> f32 { return x; }
        @kernel fn k(m: Meters) -> f32 { return take_f32(m); }
    )", "f32", "Meters");

    // 6. Direct field/swizzle on newtype without .0
    expect_err(R"(
        newtype WorldPos = vec3<f64>;
        @cpu @kernel fn k(p: WorldPos) -> f64 { return p.x; }
    )", "WorldPos", ".0");

    // 7. f64 tracking inside WorldPos in a non-@cpu GPU kernel
    expect_err(R"(
        newtype WorldPos = vec3<f64>;
        @kernel fn k(p: WorldPos) -> f32 { return p.0.x as f32; }
    )", "f64", "k");
}

static void test_sizeof_and_offsetof() {
    const char* src = R"(
        newtype Meters = f32;
        enum Mode { A = 0, B = 1 }

        struct Extra {
            a: f32,
            b: vec3<f32>,
        }

        struct Particle {
            flag: u32,
            pos: vec3<f32>,
            extra: Extra,
        }

        struct Inner {
            x: f32,
            y: f32,
        }

        struct Outer {
            head: f32,
            inner: Inner,
            tail: f32,
        }

        pub const SZ_F32: i32 = sizeof(f32);
        pub const SZ_VEC3: i32 = sizeof(vec3<f32>);
        pub const SZ_ARR: i32 = sizeof(array<f32, 4>);
        pub const SZ_BRK_ARR: i32 = sizeof([f32; 4]);
        pub const SZ_METERS: i32 = sizeof(Meters);
        pub const SZ_MODE: i32 = sizeof(Mode);
        pub const SZ_PARTICLE: i32 = sizeof(Particle);
        pub const OFF_POS: i32 = offsetof(Particle, pos);
        pub const OFF_EXTRA_B: i32 = offsetof(Particle, extra.b);
        pub const SZ430_PARTICLE: i32 = std430_sizeof(Particle);
        pub const OFF430_EXTRA_B: i32 = std430_offsetof(Particle, extra.b);
        pub const SZ430_OUTER: i32 = std430_sizeof(Outer);
        pub const SZ140_OUTER: i32 = std140_sizeof(Outer);
        pub const OFF430_INNER_Y: i32 = std430_offsetof(Outer, inner.y);
        pub const OFF140_INNER_Y: i32 = std140_offsetof(Outer, inner.y);
        pub const OFF140_TAIL: i32 = std140_offsetof(Outer, tail);
        const ARR_N: i32 = offsetof(Particle, extra.b) / 4;

        fn generic_size<T>() -> i32 {
            return sizeof(T);
        }

        @kernel fn check_queries(out: rw_store<i32, 1>) {
            let local_arr: [f32; ARR_N] = [1.0, 2.0, 3.0, 4.0, 5.0];
            out[0] = sizeof(Particle);
            out[1] = offsetof(Particle, extra.b);
            out[2] = std430_sizeof(Particle);
            out[3] = std430_offsetof(Particle, extra.b);
            out[4] = std140_sizeof(Outer);
            out[5] = std140_offsetof(Outer, inner.y);
            out[6] = generic_size<Extra>();
            out[7] = (local_arr[4] as i32);
        }
    )";

    auto res = compile_source(src, "sizeof_offsetof.bk");
    if (!res.ok) {
        std::cerr << "test_sizeof_and_offsetof failed:\n" << res.error_message() << "\n";
    }
    assert(res.ok);

    auto get_const = [&](const char* name) -> int64_t {
        for (const auto& c : res.constants) {
            if (c.name == name) return c.value.int_val;
        }
        assert(false && "missing constant");
        return -1;
    };

    assert(get_const("SZ_F32") == 4);
    assert(get_const("SZ_VEC3") == 12);
    assert(get_const("SZ_ARR") == 16);
    assert(get_const("SZ_BRK_ARR") == 16);
    assert(get_const("SZ_METERS") == 4);
    assert(get_const("SZ_MODE") == 4);
    assert(get_const("SZ_PARTICLE") == 32);
    assert(get_const("OFF_POS") == 4);
    assert(get_const("OFF_EXTRA_B") == 20);
    assert(get_const("SZ430_PARTICLE") == 64);
    assert(get_const("OFF430_EXTRA_B") == 48);
    assert(get_const("SZ430_OUTER") == 16);
    assert(get_const("SZ140_OUTER") == 48);
    assert(get_const("OFF430_INNER_Y") == 8);
    assert(get_const("OFF140_INNER_Y") == 20);
    assert(get_const("OFF140_TAIL") == 32);

    const auto* k = res.find_kernel("check_queries");
    assert(k != nullptr);
    std::vector<int32_t> cpu_out(8, 0);
    assert(k->eval_cpu({KernelArg::store(cpu_out)}));
    assert(cpu_out[0] == 32);
    assert(cpu_out[1] == 20);
    assert(cpu_out[2] == 64);
    assert(cpu_out[3] == 48);
    assert(cpu_out[4] == 48);
    assert(cpu_out[5] == 20);
    assert(cpu_out[6] == 16);
    assert(cpu_out[7] == 5);

    if (brass::gpu::vulkan_available()) {
        std::vector<int32_t> gpu_out(8, 0);
        assert(k->dispatch_gpu(GridShape(1), {KernelArg::store(gpu_out)}));
        for (int i = 0; i < 8; ++i) {
            assert(gpu_out[i] == cpu_out[i]);
        }
    }
}

static void test_compile_cache_roundtrip() {
    const std::filesystem::path cache_dir =
        std::filesystem::temp_directory_path() / "brokernel_test_lang_types_cache";
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);

    CompileCacheConfig cfg;
    cfg.dir = cache_dir;
    set_compile_cache(cfg);

    const char* src = R"(
        pub enum Kind : u32 { A = 1u, B = 7u }
        pub newtype Meters = f32;
        pub const K_DEF: Kind = Kind::B;
        pub const M_DEF: Meters = Meters(3.5);

        @kernel fn run(k: Kind, m: Meters) -> f32 {
            return (k as f32) + m.0;
        }
    )";

    CompileResult r1 = compile_source(src, "cache_lang.bk");
    assert(r1.ok);
    assert(r1.cache_status == CacheStatus::Miss);
    assert(r1.enums.size() == 1);
    assert(r1.enums[0].name == "Kind");
    assert(r1.enums[0].variants.size() == 2);
    assert(r1.types.newtypes().size() == 1);
    assert(r1.types.enums().size() == 1);

    CompileResult r2 = compile_source(src, "cache_lang.bk");
    assert(r2.ok);
    assert(r2.cache_status == CacheStatus::Hit);
    assert(r2.enums.size() == 1);
    assert(r2.enums[0].name == "Kind");
    assert(r2.enums[0].underlying == ScalarKind::U32);
    assert(r2.enums[0].variants[1].first == "B" && r2.enums[0].variants[1].second == 7);
    assert(r2.types.newtypes().size() == 1);
    assert(r2.types.newtypes()[0].name == "Meters");
    assert(r2.types.enums().size() == 1);
    assert(r2.constants.size() == 2);
    assert(r2.constants[0].type.is_enum());
    assert(r2.constants[1].type.is_newtype());
    assert(r2.constants[1].type.to_string() == "Meters");

    const auto* k2 = r2.find_kernel("run");
    assert(k2 != nullptr);
    assert(k2->params.size() == 2);
    assert(k2->params[0].type.is_enum());
    assert(k2->params[1].type.is_newtype());
    float got = k2->eval_f32({KernelArg(uint32_t(7)), KernelArg(3.5f)});
    assert(std::abs(got - 10.5f) < 1e-5f);

    set_compile_cache(CompileCacheConfig{});
    std::filesystem::remove_all(cache_dir, ec);
}

int main() {
    test_enums_basic_and_reflection();
    test_enums_cross_module_and_errors();
    test_newtypes_nominal_safety_and_parity();
    test_newtypes_rejections();
    test_sizeof_and_offsetof();
    test_compile_cache_roundtrip();
    std::cout << "test_lang_types: all tests passed!\n";
    return 0;
}
