// Semantic analysis, f64 precision discipline, stencil radius, and store access tests.

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <string>

using namespace brokernel;

static void test_f64_precision_discipline() {
    // 1. Implicit narrowing from f64 to f32 must be rejected
    {
        const char* src = R"(
            @kernel fn bad_narrow(x: f64) -> f32 {
                let y: f32 = x;
                return y;
            }
        )";
        auto res = compile_source(src, "narrow.bk");
        assert(!res.ok);
        assert(res.error_message().find("implicit narrowing") != std::string::npos);
    }

    // 2. Mixing concrete f64 and f32 without an explicit cast must be rejected
    {
        const char* src = R"(
            @kernel fn bad_mix(a: f64, b: f32) -> f64 {
                return a + b;
            }
        )";
        auto res = compile_source(src, "mix.bk");
        assert(!res.ok);
        assert(res.error_message().find("f64") != std::string::npos);
    }

    // 3. Explicit casts (`as f32`, `as f64`, `f32(x)`, `f64(x)`) and contextual literals succeed
    {
        const char* src = R"(
            @kernel fn explicit_ok(a: f64, b: f32) -> f64 {
                let a_scaled: f64 = a * 2.5 + 1.0;
                let b_wide: f64 = b as f64;
                let combined: f64 = a_scaled + f64(b) + b_wide;
                return combined;
            }
        )";
        auto res = compile_source(src, "explicit_ok.bk");
        assert(res.ok);
        const auto* k = res.default_kernel();
        assert(k != nullptr);
        assert(k->uses_f64);
        double out = k->eval_f64({KernelArg(2.0), KernelArg(1.5f)});
        // (2.0 * 2.5 + 1.0) + 1.5 + 1.5 = 9.0
        assert(std::abs(out - 9.0) < 1e-12);
    }
}

static void test_stencil_validation() {
    // 1. Valid stencil accesses within declared radius R = 1
    {
        const char* src = R"(
            @kernel fn stencil_ok(h: stencil<f64, 2, 1>, out: write_store<f64, 2>) {
                let c = h.at(0, 0);
                let l = h.at(-1, 0);
                let r = h[1, 0];
                let u = h[0, -1];
                let d = h.at(0, 1);
                out.write(0.2 * (c + l + r + u + d));
            }
        )";
        auto res = compile_source(src, "stencil_ok.bk");
        assert(res.ok);
        const auto* k = res.default_kernel();
        assert(k != nullptr);
        assert(k->max_stencil_radius == 1);
        assert(k->stencil_halo[0] == 1 && k->stencil_halo[1] == 1 && k->stencil_halo[2] == 0);
        const auto* p = k->find_param("h");
        assert(p != nullptr);
        assert(p->declared_radius == 1);
        assert(p->inferred_radius == 1);
        assert(p->stencil_offsets.size() == 5);
    }

    // 2. Stencil offset exceeding declared radius R = 1 must fail at compile time
    {
        const char* src = R"(
            @kernel fn stencil_oob(h: stencil<f64, 2, 1>, out: write_store<f64, 2>) {
                out.write(h.at(2, 0));
            }
        )";
        auto res = compile_source(src, "stencil_oob.bk");
        assert(!res.ok);
        assert(res.error_message().find("exceeds declared stencil radius") != std::string::npos);
    }

    // 3. Non-constant stencil offset must fail at compile time
    {
        const char* src = R"(
            @kernel fn stencil_dyn(h: stencil<f32, 2, 1>, dx: i32, out: write_store<f32, 2>) {
                out.write(h.at(dx, 0));
            }
        )";
        auto res = compile_source(src, "stencil_dyn.bk");
        assert(!res.ok);
        assert(res.error_message().find("compile-time integer constant") != std::string::npos);
    }

    // 4. Writing to a stencil must fail at compile time
    {
        const char* src = R"(
            @kernel fn stencil_write(h: stencil<f32, 2, 1>) {
                h[0, 0] = 1.0;
            }
        )";
        auto res = compile_source(src, "stencil_write.bk");
        assert(!res.ok);
        assert(res.error_message().find("read-only stencil") != std::string::npos);
    }
}

static void test_store_access_modes_and_mutability() {
    // 1. Writing to read_store fails
    {
        const char* src = R"(
            @kernel fn write_to_read(inp: read_store<f32, 2>) {
                inp.write(1.0);
            }
        )";
        auto res = compile_source(src, "write_to_read.bk");
        assert(!res.ok);
        assert(res.error_message().find("read-only store") != std::string::npos);
    }

    // 2. Reading from write_store fails
    {
        const char* src = R"(
            @kernel fn read_from_write(out: write_store<f32, 2>) -> f32 {
                return out.read();
            }
        )";
        auto res = compile_source(src, "read_from_write.bk");
        assert(!res.ok);
        assert(res.error_message().find("write-only store") != std::string::npos);
    }

    // 3. Reassigning an immutable `let` binding fails
    {
        const char* src = R"(
            @kernel fn reassign_let(x: f32) -> f32 {
                let a = x + 1.0;
                a = a * 2.0;
                return a;
            }
        )";
        auto res = compile_source(src, "reassign_let.bk");
        assert(!res.ok);
        assert(res.error_message().find("immutable variable") != std::string::npos);
    }

    // 4. Recursive functions are rejected
    {
        const char* src = R"(
            fn rec(n: i32) -> i32 {
                if n <= 0 { return 0; }
                return rec(n - 1);
            }
            @kernel fn entry(n: i32) -> i32 {
                return rec(n);
            }
        )";
        auto res = compile_source(src, "recursion.bk");
        assert(!res.ok);
        assert(res.error_message().find("recursive") != std::string::npos);
    }

    // 5. Invalid assignment targets (duplicate swizzle components, store sub-element, non-lvalue) are rejected
    {
        const char* src = R"(
            @kernel fn bad_swizzle(x: f32) -> f32 {
                var v: vec3<f32> = vec3<f32>(x, x, x);
                v.xx = vec2<f32>(1.0, 2.0);
                return v.x;
            }
        )";
        auto res = compile_source(src, "bad_swizzle.bk");
        assert(!res.ok);
        assert(res.error_message().find("duplicate components") != std::string::npos);
    }
    {
        const char* src = R"(
            @kernel fn bad_store_sub(s: rw_store<vec3<f32>, 1>) {
                s[0].x = 1.0;
            }
        )";
        auto res = compile_source(src, "bad_store_sub.bk");
        assert(!res.ok);
        assert(res.error_message().find("sub-element of store") != std::string::npos);
    }

    // 6. Atomic builtins on unsupported store element types are rejected
    {
        const char* src = R"(
            @kernel fn bad_atomic_vec(s: rw_store<vec3<f32>, 1>) {
                s.atomic_add(0, vec3<f32>(1.0, 2.0, 3.0));
            }
        )";
        auto res = compile_source(src, "bad_atomic_vec.bk");
        assert(!res.ok);
        assert(res.error_message().find("atomic_add requires store element type") != std::string::npos);
    }
    {
        const char* src = R"(
            @kernel fn bad_atomic_f32_min(s: rw_store<f32, 1>) {
                s.atomic_min(0, 1.0);
            }
        )";
        auto res = compile_source(src, "bad_atomic_f32_min.bk");
        assert(!res.ok);
        assert(res.error_message().find("atomic_min requires store element type") != std::string::npos);
    }
    {
        const char* src = R"(
            @kernel fn bad_atomic_f64(s: rw_store<f64, 1>) {
                s.atomic_add(0, 1.0 as f64);
            }
        )";
        auto res = compile_source(src, "bad_atomic_f64.bk");
        assert(!res.ok);
        assert(res.error_message().find("atomic_add requires store element type") != std::string::npos);
    }
}

static void test_64bit_grid_indexing_and_bitcast() {
    // 1. Grid linear index and total cells use 64-bit integers so grids > 2^31 cells do not overflow.
    {
        const char* src = R"(
            @kernel @grid(2)
            fn large_grid_idx(g: grid<2>) -> i64 {
                return g.linear_index() + g.total_cells();
            }
        )";
        auto res = compile_source(src, "large_grid.bk");
        if (!res.ok) std::cerr << res.error_message() << "\n";
        assert(res.ok);
        const auto* k = res.default_kernel();
        assert(k != nullptr);
        // 65536 * 65536 = 4,294,967,296 (> 2^31 - 1)
        // At (x=50000, y=50000), linear_index = 50000 * 65536 + 50000 = 3,276,850,000
        int64_t out = 0;
        bool ok = k->eval_grid_point_cpu(50000, 50000, 0, GridShape(65536, 65536, 1), {}, &out);
        assert(ok);
        int64_t expected_lin = 50000LL * 65536LL + 50000LL;
        int64_t expected_tot = 65536LL * 65536LL;
        assert(out == expected_lin + expected_tot);
    }

    // 2. Store indexing accepts i64 coordinates and bitcast<T> / is_f64<T>() work on f32 and f64.
    {
        const char* src = R"(
            fn type_tag<T>(x: T) -> i32 {
                if is_f64<T>() {
                    return 64;
                } else {
                    return bitcast<i32>(x);
                }
            }

            @kernel fn test_intrinsics(buf: store<f32, 1>, idx: i64, val: f32) -> i32 {
                buf[idx] = val;
                let read_back: f32 = buf[idx];
                return type_tag<f32>(read_back) + type_tag<f64>(1.0 as f64);
            }
        )";
        auto res = compile_source(src, "intrinsics.bk");
        if (!res.ok) std::cerr << res.error_message() << "\n";
        assert(res.ok);
        const auto* k = res.default_kernel();
        assert(k != nullptr);
        std::vector<float> data(16, 0.0f);
        int32_t got = k->eval_i32({KernelArg::store(data), KernelArg(int64_t(11)), KernelArg(1.0f)});
        // bitcast<i32>(1.0f) == 0x3f800000 (1065353216)
        assert(data[11] == 1.0f);
        assert(got == 0x3f800000 + 64);
    }
}

int main() {
    test_f64_precision_discipline();
    test_stencil_validation();
    test_store_access_modes_and_mutability();
    test_64bit_grid_indexing_and_bitcast();
    std::cout << "test_sema_types: ALL PASSED\n";
    return 0;
}
