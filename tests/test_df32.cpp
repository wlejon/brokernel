#include "brokernel/brokernel.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace brokernel;

struct DfPairHost {
    float hi;
    float lo;
};

static double df_to_f64(DfPairHost p) {
    return static_cast<double>(p.hi) + static_cast<double>(p.lo);
}

static DfPairHost f64_to_df(double v) {
    float hi = static_cast<float>(v);
    float lo = static_cast<float>(v - static_cast<double>(hi));
    return {hi, lo};
}

int main() {
    const char* src = R"(
        struct AnchorSample {
            pos: vec3<df32>,
            height: df32,
            tag: i32,
        }

        @kernel @grid(1)
        fn test_arith(
            g: grid<1>,
            inp: read_store<df32, 1>,
            out: write_store<df32, 1>
        ) {
            let big: df32 = 16384.0;
            let eps = inp.read();
            let sum = big + eps;
            let diff = sum - big;
            let prod = df32(123456.789) * df32(0.000125) + df32(1.0) / df32(3.0);
            let sq = sqrt(df32(2.0));
            let fl = floor(big + df32(0.75));
            let res = diff + (prod - prod) + (sq - df32(1.4142135623730951)) + (fl - big);
            out.write(res);
        }

        @kernel @grid(1)
        fn test_struct_and_vec(
            g: grid<1>,
            inp: read_store<AnchorSample, 1>,
            offset: vec3<df32>,
            out: write_store<df32, 1>
        ) {
            let s = inp.read();
            let shifted = s.pos + offset;
            let d = length(shifted) + abs(s.height - df32(16384.0));
            out.write(df32(shifted.x.hi, shifted.x.lo) + (d - length(shifted)));
        }

        @kernel
        fn test_point(a: df32, c: df32) -> df32 {
            return (df32(16384.0) + a) - df32(16384.0) + a * c;
        }
    )";

    Compiler c;
    auto cr = c.compile_source(src, "df32_test.bk");
    if (!cr.ok) {
        std::cerr << cr.error_message() << "\n";
        return 1;
    }

    const CompiledKernel* k_pt = cr.find_kernel("test_point");
    assert(k_pt && k_pt->cpu_only == false && k_pt->valid());
    DfPairHost pt_ret{0.0f, 0.0f};
    std::string err;
    const double a_val = 0.1251220703125;
    const double c_val = 1.75;
    const bool pt_ok = k_pt->eval_cpu({KernelArg(a_val), KernelArg(c_val)}, &pt_ret, &err);
    assert(pt_ok);
    (void)pt_ok;
    const double want_pt = a_val + a_val * c_val;
    assert(std::abs(df_to_f64(pt_ret) - want_pt) < 1e-12);
    (void)want_pt;

    const CompiledKernel* k_arith = cr.find_kernel("test_arith");
    assert(k_arith && k_arith->valid() && !k_arith->spirv.words.empty());
#if defined(BROKERNEL_SPIRV_VAL)
    if (std::string(BROKERNEL_SPIRV_VAL).size() > 0) {
        const std::string spv_path = "/tmp/brokernel_df32_test.spv";
        std::FILE* fp = std::fopen(spv_path.c_str(), "wb");
        if (fp) {
            std::fwrite(k_arith->spirv.words.data(), sizeof(uint32_t), k_arith->spirv.words.size(), fp);
            std::fclose(fp);
            const std::string cmd = std::string(BROKERNEL_SPIRV_VAL) + " --target-env vulkan1.2 " + spv_path;
            const int rc = std::system(cmd.c_str());
            std::remove(spv_path.c_str());
            assert(rc == 0);
            (void)rc;
        }
    }
#endif

    constexpr uint32_t N = 16;
    std::vector<DfPairHost> inp(N), cpu_out(N), gpu_out(N);
    for (uint32_t i = 0; i < N; ++i) {
        inp[i] = f64_to_df(0.0625 + 0.000123456789 * i);
    }
    const bool cpu_ok = k_arith->dispatch_cpu(GridShape(N), {KernelArg::store(inp), KernelArg::store(cpu_out)}, nullptr, &err);
    assert(cpu_ok);
    (void)cpu_ok;
    for (uint32_t i = 0; i < N; ++i) {
        const double got = df_to_f64(cpu_out[i]);
        const double expect = 0.0625 + 0.000123456789 * i;
        assert(std::abs(got - expect) <= 1e-11);
        (void)got;
        (void)expect;
    }

    if (k_arith->dispatch_gpu(GridShape(N), {KernelArg::store(inp), KernelArg::store(gpu_out)}, nullptr, &err)) {
        for (uint32_t i = 0; i < N; ++i) {
            const double got = df_to_f64(gpu_out[i]);
            const double expect = 0.0625 + 0.000123456789 * i;
            if (std::abs(got - expect) > 1e-9) {
                std::fprintf(stderr, "df32 GPU mismatch [%u]: got %.17g expect %.17g (diff=%.3e) pair=(%.9g, %.9g)\n",
                             i, got, expect, std::abs(got - expect), gpu_out[i].hi, gpu_out[i].lo);
                return 1;
            }
        }
        std::cout << "test_df32 arith: CPU & GPU precision PASS\n";
    } else {
        std::cout << "test_df32 arith: CPU PASS (GPU skipped: " << err << ")\n";
    }

    struct AnchorHost {
        DfPairHost pos[3];
        DfPairHost height;
        int32_t tag;
    };
    const CompiledKernel* k_st = cr.find_kernel("test_struct_and_vec");
    assert(k_st && k_st->valid());
    std::vector<AnchorHost> anchors(N);
    for (uint32_t i = 0; i < N; ++i) {
        anchors[i] = AnchorHost{
            {f64_to_df(16384.0 + i * 0.25), f64_to_df(2.0), f64_to_df(3.0)},
            f64_to_df(16384.0 + 0.5 * i),
            static_cast<int32_t>(i),
        };
    }
    KernelArg off_arg = KernelArg::vec3d(-16384.0, 1.0, 2.0);
    std::vector<DfPairHost> st_cpu(N), st_gpu(N);
    const bool st_ok = k_st->dispatch_cpu(GridShape(N),
                                          {KernelArg::store(anchors), off_arg, KernelArg::store(st_cpu)},
                                          nullptr, &err);
    assert(st_ok);
    (void)st_ok;
    for (uint32_t i = 0; i < N; ++i) {
        const double want = i * 0.25 + 0.5 * i;
        assert(std::abs(df_to_f64(st_cpu[i]) - want) <= 1e-11);
        (void)want;
    }
    if (k_st->dispatch_gpu(GridShape(N),
                           {KernelArg::store(anchors), off_arg, KernelArg::store(st_gpu)},
                           nullptr, &err)) {
        for (uint32_t i = 0; i < N; ++i) {
            const double want = i * 0.25 + 0.5 * i;
            assert(std::abs(df_to_f64(st_gpu[i]) - want) <= 1e-9);
            (void)want;
        }
        std::cout << "test_df32 struct+vec: CPU & GPU precision PASS\n";
    }

    std::cout << "ALL test_df32 PASSED\n";
    return 0;
}
