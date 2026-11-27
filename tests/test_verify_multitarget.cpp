#include "brokernel/brokernel.h"
#include "brokernel/compile_cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace brokernel;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK FAILED (%s:%d): %s\n", __FILE__, __LINE__, #cond);         \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

static void test_formal_verify_and_multitarget() {
    const char* src = R"(
        shared var tile: array<f32, 64>;

        @kernel fn smooth_copy(g: grid1d, inp: read_store<f32, 1>, out: write_store<f32, 1>) {
            let lid: i32 = g.x() % 64;
            tile[lid] = inp[g.x()];
            workgroup_barrier();
            out[g.x()] = tile[lid] * 0.5 + tile[63 - lid] * 0.5;
        }
    )";

    CompileOptions opts;
    opts.emit_spirv = true;
    opts.emit_spirv_disassembly = true;
    opts.emit_ptx = true;
    opts.emit_msl = true;

    auto res = compile_source(src, "verify_mt.bk", opts);
    if (!res.ok) std::fprintf(stderr, "compile err: %s\n", res.error_message().c_str());
    CHECK(res.ok);
    const CompiledKernel* k = res.find_kernel("smooth_copy");
    CHECK(k != nullptr);

    KernelVerifyReport rep = k->verify();
    std::printf("verify: parsed=%d mem_safe=%d bar_uniform=%d shared=%zu global=%zu bar=%zu diag=%s\n",
                rep.spirv_parsed ? 1 : 0, rep.memory_safe ? 1 : 0, rep.barrier_uniform ? 1 : 0,
                rep.shared_accesses_proven, rep.global_accesses_proven, rep.barriers_proven,
                rep.diagnostic.c_str());
    CHECK(rep.spirv_parsed);
    CHECK(rep.memory_safe);
    CHECK(rep.barrier_uniform);
    CHECK(rep.barriers_proven >= 1);

    CHECK(!k->spirv_disassembly.empty());
    CHECK(!k->msl.empty());
    CHECK(k->msl.find("kernel ") != std::string::npos);
    CHECK(!k->msl_entry.empty());
    CHECK(k->msl_num_buffers >= 2);
    CHECK(!k->msl_scalar_param_offsets.empty());
    CHECK(!k->ptx.empty());
    CHECK(k->ptx.find(".entry") != std::string::npos);

    KernelSandboxReport sb = k->verify_sandbox();
    CHECK(sb.passed);
    CHECK(sb.portability_ok);
    CHECK(sb.memory_safe);
    CHECK(sb.barrier_uniform);
    CHECK(sb.proof_tier == brass::SandboxProofTier::FullSmt);
    CHECK(sb.hardening_tier == brass::HardeningTier::PureSmtZeroOverhead);
    CHECK(sb.global_bounds_proven);

    // Verify dispatch bounds enforcement against prover dimension caps on `smooth_copy`
    // and proved grid-cover store length on a `@grid(1)` kernel without barriers.
    {
        std::vector<float> buf_in(64, 1.0f);
        std::vector<float> buf_out(64, 0.0f);
        std::vector<KernelArg> ok_args = {
            KernelArg::buffer(buf_in.data(), buf_in.size() * sizeof(float)),
            KernelArg::buffer(buf_out.data(), buf_out.size() * sizeof(float)),
        };
        std::string berr;
        CHECK(k->verify_dispatch_bounds(GridShape(64), ok_args, &berr));
        berr.clear();
        std::vector<KernelArg> huge_args = {
            KernelArg::buffer(buf_in.data(), (brass::kMaxProverDim1D + 1) * sizeof(float)),
            KernelArg::buffer(buf_out.data(), (brass::kMaxProverDim1D + 1) * sizeof(float)),
        };
        CHECK(!k->verify_dispatch_bounds(GridShape(4), huge_args, &berr));
        CHECK(!berr.empty());

        const char* grid_src = R"(
            @kernel @grid(1) fn grid_copy(g: grid1d, inp: read_store<f32, 1>, out: write_store<f32, 1>) {
                let i: i32 = g.x();
                out[i] = inp[i];
            }
        )";
        CompileOptions g_opts;
        g_opts.emit_spirv = true;
        auto gres = compile_source(grid_src, "verify_grid_bounds.bk", g_opts);
        CHECK(gres.ok);
        const CompiledKernel* gk = gres.find_kernel("grid_copy");
        CHECK(gk != nullptr && gk->verify_sandbox().passed);
        berr.clear();
        CHECK(gk->verify_dispatch_bounds(GridShape(64), ok_args, &berr));
        berr.clear();
        CHECK(!gk->verify_dispatch_bounds(GridShape(65), ok_args, &berr));
        CHECK(!berr.empty());
    }

    // Verify lazy ensure_msl() when emit_msl = false
    CompileOptions lazy_opts;
    lazy_opts.emit_spirv = true;
    lazy_opts.emit_msl = false;
    auto lazy_res = compile_source(src, "verify_lazy_msl.bk", lazy_opts);
    CHECK(lazy_res.ok);
    const CompiledKernel* lk = lazy_res.find_kernel("smooth_copy");
    CHECK(lk != nullptr && lk->msl.empty());
    std::string msl_err;
    const MslArtifact& art = lk->ensure_msl(&msl_err);
    CHECK(!art.source.empty() && art.source.find("kernel ") != std::string::npos);
    CHECK(!art.entry_name.empty());
    CHECK(art.num_buffers == k->msl_num_buffers);
    CHECK(art.scalar_param_offsets == k->msl_scalar_param_offsets);

    // @cpu kernel must fail sandbox verification cleanly
    const char* cpu_src = R"(
        @cpu @kernel fn cpu_only(g: grid1d, out: write_store<f32, 1>) {
            out[g.x()] = 1.0;
        }
    )";
    auto cpu_res = compile_source(cpu_src, "verify_cpu_sb.bk", lazy_opts);
    CHECK(cpu_res.ok);
    const CompiledKernel* ck = cpu_res.find_kernel("cpu_only");
    CHECK(ck != nullptr && !ck->verify_sandbox().passed);

    // Also verify compile-cache serialization preserves msl & ptx
    namespace fs = std::filesystem;
    fs::path tmp_dir = fs::temp_directory_path() / "brokernel_test_mt_cache";
    fs::remove_all(tmp_dir);
    fs::create_directories(tmp_dir);
    CompileCacheConfig cfg;
    cfg.dir = tmp_dir;
    cfg.max_bytes = 16 * 1024 * 1024;
    set_compile_cache(cfg);

    auto miss_res = compile_source(src, "verify_mt_cache.bk", opts);
    CHECK(miss_res.ok);
    CHECK(miss_res.cache_status == CacheStatus::Miss);

    auto hit_res = compile_source(src, "verify_mt_cache.bk", opts);
    CHECK(hit_res.ok);
    CHECK(hit_res.cache_status == CacheStatus::Hit);
    const CompiledKernel* hk = hit_res.find_kernel("smooth_copy");
    CHECK(hk != nullptr);
    CHECK(hk->msl == miss_res.kernels[0].msl);
    CHECK(hk->ptx == miss_res.kernels[0].ptx);
    CHECK(hk->verify().memory_safe);

    // Verify harden_sandbox compiles & caches a dynamic-index kernel with HybridSmtClamped.
    const char* dyn_src = R"(
        @kernel fn scatter_add(g: grid1d, idx_buf: read_store<i32, 1>, inp: read_store<f32, 1>,
                              out: rw_store<f32, 1>, n: i32) {
            let tid: i32 = g.x();
            let target: i32 = idx_buf[tid];
            var acc: f32 = 0.0;
            for (var k: i32 = 0; k < n; k = k + 1) {
                acc = acc + inp[k];
            }
            out[target] = acc;
        }
    )";
    CompileOptions h_opts;
    h_opts.emit_spirv = true;
    h_opts.harden_sandbox = true;
    h_opts.sandbox_max_loop_iterations = 4096;
    auto h_miss = compile_source(dyn_src, "verify_harden_cache.bk", h_opts);
    CHECK(h_miss.ok);
    CHECK(h_miss.cache_status == CacheStatus::Miss);
    const CompiledKernel* dk = h_miss.find_kernel("scatter_add");
    CHECK(dk != nullptr);
    const KernelSandboxReport dsb = dk->verify_sandbox();
    CHECK(dsb.passed);
    CHECK(dsb.hardening_tier == brass::HardeningTier::HybridSmtClamped);
    CHECK((dsb.clamped_global_loads + dsb.clamped_global_stores + dsb.loops_fuel_instrumented) > 0);

    auto h_hit = compile_source(dyn_src, "verify_harden_cache.bk", h_opts);
    CHECK(h_hit.ok);
    CHECK(h_hit.cache_status == CacheStatus::Hit);
    const CompiledKernel* dkh = h_hit.find_kernel("scatter_add");
    CHECK(dkh != nullptr);
    const KernelSandboxReport dsb_hit = dkh->verify_sandbox();
    CHECK(dsb_hit.passed);
    CHECK(dsb_hit.hardening_tier == brass::HardeningTier::HybridSmtClamped);
    CHECK(dsb_hit.clamped_global_loads == dsb.clamped_global_loads);
    CHECK(dsb_hit.clamped_global_stores == dsb.clamped_global_stores);
    CHECK(dsb_hit.loops_fuel_instrumented == dsb.loops_fuel_instrumented);

    set_compile_cache(CompileCacheConfig{});
    fs::remove_all(tmp_dir);
    std::printf("test_formal_verify_and_multitarget OK\n");
}

int main() {
    test_formal_verify_and_multitarget();
    return 0;
}
