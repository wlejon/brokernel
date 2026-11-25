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

    set_compile_cache(CompileCacheConfig{});
    fs::remove_all(tmp_dir);
    std::printf("test_formal_verify_and_multitarget OK\n");
}

int main() {
    test_formal_verify_and_multitarget();
    return 0;
}
