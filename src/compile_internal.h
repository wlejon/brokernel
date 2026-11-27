#pragma once
// What a compile and a compile-cache hit build alike.
#include "brokernel/brokernel.h"

#include <functional>
#include <memory>
#include <string>

namespace brokernel::detail {

// A kernel's probed variant (CompiledKernel::build_probed_variant): `res`
// compiled again from its source, modules and options with probes on, for
// the kernel named `kernel` alone.
std::function<std::shared_ptr<const CompiledKernel>()> probed_variant_builder(const CompileResult& res,
                                                                              const std::string& kernel);
std::function<std::shared_ptr<const CompiledKernel>(bool trap_nf)> checked_variant_builder(const CompileResult& res,
                                                                                           const std::string& kernel);

// Thread-local CPU diagnostic buffers & runtime formatting helpers (runtime_checks.cpp).
uint32_t* active_cpu_check_buf();
uint32_t* active_cpu_probe_if_buf();
struct ScopedCpuCheckBuf {
    uint32_t* prev;
    explicit ScopedCpuCheckBuf(uint32_t* buf);
    ~ScopedCpuCheckBuf();
};
struct ScopedCpuProbeIfBuf {
    uint32_t* prev;
    explicit ScopedCpuProbeIfBuf(uint32_t* buf);
    ~ScopedCpuProbeIfBuf();
};
CheckReport decode_check_report(const CompiledKernel& k, const uint32_t words[8]);
void decode_probe_if_samples(const CompiledKernel& k, const std::vector<uint32_t>& words, uint32_t max_samples,
                             std::vector<ConditionalProbeSample>& out);

std::function<std::string(std::string* err)> ptx_variant_builder(const CompileResult& res,
                                                                 const std::string& kernel);
std::function<MslArtifact(std::string* err)> msl_variant_builder(const CompileResult& res,
                                                                 const std::string& kernel);

// A CPU build with the code generator's options for `opts`, nothing lowered.
std::shared_ptr<CompiledKernel::CpuBuild> new_cpu_build(const CompileOptions& opts);

// Appends GPU runtime `@knob` leaf arguments to `vk_args` (runtime_knobs.cpp).
void append_gpu_runtime_knob_args(const CompiledKernel& k, std::vector<brass::gpu::VulkanArg>& vk_args);

// Appends GPU telemetry buffer pointer + clamped slot capacity to `vk_args` (lowering_telemetry.cpp).
void append_gpu_telemetry_args(const CompiledKernel& k, GpuLaunch& out, std::vector<brass::gpu::VulkanArg>& vk_args);

struct TelemetryVkBounce {
    brass::gpu::VulkanBuffer dev_buf;
    void* host_ptr = nullptr;
    size_t byte_size = 0;
};
void setup_gpu_telemetry_bounce(const CompiledKernel& k, GpuLaunch& launch, TelemetryVkBounce& bounce);
void finish_gpu_telemetry_bounce(const TelemetryVkBounce& bounce);

// Emits optional PTX/MSL text targets onto `ck` (kernel_verify.cpp).
void emit_optional_gpu_targets(const CompileOptions& opts, const CheckedFunction& fn,
                               const brass::Function& gpu_fn, CompiledKernel& ck);

inline size_t count_non_grid_params(const std::vector<KernelParamInfo>& params) noexcept {
    size_t c = 0;
    for (const auto& p : params) if (!is_bound_param(p.role)) ++c;
    return c;
}

inline size_t count_param_blocks(const std::vector<KernelParamInfo>& params) noexcept {
    size_t c = 0;
    for (const auto& p : params) {
        if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) ++c;
    }
    return c;
}

} // namespace brokernel::detail
