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

std::function<std::string(std::string* err)> ptx_variant_builder(const CompileResult& res,
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

} // namespace brokernel::detail
