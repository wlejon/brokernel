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

// A CPU build with the code generator's options for `opts`, nothing lowered.
std::shared_ptr<CompiledKernel::CpuBuild> new_cpu_build(const CompileOptions& opts);

} // namespace brokernel::detail
