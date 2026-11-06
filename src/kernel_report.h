#pragma once
// What compile_source reports of a kernel it built: its code's hash and its
// instructions by source function.

#include "brokernel/brokernel.h"

#include <cstdint>
#include <string>
#include <vector>

namespace brokernel::detail {

// FNV-1a over the kernel's SPIR-V, or with none, its lowered CPU functions'
// text (printed before the CPU build, which consumes them).
uint64_t code_hash(const CompiledKernel& ck);

// The function's instructions by the inlined scope each carries (lowering
// stamps every inlined call's body with one naming the callee), as self and
// total counts per source function, largest total first.
std::vector<CompiledKernel::FunctionCost> function_costs(const brass::Function& fn, const brass::DebugContext& dc,
                                                         const std::string& entry);

} // namespace brokernel::detail
