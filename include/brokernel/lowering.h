#pragma once
// Lowering from CheckedModule / CheckedFunction to brass MIR (KernelBuilder),
// SPIR-V (SpirvTarget), PTX (PtxTarget), and CPU native code (KernelJit).

#include "brokernel/diagnostics.h"
#include "brokernel/sema.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/spirv_target.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace brokernel {

/// Metadata for a single parameter of a lowered kernel, mapping source
/// parameters (and implicit grid dimensions / output stores) to MIR and SPIR-V.
struct KernelParamInfo {
    std::string name;
    Type type;
    ParamRole role = ParamRole::Scalar;
    AccessMode access = AccessMode::Read;
    uint32_t dimension = 0;            // 1..3 for stores/stencils/images, lane count for vectors
    int32_t declared_radius = -1;      // Stencil declared radius (or -1 if inferred)
    int32_t inferred_radius = 0;       // Max |offset| seen by compiler across all axes
    int32_t halo[3] = {0, 0, 0};       // Max |offset| per axis (x, y, z)
    std::vector<StencilOffset> stencil_offsets;
    uint32_t mir_param_index = 0;      // Starting index in SpirvKernel::params
    uint32_t mir_param_count = 1;      // Number of lowered scalar/ptr MIR params (a store: its address and element count)
    uint32_t element_bytes = 0;        // A store's element size: its count is its bytes / this
    // A store the kernel reads or writes at the grid's cells (a stencil, or
    // at the cell's own coordinates), over the grid's first this-many axes:
    // the dispatch requires it to hold at least that many cells (width,
    // times height from 2, times depth at 3). Its other accesses are
    // bounded: outside the store a read is zero and a write does nothing.
    // 0 when no access is at the cells.
    uint32_t grid_cover_dim = 0;
    uint32_t push_constant_offset = 0; // Byte offset in Vulkan push-constant block
    uint32_t push_constant_size = 0;   // Byte size in Vulkan push-constant block
    uint32_t descriptor_set = 0;       // For Image / Sampler
    uint32_t descriptor_binding = 0;   // For Image / Sampler
};

struct LoweredKernelMir {
    std::unique_ptr<brass::Module> gpu_module;
    brass::Function* gpu_function = nullptr;

    std::unique_ptr<brass::Module> cpu_module;
    brass::Function* cpu_point_function = nullptr;
    brass::Function* cpu_grid_function = nullptr;
    // The cells the grid function runs at once along x: 8 or 4 (AVX2 lanes),
    // 0 when it runs one cell at a time.
    uint32_t cpu_grid_lanes = 0;

    std::vector<KernelParamInfo> param_infos;
    uint32_t grid_dim_mir_index = UINT32_MAX;   // Index of __grid_w in GPU MIR params (if grid_dim > 0)
    uint32_t return_out_mir_index = UINT32_MAX; // Index of __out in GPU MIR params (if return_type != void)
};

/// Lowers a single checked function `fn` (and any helper functions it calls from `mod`)
/// into brass MIR functions for both GPU (SpirvTarget / PtxTarget) and CPU (KernelJit).
LoweredKernelMir lower_kernel_to_mir(
    const CheckedModule& mod,
    const CheckedFunction& fn,
    DiagnosticBag& diags);

} // namespace brokernel
