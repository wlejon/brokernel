#include "brokernel/brokernel.h"

#include <brass/target/ptx_target.hpp>
#include <brass/verify/gpu_equiv.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

namespace brokernel {

const char* version() {
    return "0.1.0";
}

// ---------------------------------------------------------------------------
// KernelArg Accessors
// ---------------------------------------------------------------------------

int32_t KernelArg::as_i32() const noexcept {
    switch (kind_) {
        case Kind::I32:
        case Kind::I64: return static_cast<int32_t>(i64_);
        case Kind::U32:
        case Kind::U64: return static_cast<int32_t>(u64_);
        case Kind::F32:
        case Kind::F64: return static_cast<int32_t>(f64_);
        case Kind::ScalarsI32: return i32_vals_.empty() ? 0 : i32_vals_[0];
        default: return 0;
    }
}

uint32_t KernelArg::as_u32() const noexcept {
    switch (kind_) {
        case Kind::U32:
        case Kind::U64: return static_cast<uint32_t>(u64_);
        case Kind::I32:
        case Kind::I64: return static_cast<uint32_t>(i64_);
        case Kind::F32:
        case Kind::F64: return static_cast<uint32_t>(f64_);
        default: return 0;
    }
}

int64_t KernelArg::as_i64() const noexcept {
    switch (kind_) {
        case Kind::I32:
        case Kind::I64: return i64_;
        case Kind::U32:
        case Kind::U64: return static_cast<int64_t>(u64_);
        case Kind::F32:
        case Kind::F64: return static_cast<int64_t>(f64_);
        default: return 0;
    }
}

uint64_t KernelArg::as_u64() const noexcept {
    switch (kind_) {
        case Kind::U32:
        case Kind::U64: return u64_;
        case Kind::I32:
        case Kind::I64: return static_cast<uint64_t>(i64_);
        case Kind::F32:
        case Kind::F64: return static_cast<uint64_t>(f64_);
        default: return 0;
    }
}

float KernelArg::as_f32() const noexcept {
    switch (kind_) {
        case Kind::F32:
        case Kind::F64: return static_cast<float>(f64_);
        case Kind::I32:
        case Kind::I64: return static_cast<float>(i64_);
        case Kind::U32:
        case Kind::U64: return static_cast<float>(u64_);
        case Kind::ScalarsF32: return f32_vals_.empty() ? 0.0f : f32_vals_[0];
        case Kind::ScalarsF64: return f64_vals_.empty() ? 0.0f : static_cast<float>(f64_vals_[0]);
        default: return 0.0f;
    }
}

double KernelArg::as_f64() const noexcept {
    switch (kind_) {
        case Kind::F32:
        case Kind::F64: return f64_;
        case Kind::I32:
        case Kind::I64: return static_cast<double>(i64_);
        case Kind::U32:
        case Kind::U64: return static_cast<double>(u64_);
        case Kind::ScalarsF64: return f64_vals_.empty() ? 0.0 : f64_vals_[0];
        case Kind::ScalarsF32: return f32_vals_.empty() ? 0.0 : static_cast<double>(f32_vals_[0]);
        default: return 0.0;
    }
}

// ---------------------------------------------------------------------------
// CompiledKernel & CompileResult Lookups
// ---------------------------------------------------------------------------

const KernelParamInfo* CompiledKernel::find_param(std::string_view pname) const noexcept {
    for (const auto& p : params) {
        if (p.name == pname) return &p;
    }
    return nullptr;
}

const CompiledKernel* CompileResult::find_kernel(std::string_view name) const noexcept {
    for (const auto& k : kernels) {
        if (k.name == name) return &k;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// CPU Argument Packing & Execution
// ---------------------------------------------------------------------------

namespace {

uint64_t pack_f32_bits(float v) noexcept {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(float));
    return static_cast<uint64_t>(u);
}

uint64_t pack_f64_bits(double v) noexcept {
    uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(double));
    return u;
}

size_t count_non_grid_params(const std::vector<KernelParamInfo>& params) noexcept {
    size_t c = 0;
    for (const auto& p : params) {
        if (p.role != ParamRole::Grid && p.role != ParamRole::Image && p.role != ParamRole::Sampler) {
            ++c;
        }
    }
    return c;
}

// A store's element count: what every access to it is bounded by.
uint64_t store_count(const KernelParamInfo& p, const KernelArg& a) {
    return p.element_bytes ? a.buffer_bytes() / p.element_bytes : 0;
}

// A store the kernel accesses at the grid's cells holds them all
// (KernelParamInfo::grid_cover_dim): those accesses are not bounded.
bool store_covers(const KernelParamInfo& p, const KernelArg& a, GridShape shape, std::string* err) {
    if (p.grid_cover_dim == 0) return true;
    uint64_t cells = std::max(1u, shape.width);
    if (p.grid_cover_dim >= 2) cells *= std::max(1u, shape.height);
    if (p.grid_cover_dim >= 3) cells *= std::max(1u, shape.depth);
    const uint64_t count = store_count(p, a);
    if (count >= cells) return true;
    if (err) {
        *err = "store '" + p.name + "' has " + std::to_string(count) + " element(s); the kernel accesses it at the grid's " +
               std::to_string(cells) + " cell(s)";
    }
    return false;
}

// What an empty store (or one smaller than an element) points at: an access
// outside a store still reads its element 0 and discards it, so there has to
// be an element there.
const void* empty_store_block() {
    alignas(64) static const uint8_t zeros[4096] = {};
    return zeros;
}

bool pack_cpu_args(
    const CompiledKernel& kernel,
    std::span<const KernelArg> args,
    GridShape shape,
    std::vector<uint64_t>& slots_out,
    std::string* err) {
    const size_t non_grid = count_non_grid_params(kernel.params);
    const bool skip_domain_args = (args.size() == non_grid && kernel.params.size() != non_grid);
    if (args.size() != kernel.params.size() && !skip_domain_args) {
        if (err) {
            *err = "kernel '" + kernel.name + "' expects " + std::to_string(non_grid) +
                   " argument(s), got " + std::to_string(args.size());
        }
        return false;
    }

    slots_out.clear();
    size_t arg_idx = 0;
    for (const auto& p : kernel.params) {
        if (p.role == ParamRole::Grid || p.role == ParamRole::Image || p.role == ParamRole::Sampler) {
            if (!skip_domain_args) ++arg_idx;
            continue;
        }
        const KernelArg& a = args[arg_idx++];
        if (p.role == ParamRole::Store || p.role == ParamRole::Stencil) {
            if (a.kind() != KernelArg::Kind::Buffer || !a.buffer_ptr()) {
                if (err) *err = "parameter '" + p.name + "' requires a buffer/store argument";
                return false;
            }
            if (!store_covers(p, a, shape, err)) return false;
            const uint64_t count = store_count(p, a);
            slots_out.push_back(reinterpret_cast<uint64_t>(count ? a.buffer_ptr() : empty_store_block()));
            slots_out.push_back(count);
        } else if (p.role == ParamRole::Scalar) {
            switch (p.type.scalar) {
                case ScalarKind::F32: slots_out.push_back(pack_f32_bits(a.as_f32())); break;
                case ScalarKind::F64: slots_out.push_back(pack_f64_bits(a.as_f64())); break;
                case ScalarKind::U32: slots_out.push_back(static_cast<uint64_t>(a.as_u32())); break;
                case ScalarKind::U64: slots_out.push_back(a.as_u64()); break;
                case ScalarKind::I64: slots_out.push_back(static_cast<uint64_t>(a.as_i64())); break;
                default: slots_out.push_back(static_cast<uint64_t>(static_cast<uint32_t>(a.as_i32()))); break;
            }
        } else {
            // Vector, Array, or Struct parameter
            if (a.kind() == KernelArg::Kind::ScalarsF32) {
                for (float v : a.f32_values()) slots_out.push_back(pack_f32_bits(v));
            } else if (a.kind() == KernelArg::Kind::ScalarsF64) {
                for (double v : a.f64_values()) slots_out.push_back(pack_f64_bits(v));
            } else if (a.kind() == KernelArg::Kind::ScalarsI32) {
                for (int32_t v : a.i32_values()) slots_out.push_back(static_cast<uint64_t>(static_cast<uint32_t>(v)));
            } else if (a.kind() == KernelArg::Kind::Buffer && a.buffer_ptr()) {
                // Unpack struct/vector from raw host pointer
                const auto* raw = static_cast<const uint8_t*>(a.buffer_ptr());
                size_t off = 0;
                uint32_t elem_sz = p.type.scalar_size_bytes();
                if (elem_sz == 0) elem_sz = 4;
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    uint64_t word = 0;
                    if (off + elem_sz <= a.buffer_bytes()) {
                        std::memcpy(&word, raw + off, elem_sz);
                    }
                    slots_out.push_back(word);
                    off += elem_sz;
                }
            } else {
                // Splat scalar across leaves
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    if (p.type.scalar == ScalarKind::F64) slots_out.push_back(pack_f64_bits(a.as_f64()));
                    else if (p.type.scalar == ScalarKind::F32) slots_out.push_back(pack_f32_bits(a.as_f32()));
                    else slots_out.push_back(static_cast<uint64_t>(static_cast<uint32_t>(a.as_i32())));
                }
            }
        }
    }
    return true;
}

} // namespace

bool CompiledKernel::eval_grid_point_cpu(
    int32_t gx, int32_t gy, int32_t gz,
    GridShape shape,
    std::span<const KernelArg> args,
    void* out_return,
    std::string* err) const {
    if (!cpu_point_fn.is_valid()) {
        if (err) *err = "kernel '" + name + "' was not compiled with CPU support";
        return false;
    }
    // A point query is a call per position (a height for physics or the
    // camera), so its argument slots reuse one buffer per thread rather than
    // allocating per call. Kernels do not call back into the host, so the
    // buffer is never in use twice on one thread.
    // The cell's own coordinates index its stores unbounded, so it has to
    // be one of the shape's.
    if (gx < 0 || gy < 0 || gz < 0 || uint32_t(gx) >= std::max(1u, shape.width) ||
        uint32_t(gy) >= std::max(1u, shape.height) || uint32_t(gz) >= std::max(1u, shape.depth)) {
        if (err) *err = "cell (" + std::to_string(gx) + ", " + std::to_string(gy) + ", " + std::to_string(gz) +
                        ") is outside the grid";
        return false;
    }
    thread_local std::vector<uint64_t> slots;
    if (!pack_cpu_args(*this, args, shape, slots, err)) return false;

    const int32_t coords_shape[6] = {
        gx, gy, gz,
        static_cast<int32_t>( std::max(1u, shape.width)),
        static_cast<int32_t>(std::max(1u, shape.height)),
        static_cast<int32_t>(std::max(1u, shape.depth)),
    };

    alignas(16) uint8_t scratch_ret[256] = {};
    void* ret_ptr = out_return ? out_return : static_cast<void*>(scratch_ret);

    using PointFn = void (*)(const uint64_t*, const int32_t*, void*);
    auto fn_ptr = cpu_point_fn.as<PointFn>();
    fn_ptr(slots.data(), coords_shape, ret_ptr);
    return true;
}

bool CompiledKernel::eval_cpu(
    std::span<const KernelArg> args,
    void* out_return,
    std::string* err) const {
    return eval_grid_point_cpu(0, 0, 0, GridShape(1, 1, 1), args, out_return, err);
}

float CompiledKernel::eval_f32(std::span<const KernelArg> args) const {
    float res = 0.0f;
    eval_cpu(args, &res, nullptr);
    return res;
}

double CompiledKernel::eval_f64(std::span<const KernelArg> args) const {
    double res = 0.0;
    eval_cpu(args, &res, nullptr);
    return res;
}

int32_t CompiledKernel::eval_i32(std::span<const KernelArg> args) const {
    int32_t res = 0;
    eval_cpu(args, &res, nullptr);
    return res;
}

int64_t CompiledKernel::eval_i64(std::span<const KernelArg> args) const {
    int64_t res = 0;
    eval_cpu(args, &res, nullptr);
    return res;
}

float CompiledKernel::eval_point_2d_f32(
    int32_t gx, int32_t gy, uint32_t width, uint32_t height,
    std::span<const KernelArg> args) const {
    float res = 0.0f;
    eval_grid_point_cpu(gx, gy, 0, GridShape(width, height, 1), args, &res, nullptr);
    return res;
}

double CompiledKernel::eval_point_2d_f64(
    int32_t gx, int32_t gy, uint32_t width, uint32_t height,
    std::span<const KernelArg> args) const {
    double res = 0.0;
    eval_grid_point_cpu(gx, gy, 0, GridShape(width, height, 1), args, &res, nullptr);
    return res;
}

bool CompiledKernel::dispatch_cpu(
    GridShape shape,
    std::span<const KernelArg> args,
    void* return_store_out,
    std::string* err) const {
    if (!cpu_grid_fn.is_valid()) {
        if (err) *err = "kernel '" + name + "' was not compiled with CPU support";
        return false;
    }
    std::vector<uint64_t> slots;
    if (!pack_cpu_args(*this, args, shape, slots, err)) return false;

    const int32_t shape4[4] = {
        static_cast<int32_t>(std::max(1u, shape.width)),
        static_cast<int32_t>(std::max(1u, shape.height)),
        static_cast<int32_t>(std::max(1u, shape.depth)),
        return_store_out != nullptr ? 1 : 0,
    };

    // With no return store the cells' returns are written to scratch elements
    // 0 .. cpu_grid_lanes - 1 (the grid loop selects that index rather than
    // branching).
    std::vector<uint64_t> scratch_ret;
    void* ret_ptr = return_store_out;
    if (!ret_ptr) {
        scratch_ret.resize((std::max<size_t>(return_byte_size, 8) * cpu_grid_lanes + 7) / 8);
        ret_ptr = scratch_ret.data();
    }

    using GridFn = void (*)(const uint64_t*, const int32_t*, void*);
    auto fn_ptr = cpu_grid_fn.as<GridFn>();
    fn_ptr(slots.data(), shape4, ret_ptr);
    return true;
}

// ---------------------------------------------------------------------------
// GPU (Vulkan) Dispatch & Evaluation
// ---------------------------------------------------------------------------

size_t CompiledKernel::gpu_return_bytes(GridShape shape) const noexcept {
    if (return_type.is_void()) return 0;
    const size_t cells = grid_dim > 0 ? shape.total_points() : 1;
    return cells * std::max<size_t>(return_byte_size, 4);
}

bool CompiledKernel::gpu_launch(
    GridShape shape,
    std::span<const KernelArg> args,
    uint64_t return_address,
    GpuLaunch& out,
    std::string* err) const {
    if (stage != Stage::Compute) {
        if (err) *err = "a GPU dispatch is only supported for @compute / @kernel stages";
        return false;
    }
    if (spirv.words.empty()) {
        if (err) *err = "kernel '" + name + "' has no compiled SPIR-V binary";
        return false;
    }

    const size_t non_grid = count_non_grid_params(params);
    const bool skip_domain_args = (args.size() == non_grid && params.size() != non_grid);
    if (args.size() != params.size() && !skip_domain_args) {
        if (err) {
            *err = "kernel '" + name + "' expects " + std::to_string(non_grid) +
                   " argument(s), got " + std::to_string(args.size());
        }
        return false;
    }

    std::vector<brass::gpu::VulkanArg>& vk_args = out.args;
    vk_args.clear();
    size_t arg_idx = 0;
    for (const auto& p : params) {
        if (p.role == ParamRole::Grid || p.role == ParamRole::Image || p.role == ParamRole::Sampler) {
            if (!skip_domain_args) ++arg_idx;
            continue;
        }
        const KernelArg& a = args[arg_idx++];
        if (p.role == ParamRole::Store || p.role == ParamRole::Stencil) {
            if (a.kind() != KernelArg::Kind::Device || !a.device_address()) {
                if (err) *err = "parameter '" + p.name + "' requires a GPU store argument";
                return false;
            }
            const size_t need = std::max<size_t>(a.buffer_bytes(), p.element_bytes);
            if (a.device_capacity() < need) {
                if (err) {
                    *err = "store '" + p.name + "' has " + std::to_string(a.device_capacity()) +
                           " byte(s) behind its address; it needs " + std::to_string(need);
                }
                return false;
            }
            if (!store_covers(p, a, shape, err)) return false;
            vk_args.emplace_back(a.device_address());
            vk_args.emplace_back(static_cast<int64_t>(store_count(p, a)));
        } else if (p.role == ParamRole::Scalar) {
            switch (p.type.scalar) {
                case ScalarKind::F32: vk_args.emplace_back(a.as_f32()); break;
                case ScalarKind::F64: vk_args.emplace_back(a.as_f64()); break;
                case ScalarKind::U32: vk_args.emplace_back(a.as_u32()); break;
                case ScalarKind::U64: vk_args.emplace_back(a.as_u64()); break;
                case ScalarKind::I64: vk_args.emplace_back(a.as_i64()); break;
                default: vk_args.emplace_back(a.as_i32()); break;
            }
        } else {
            // Vector or Struct flattened into scalar push constants
            if (a.kind() == KernelArg::Kind::ScalarsF32) {
                for (float v : a.f32_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::ScalarsF64) {
                for (double v : a.f64_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::ScalarsI32) {
                for (int32_t v : a.i32_values()) vk_args.emplace_back(v);
            } else {
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    if (p.type.scalar == ScalarKind::F64) vk_args.emplace_back(a.as_f64());
                    else if (p.type.scalar == ScalarKind::F32) vk_args.emplace_back(a.as_f32());
                    else vk_args.emplace_back(a.as_i32());
                }
            }
        }
    }

    // Implicit grid extent arguments (__grid_w, __grid_h, __grid_d)
    const uint32_t w = std::max(1u, shape.width);
    const uint32_t h = std::max(1u, shape.height);
    const uint32_t d = std::max(1u, shape.depth);
    if (grid_dim >= 1) vk_args.emplace_back(static_cast<int32_t>(w));
    if (grid_dim >= 2) vk_args.emplace_back(static_cast<int32_t>(h));
    if (grid_dim >= 3) vk_args.emplace_back(static_cast<int32_t>(d));

    // Implicit output buffer argument (__out) when return_type != void
    if (!return_type.is_void()) {
        if (!return_address) {
            if (err) *err = "kernel '" + name + "' returns a value per cell; its dispatch needs a return buffer";
            return false;
        }
        vk_args.emplace_back(return_address);
    }

    for (int i = 0; i < 3; ++i) out.local_size[i] = std::max(1u, local_size[i]);
    out.groups[0] = (grid_dim >= 1) ? (w + out.local_size[0] - 1) / out.local_size[0] : 1;
    out.groups[1] = (grid_dim >= 2) ? (h + out.local_size[1] - 1) / out.local_size[1] : 1;
    out.groups[2] = (grid_dim >= 3) ? (d + out.local_size[2] - 1) / out.local_size[2] : 1;
    return true;
}

bool CompiledKernel::dispatch_gpu(
    GridShape shape,
    std::span<const KernelArg> args,
    void* return_store_out,
    std::string* err) const {
    if (spirv.words.empty()) {
        if (err) *err = "kernel '" + name + "' has no compiled SPIR-V binary";
        return false;
    }
    if (!brass::gpu::vulkan_available()) {
        if (err) *err = "no Vulkan compute device available";
        return false;
    }

    try {
        std::string vk_err;
        const brass::gpu::VulkanModule* vk_module = nullptr;
        {
            std::lock_guard<std::mutex> lock(gpu_cache_->mu);
            if (!gpu_cache_->module) {
                auto loaded = brass::gpu::VulkanModule::load(spirv, &vk_err);
                if (!loaded.valid()) {
                    if (err) *err = vk_err.empty() ? "failed to load Vulkan module" : vk_err;
                    return false;
                }
                gpu_cache_->module = std::make_unique<brass::gpu::VulkanModule>(std::move(loaded));
            }
            vk_module = gpu_cache_->module.get();
        }
        const brass::gpu::VulkanModule& vk = *vk_module;

        struct BoundBuffer {
            brass::gpu::VulkanBuffer dev_buf;
            void* host_ptr = nullptr;
            size_t byte_size = 0;
            bool write_back = false;
        };
        std::vector<BoundBuffer> bound_buffers;
        bound_buffers.reserve(params.size() + 1);

        // Each host store goes into a buffer of brass's device, and its
        // argument becomes that buffer's address.
        std::vector<KernelArg> dev_args(args.begin(), args.end());
        const size_t non_grid = count_non_grid_params(params);
        const bool skip_domain_args = (args.size() == non_grid && params.size() != non_grid);
        size_t arg_idx = 0;
        for (const auto& p : params) {
            if (p.role == ParamRole::Grid || p.role == ParamRole::Image || p.role == ParamRole::Sampler) {
                if (!skip_domain_args) ++arg_idx;
                continue;
            }
            const size_t i = arg_idx++;
            if (i >= args.size()) break;  // gpu_launch reports the count
            if (p.role != ParamRole::Store && p.role != ParamRole::Stencil) continue;
            const KernelArg& a = args[i];
            if (a.kind() != KernelArg::Kind::Buffer || !a.buffer_ptr()) {
                if (err) *err = "parameter '" + p.name + "' requires a buffer/store argument";
                return false;
            }
            if (!store_covers(p, a, shape, err)) return false;
            // Room for element 0 however small the store (Ops::Bound).
            size_t bytes = std::max<size_t>({a.buffer_bytes(), size_t(p.element_bytes), 16});
            BoundBuffer bb;
            bb.dev_buf = brass::gpu::VulkanBuffer::alloc(bytes);
            bb.host_ptr = a.buffer_ptr();
            bb.byte_size = a.buffer_bytes();
            bb.write_back = (p.role == ParamRole::Store && can_Write(p.access) && !a.is_read_only());
            if (a.buffer_bytes() > 0 && (p.role == ParamRole::Stencil || can_Read(p.access))) {
                bb.dev_buf.upload(a.buffer_ptr(), a.buffer_bytes());
            } else {
                bb.dev_buf.zero();
            }
            bound_buffers.push_back(std::move(bb));
            dev_args[i] = KernelArg::device(bound_buffers.back().dev_buf.device_address(), a.buffer_bytes(), bytes);
        }

        uint64_t return_address = 0;
        if (const size_t ret_total_bytes = gpu_return_bytes(shape)) {
            BoundBuffer ret_bb;
            ret_bb.dev_buf = brass::gpu::VulkanBuffer::alloc(std::max<size_t>(ret_total_bytes, 16));
            ret_bb.dev_buf.zero();
            ret_bb.host_ptr = return_store_out;
            ret_bb.byte_size = ret_total_bytes;
            ret_bb.write_back = (return_store_out != nullptr);
            bound_buffers.push_back(std::move(ret_bb));
            return_address = bound_buffers.back().dev_buf.device_address();
        }

        GpuLaunch launch;
        if (!gpu_launch(shape, dev_args, return_address, launch, err)) return false;

        brass::gpu::VulkanDispatch disp;
        for (int i = 0; i < 3; ++i) {
            disp.grid[i] = launch.groups[i];
            disp.block[i] = launch.local_size[i];
        }

        if (!vk.launch(disp, launch.args, &vk_err)) {
            if (err) *err = vk_err;
            return false;
        }

        for (const auto& bb : bound_buffers) {
            if (bb.write_back && bb.host_ptr && bb.byte_size > 0) {
                bb.dev_buf.download(bb.host_ptr, bb.byte_size);
            }
        }
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = ex.what();
        return false;
    }
}

bool CompiledKernel::eval_gpu(std::span<const KernelArg> args, void* out_return, std::string* err) const {
    return dispatch_gpu(GridShape(1, 1, 1), args, out_return, err);
}

float CompiledKernel::eval_gpu_f32(std::span<const KernelArg> args, std::string* err) const {
    float res = 0.0f;
    eval_gpu(args, &res, err);
    return res;
}

double CompiledKernel::eval_gpu_f64(std::span<const KernelArg> args, std::string* err) const {
    double res = 0.0;
    eval_gpu(args, &res, err);
    return res;
}

// ---------------------------------------------------------------------------
// CPU vs GPU Target Agreement Check
// ---------------------------------------------------------------------------

ParityReport CompiledKernel::compare_cpu_gpu(
    GridShape shape,
    std::span<const KernelArg> args,
    double rel_tol,
    double abs_tol) const {
    // Agreement means both targets ran: without a device the report fails
    // (ran_gpu false) rather than passing on the CPU's word alone.
    ParityReport rep;
    if (!brass::gpu::vulkan_available()) {
        rep.ran_gpu = false;
        rep.ok = false;
        rep.message = "no Vulkan compute device: the GPU build did not run";
        return rep;
    }

    const size_t non_grid = count_non_grid_params(params);
    const bool skip_domain_args = (args.size() == non_grid && params.size() != non_grid);

    // Clone buffer arguments so CPU and GPU dispatches operate on independent buffers
    std::vector<std::vector<uint8_t>> cpu_bufs(args.size());
    std::vector<std::vector<uint8_t>> gpu_bufs(args.size());
    std::vector<KernelArg> cpu_args(args.begin(), args.end());
    std::vector<KernelArg> gpu_args(args.begin(), args.end());

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].kind() == KernelArg::Kind::Buffer && args[i].buffer_ptr() && args[i].buffer_bytes() > 0) {
            const auto* src = static_cast<const uint8_t*>(args[i].buffer_ptr());
            cpu_bufs[i].assign(src, src + args[i].buffer_bytes());
            gpu_bufs[i].assign(src, src + args[i].buffer_bytes());
            cpu_args[i] = KernelArg::buffer(cpu_bufs[i].data(), cpu_bufs[i].size());
            gpu_args[i] = KernelArg::buffer(gpu_bufs[i].data(), gpu_bufs[i].size());
        }
    }

    size_t ret_bytes = 0;
    if (!return_type.is_void()) {
        size_t num_pts = (grid_dim > 0) ? shape.total_points() : 1;
        ret_bytes = num_pts * std::max<size_t>(return_byte_size, 4);
    }
    std::vector<uint8_t> cpu_ret(ret_bytes, 0);
    std::vector<uint8_t> gpu_ret(ret_bytes, 0);

    std::string err;
    if (!dispatch_cpu(shape, cpu_args, ret_bytes > 0 ? cpu_ret.data() : nullptr, &err)) {
        rep.ok = false;
        rep.message = "CPU dispatch failed: " + err;
        return rep;
    }
    if (!dispatch_gpu(shape, gpu_args, ret_bytes > 0 ? gpu_ret.data() : nullptr, &err)) {
        rep.ok = false;
        rep.message = "GPU dispatch failed: " + err;
        return rep;
    }
    rep.ran_gpu = true;
    rep.ok = true;

    // Floats agree within abs_tol or rel_tol; a NaN agrees only with a NaN.
    // Integers agree exactly.
    auto compare_float = [&](double x, double y, uint64_t ulps) {
        ++rep.elements_compared;
        if (std::isnan(x) || std::isnan(y)) {
            if (std::isnan(x) != std::isnan(y)) {
                rep.ok = false;
                rep.max_abs_diff = std::numeric_limits<double>::infinity();
            }
            return;
        }
        if (x == y) return; // equal infinities included
        double diff = std::abs(x - y);
        double rdiff = diff / std::max({std::abs(x), std::abs(y), 1e-300});
        rep.max_abs_diff = std::max(rep.max_abs_diff, diff);
        rep.max_rel_diff = std::max(rep.max_rel_diff, rdiff);
        rep.max_ulp_diff = std::max(rep.max_ulp_diff, ulps);
        if (!(diff <= abs_tol || rdiff <= rel_tol)) rep.ok = false;
    };
    // Distance in representable values (both finite, neither NaN).
    auto ulp_distance = [](auto x, auto y) -> uint64_t {
        using F = decltype(x);
        using I = std::conditional_t<sizeof(F) == 8, int64_t, int32_t>;
        auto ordered = [](F v) {
            I i;
            std::memcpy(&i, &v, sizeof(v));
            return i < 0 ? static_cast<int64_t>(std::numeric_limits<I>::min()) - i : static_cast<int64_t>(i);
        };
        int64_t a = ordered(x), b = ordered(y);
        return a > b ? static_cast<uint64_t>(a) - static_cast<uint64_t>(b)
                     : static_cast<uint64_t>(b) - static_cast<uint64_t>(a);
    };
    auto compare_int = [&](int64_t x, int64_t y) {
        ++rep.elements_compared;
        if (x == y) return;
        rep.ok = false;
        rep.max_abs_diff = std::max(rep.max_abs_diff, std::abs(static_cast<double>(x) - static_cast<double>(y)));
    };
    auto compare_bytes = [&](const uint8_t* a, const uint8_t* b, size_t nbytes, ScalarKind sk) {
        if (sk == ScalarKind::F64) {
            for (size_t off = 0; off + 8 <= nbytes; off += 8) {
                double x, y;
                std::memcpy(&x, a + off, 8);
                std::memcpy(&y, b + off, 8);
                compare_float(x, y, std::isfinite(x) && std::isfinite(y) ? ulp_distance(x, y) : 0);
            }
        } else if (sk == ScalarKind::F32) {
            for (size_t off = 0; off + 4 <= nbytes; off += 4) {
                float x, y;
                std::memcpy(&x, a + off, 4);
                std::memcpy(&y, b + off, 4);
                compare_float(x, y, std::isfinite(x) && std::isfinite(y) ? ulp_distance(x, y) : 0);
            }
        } else if (sk == ScalarKind::I64 || sk == ScalarKind::U64) {
            for (size_t off = 0; off + 8 <= nbytes; off += 8) {
                int64_t x, y;
                std::memcpy(&x, a + off, 8);
                std::memcpy(&y, b + off, 8);
                compare_int(x, y);
            }
        } else {
            for (size_t off = 0; off + 4 <= nbytes; off += 4) {
                int32_t x, y;
                std::memcpy(&x, a + off, 4);
                std::memcpy(&y, b + off, 4);
                compare_int(x, y);
            }
        }
    };

    // Compare writable store buffers (and copy GPU output back to caller's writable buffers)
    size_t arg_idx = 0;
    for (const auto& p : params) {
        if (p.role == ParamRole::Grid || p.role == ParamRole::Image || p.role == ParamRole::Sampler) {
            if (!skip_domain_args) ++arg_idx;
            continue;
        }
        size_t cur = arg_idx++;
        if (p.role == ParamRole::Store && can_Write(p.access) && !cpu_bufs[cur].empty()) {
            compare_bytes(cpu_bufs[cur].data(), gpu_bufs[cur].data(), cpu_bufs[cur].size(), p.type.scalar);
            if (!args[cur].is_read_only() && args[cur].buffer_ptr()) {
                std::memcpy(args[cur].buffer_ptr(), gpu_bufs[cur].data(), gpu_bufs[cur].size());
            }
        }
    }

    if (ret_bytes > 0) {
        // One element per point; its leaves are packed in order (a struct's
        // may differ in kind).
        const size_t stride = std::max<size_t>(return_byte_size, 4);
        for (size_t base = 0; base + stride <= ret_bytes; base += stride) {
            size_t off = 0;
            for (ScalarKind sk : return_leaf_kinds) {
                size_t sz = (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) ? 8 : 4;
                if (off + sz > stride) break;
                compare_bytes(cpu_ret.data() + base + off, gpu_ret.data() + base + off, sz, sk);
                off += sz;
            }
        }
    }

    if (!rep.ok) {
        std::ostringstream oss;
        oss << "CPU/GPU mismatch: max_abs_diff=" << rep.max_abs_diff
            << ", max_rel_diff=" << rep.max_rel_diff;
        rep.message = oss.str();
    }
    return rep;
}

// ---------------------------------------------------------------------------
// Compiler Implementation
// ---------------------------------------------------------------------------

Compiler::Compiler() = default;

void Compiler::register_module(std::string name, std::string source) {
    modules_.register_module(std::move(name), std::move(source));
}

void Compiler::add_search_path(std::filesystem::path dir) {
    modules_.add_search_path(std::move(dir));
}

CompileResult Compiler::compile_source(
    std::string_view source,
    std::string file_name,
    const CompileOptions& opts) {
    CompileResult res;

    auto ast = parse_source(source, file_name, res.diagnostics);
    if (!ast || res.diagnostics.has_errors()) {
        res.ok = false;
        return res;
    }

    std::filesystem::path root_dir;
    if (!file_name.empty()) {
        std::filesystem::path p(file_name);
        if (p.has_parent_path()) root_dir = p.parent_path();
    }

    auto checked = check_module(*ast, modules_, res.diagnostics, root_dir);
    if (!checked || res.diagnostics.has_errors()) {
        res.ok = false;
        return res;
    }

    res.module_name = checked->module_name;
    res.types = checked->types;

    // Determine which functions in root module should be compiled as entry points
    bool any_explicit_kernel = false;
    for (const auto& fn : checked->functions) {
        if (fn.module_name == checked->module_name && fn.is_entry_kernel) {
            any_explicit_kernel = true;
            break;
        }
    }

    std::vector<const CheckedFunction*> targets;
    for (const auto& fn : checked->functions) {
        if (fn.module_name != checked->module_name) continue;
        if (!opts.entry_name.empty()) {
            if (fn.name == opts.entry_name || fn.source_name == opts.entry_name) {
                targets.push_back(&fn);
            }
            continue;
        }
        if (opts.compile_all_functions || !any_explicit_kernel || fn.is_entry_kernel) {
            targets.push_back(&fn);
        }
    }

    if (!opts.entry_name.empty() && targets.empty()) {
        res.diagnostics.error(ast->span, "entry function '" + opts.entry_name + "' not found in module");
        res.ok = false;
        return res;
    }

    for (const CheckedFunction* fn : targets) {
        LoweredKernelMir lowered = lower_kernel_to_mir(*checked, *fn, res.diagnostics);
        if (res.diagnostics.has_errors()) {
            res.ok = false;
            return res;
        }

        CompiledKernel ck;
        ck.name = fn->source_name;
        ck.stage = fn->stage;
        ck.grid_dim = fn->grid_dim;
        for (int i = 0; i < 3; ++i) {
            ck.local_size[i] = fn->has_explicit_local_size ? fn->local_size[i]
                               : (fn->grid_dim >= 2 ? fn->local_size[i] : opts.local_size[i]);
        }
        ck.params = lowered.param_infos;
        ck.return_type = fn->return_type;
        ck.return_leaf_kinds = checked->types.flatten_scalars(fn->return_type);
        ck.return_flat_leaves = static_cast<uint32_t>(ck.return_leaf_kinds.size());
        ck.return_byte_size = checked->types.byte_size(fn->return_type);
        ck.uses_f64 = fn->uses_f64;
        ck.uses_grid_coords = fn->uses_grid_coords;
        ck.max_stencil_radius = fn->max_stencil_radius;
        ck.stencil_halo[0] = fn->stencil_halo[0];
        ck.stencil_halo[1] = fn->stencil_halo[1];
        ck.stencil_halo[2] = fn->stencil_halo[2];

        try {
            if (opts.emit_spirv) {
                brass::target::SpirvOptions spv_opts;
                spv_opts.stage = ck.stage;
                spv_opts.local_size_x = ck.local_size[0];
                spv_opts.local_size_y = ck.local_size[1];
                spv_opts.local_size_z = ck.local_size[2];
                spv_opts.local_size_spec_constants = opts.local_size_spec_constants;
                spv_opts.spirv_version = opts.spirv_version;
                spv_opts.precise_float = true; // the GPU rounds as the CPU build does

                ck.spirv = brass::target::SpirvTarget::compile(*lowered.gpu_function, spv_opts);
                if (opts.emit_spirv_disassembly) {
                    ck.spirv_disassembly = brass::target::SpirvTarget::dump_function(*lowered.gpu_function, spv_opts);
                }

                // Populate push constant offsets on `ck.params`
                for (auto& p : ck.params) {
                    if (p.mir_param_count > 0 && p.mir_param_index < ck.spirv.params.size()) {
                        p.push_constant_offset = ck.spirv.params[p.mir_param_index].offset;
                        uint32_t total_sz = 0;
                        for (uint32_t k = 0; k < p.mir_param_count && (p.mir_param_index + k) < ck.spirv.params.size(); ++k) {
                            total_sz += ck.spirv.params[p.mir_param_index + k].size;
                        }
                        p.push_constant_size = total_sz;
                    }
                }

                if (opts.verify_spirv_equivalence && ck.stage == Stage::Compute) {
                    auto eq = brass::verify::verify_spirv_equivalence(*lowered.gpu_function, spv_opts);
                    ck.spirv_equivalent = eq.is_equivalent();
                }
            }

            if (opts.emit_ptx && ck.stage == Stage::Compute) {
                ck.ptx = brass::target::PtxTarget::emit_function(*lowered.gpu_function);
            }

            if (opts.compile_cpu && lowered.cpu_point_function) {
                brass::codegen::KernelOptions kopts;
                kopts.opt_level = opts.opt_level;
                kopts.enable_optimizations = (opts.opt_level > 0);
                kopts.enable_fma = false;
                kopts.enable_parallel = false;
                kopts.enable_superopt = false;
                kopts.enable_smt_guard_elim = false;

                brass::codegen::KernelJit jit_point(kopts);
                ck.cpu_point_fn = jit_point.compile(*lowered.cpu_point_function);

                brass::codegen::KernelJit jit_grid(kopts);
                ck.cpu_grid_fn = jit_grid.compile(*lowered.cpu_grid_function);
                ck.cpu_grid_lanes = lowered.cpu_grid_lanes ? lowered.cpu_grid_lanes : 1;
            }
        } catch (const std::exception& ex) {
            res.diagnostics.error(fn->span, std::string("backend compilation error: ") + ex.what());
            res.ok = false;
            return res;
        }

        res.kernels.push_back(std::move(ck));
    }

    res.ok = !res.diagnostics.has_errors();
    return res;
}

CompileResult Compiler::compile_file(
    const std::filesystem::path& path,
    const CompileOptions& opts) {
    CompileResult res;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        res.diagnostics.error({path.string(), 1, 1, 0, 0},
                              "could not open kernel file '" + path.string() + "'");
        res.ok = false;
        return res;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return compile_source(ss.str(), path.string(), opts);
}

CompileResult compile_source(
    std::string_view source,
    std::string file_name,
    const CompileOptions& opts) {
    Compiler compiler;
    return compiler.compile_source(source, std::move(file_name), opts);
}

CompileResult compile_file(
    const std::filesystem::path& path,
    const CompileOptions& opts) {
    Compiler compiler;
    return compiler.compile_file(path, opts);
}

} // namespace brokernel
