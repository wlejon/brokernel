#include "brokernel/brokernel.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace brokernel {

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

namespace {

struct alignas(8) CpuPointContext {
    int32_t gx, gy, gz;
    int32_t gw, gh, gd;
    int32_t lid_x, lid_y, lid_z;
    int32_t wgid_x, wgid_y, wgid_z;
    int32_t lsz_x, lsz_y, lsz_z;
    int32_t nwg_x, nwg_y, nwg_z;
    void* shared_base;
    void (*barrier_fn)(void*);
    void* barrier_ctx;
};
static_assert(offsetof(CpuPointContext, shared_base) == 72);
static_assert(offsetof(CpuPointContext, barrier_fn) == 80);
static_assert(offsetof(CpuPointContext, barrier_ctx) == 88);

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
        if (!is_bound_param(p.role)) ++c;
    }
    return c;
}

size_t count_param_blocks(const std::vector<KernelParamInfo>& params) noexcept {
    size_t c = 0;
    for (const auto& p : params) {
        if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) ++c;
    }
    return c;
}

uint64_t store_count(const KernelParamInfo& p, const KernelArg& a) {
    return p.element_bytes ? a.buffer_bytes() / p.element_bytes : 0;
}

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
    const size_t param_blocks = count_param_blocks(kernel.params);
    const size_t global_blocks = kernel.block_bindings.size() >= param_blocks
                                     ? kernel.block_bindings.size() - param_blocks
                                     : 0;
    const size_t expected_with_blocks = non_grid + kernel.block_bindings.size();
    const size_t expected_full = kernel.params.size() + global_blocks;
    const bool skip_all_bound = (args.size() == non_grid && kernel.block_bindings.empty());
    const bool skip_non_block_bound = (args.size() == expected_with_blocks && expected_with_blocks != expected_full);
    if (args.size() != expected_full && !skip_all_bound && !skip_non_block_bound) {
        if (err) {
            *err = "kernel '" + kernel.name + "' expects " + std::to_string(expected_with_blocks) +
                   " argument(s), got " + std::to_string(args.size());
        }
        return false;
    }

    slots_out.clear();
    std::vector<const KernelArg*> param_block_args(kernel.params.size(), nullptr);
    size_t arg_idx = 0;
    for (size_t i = 0; i < kernel.params.size(); ++i) {
        const auto& p = kernel.params[i];
        if (is_bound_param(p.role)) {
            if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) {
                if (!skip_all_bound && arg_idx < args.size()) {
                    param_block_args[i] = &args[arg_idx++];
                }
            } else if (!skip_all_bound && !skip_non_block_bound) {
                ++arg_idx;
            }
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
            if (a.kind() == KernelArg::Kind::ScalarsF32) {
                for (float v : a.f32_values()) slots_out.push_back(pack_f32_bits(v));
            } else if (a.kind() == KernelArg::Kind::ScalarsF64) {
                for (double v : a.f64_values()) slots_out.push_back(pack_f64_bits(v));
            } else if (a.kind() == KernelArg::Kind::ScalarsI32) {
                for (int32_t v : a.i32_values()) slots_out.push_back(static_cast<uint64_t>(static_cast<uint32_t>(v)));
            } else if (a.kind() == KernelArg::Kind::Buffer && a.buffer_ptr()) {
                const auto* raw = static_cast<const uint8_t*>(a.buffer_ptr());
                size_t off = 0;
                uint32_t elem_sz = p.type.scalar_size_bytes();
                if (elem_sz == 0) elem_sz = 4;
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    uint64_t word = 0;
                    if (off + elem_sz <= a.buffer_bytes()) std::memcpy(&word, raw + off, elem_sz);
                    slots_out.push_back(word);
                    off += elem_sz;
                }
            } else {
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    if (p.type.scalar == ScalarKind::F64) slots_out.push_back(pack_f64_bits(a.as_f64()));
                    else if (p.type.scalar == ScalarKind::F32) slots_out.push_back(pack_f32_bits(a.as_f32()));
                    else slots_out.push_back(static_cast<uint64_t>(static_cast<uint32_t>(a.as_i32())));
                }
            }
        }
    }
    for (const auto& bb : kernel.block_bindings) {
        const KernelArg* ba = nullptr;
        if (bb.mir_param_index < kernel.params.size()) {
            ba = param_block_args[bb.mir_param_index];
        } else if (arg_idx < args.size()) {
            ba = &args[arg_idx++];
        }
        if (!ba || ba->kind() != KernelArg::Kind::Buffer || !ba->buffer_ptr()) {
            if (err) *err = "binding '" + bb.name + "' requires a buffer argument";
            return false;
        }
        slots_out.push_back(reinterpret_cast<uint64_t>(ba->buffer_ptr()));
        slots_out.push_back(static_cast<uint64_t>(ba->buffer_bytes()));
    }
    return true;
}

const CompiledKernel::CpuBuild* cpu_build(const std::shared_ptr<CompiledKernel::CpuBuild>& cpu) {
    if (!cpu) return nullptr;
    std::call_once(cpu->once, [&] {
        try {
            if (cpu->point) {
                brass::codegen::KernelJit jit_point(cpu->options);
                cpu->point_fn = jit_point.compile(*cpu->point);
            }
            if (cpu->grid) {
                brass::codegen::KernelJit jit_grid(cpu->options);
                cpu->grid_fn = jit_grid.compile(*cpu->grid);
            }
        } catch (const std::exception& ex) {
            cpu->error = std::string("backend compilation error: ") + ex.what();
        }
        cpu->module.reset();
    });
    return cpu.get();
}

const brass::codegen::KernelFunction kNoFunction{};
const std::string kNoError;

#ifdef _WIN32
using PointFn = void (*)(const uint64_t*, const void*, void*);

struct WorkgroupFiberPool {
    PointFn fn_ptr = nullptr;
    const uint64_t* slots = nullptr;
    void* main_fiber = nullptr;
    std::vector<void*> fibers;
    std::vector<CpuPointContext> contexts;
    std::vector<void*> ret_ptrs;
    std::vector<uint8_t> done;
    size_t wg_size = 0;
    size_t current_idx = 0;
    bool shutdown = false;

    void step_from(size_t idx) {
        for (size_t step = 1; step <= wg_size; ++step) {
            size_t cand = (idx + step) % wg_size;
            if (!done[cand]) {
                current_idx = cand;
                SwitchToFiber(fibers[cand]);
                return;
            }
        }
        SwitchToFiber(main_fiber);
    }

    static void WINAPI fiber_proc(void* param) {
        auto* pool = static_cast<WorkgroupFiberPool*>(param);
        const size_t idx = pool->current_idx;
        while (!pool->shutdown) {
            pool->fn_ptr(pool->slots, &pool->contexts[idx], pool->ret_ptrs[idx]);
            pool->done[idx] = 1;
            pool->step_from(idx);
        }
        SwitchToFiber(pool->main_fiber);
    }

    static void barrier_callback(void* param) {
        auto* pool = static_cast<WorkgroupFiberPool*>(param);
        pool->step_from(pool->current_idx);
    }
};
#endif

} // namespace

const brass::codegen::KernelFunction& CompiledKernel::cpu_point_fn() const {
    const CpuBuild* cpu = cpu_build(cpu_);
    return cpu ? cpu->point_fn : kNoFunction;
}

const brass::codegen::KernelFunction& CompiledKernel::cpu_grid_fn() const {
    const CpuBuild* cpu = cpu_build(cpu_);
    return cpu ? cpu->grid_fn : kNoFunction;
}

const std::string& CompiledKernel::cpu_error() const {
    const CpuBuild* cpu = cpu_build(cpu_);
    return cpu ? cpu->error : kNoError;
}

bool CompiledKernel::eval_grid_point_cpu(
    int32_t gx, int32_t gy, int32_t gz,
    GridShape shape,
    std::span<const KernelArg> args,
    void* out_return,
    std::string* err) const {
    if (!cpu_point_fn().is_valid()) {
        if (err) *err = "kernel '" + name + "' has no CPU build" + (cpu_error().empty() ? std::string() : ": " + cpu_error());
        return false;
    }
    if (gx < 0 || gy < 0 || gz < 0 || uint32_t(gx) >= std::max(1u, shape.width) ||
        uint32_t(gy) >= std::max(1u, shape.height) || uint32_t(gz) >= std::max(1u, shape.depth)) {
        if (err) *err = "cell (" + std::to_string(gx) + ", " + std::to_string(gy) + ", " + std::to_string(gz) +
                        ") is outside the grid";
        return false;
    }
    thread_local std::vector<uint64_t> slots;
    if (!pack_cpu_args(*this, args, shape, slots, err)) return false;

    const int32_t gw = static_cast<int32_t>(std::max(1u, shape.width));
    const int32_t gh = static_cast<int32_t>(std::max(1u, shape.height));
    const int32_t gd = static_cast<int32_t>(std::max(1u, shape.depth));
    const int32_t lx = static_cast<int32_t>(std::max(1u, local_size[0]));
    const int32_t ly = static_cast<int32_t>(std::max(1u, local_size[1]));
    const int32_t lz = static_cast<int32_t>(std::max(1u, local_size[2]));
    std::vector<uint8_t> shared_buf(std::max<size_t>(cpu_shared_bytes, 64), 0);
    CpuPointContext ctx{
        gx, gy, gz, gw, gh, gd,
        gx % lx, gy % ly, gz % lz,
        gx / lx, gy / ly, gz / lz,
        lx, ly, lz,
        (gw + lx - 1) / lx, (gh + ly - 1) / ly, (gd + lz - 1) / lz,
        shared_buf.data(), nullptr, nullptr,
    };

    alignas(16) uint8_t scratch_ret[256] = {};
    void* ret_ptr = out_return ? out_return : static_cast<void*>(scratch_ret);

    using PointFn = void (*)(const uint64_t*, const void*, void*);
    auto fn_ptr = cpu_point_fn().as<PointFn>();
    fn_ptr(slots.data(), &ctx, ret_ptr);
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
    if (cpu_grid_fn().is_valid()) {
        std::vector<uint64_t> slots;
        if (!pack_cpu_args(*this, args, shape, slots, err)) return false;

        const int32_t shape7[7] = {
            static_cast<int32_t>(std::max(1u, shape.width)),
            static_cast<int32_t>(std::max(1u, shape.height)),
            static_cast<int32_t>(std::max(1u, shape.depth)),
            return_store_out != nullptr ? 1 : 0,
            static_cast<int32_t>(std::max(1u, local_size[0])),
            static_cast<int32_t>(std::max(1u, local_size[1])),
            static_cast<int32_t>(std::max(1u, local_size[2])),
        };

        std::vector<uint64_t> scratch_ret;
        void* ret_ptr = return_store_out;
        if (!ret_ptr) {
            scratch_ret.resize((std::max<size_t>(return_byte_size, 8) * cpu_grid_lanes + 7) / 8);
            ret_ptr = scratch_ret.data();
        }

        using GridFn = void (*)(const uint64_t*, const int32_t*, void*);
        auto fn_ptr = cpu_grid_fn().as<GridFn>();
        fn_ptr(slots.data(), shape7, ret_ptr);
        return true;
    }

    if (!cpu_point_fn().is_valid()) {
        if (err) *err = "kernel '" + name + "' has no CPU build" + (cpu_error().empty() ? std::string() : ": " + cpu_error());
        return false;
    }

    std::vector<uint64_t> slots;
    if (!pack_cpu_args(*this, args, shape, slots, err)) return false;

    const uint32_t w = std::max(1u, shape.width);
    const uint32_t h = std::max(1u, shape.height);
    const uint32_t d = std::max(1u, shape.depth);
    const uint32_t lsz[3] = {
        std::max(1u, local_size[0]),
        std::max(1u, local_size[1]),
        std::max(1u, local_size[2]),
    };
    const uint32_t nwg[3] = {
        (w + lsz[0] - 1) / lsz[0],
        (grid_dim >= 2 || grid_dim == 0) ? (h + lsz[1] - 1) / lsz[1] : 1u,
        (grid_dim >= 3 || grid_dim == 0) ? (d + lsz[2] - 1) / lsz[2] : 1u,
    };
    const size_t wg_size = size_t(lsz[0]) * lsz[1] * lsz[2];
    const size_t ret_stride = std::max<size_t>(return_byte_size, 4);
    alignas(16) uint8_t scratch_ret[256] = {};
    std::vector<uint8_t> shared_buf(std::max<size_t>(cpu_shared_bytes, 64), 0);

    using PointFn = void (*)(const uint64_t*, const void*, void*);
    auto fn_ptr = cpu_point_fn().as<PointFn>();

#ifdef _WIN32
    if (uses_barriers_or_shared && wg_size > 1) {
        const bool was_fiber = IsThreadAFiber() != FALSE;
        void* main_fiber = was_fiber ? GetCurrentFiber() : ConvertThreadToFiber(nullptr);
        WorkgroupFiberPool pool;
        pool.fn_ptr = fn_ptr;
        pool.slots = slots.data();
        pool.main_fiber = main_fiber;
        pool.wg_size = wg_size;
        pool.fibers.resize(wg_size, nullptr);
        pool.contexts.resize(wg_size);
        pool.ret_ptrs.resize(wg_size, scratch_ret);
        pool.done.resize(wg_size, 0);
        for (size_t i = 0; i < wg_size; ++i) {
            pool.fibers[i] = CreateFiber(0, &WorkgroupFiberPool::fiber_proc, &pool);
        }

        for (uint32_t wz = 0; wz < nwg[2]; ++wz) {
            for (uint32_t wy = 0; wy < nwg[1]; ++wy) {
                for (uint32_t wx = 0; wx < nwg[0]; ++wx) {
                    std::fill(shared_buf.begin(), shared_buf.end(), 0);
                    size_t idx = 0;
                    for (uint32_t lz = 0; lz < lsz[2]; ++lz) {
                        for (uint32_t ly = 0; ly < lsz[1]; ++ly) {
                            for (uint32_t lx = 0; lx < lsz[0]; ++lx, ++idx) {
                                const uint32_t gx = wx * lsz[0] + lx;
                                const uint32_t gy = wy * lsz[1] + ly;
                                const uint32_t gz = wz * lsz[2] + lz;
                                pool.contexts[idx] = CpuPointContext{
                                    static_cast<int32_t>(gx), static_cast<int32_t>(gy), static_cast<int32_t>(gz),
                                    static_cast<int32_t>(w), static_cast<int32_t>(h), static_cast<int32_t>(d),
                                    static_cast<int32_t>(lx), static_cast<int32_t>(ly), static_cast<int32_t>(lz),
                                    static_cast<int32_t>(wx), static_cast<int32_t>(wy), static_cast<int32_t>(wz),
                                    static_cast<int32_t>(lsz[0]), static_cast<int32_t>(lsz[1]), static_cast<int32_t>(lsz[2]),
                                    static_cast<int32_t>(nwg[0]), static_cast<int32_t>(nwg[1]), static_cast<int32_t>(nwg[2]),
                                    shared_buf.data(), &WorkgroupFiberPool::barrier_callback, &pool,
                                };
                                if (return_store_out && gx < w && gy < h && gz < d) {
                                    size_t lin = (size_t(gz) * h + gy) * w + gx;
                                    pool.ret_ptrs[idx] = static_cast<uint8_t*>(return_store_out) + lin * ret_stride;
                                } else {
                                    pool.ret_ptrs[idx] = scratch_ret;
                                }
                                pool.done[idx] = 0;
                            }
                        }
                    }
                    pool.current_idx = 0;
                    SwitchToFiber(pool.fibers[0]);
                }
            }
        }

        for (void* f : pool.fibers) {
            if (f) DeleteFiber(f);
        }
        if (!was_fiber) ConvertFiberToThread();
        return true;
    }
#endif

    for (uint32_t wz = 0; wz < nwg[2]; ++wz) {
        for (uint32_t wy = 0; wy < nwg[1]; ++wy) {
            for (uint32_t wx = 0; wx < nwg[0]; ++wx) {
                std::fill(shared_buf.begin(), shared_buf.end(), 0);
                for (uint32_t lz = 0; lz < lsz[2]; ++lz) {
                    for (uint32_t ly = 0; ly < lsz[1]; ++ly) {
                        for (uint32_t lx = 0; lx < lsz[0]; ++lx) {
                            const uint32_t gx = wx * lsz[0] + lx;
                            const uint32_t gy = wy * lsz[1] + ly;
                            const uint32_t gz = wz * lsz[2] + lz;
                            CpuPointContext ctx{
                                static_cast<int32_t>(gx), static_cast<int32_t>(gy), static_cast<int32_t>(gz),
                                static_cast<int32_t>(w), static_cast<int32_t>(h), static_cast<int32_t>(d),
                                static_cast<int32_t>(lx), static_cast<int32_t>(ly), static_cast<int32_t>(lz),
                                static_cast<int32_t>(wx), static_cast<int32_t>(wy), static_cast<int32_t>(wz),
                                static_cast<int32_t>(lsz[0]), static_cast<int32_t>(lsz[1]), static_cast<int32_t>(lsz[2]),
                                static_cast<int32_t>(nwg[0]), static_cast<int32_t>(nwg[1]), static_cast<int32_t>(nwg[2]),
                                shared_buf.data(), nullptr, nullptr,
                            };
                            void* cell_ret = scratch_ret;
                            if (return_store_out && gx < w && gy < h && gz < d) {
                                size_t lin = (size_t(gz) * h + gy) * w + gx;
                                cell_ret = static_cast<uint8_t*>(return_store_out) + lin * ret_stride;
                            }
                            fn_ptr(slots.data(), &ctx, cell_ret);
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace brokernel
