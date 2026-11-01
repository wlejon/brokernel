#include "brokernel/brokernel.h"

#include <brass/target/host_cpu.hpp>
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

namespace {

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

} // namespace

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
        if (err) *err = "kernel '" + name + (cpu_only ? "' is a @cpu kernel: it has no GPU build" : "' has no compiled SPIR-V binary");
        return false;
    }

    const size_t non_grid = count_non_grid_params(params);
    const size_t param_blocks = count_param_blocks(params);
    const size_t global_blocks = block_bindings.size() >= param_blocks ? block_bindings.size() - param_blocks : 0;
    const size_t expected_with_blocks = non_grid + block_bindings.size();
    const size_t expected_full = params.size() + global_blocks;
    const bool skip_all_bound = (args.size() == non_grid);
    const bool skip_non_block_bound = (args.size() == expected_with_blocks && expected_with_blocks != expected_full);
    if (args.size() != params.size() && args.size() != expected_full && !skip_all_bound && !skip_non_block_bound) {
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
        if (is_bound_param(p.role)) {
            if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) {
                if (!skip_all_bound) ++arg_idx;
            } else if (!skip_all_bound && !skip_non_block_bound) {
                ++arg_idx;
            }
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
            if (a.kind() == KernelArg::Kind::ScalarsF32) {
                for (float v : a.f32_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::ScalarsF64) {
                for (double v : a.f64_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::ScalarsI32) {
                for (int32_t v : a.i32_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::Buffer && a.buffer_ptr()) {
                const auto* raw = static_cast<const uint8_t*>(a.buffer_ptr());
                uint32_t elem_sz = p.type.scalar_size_bytes();
                if (elem_sz == 0) elem_sz = 4;
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    uint64_t word = 0;
                    const size_t off = size_t{k} * elem_sz;
                    if (off + elem_sz <= a.buffer_bytes()) std::memcpy(&word, raw + off, elem_sz);
                    const size_t leaf = p.mir_param_index + k;
                    const brass::Type lt = leaf < spirv.params.size() ? spirv.params[leaf].type : brass::Type();
                    if (lt.kind() == brass::TypeKind::F32) {
                        float f;
                        const uint32_t bits = static_cast<uint32_t>(word);
                        std::memcpy(&f, &bits, 4);
                        vk_args.emplace_back(f);
                    } else if (lt.kind() == brass::TypeKind::F64) {
                        double f;
                        std::memcpy(&f, &word, 8);
                        vk_args.emplace_back(f);
                    } else if (lt.is_i32()) {
                        vk_args.emplace_back(static_cast<uint32_t>(word));
                    } else {
                        vk_args.emplace_back(word);
                    }
                }
            } else {
                for (uint32_t k = 0; k < p.mir_param_count; ++k) {
                    if (p.type.scalar == ScalarKind::F64) vk_args.emplace_back(a.as_f64());
                    else if (p.type.scalar == ScalarKind::F32) vk_args.emplace_back(a.as_f32());
                    else vk_args.emplace_back(a.as_i32());
                }
            }
        }
    }

    const uint32_t w = std::max(1u, shape.width);
    const uint32_t h = std::max(1u, shape.height);
    const uint32_t d = std::max(1u, shape.depth);
    if (grid_dim >= 1) vk_args.emplace_back(static_cast<int32_t>(w));
    if (grid_dim >= 2) vk_args.emplace_back(static_cast<int32_t>(h));
    if (grid_dim >= 3) vk_args.emplace_back(static_cast<int32_t>(d));

    if (!return_type.is_void()) {
        if (!return_address) {
            if (err) *err = "kernel '" + name + "' returns a value per cell; its dispatch needs a return buffer";
            return false;
        }
        vk_args.emplace_back(return_address);
    }

    for (int i = 0; i < 3; ++i) out.local_size[i] = std::max(1u, local_size[i]);
    out.groups[0] = (w + out.local_size[0] - 1) / out.local_size[0];
    out.groups[1] = (grid_dim >= 2 || grid_dim == 0) ? (h + out.local_size[1] - 1) / out.local_size[1] : 1;
    out.groups[2] = (grid_dim >= 3 || grid_dim == 0) ? (d + out.local_size[2] - 1) / out.local_size[2] : 1;
    return true;
}

bool CompiledKernel::dispatch_gpu(
    GridShape shape,
    std::span<const KernelArg> args,
    void* return_store_out,
    std::string* err) const {
    if (spirv.words.empty()) {
        if (err) *err = "kernel '" + name + (cpu_only ? "' is a @cpu kernel: it has no GPU build" : "' has no compiled SPIR-V binary");
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
        bound_buffers.reserve(params.size() + block_bindings.size() + 1);

        std::vector<KernelArg> dev_args(args.begin(), args.end());
        const size_t non_grid = count_non_grid_params(params);
        const size_t param_blocks = count_param_blocks(params);
        const size_t global_blocks = block_bindings.size() >= param_blocks ? block_bindings.size() - param_blocks : 0;
        const size_t expected_with_blocks = non_grid + block_bindings.size();
        const size_t expected_full = params.size() + global_blocks;
        const bool skip_all_bound = (args.size() == non_grid && block_bindings.empty());
        const bool skip_non_block_bound = (args.size() == expected_with_blocks && expected_with_blocks != expected_full);

        std::vector<size_t> param_block_arg_idx(params.size(), SIZE_MAX);
        size_t arg_idx = 0;
        for (size_t p_i = 0; p_i < params.size(); ++p_i) {
            const auto& p = params[p_i];
            if (is_bound_param(p.role)) {
                if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) {
                    if (!skip_all_bound && arg_idx < args.size()) {
                        param_block_arg_idx[p_i] = arg_idx++;
                    }
                } else if (!skip_all_bound && !skip_non_block_bound) {
                    ++arg_idx;
                }
                continue;
            }
            const size_t i = arg_idx++;
            if (i >= args.size()) break;
            if (p.role != ParamRole::Store && p.role != ParamRole::Stencil) continue;
            const KernelArg& a = args[i];
            if (a.kind() != KernelArg::Kind::Buffer || !a.buffer_ptr()) {
                if (err) *err = "parameter '" + p.name + "' requires a buffer/store argument";
                return false;
            }
            if (!store_covers(p, a, shape, err)) return false;
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

        std::vector<brass::gpu::VulkanDescriptorBuffer> vk_descriptors;
        vk_descriptors.reserve(block_bindings.size());
        for (const auto& binfo : block_bindings) {
            size_t bi = SIZE_MAX;
            if (binfo.mir_param_index < params.size()) {
                bi = param_block_arg_idx[binfo.mir_param_index];
            } else if (arg_idx < args.size()) {
                bi = arg_idx++;
            }
            if (bi >= args.size() || args[bi].kind() != KernelArg::Kind::Buffer || !args[bi].buffer_ptr()) {
                if (err) *err = "binding '" + binfo.name + "' requires a buffer argument";
                return false;
            }
            const KernelArg& a = args[bi];
            size_t bytes = std::max<size_t>({a.buffer_bytes(), size_t(binfo.element_bytes), 16});
            BoundBuffer bb;
            bb.dev_buf = brass::gpu::VulkanBuffer::alloc(bytes);
            bb.host_ptr = a.buffer_ptr();
            bb.byte_size = a.buffer_bytes();
            bb.write_back = (binfo.role == ParamRole::Buffer && can_Write(binfo.access) && !a.is_read_only());
            if (a.buffer_bytes() > 0) {
                bb.dev_buf.upload(a.buffer_ptr(), a.buffer_bytes());
            } else {
                bb.dev_buf.zero();
            }
            vk_descriptors.push_back(brass::gpu::VulkanDescriptorBuffer{
                binfo.descriptor_set,
                binfo.descriptor_binding,
                bb.dev_buf.handle(),
                bytes,
            });
            bound_buffers.push_back(std::move(bb));
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
        disp.descriptors = std::move(vk_descriptors);

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
    ParityReport rep;
    if (!brass::gpu::vulkan_available()) {
        rep.ran_gpu = false;
        rep.ok = false;
        rep.message = "no Vulkan compute device: the GPU build did not run";
        return rep;
    }

    const size_t non_grid = count_non_grid_params(params);
    const size_t param_blocks = count_param_blocks(params);
    const size_t global_blocks = block_bindings.size() >= param_blocks ? block_bindings.size() - param_blocks : 0;
    const size_t expected_with_blocks = non_grid + block_bindings.size();
    const size_t expected_full = params.size() + global_blocks;
    const bool skip_all_bound = (args.size() == non_grid && block_bindings.empty());
    const bool skip_non_block_bound = (args.size() == expected_with_blocks && expected_with_blocks != expected_full);

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

    auto compare_float = [&](double x, double y, uint64_t ulps) {
        ++rep.elements_compared;
        if (std::isnan(x) || std::isnan(y)) {
            if (std::isnan(x) != std::isnan(y)) {
                rep.ok = false;
                rep.max_abs_diff = std::numeric_limits<double>::infinity();
            }
            return;
        }
        if (x == y) return;
        double diff = std::abs(x - y);
        double rdiff = diff / std::max({std::abs(x), std::abs(y), 1e-300});
        rep.max_abs_diff = std::max(rep.max_abs_diff, diff);
        rep.max_rel_diff = std::max(rep.max_rel_diff, rdiff);
        rep.max_ulp_diff = std::max(rep.max_ulp_diff, ulps);
        if (!(diff <= abs_tol || rdiff <= rel_tol)) rep.ok = false;
    };
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

    std::vector<size_t> param_block_arg_idx(params.size(), SIZE_MAX);
    size_t arg_idx = 0;
    for (size_t p_i = 0; p_i < params.size(); ++p_i) {
        const auto& p = params[p_i];
        if (is_bound_param(p.role)) {
            if (p.role == ParamRole::Uniform || p.role == ParamRole::Buffer) {
                if (!skip_all_bound && arg_idx < args.size()) {
                    param_block_arg_idx[p_i] = arg_idx++;
                }
            } else if (!skip_all_bound && !skip_non_block_bound) {
                ++arg_idx;
            }
            continue;
        }
        size_t cur = arg_idx++;
        if (p.role == ParamRole::Store && can_Write(p.access) && cur < cpu_bufs.size() && !cpu_bufs[cur].empty()) {
            compare_bytes(cpu_bufs[cur].data(), gpu_bufs[cur].data(), cpu_bufs[cur].size(), p.type.scalar);
            if (!args[cur].is_read_only() && args[cur].buffer_ptr()) {
                std::memcpy(args[cur].buffer_ptr(), gpu_bufs[cur].data(), gpu_bufs[cur].size());
            }
        }
    }
    for (const auto& binfo : block_bindings) {
        size_t bi = SIZE_MAX;
        if (binfo.mir_param_index < params.size()) {
            bi = param_block_arg_idx[binfo.mir_param_index];
        } else if (arg_idx < args.size()) {
            bi = arg_idx++;
        }
        if (binfo.role == ParamRole::Buffer && can_Write(binfo.access) && bi < cpu_bufs.size() && !cpu_bufs[bi].empty()) {
            compare_bytes(cpu_bufs[bi].data(), gpu_bufs[bi].data(), cpu_bufs[bi].size(), binfo.type.scalar);
            if (!args[bi].is_read_only() && args[bi].buffer_ptr()) {
                std::memcpy(args[bi].buffer_ptr(), gpu_bufs[bi].data(), gpu_bufs[bi].size());
            }
        }
    }

    if (ret_bytes > 0) {
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

    auto checked = check_module(*ast, modules_, res.diagnostics, root_dir, opts.entry_name);
    if (!checked || res.diagnostics.has_errors()) {
        res.ok = false;
        return res;
    }

    res.module_name = checked->module_name;
    res.types = checked->types;
    for (const auto& [key, index] : checked->global_index) {
        const CheckedParam& g = checked->globals[index];
        if (g.role == ParamRole::Shared) continue;
        res.resources.push_back({g.name, key.substr(0, key.size() - g.name.size() - 2), g.descriptor_set,
                                 g.descriptor_binding, g.type, g.image_format});
    }
    std::sort(res.resources.begin(), res.resources.end(), [](const ResourceInfo& a, const ResourceInfo& b) {
        return a.set != b.set ? a.set < b.set : a.binding < b.binding;
    });

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
        // GPU code is f32: a kernel built for the GPU that reaches f64 is an
        // error at the f64. A @cpu kernel has no GPU build and may use it; a
        // plain function compiled as an entry (no @kernel in the module) has
        // only its CPU build when it does.
        bool gpu_build = opts.emit_spirv && !fn->cpu_only;
        if (fn->cpu_only && fn->stage != Stage::Compute) {
            res.diagnostics.error(fn->span, "'" + fn->source_name + "' is a vertex or fragment stage: it cannot be @cpu");
            continue;
        }
        if (gpu_build && fn->uses_f64) {
            if (fn->is_entry_kernel) {
                res.diagnostics.error(fn->f64_site ? *fn->f64_site : fn->span,
                                      "f64 in GPU kernel '" + fn->source_name +
                                          "': GPU code is f32 (compute in f64 on the CPU and pass f32 offsets from an "
                                          "anchor near the work), or mark the kernel @cpu");
                continue;
            }
            gpu_build = false;
        }
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
        ck.block_bindings = lowered.block_bindings;
        ck.uses_barriers_or_shared = lowered.uses_barriers_or_shared;
        ck.cpu_shared_bytes = lowered.cpu_shared_bytes;
        ck.return_type = fn->return_type;
        ck.return_leaf_kinds = checked->types.flatten_scalars(fn->return_type);
        ck.return_flat_leaves = static_cast<uint32_t>(ck.return_leaf_kinds.size());
        ck.return_byte_size = checked->types.byte_size(fn->return_type);
        ck.uses_f64 = fn->uses_f64;
        ck.cpu_only = fn->cpu_only;
        ck.uses_grid_coords = fn->uses_grid_coords;
        ck.max_stencil_radius = fn->max_stencil_radius;
        ck.stencil_halo[0] = fn->stencil_halo[0];
        ck.stencil_halo[1] = fn->stencil_halo[1];
        ck.stencil_halo[2] = fn->stencil_halo[2];

        try {
            if (gpu_build) {
                brass::target::SpirvOptions spv_opts;
                spv_opts.stage = ck.stage;
                spv_opts.local_size_x = ck.local_size[0];
                spv_opts.local_size_y = ck.local_size[1];
                spv_opts.local_size_z = ck.local_size[2];
                spv_opts.local_size_spec_constants = opts.local_size_spec_constants;
                spv_opts.spirv_version = opts.spirv_version;
                spv_opts.precise_float = !fn->fast_math;

                ck.spirv = brass::target::SpirvTarget::compile(*lowered.gpu_function, spv_opts);
                if (opts.emit_spirv_disassembly) {
                    ck.spirv_disassembly = brass::target::SpirvTarget::dump_function(*lowered.gpu_function, spv_opts);
                }

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

            if (opts.emit_ptx && ck.stage == Stage::Compute && !fn->cpu_only && !fn->uses_f64) {
                ck.ptx = brass::target::PtxTarget::emit_function(*lowered.gpu_function);
            }

            if (opts.compile_cpu && lowered.cpu_point_function) {
                auto cpu = std::make_shared<CompiledKernel::CpuBuild>();
                cpu->options.opt_level = opts.opt_level;
                cpu->options.enable_optimizations = (opts.opt_level > 0);
                // Fast math fuses multiply-adds where the host has FMA.
                cpu->options.enable_fma = fn->fast_math && brass::host_has_fma();
                cpu->options.enable_parallel = false;
                cpu->options.enable_superopt = false;
                cpu->options.enable_smt_guard_elim = false;
                cpu->point = lowered.cpu_point_function;
                cpu->grid = lowered.cpu_grid_function;
                cpu->module = std::move(lowered.cpu_module);
                ck.cpu_ = std::move(cpu);
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
