#include "brokernel/brokernel.h"
#include "compile_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace brokernel {

const char* version() { return "0.1.0"; }

namespace {

size_t count_non_grid_params(const std::vector<KernelParamInfo>& params) noexcept {
    size_t c = 0;
    for (const auto& p : params) if (!is_bound_param(p.role)) ++c;
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

struct GpuBoundBuffer {
    brass::gpu::VulkanBuffer dev_buf;
    void* host_ptr = nullptr;
    size_t byte_size = 0;
    bool write_back = false;
};

bool prepare_gpu_args_and_descriptors(
    const std::vector<KernelParamInfo>& params,
    const std::vector<KernelParamInfo>& block_bindings,
    GridShape shape,
    std::span<const KernelArg> args,
    bool check_coverage,
    bool record_writeback,
    bool allow_device_store_args,
    std::vector<KernelArg>& dev_args,
    std::vector<GpuBoundBuffer>& bound_buffers,
    std::vector<brass::gpu::VulkanDescriptorBuffer>& vk_descriptors,
    std::string* err) {
    bound_buffers.reserve(params.size() + block_bindings.size() + 1);
    dev_args.assign(args.begin(), args.end());

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
        if (allow_device_store_args && a.kind() == KernelArg::Kind::Device) continue;
        if (a.kind() != KernelArg::Kind::Buffer || !a.buffer_ptr()) {
            if (err) *err = "parameter '" + p.name + "' requires a buffer/store argument";
            return false;
        }
        if (check_coverage && !store_covers(p, a, shape, err)) return false;
        size_t bytes = std::max<size_t>({a.buffer_bytes(), size_t(p.element_bytes), 16});
        GpuBoundBuffer bb;
        bb.dev_buf = brass::gpu::VulkanBuffer::alloc(bytes);
        bb.host_ptr = a.buffer_ptr();
        bb.byte_size = a.buffer_bytes();
        bb.write_back = record_writeback && (p.role == ParamRole::Store && can_Write(p.access) && !a.is_read_only());
        if (a.buffer_bytes() > 0 && (p.role == ParamRole::Stencil || can_Read(p.access))) {
            bb.dev_buf.upload(a.buffer_ptr(), a.buffer_bytes());
        } else {
            bb.dev_buf.zero();
        }
        bound_buffers.push_back(std::move(bb));
        dev_args[i] = KernelArg::device(bound_buffers.back().dev_buf.device_address(), a.buffer_bytes(), bytes);
    }

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
        GpuBoundBuffer bb;
        bb.dev_buf = brass::gpu::VulkanBuffer::alloc(bytes);
        bb.host_ptr = a.buffer_ptr();
        bb.byte_size = a.buffer_bytes();
        bb.write_back = record_writeback && (binfo.role == ParamRole::Buffer && can_Write(binfo.access) && !a.is_read_only());
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
    return true;
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
    if (probes_compiled && stage == Stage::Compute && !probes.empty()) {
        out.probe_arg_index = vk_args.size();
        vk_args.emplace_back(uint64_t(0));
    }
    detail::append_gpu_runtime_knob_args(*this, vk_args);
    detail::append_gpu_telemetry_args(*this, out, vk_args);

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

        std::vector<GpuBoundBuffer> bound_buffers;
        std::vector<KernelArg> dev_args;
        std::vector<brass::gpu::VulkanDescriptorBuffer> vk_descriptors;
        if (!prepare_gpu_args_and_descriptors(params, block_bindings, shape, args,
                                              /*check_coverage=*/true,
                                              /*record_writeback=*/true,
                                              /*allow_device_store_args=*/false,
                                              dev_args, bound_buffers, vk_descriptors, err)) {
            return false;
        }

        uint64_t return_address = 0;
        if (const size_t ret_total_bytes = gpu_return_bytes(shape)) {
            GpuBoundBuffer ret_bb;
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
        if (probes_compiled && stage == Stage::Compute && !probes.empty() && launch.probe_arg_index < launch.args.size()) {
            const int32_t inactive_target[4] = {-1, -1, -1, 0};
            GpuBoundBuffer pbb;
            pbb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pbb.dev_buf.upload(inactive_target, 16);
            launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(pbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pbb));
        }
        detail::TelemetryVkBounce telem_bounce;
        detail::setup_gpu_telemetry_bounce(*this, launch, telem_bounce);

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
        detail::finish_gpu_telemetry_bounce(telem_bounce);

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

bool CompiledKernel::probe_gpu(
    int32_t gx, int32_t gy, int32_t gz,
    GridShape shape,
    std::span<const KernelArg> args,
    std::vector<ProbeSample>& out,
    std::string* err) const {
    out.clear();
    if (!probes_compiled && !probes.empty() && build_probed_variant && gpu_cache_) {
        std::shared_ptr<const CompiledKernel> pv;
        {
            std::lock_guard<std::mutex> lock(gpu_cache_->mu);
            if (!gpu_cache_->probed_variant) {
                gpu_cache_->probed_variant = build_probed_variant();
            }
            pv = gpu_cache_->probed_variant;
        }
        if (pv) return pv->probe_gpu(gx, gy, gz, shape, args, out, err);
    }
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

        std::vector<GpuBoundBuffer> bound_buffers;
        std::vector<KernelArg> dev_args;
        std::vector<brass::gpu::VulkanDescriptorBuffer> vk_descriptors;
        if (!prepare_gpu_args_and_descriptors(params, block_bindings, shape, args,
                                              /*check_coverage=*/false,
                                              /*record_writeback=*/false,
                                              /*allow_device_store_args=*/true,
                                              dev_args, bound_buffers, vk_descriptors, err)) {
            return false;
        }

        uint64_t return_address = 0;
        if (const size_t ret_bytes = gpu_return_bytes(shape)) {
            GpuBoundBuffer rbb;
            rbb.dev_buf = brass::gpu::VulkanBuffer::alloc(std::max<size_t>(ret_bytes, 16));
            rbb.dev_buf.zero();
            return_address = rbb.dev_buf.device_address();
            bound_buffers.push_back(std::move(rbb));
        }

        uint32_t max_idx = 0;
        for (const auto& p : probes) max_idx = std::max(max_idx, p.index + 1);
        size_t probe_words = 4 + static_cast<size_t>(max_idx) * 5;
        std::vector<uint32_t> host_probe(probe_words, 0);
        host_probe[0] = static_cast<uint32_t>(gx);
        host_probe[1] = static_cast<uint32_t>(gy);
        host_probe[2] = static_cast<uint32_t>(gz);
        auto probe_buf = brass::gpu::VulkanBuffer::alloc(probe_words * sizeof(uint32_t));
        probe_buf.upload(host_probe.data(), probe_words * sizeof(uint32_t));

        GpuLaunch launch;
        if (!gpu_launch(shape, dev_args, return_address, launch, err)) return false;
        if (probes_compiled && stage == Stage::Compute && !probes.empty() && launch.probe_arg_index < launch.args.size()) {
            launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(probe_buf.device_address());
        }

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

        probe_buf.download(host_probe.data(), probe_words * sizeof(uint32_t));
        out.reserve(probes.size());
        for (const auto& p : probes) {
            ProbeSample s;
            s.name = p.name;
            s.type = p.type;
            s.lanes = p.type.is_vector() ? p.type.dimension : 1;
            size_t base = 4 + static_cast<size_t>(p.index) * 5;
            if (base + 5 <= host_probe.size()) {
                float hit_f = 0.0f;
                std::memcpy(&hit_f, &host_probe[base], sizeof(float));
                s.hit = (hit_f != 0.0f);
                for (int i = 0; i < 4; ++i) {
                    std::memcpy(&s.values[i], &host_probe[base + 1 + i], sizeof(float));
                }
            }
            out.push_back(std::move(s));
        }
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("Vulkan probe error: ") + ex.what();
        return false;
    }
}

} // namespace brokernel
