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

    const size_t non_grid = detail::count_non_grid_params(params);
    const size_t param_blocks = detail::count_param_blocks(params);
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

    const size_t non_grid = detail::count_non_grid_params(params);
    const size_t param_blocks = detail::count_param_blocks(params);
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
                case ScalarKind::DF32: {
                    const double d = a.as_f64();
                    const float hi = static_cast<float>(d);
                    const float lo = static_cast<float>(d - static_cast<double>(hi));
                    vk_args.emplace_back(hi);
                    vk_args.emplace_back(lo);
                    break;
                }
                case ScalarKind::U32: vk_args.emplace_back(a.as_u32()); break;
                case ScalarKind::U64: vk_args.emplace_back(a.as_u64()); break;
                case ScalarKind::I64: vk_args.emplace_back(a.as_i64()); break;
                default: vk_args.emplace_back(a.as_i32()); break;
            }
        } else {
            if (a.kind() == KernelArg::Kind::ScalarsF32) {
                for (float v : a.f32_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::ScalarsF64) {
                for (double v : a.f64_values()) {
                    if (p.type.scalar == ScalarKind::DF32) {
                        const float hi = static_cast<float>(v);
                        const float lo = static_cast<float>(v - static_cast<double>(hi));
                        vk_args.emplace_back(hi);
                        vk_args.emplace_back(lo);
                    } else {
                        vk_args.emplace_back(v);
                    }
                }
            } else if (a.kind() == KernelArg::Kind::ScalarsI32) {
                for (int32_t v : a.i32_values()) vk_args.emplace_back(v);
            } else if (a.kind() == KernelArg::Kind::Buffer && a.buffer_ptr()) {
                const auto* raw = static_cast<const uint8_t*>(a.buffer_ptr());
                uint32_t elem_sz = p.type.scalar == ScalarKind::DF32 ? 4u : p.type.scalar_size_bytes();
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
    if (probes_compiled && stage == Stage::Compute && has_unconditional_probes()) {
        out.probe_arg_index = vk_args.size();
        vk_args.emplace_back(uint64_t(0));
    }
    detail::append_gpu_runtime_knob_args(*this, vk_args);
    detail::append_gpu_telemetry_args(*this, out, vk_args);
    if (probes_compiled && stage == Stage::Compute && has_conditional_probes()) {
        out.probe_if_arg_index = vk_args.size();
        vk_args.emplace_back(uint64_t(0));
    }
    if ((checks_compiled || trap_non_finite_compiled) && stage == Stage::Compute) {
        out.check_arg_index = vk_args.size();
        vk_args.emplace_back(uint64_t(0));
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
        if (launch.probe_arg_index < launch.args.size()) {
            const int32_t inactive_target[4] = {-1, -1, -1, 0};
            GpuBoundBuffer pbb;
            pbb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pbb.dev_buf.upload(inactive_target, 16);
            launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(pbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pbb));
        }
        if (launch.probe_if_arg_index < launch.args.size()) {
            GpuBoundBuffer pif_bb;
            pif_bb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pif_bb.dev_buf.zero();
            launch.args[launch.probe_if_arg_index] = brass::gpu::VulkanArg(pif_bb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pif_bb));
        }
        if (launch.check_arg_index < launch.args.size()) {
            GpuBoundBuffer cbb;
            cbb.dev_buf = brass::gpu::VulkanBuffer::alloc(32);
            cbb.dev_buf.zero();
            launch.args[launch.check_arg_index] = brass::gpu::VulkanArg(cbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(cbb));
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
        if (launch.probe_arg_index < launch.args.size()) {
            launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(probe_buf.device_address());
        }
        if (launch.probe_if_arg_index < launch.args.size()) {
            GpuBoundBuffer pif_bb;
            pif_bb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pif_bb.dev_buf.zero();
            launch.args[launch.probe_if_arg_index] = brass::gpu::VulkanArg(pif_bb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pif_bb));
        }
        if (launch.check_arg_index < launch.args.size()) {
            GpuBoundBuffer cbb;
            cbb.dev_buf = brass::gpu::VulkanBuffer::alloc(32);
            cbb.dev_buf.zero();
            launch.args[launch.check_arg_index] = brass::gpu::VulkanArg(cbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(cbb));
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

CheckReport CompiledKernel::check_gpu(
    GridShape shape,
    std::span<const KernelArg> args,
    bool trap_non_finite,
    std::string* err) const {
    if (auto variant = ensure_checked_variant(trap_non_finite)) {
        return variant->check_gpu(shape, args, trap_non_finite, err);
    }
    CheckReport rep;
    if (!checks_compiled && !trap_non_finite_compiled) {
        if (err) *err = "kernel '" + name + "' has no checked variant available";
        rep.ok = false;
        rep.message = err ? *err : "kernel has no checked variant available";
        return rep;
    }
    if (spirv.words.empty()) {
        if (err) *err = "kernel '" + name + (cpu_only ? "' is a @cpu kernel: it has no GPU build" : "' has no compiled SPIR-V binary");
        rep.ok = false;
        rep.message = err ? *err : "no SPIR-V binary";
        return rep;
    }
    if (!brass::gpu::vulkan_available()) {
        if (err) *err = "no Vulkan compute device available";
        rep.ok = false;
        rep.message = "no Vulkan compute device available";
        return rep;
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
                    rep.ok = false;
                    rep.message = err ? *err : "failed to load Vulkan module";
                    return rep;
                }
                gpu_cache_->module = std::make_unique<brass::gpu::VulkanModule>(std::move(loaded));
            }
            vk_module = gpu_cache_->module.get();
        }
        const brass::gpu::VulkanModule& vk = *vk_module;

        std::vector<std::vector<uint8_t>> host_copies(args.size());
        std::vector<KernelArg> copy_args(args.begin(), args.end());
        for (size_t i = 0; i < args.size(); ++i) {
            if (args[i].kind() == KernelArg::Kind::Buffer && args[i].buffer_ptr() && args[i].buffer_bytes() > 0) {
                const auto* src = static_cast<const uint8_t*>(args[i].buffer_ptr());
                host_copies[i].assign(src, src + args[i].buffer_bytes());
                copy_args[i] = KernelArg::buffer(host_copies[i].data(), host_copies[i].size());
            }
        }

        std::vector<GpuBoundBuffer> bound_buffers;
        std::vector<KernelArg> dev_args;
        std::vector<brass::gpu::VulkanDescriptorBuffer> vk_descriptors;
        if (!prepare_gpu_args_and_descriptors(params, block_bindings, shape, copy_args,
                                              /*check_coverage=*/true,
                                              /*record_writeback=*/false,
                                              /*allow_device_store_args=*/true,
                                              dev_args, bound_buffers, vk_descriptors, err)) {
            rep.ok = false;
            rep.message = err ? *err : "invalid check arguments";
            return rep;
        }

        uint64_t return_address = 0;
        if (const size_t ret_bytes = gpu_return_bytes(shape)) {
            GpuBoundBuffer rbb;
            rbb.dev_buf = brass::gpu::VulkanBuffer::alloc(std::max<size_t>(ret_bytes, 16));
            rbb.dev_buf.zero();
            return_address = rbb.dev_buf.device_address();
            bound_buffers.push_back(std::move(rbb));
        }

        GpuLaunch launch;
        if (!gpu_launch(shape, dev_args, return_address, launch, err)) {
            rep.ok = false;
            rep.message = err ? *err : "gpu_launch failed";
            return rep;
        }

        auto check_buf = brass::gpu::VulkanBuffer::alloc(32);
        check_buf.zero();
        if (launch.check_arg_index < launch.args.size()) {
            launch.args[launch.check_arg_index] = brass::gpu::VulkanArg(check_buf.device_address());
        }
        if (launch.probe_arg_index < launch.args.size()) {
            const int32_t inactive_target[4] = {-1, -1, -1, 0};
            GpuBoundBuffer pbb;
            pbb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pbb.dev_buf.upload(inactive_target, 16);
            launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(pbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pbb));
        }
        if (launch.probe_if_arg_index < launch.args.size()) {
            GpuBoundBuffer pif_bb;
            pif_bb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pif_bb.dev_buf.zero();
            launch.args[launch.probe_if_arg_index] = brass::gpu::VulkanArg(pif_bb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pif_bb));
        }

        brass::gpu::VulkanDispatch disp;
        for (int i = 0; i < 3; ++i) {
            disp.grid[i] = launch.groups[i];
            disp.block[i] = launch.local_size[i];
        }
        disp.descriptors = std::move(vk_descriptors);
        if (!vk.launch(disp, launch.args, &vk_err)) {
            if (err) *err = vk_err;
            rep.ok = false;
            rep.message = vk_err;
            return rep;
        }

        uint32_t check_words[8] = {};
        check_buf.download(check_words, sizeof(check_words));
        return detail::decode_check_report(*this, check_words);
    } catch (const std::exception& ex) {
        if (err) *err = std::string("Vulkan check error: ") + ex.what();
        rep.ok = false;
        rep.message = err ? *err : "Vulkan check error";
        return rep;
    }
}

bool CompiledKernel::probe_if_gpu(
    GridShape shape,
    std::span<const KernelArg> args,
    std::vector<ConditionalProbeSample>& out,
    uint32_t max_samples,
    std::string* err) const {
    out.clear();
    if (auto variant = ensure_probed_variant()) {
        return variant->probe_if_gpu(shape, args, out, max_samples, err);
    }
    if (!has_conditional_probes()) return true;
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

        const size_t buf_words = 2 + static_cast<size_t>(max_samples) * 9;
        std::vector<uint32_t> host_buf(buf_words, 0);
        host_buf[0] = 0;
        host_buf[1] = max_samples;
        auto pif_buf = brass::gpu::VulkanBuffer::alloc(std::max<size_t>(buf_words * sizeof(uint32_t), 16));
        pif_buf.upload(host_buf.data(), buf_words * sizeof(uint32_t));

        GpuLaunch launch;
        if (!gpu_launch(shape, dev_args, return_address, launch, err)) return false;
        if (launch.probe_if_arg_index < launch.args.size()) {
            launch.args[launch.probe_if_arg_index] = brass::gpu::VulkanArg(pif_buf.device_address());
        }
        if (launch.probe_arg_index < launch.args.size()) {
            const int32_t inactive_target[4] = {-1, -1, -1, 0};
            GpuBoundBuffer pbb;
            pbb.dev_buf = brass::gpu::VulkanBuffer::alloc(16);
            pbb.dev_buf.upload(inactive_target, 16);
            launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(pbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(pbb));
        }
        if (launch.check_arg_index < launch.args.size()) {
            GpuBoundBuffer cbb;
            cbb.dev_buf = brass::gpu::VulkanBuffer::alloc(32);
            cbb.dev_buf.zero();
            launch.args[launch.check_arg_index] = brass::gpu::VulkanArg(cbb.dev_buf.device_address());
            bound_buffers.push_back(std::move(cbb));
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

        pif_buf.download(host_buf.data(), buf_words * sizeof(uint32_t));
        detail::decode_probe_if_samples(*this, host_buf, max_samples, out);
        return true;
    } catch (const std::exception& ex) {
        if (err) *err = std::string("Vulkan probe_if error: ") + ex.what();
        return false;
    }
}

} // namespace brokernel
