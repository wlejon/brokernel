// Host runtime diagnostics: check_cpu, probe_if_cpu, variant resolution, and report decoding.
#include "brokernel/brokernel.h"
#include "compile_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace brokernel {

namespace detail {

namespace {

thread_local uint32_t* t_active_cpu_check_buf = nullptr;
thread_local uint32_t* t_active_cpu_probe_if_buf = nullptr;

std::vector<KernelArg> clone_buffer_args(
    std::span<const KernelArg> args,
    std::vector<std::vector<uint8_t>>& backing) {
    backing.resize(args.size());
    std::vector<KernelArg> out(args.begin(), args.end());
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i].kind() == KernelArg::Kind::Buffer && args[i].buffer_ptr() && args[i].buffer_bytes() > 0) {
            const auto* src = static_cast<const uint8_t*>(args[i].buffer_ptr());
            backing[i].assign(src, src + args[i].buffer_bytes());
            out[i] = KernelArg::buffer(backing[i].data(), backing[i].size());
        }
    }
    return out;
}

} // namespace

uint32_t* active_cpu_check_buf() { return t_active_cpu_check_buf; }
uint32_t* active_cpu_probe_if_buf() { return t_active_cpu_probe_if_buf; }

ScopedCpuCheckBuf::ScopedCpuCheckBuf(uint32_t* buf) : prev(t_active_cpu_check_buf) {
    t_active_cpu_check_buf = buf;
}
ScopedCpuCheckBuf::~ScopedCpuCheckBuf() {
    t_active_cpu_check_buf = prev;
}

ScopedCpuProbeIfBuf::ScopedCpuProbeIfBuf(uint32_t* buf) : prev(t_active_cpu_probe_if_buf) {
    t_active_cpu_probe_if_buf = buf;
}
ScopedCpuProbeIfBuf::~ScopedCpuProbeIfBuf() {
    t_active_cpu_probe_if_buf = prev;
}

CheckReport decode_check_report(const CompiledKernel& k, const uint32_t words[8]) {
    CheckReport rep;
    if (words[0] == 0) {
        rep.ok = true;
        return rep;
    }
    rep.ok = false;
    rep.kind = words[1];
    rep.coord[0] = static_cast<int32_t>(words[2]);
    rep.coord[1] = static_cast<int32_t>(words[3]);
    rep.coord[2] = static_cast<int32_t>(words[4]);
    const uint32_t lid = words[5];
    if (lid < k.check_labels.size()) rep.label = k.check_labels[lid];

    if (rep.kind == 2) {
        if (words[7] != 0) {
            const uint64_t u64 = (static_cast<uint64_t>(words[7]) << 32) | static_cast<uint64_t>(words[6]);
            std::memcpy(&rep.value, &u64, sizeof(double));
        } else {
            float f32 = 0.0f;
            std::memcpy(&f32, &words[6], sizeof(float));
            rep.value = static_cast<double>(f32);
        }
    }

    std::ostringstream oss;
    const char* coord_open = "(";
    if (rep.kind == 1) {
        oss << "check '" << rep.label << "' failed at " << coord_open
            << rep.coord[0] << ", " << rep.coord[1] << ", " << rep.coord[2] << ")";
    } else if (rep.kind == 2) {
        const char* kind_str = std::isnan(rep.value) ? "nan" : (rep.value < 0.0 ? "-inf" : "inf");
        if (rep.label == "<return>" || rep.label.empty()) {
            oss << "non-finite return value (" << kind_str << ") at ("
                << rep.coord[0] << ", " << rep.coord[1] << ", " << rep.coord[2] << ")";
        } else {
            oss << "non-finite value (" << kind_str << ") written to '" << rep.label << "' at ("
                << rep.coord[0] << ", " << rep.coord[1] << ", " << rep.coord[2] << ")";
        }
    } else {
        oss << "runtime check violation (kind=" << rep.kind << ") at ("
            << rep.coord[0] << ", " << rep.coord[1] << ", " << rep.coord[2] << ")";
    }
    rep.message = oss.str();
    return rep;
}

void decode_probe_if_samples(
    const CompiledKernel& k,
    const std::vector<uint32_t>& words,
    uint32_t max_samples,
    std::vector<ConditionalProbeSample>& out) {
    out.clear();
    if (words.size() < 2) return;
    std::unordered_map<uint32_t, const ProbeInfo*> by_id;
    for (const auto& p : k.probes) {
        by_id[p.index] = &p;
    }
    const uint32_t raw_count = words[0];
    const uint32_t cap = std::min<uint32_t>(words[1], max_samples);
    const uint32_t count = std::min(raw_count, cap);
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const size_t base = 2 + static_cast<size_t>(i) * 9;
        if (base + 9 > words.size()) break;
        ConditionalProbeSample s;
        s.coord[0] = static_cast<int32_t>(words[base + 0]);
        s.coord[1] = static_cast<int32_t>(words[base + 1]);
        s.coord[2] = static_cast<int32_t>(words[base + 2]);
        const uint32_t probe_id = words[base + 3];
        const uint32_t lanes = std::clamp<uint32_t>(words[base + 4], 1u, 4u);
        const auto pit = by_id.find(probe_id);
        const ProbeInfo* pi = (pit != by_id.end()) ? pit->second : nullptr;
        if (pi) s.label = pi->name;
        const ScalarKind sk = pi ? pi->type.scalar : ScalarKind::F32;
        s.values.reserve(lanes);
        for (uint32_t l = 0; l < lanes; ++l) {
            const uint32_t w = words[base + 5 + l];
            if (sk == ScalarKind::F32 || sk == ScalarKind::F64) {
                float f = 0.0f;
                std::memcpy(&f, &w, sizeof(float));
                s.values.push_back(static_cast<double>(f));
            } else if (sk == ScalarKind::I32) {
                int32_t iv = 0;
                std::memcpy(&iv, &w, sizeof(int32_t));
                s.values.push_back(static_cast<double>(iv));
            } else if (sk == ScalarKind::Bool) {
                s.values.push_back(w != 0 ? 1.0 : 0.0);
            } else {
                s.values.push_back(static_cast<double>(w));
            }
        }
        out.push_back(std::move(s));
    }
}

} // namespace detail

CheckReport CompiledKernel::check_cpu(
    GridShape shape,
    std::span<const KernelArg> args,
    bool trap_non_finite,
    std::string* err) const {
    if (auto variant = ensure_checked_variant(trap_non_finite)) {
        return variant->check_cpu(shape, args, trap_non_finite, err);
    }
    CheckReport rep;
    if (!checks_compiled && !trap_non_finite_compiled) {
        if (err) *err = "kernel '" + name + "' has no checked variant available";
        rep.ok = false;
        rep.message = err ? *err : "kernel has no checked variant available";
        return rep;
    }

    std::vector<std::vector<uint8_t>> backing;
    std::vector<KernelArg> copy_args = detail::clone_buffer_args(args, backing);

    uint32_t check_buf[8] = {};
    detail::ScopedCpuCheckBuf guard(check_buf);
    if (!dispatch_cpu(shape, copy_args, nullptr, err)) {
        rep.ok = false;
        rep.message = err ? *err : "CPU check dispatch failed";
        return rep;
    }
    return detail::decode_check_report(*this, check_buf);
}

bool CompiledKernel::probe_if_cpu(
    GridShape shape,
    std::span<const KernelArg> args,
    std::vector<ConditionalProbeSample>& out,
    uint32_t max_samples,
    std::string* err) const {
    out.clear();
    if (auto variant = ensure_probed_variant()) {
        return variant->probe_if_cpu(shape, args, out, max_samples, err);
    }
    if (!has_conditional_probes()) return true;

    std::vector<uint32_t> buf(2 + static_cast<size_t>(max_samples) * 9, 0);
    buf[0] = 0;
    buf[1] = max_samples;
    detail::ScopedCpuProbeIfBuf guard(buf.data());
    if (!dispatch_cpu(shape, args, nullptr, err)) return false;
    detail::decode_probe_if_samples(*this, buf, max_samples, out);
    return true;
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

    const size_t non_grid = detail::count_non_grid_params(params);
    const size_t param_blocks = detail::count_param_blocks(params);
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

    const size_t w = std::max<size_t>(1, shape.width);
    const size_t h = std::max<size_t>(1, shape.height);
    const size_t cells = (grid_dim > 0) ? shape.total_points() : 1;
    bool have_any = false;
    bool have_fail = false;
    double worst_fail_diff = -1.0;

    auto set_worst = [&](int32_t gx, int32_t gy, int32_t gz, double x, double y) {
        rep.worst_coord[0] = gx;
        rep.worst_coord[1] = gy;
        rep.worst_coord[2] = gz;
        rep.cpu_val = x;
        rep.gpu_val = y;
    };

    auto record_diff = [&](int32_t gx, int32_t gy, int32_t gz, double x, double y, double diff, bool failed) {
        if (failed) {
            if (!have_fail || (std::isinf(diff) && !std::isinf(worst_fail_diff)) || diff > worst_fail_diff) {
                have_fail = true;
                worst_fail_diff = diff;
                set_worst(gx, gy, gz, x, y);
            }
        } else if (!have_fail) {
            if (!have_any || diff > rep.max_abs_diff) {
                have_any = true;
                set_worst(gx, gy, gz, x, y);
            }
        }
    };

    auto coord_for = [&](size_t lin, bool is_grid) {
        struct Coord { int32_t x, y, z; };
        if (is_grid && grid_dim > 0 && lin < cells) {
            return Coord{
                static_cast<int32_t>(lin % w),
                static_cast<int32_t>((lin / w) % h),
                static_cast<int32_t>(lin / (w * h))
            };
        }
        return Coord{static_cast<int32_t>(lin), 0, 0};
    };

    auto compare_float = [&](double x, double y, uint64_t ulps, int32_t gx, int32_t gy, int32_t gz) {
        ++rep.elements_compared;
        if (std::isnan(x) || std::isnan(y)) {
            if (std::isnan(x) != std::isnan(y)) {
                rep.ok = false;
                record_diff(gx, gy, gz, x, y, std::numeric_limits<double>::infinity(), true);
                rep.max_abs_diff = std::numeric_limits<double>::infinity();
            }
            return;
        }
        if (x == y) return;
        double diff = std::abs(x - y);
        double rdiff = (std::isinf(x) || std::isinf(y))
                           ? 1.0
                           : diff / std::max({std::abs(x), std::abs(y), 1e-300});
        const bool failed = !(diff <= abs_tol || rdiff <= rel_tol);
        if (failed) rep.ok = false;
        record_diff(gx, gy, gz, x, y, diff, failed);
        rep.max_abs_diff = std::max(rep.max_abs_diff, diff);
        rep.max_rel_diff = std::max(rep.max_rel_diff, rdiff);
        rep.max_ulp_diff = std::max(rep.max_ulp_diff, ulps);
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
    auto compare_int = [&](int64_t x, int64_t y, int32_t gx, int32_t gy, int32_t gz) {
        ++rep.elements_compared;
        if (x == y) return;
        rep.ok = false;
        const uint64_t idiff = x > y ? static_cast<uint64_t>(x) - static_cast<uint64_t>(y)
                                     : static_cast<uint64_t>(y) - static_cast<uint64_t>(x);
        const double diff = static_cast<double>(idiff);
        record_diff(gx, gy, gz, static_cast<double>(x), static_cast<double>(y), diff, true);
        rep.max_abs_diff = std::max(rep.max_abs_diff, diff);
    };
    auto compare_uint = [&](uint64_t x, uint64_t y, int32_t gx, int32_t gy, int32_t gz) {
        ++rep.elements_compared;
        if (x == y) return;
        rep.ok = false;
        const uint64_t idiff = x > y ? (x - y) : (y - x);
        const double diff = static_cast<double>(idiff);
        record_diff(gx, gy, gz, static_cast<double>(x), static_cast<double>(y), diff, true);
        rep.max_abs_diff = std::max(rep.max_abs_diff, diff);
    };
    auto compare_bytes = [&](const uint8_t* a, const uint8_t* b, size_t nbytes, ScalarKind sk,
                             size_t cell_stride, bool is_grid, size_t fixed_lin = SIZE_MAX) {
        auto step_coord = [&](size_t off, size_t elem_sz) {
            if (fixed_lin != SIZE_MAX) return coord_for(fixed_lin, is_grid);
            const size_t stride = cell_stride > 0 ? cell_stride : elem_sz;
            return coord_for(off / stride, is_grid);
        };
        if (sk == ScalarKind::F64) {
            for (size_t off = 0; off + 8 <= nbytes; off += 8) {
                double x, y;
                std::memcpy(&x, a + off, 8);
                std::memcpy(&y, b + off, 8);
                auto c = step_coord(off, 8);
                compare_float(x, y, std::isfinite(x) && std::isfinite(y) ? ulp_distance(x, y) : 0, c.x, c.y, c.z);
            }
        } else if (sk == ScalarKind::F32) {
            for (size_t off = 0; off + 4 <= nbytes; off += 4) {
                float x, y;
                std::memcpy(&x, a + off, 4);
                std::memcpy(&y, b + off, 4);
                auto c = step_coord(off, 4);
                compare_float(x, y, std::isfinite(x) && std::isfinite(y) ? ulp_distance(x, y) : 0, c.x, c.y, c.z);
            }
        } else if (sk == ScalarKind::U64) {
            for (size_t off = 0; off + 8 <= nbytes; off += 8) {
                uint64_t x, y;
                std::memcpy(&x, a + off, 8);
                std::memcpy(&y, b + off, 8);
                auto c = step_coord(off, 8);
                compare_uint(x, y, c.x, c.y, c.z);
            }
        } else if (sk == ScalarKind::I64) {
            for (size_t off = 0; off + 8 <= nbytes; off += 8) {
                int64_t x, y;
                std::memcpy(&x, a + off, 8);
                std::memcpy(&y, b + off, 8);
                auto c = step_coord(off, 8);
                compare_int(x, y, c.x, c.y, c.z);
            }
        } else if (sk == ScalarKind::U32) {
            for (size_t off = 0; off + 4 <= nbytes; off += 4) {
                uint32_t x, y;
                std::memcpy(&x, a + off, 4);
                std::memcpy(&y, b + off, 4);
                auto c = step_coord(off, 4);
                compare_uint(x, y, c.x, c.y, c.z);
            }
        } else {
            for (size_t off = 0; off + 4 <= nbytes; off += 4) {
                int32_t x, y;
                std::memcpy(&x, a + off, 4);
                std::memcpy(&y, b + off, 4);
                auto c = step_coord(off, 4);
                compare_int(x, y, c.x, c.y, c.z);
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
            const size_t scalar_sz = (p.type.scalar == ScalarKind::F64 || p.type.scalar == ScalarKind::I64 ||
                                      p.type.scalar == ScalarKind::U64) ? 8 : 4;
            const size_t cell_stride = p.element_bytes ? p.element_bytes : scalar_sz;
            const bool is_grid = (grid_dim > 0) && (p.grid_cover_dim > 0 || cpu_bufs[cur].size() == cells * cell_stride);
            compare_bytes(cpu_bufs[cur].data(), gpu_bufs[cur].data(), cpu_bufs[cur].size(),
                          p.type.scalar, cell_stride, is_grid);
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
            const size_t scalar_sz = (binfo.type.scalar == ScalarKind::F64 || binfo.type.scalar == ScalarKind::I64 ||
                                      binfo.type.scalar == ScalarKind::U64) ? 8 : 4;
            const size_t cell_stride = binfo.element_bytes ? binfo.element_bytes : scalar_sz;
            const bool is_grid = (grid_dim > 0) && (cpu_bufs[bi].size() == cells * cell_stride);
            compare_bytes(cpu_bufs[bi].data(), gpu_bufs[bi].data(), cpu_bufs[bi].size(),
                          binfo.type.scalar, cell_stride, is_grid);
            if (!args[bi].is_read_only() && args[bi].buffer_ptr()) {
                std::memcpy(args[bi].buffer_ptr(), gpu_bufs[bi].data(), gpu_bufs[bi].size());
            }
        }
    }

    if (ret_bytes > 0) {
        const size_t stride = std::max<size_t>(return_byte_size, 4);
        for (size_t cell = 0, base = 0; base + stride <= ret_bytes; ++cell, base += stride) {
            size_t off = 0;
            for (ScalarKind sk : return_leaf_kinds) {
                size_t sz = (sk == ScalarKind::F64 || sk == ScalarKind::I64 || sk == ScalarKind::U64) ? 8 : 4;
                if (off + sz > stride) break;
                compare_bytes(cpu_ret.data() + base + off, gpu_ret.data() + base + off, sz, sk,
                              stride, grid_dim > 0, cell);
                off += sz;
            }
        }
    }

    if (!rep.ok) {
        std::ostringstream oss;
        oss << "CPU/GPU mismatch at (" << rep.worst_coord[0] << ", " << rep.worst_coord[1] << ", "
            << rep.worst_coord[2] << "): cpu=" << rep.cpu_val << ", gpu=" << rep.gpu_val
            << ", max_abs_diff=" << rep.max_abs_diff
            << ", max_rel_diff=" << rep.max_rel_diff;
        rep.message = oss.str();
    }
    return rep;
}

} // namespace brokernel
