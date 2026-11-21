// Runtime multi-entity telemetry storage and MIR/GPU argument wiring.
#include "brokernel/brokernel.h"
#include "compile_internal.h"
#include "lowering_internal.h"

#include <algorithm>
#include <mutex>

namespace brokernel {

struct TelemetryStorage {
    mutable std::mutex mu;
    float* host_ptr = nullptr;
    uint32_t capacity_floats = 0;
    uint64_t gpu_address = 0;
};

void CompiledKernel::init_telemetry() {
    if (telemetry_slots == 0 || !uses_telemetry) {
        telemetry_storage.reset();
        return;
    }
    telemetry_storage = std::make_shared<TelemetryStorage>();
}

void CompiledKernel::set_telemetry_buffer(float* host_ptr, uint32_t capacity_floats, uint64_t gpu_address) const {
    if (!telemetry_storage) return;
    std::lock_guard<std::mutex> lock(telemetry_storage->mu);
    telemetry_storage->host_ptr = host_ptr;
    telemetry_storage->capacity_floats = capacity_floats;
    telemetry_storage->gpu_address = gpu_address;
}

float* CompiledKernel::telemetry_host_ptr() const {
    if (!telemetry_storage) return nullptr;
    std::lock_guard<std::mutex> lock(telemetry_storage->mu);
    return telemetry_storage->host_ptr;
}

uint32_t CompiledKernel::telemetry_capacity_floats() const {
    if (!telemetry_storage) return 0;
    std::lock_guard<std::mutex> lock(telemetry_storage->mu);
    return telemetry_storage->capacity_floats;
}

uint64_t CompiledKernel::telemetry_gpu_address() const {
    if (!telemetry_storage) return 0;
    std::lock_guard<std::mutex> lock(telemetry_storage->mu);
    return telemetry_storage->gpu_address;
}

namespace detail {

void append_gpu_telemetry_args(
    const CompiledKernel& k,
    GpuLaunch& out,
    std::vector<brass::gpu::VulkanArg>& vk_args) {
    if (k.telemetry_slots == 0 || !k.uses_telemetry) return;
    out.telemetry_arg_index = vk_args.size();
    const uint64_t addr = k.telemetry_gpu_address();
    const uint32_t cap = addr ? std::min(k.telemetry_slots, k.telemetry_capacity_floats()) : 0u;
    vk_args.emplace_back(addr);
    vk_args.emplace_back(static_cast<int32_t>(cap));
}

void setup_gpu_telemetry_bounce(const CompiledKernel& k, GpuLaunch& launch, TelemetryVkBounce& bounce) {
    if (k.telemetry_slots == 0 || !k.uses_telemetry) return;
    if (launch.telemetry_arg_index + 1 >= launch.args.size()) return;
    float* host_ptr = k.telemetry_host_ptr();
    if (!host_ptr || k.telemetry_gpu_address() != 0) return;
    const uint32_t cap = std::min(k.telemetry_slots, k.telemetry_capacity_floats());
    if (cap == 0) return;
    const size_t bytes = static_cast<size_t>(cap) * sizeof(float);
    bounce.dev_buf = brass::gpu::VulkanBuffer::alloc(std::max<size_t>(bytes, 16));
    bounce.dev_buf.upload(host_ptr, bytes);
    bounce.host_ptr = host_ptr;
    bounce.byte_size = bytes;
    launch.args[launch.telemetry_arg_index] = brass::gpu::VulkanArg(bounce.dev_buf.device_address());
    launch.args[launch.telemetry_arg_index + 1] = brass::gpu::VulkanArg(static_cast<int32_t>(cap));
}

void finish_gpu_telemetry_bounce(const TelemetryVkBounce& bounce) {
    if (bounce.host_ptr && bounce.byte_size > 0) {
        bounce.dev_buf.download(bounce.host_ptr, bounce.byte_size);
    }
}

void MirBodyLowerer::lower_telemetry(
    brass::Value* slot_v,
    ScalarKind slot_sk,
    brass::Value* val_v,
    ScalarKind val_sk) {
    if (mod.telemetry_slots == 0) return;

    brass::Value* ptr = is_gpu ? gpu_telemetry_ptr : cpu_telemetry_ptr;
    brass::Value* cap64 = is_gpu
        ? (gpu_telemetry_cap ? b.build_zext_i64(gpu_telemetry_cap) : nullptr)
        : cpu_telemetry_cap;
    if (!ptr || !cap64) return;

    brass::Value* slot_i32 = (slot_sk == ScalarKind::I32 || slot_sk == ScalarKind::U32)
        ? slot_v
        : lower_scalar_cast(slot_v, slot_sk, ScalarKind::I32);
    brass::Value* val_f32 = (val_sk == ScalarKind::F32)
        ? val_v
        : lower_scalar_cast(val_v, val_sk, ScalarKind::F32);

    // Sign-extending to i64 produces a scalar i64 or an Ops index marker in
    // AVX2 vector mode; `Bound{idx64, cap64}` rejects both negative (< 0 signed)
    // and >= cap via unsigned comparison, and vector-mode out-of-bounds/null
    // writes route safely to the stack scratch slot instead of dereferencing ptr.
    brass::Value* idx64 = b.build_sext_i64(slot_i32);
    const Ops::Bound bound{idx64, cap64};
    b.build_store_indexed(brass::Type::f32(), ptr, idx64, 4, 0, val_f32, &bound);
}

} // namespace detail
} // namespace brokernel
