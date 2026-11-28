#include "api_internal.h"
#include "embed/embed_native.h"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace brokernel::api {

namespace {

struct BoundSlot {
    bool pinned = false;
    ev::Persistent value;
};

struct BoundClosure {
    bool is_fn = false;
    ev::Persistent target;
    std::vector<BoundSlot> slots;
    size_t placeholder_count = 0;

    // Cached fast-eval state (re-initialized when target kernel changes).
    const CompiledKernel* fast_kernel = nullptr;
    std::shared_ptr<FastEvalPlan> fast_plan;
    bool fast_checked = false;
    bool fast_eligible = false;
    uint64_t pinned_slots[16]{};
    bool has_cached_slot[16]{};
    std::shared_ptr<GpuBuffer> pinned_writable_bufs[16];
};

std::mutex g_fieldExportMu;
std::vector<brass::codegen::KernelFunction> g_fieldExportTrampolines;

void refreshFastBoundPlan(BoundClosure& closure, const CompiledKernel* k) {
    if (closure.fast_kernel == k && closure.fast_checked) return;
    closure.fast_kernel = k;
    closure.fast_checked = true;
    closure.fast_eligible = false;
    closure.fast_plan.reset();
    for (size_t i = 0; i < 16; ++i) {
        closure.has_cached_slot[i] = false;
        closure.pinned_writable_bufs[i].reset();
    }
    if (!k || k->uses_grid_coords || !k->cpu_built()) return;
    closure.fast_plan = buildFastEvalPlan(*k);
    if (!closure.fast_plan || !fastPlanIsEligible(*closure.fast_plan)) return;

    const size_t nparams = fastPlanParamCount(*closure.fast_plan);
    if (closure.slots.size() > nparams) return;

    for (size_t i = 0; i < closure.slots.size(); ++i) {
        if (!closure.slots[i].pinned) continue;
        std::shared_ptr<GpuBuffer> wbuf;
        bool isPinnedBuf = false;
        if (!packFastArgIntoSlots(*closure.fast_plan, i, closure.slots[i].value.get(),
                                  closure.pinned_slots, &wbuf, &isPinnedBuf)) {
            return;
        }
        // Cache pinned scalar slots and pinned KernelBuffer (external heap storage);
        // TypedArray pointers are re-queried on each invocation in case of backing shifts.
        if (isFastScalarParam(*closure.fast_plan, i) || isPinnedBuf) {
            closure.has_cached_slot[i] = true;
            closure.pinned_writable_bufs[i] = std::move(wbuf);
        }
    }
    closure.fast_eligible = true;
}

bool evalFastBound(BoundClosure& closure, const CompiledKernel* k, std::span<const Value> callArgs, Value& out) {
    if (!closure.fast_eligible || !closure.fast_plan || closure.fast_kernel != k) {
        refreshFastBoundPlan(closure, k);
        if (!closure.fast_eligible) return false;
    }
    const FastEvalPlan& plan = *closure.fast_plan;
    const size_t nparams = fastPlanParamCount(plan);
    const size_t trailing = nparams > closure.slots.size() ? nparams - closure.slots.size() : 0;
    if (callArgs.size() != closure.placeholder_count + trailing) return false;

    uint64_t slots[16]{};
    std::shared_ptr<GpuBuffer> dirtyBufs[16];
    std::memcpy(slots, closure.pinned_slots, sizeof(slots));

    size_t callIdx = 0;
    for (size_t i = 0; i < closure.slots.size(); ++i) {
        if (!closure.slots[i].pinned) {
            if (!packFastArgIntoSlots(plan, i, callArgs[callIdx++], slots, &dirtyBufs[i])) {
                return false;
            }
        } else if (!closure.has_cached_slot[i]) {
            if (!packFastArgIntoSlots(plan, i, closure.slots[i].value.get(), slots, &dirtyBufs[i])) {
                return false;
            }
        } else if (closure.pinned_writable_bufs[i]) {
            dirtyBufs[i] = closure.pinned_writable_bufs[i];
        }
    }
    for (size_t i = closure.slots.size(); i < nparams; ++i) {
        if (!packFastArgIntoSlots(plan, i, callArgs[callIdx++], slots, &dirtyBufs[i])) {
            return false;
        }
    }

    out = evalFastPlanSlots(plan, *k, slots);
    for (size_t i = 0; i < nparams; ++i) {
        if (dirtyBufs[i]) dirtyBufs[i]->markHostWritten();
    }
    return true;
}

Value invokeBound(BoundClosure& closure, std::span<const Value> callArgs) {
    if (closure.is_fn) {
        auto sp = unwrapFnPointKernelShared(closure.target.get());
        if (!sp) return ev::throwTypeError("KernelFn.bind: target KernelFn is invalid");
        if (!sp->uses_grid_coords && !sp->cpu_built()) ensureCpuBuilt(*sp);
        Value fastOut;
        if (evalFastBound(closure, sp.get(), callArgs, fastOut)) {
            return fastOut;
        }
        std::vector<Value> merged;
        merged.reserve(closure.slots.size() + callArgs.size());
        size_t ci = 0;
        for (const auto& s : closure.slots) {
            if (s.pinned) {
                merged.push_back(s.value.get());
            } else {
                if (ci >= callArgs.size()) {
                    return ev::throwRangeError("KernelFn.bind: missing call argument for unbound slot");
                }
                merged.push_back(callArgs[ci++]);
            }
        }
        while (ci < callArgs.size()) merged.push_back(callArgs[ci++]);
        return evalFnWithArgs(closure.target.get(), merged);
    }

    KernelHandle* kh = kernelHandle(closure.target.get());
    if (!kh || !kh->kernel) return ev::throwTypeError("Kernel.bind: target Kernel is invalid");
    const CompiledKernel* k = kh->kernel.get();
    if (k->grid_dim == 0) {
        if (!k->cpu_built()) ensureCpuBuilt(*k);
        Value fastOut;
        if (evalFastBound(closure, k, callArgs, fastOut)) {
            return fastOut;
        }
    }

    std::vector<Value> merged;
    merged.reserve(closure.slots.size() + callArgs.size());
    size_t ci = 0;
    for (const auto& s : closure.slots) {
        if (s.pinned) {
            merged.push_back(s.value.get());
        } else {
            if (ci >= callArgs.size()) {
                return ev::throwRangeError("Kernel.bind: missing call argument for unbound slot");
            }
            merged.push_back(callArgs[ci++]);
        }
    }
    while (ci < callArgs.size()) merged.push_back(callArgs[ci++]);
    if (k->grid_dim != 0) return runResolvedKernel(*kh, merged);
    return evalResolvedKernel(*kh, merged, "Kernel.bind");
}

Value makeBoundFunction(bool isFn, Value target, std::span<const Value> boundArgs) {
    if (isFn) {
        if (!unwrapFnPointKernelShared(target)) {
            return ev::throwTypeError("KernelFn.bind: invalid receiver");
        }
    } else {
        if (!kernelHandle(target)) {
            return ev::throwTypeError("Kernel.bind: invalid receiver");
        }
    }

    auto closure = std::make_shared<BoundClosure>();
    closure->is_fn = isFn;
    closure->target.set(target);
    closure->slots.resize(boundArgs.size());
    for (size_t i = 0; i < boundArgs.size(); ++i) {
        if (ev::isUndefined(boundArgs[i])) {
            closure->slots[i].pinned = false;
            ++closure->placeholder_count;
        } else {
            closure->slots[i].pinned = true;
            closure->slots[i].value.set(boundArgs[i]);
        }
    }

    uint32_t arity = static_cast<uint32_t>(closure->placeholder_count);
    return ev::makeFunction(
        [closure](Value, std::span<const Value> callArgs) -> Value {
            return invokeBound(*closure, callArgs);
        },
        arity,
        isFn ? "boundKernelFn" : "boundKernel");
}

Value js_structViewExportFieldNative(Value self, std::span<const Value> args) {
    if (args.size() < 2 || !ev::isString(args[0]) || !ev::isString(args[1])) {
        return ev::throwTypeError(
            "StructView.exportFieldNative: requires (fieldName, jsPath[, arrayType])");
    }
    const std::string fieldName = ev::toUtf8(args[0]);
    const std::string jsPath = ev::toUtf8(args[1]);
    std::string arrayType = "u8[]";
    if (args.size() >= 3 && !ev::isUndefined(args[2])) {
        if (!ev::isString(args[2])) {
            return ev::throwTypeError("StructView.exportFieldNative: arrayType must be 'u8[]' or 'f32[]'");
        }
        arrayType = ev::toUtf8(args[2]);
    }

    uint32_t byteStride = 0;
    uint32_t byteOffset = 0;
    ScalarKind sk = ScalarKind::Void;
    std::string err;
    if (!structViewFieldMeta(self, fieldName, byteStride, byteOffset, sk, &err)) {
        return ev::throwError("StructView.exportFieldNative: " + err);
    }

    uint32_t unitBytes = 1;
    if (arrayType == "u8[]") {
        unitBytes = 1;
    } else if (arrayType == "f32[]") {
        if (byteStride % 4 != 0 || byteOffset % 4 != 0) {
            return ev::throwError(
                "StructView.exportFieldNative: 'f32[]' requires 4-byte aligned stride and field offset");
        }
        unitBytes = 4;
    } else {
        return ev::throwTypeError(
            "StructView.exportFieldNative: arrayType must be 'u8[]' or 'f32[]'");
    }

    const char* retName = nullptr;
    brass::Type mirRet = brass::Type::void_type();
    uint32_t fieldBytes = 0;
    switch (sk) {
        case ScalarKind::F32:  retName = "f64"; mirRet = brass::Type::f64(); fieldBytes = 4; break;
        case ScalarKind::F64:  retName = "f64"; mirRet = brass::Type::f64(); fieldBytes = 8; break;
        case ScalarKind::DF32: retName = "f64"; mirRet = brass::Type::f64(); fieldBytes = 8; break;
        case ScalarKind::I32:
        case ScalarKind::U32:
        case ScalarKind::Bool: retName = "i32"; mirRet = brass::Type::i32(); fieldBytes = 4; break;
        default:
            return ev::throwError(
                "StructView.exportFieldNative: field '" + fieldName + "' must be a scalar or df32");
    }

    auto mirMod = std::make_shared<brass::Module>("brokernel_struct_field_" + fieldName);
    const std::string entryName = "__bk_field_" + fieldName;
    brass::Function* fn = mirMod->create_function(
        entryName, mirRet, {brass::Type::i64(), brass::Type::i32(), brass::Type::i32()});
    brass::Builder b(*mirMod);
    b.set_function(fn);

    brass::BasicBlock* entry = b.append_block("entry");
    brass::BasicBlock* okBb = b.append_block("ok");
    brass::BasicBlock* oobBb = b.append_block("oob");

    brass::Value* ptr = b.add_block_param(entry, brass::Type::i64());
    brass::Value* taLen = b.add_block_param(entry, brass::Type::i32());
    brass::Value* idx = b.add_block_param(entry, brass::Type::i32());

    b.position_at_end(entry);
    brass::Value* nonNeg = b.build_sge(idx, b.build_iconst_i32(0));
    brass::Value* idx64 = b.build_zext_i64(idx);
    brass::Value* totalBytes64 = (unitBytes == 1)
                                     ? b.build_zext_i64(taLen)
                                     : b.build_mul(b.build_zext_i64(taLen), b.build_iconst_i64(unitBytes));
    brass::Value* baseOff = b.build_mul(idx64, b.build_iconst_i64(byteStride));
    brass::Value* fieldOff = b.build_add(baseOff, b.build_iconst_i64(byteOffset));
    brass::Value* endOff = b.build_add(fieldOff, b.build_iconst_i64(fieldBytes));
    brass::Value* inBounds = b.build_and(nonNeg, b.build_ule(endOff, totalBytes64));
    b.build_br_if(inBounds, okBb, oobBb);

    b.position_at_end(okBb);
    brass::Value* addr = b.build_add(ptr, fieldOff);
    brass::Value* val = nullptr;
    switch (sk) {
        case ScalarKind::F32:
            val = b.build_fpext_f64_f32(b.build_load(brass::Type::f32(), addr, 0));
            break;
        case ScalarKind::F64:
            val = b.build_load(brass::Type::f64(), addr, 0);
            break;
        case ScalarKind::DF32: {
            brass::Value* hi = b.build_load(brass::Type::f32(), addr, 0);
            brass::Value* lo = b.build_load(brass::Type::f32(), addr, 4);
            val = b.build_add(b.build_fpext_f64_f32(hi), b.build_fpext_f64_f32(lo));
            break;
        }
        case ScalarKind::Bool:
            val = b.build_ne(b.build_load(brass::Type::i32(), addr, 0), b.build_iconst_i32(0));
            break;
        default:
            val = b.build_load(brass::Type::i32(), addr, 0);
            break;
    }
    b.build_ret(val);

    b.position_at_end(oobBb);
    if (mirRet == brass::Type::f64()) {
        b.build_ret(b.build_fconst_f64(0.0));
    } else {
        b.build_ret(b.build_iconst_i32(0));
    }
    fn->rebuild_cfg_predecessors();

    brass::codegen::KernelJit jit;
    brass::codegen::KernelFunction compiled = jit.compile(*fn);
    if (!compiled.is_valid()) {
        return ev::throwError("StructView.exportFieldNative: failed to compile field reader");
    }
    void* raw = compiled.entry_point();
    {
        std::lock_guard<std::mutex> lk(g_fieldExportMu);
        g_fieldExportTrampolines.push_back(std::move(compiled));
    }

    ev::NativeSignature sig;
    sig.returnType = retName;
    sig.paramTypes = {arrayType, "i32"};

    ev::NativeMirSpec spec;
    spec.mirModule = mirMod;
    spec.mirFunction = entryName;
    spec.epochPtr = &reloadEpochCounter();
    spec.expectedEpoch = currentReloadEpoch();
    spec.maxInlinedInsts = 512;

    if (!ev::registerNativeMir(jsPath, raw, sig, spec, &err)) {
        return ev::throwError("StructView.exportFieldNative: " + err);
    }
    return ev::undefined();
}

} // namespace

void decorateBind(ev::ObjectBuilder& b, bool isFn) {
    b.def("bind", 0, [isFn](Value self, std::span<const Value> args) -> Value {
        return makeBoundFunction(isFn, self, args);
    });
}

void decorateStructViewBind(ev::ObjectBuilder& b) {
    b.def("exportFieldNative", 2, js_structViewExportFieldNative);
}

} // namespace brokernel::api
