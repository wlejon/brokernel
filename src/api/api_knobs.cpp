// JS API surface for runtime `@knob`s and `mod.knobGrad(...)`.
#include "api_internal.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace brokernel::api {

Value js_kernelSetKnobs(Value self, std::span<const Value> args) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) {
        return ev::throwTypeError("Kernel.setKnobs: invalid receiver");
    }
    if (h->kernel->runtime_knobs.empty()) {
        return ev::throwError("Kernel.setKnobs: kernel was not compiled with { runtimeKnobs: true }");
    }
    if (args.empty() || !ev::isObject(args[0])) {
        return ev::throwTypeError("Kernel.setKnobs: expects an object of knob overrides");
    }
    auto overrides = parseKnobOverrides(args[0], "Kernel.setKnobs");
    if (!h->kernel->set_knobs(overrides)) {
        return ev::throwError("Kernel.setKnobs: unknown knob or incompatible type");
    }
    return self;
}

Value js_moduleKnobGrad(Value self, std::span<const Value> args) {
    auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
    if (!h || !h->cell || !h->cell->result) {
        return ev::throwTypeError("KernelModule.knobGrad: invalid receiver");
    }
    if (args.size() < 2 || !ev::isString(args[0])) {
        return ev::throwTypeError("KernelModule.knobGrad: expects (fnName, knobNameOrArray)");
    }
    const std::string fnName = ev::toUtf8(args[0]);
    std::vector<std::string> knobNames;
    bool returnMap = false;
    if (ev::isString(args[1])) {
        knobNames.push_back(ev::toUtf8(args[1]));
    } else if (ev::isObject(args[1])) {
        returnMap = true;
        ev::Persistent arr(args[1]);
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(arr.get(), "length")));
        for (uint32_t i = 0; i < n; ++i) {
            const Value el = ev::getElement(arr.get(), i);
            if (!ev::isString(el)) {
                return ev::throwTypeError("KernelModule.knobGrad: knob array elements must be strings");
            }
            knobNames.push_back(ev::toUtf8(el));
        }
    } else {
        return ev::throwTypeError("KernelModule.knobGrad: second argument must be a knob name string or array");
    }

    auto gradRes = compileOff([&] {
        return h->cell->result->compile_knob_grads(fnName, knobNames);
    });
    if (!gradRes->ok) throwCompileError(*gradRes);
    std::shared_ptr<const CompileResult> sharedRes = std::move(gradRes);
    auto gradCell = std::make_shared<ModuleCell>();
    gradCell->result = sharedRes;

    if (!returnMap) {
        if (sharedRes->kernels.empty()) {
            return ev::throwError("KernelModule.knobGrad: no gradient kernel generated");
        }
        return makeFnValueFromResult(gradCell, sharedRes, sharedRes->kernels[0].name);
    }
    ev::ObjectBuilder out;
    for (size_t i = 0; i < knobNames.size() && i < sharedRes->kernels.size(); ++i) {
        ev::Persistent fnVal(makeFnValueFromResult(gradCell, sharedRes, sharedRes->kernels[i].name));
        out.set(knobNames[i], fnVal.get());
    }
    return out.get();
}

Value js_moduleKnobGradGrid(Value self, std::span<const Value> args) {
    auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
    if (!h || !h->cell || !h->cell->result) {
        return ev::throwTypeError("KernelModule.knobGradGrid: invalid receiver");
    }
    if (args.size() < 2 || !ev::isString(args[0])) {
        return ev::throwTypeError("KernelModule.knobGradGrid: expects (fnName, knobNameOrArray, gridDim?)");
    }
    const std::string fnName = ev::toUtf8(args[0]);
    std::vector<std::string> knobNames;
    if (ev::isString(args[1])) {
        knobNames.push_back(ev::toUtf8(args[1]));
    } else if (ev::isObject(args[1])) {
        ev::Persistent arr(args[1]);
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(arr.get(), "length")));
        for (uint32_t i = 0; i < n; ++i) {
            const Value el = ev::getElement(arr.get(), i);
            if (!ev::isString(el)) {
                return ev::throwTypeError("KernelModule.knobGradGrid: knob array elements must be strings");
            }
            knobNames.push_back(ev::toUtf8(el));
        }
    } else {
        return ev::throwTypeError("KernelModule.knobGradGrid: second argument must be a knob name string or array");
    }
    uint32_t gridDim = 1;
    if (args.size() >= 3 && !ev::isUndefined(args[2]) && !ev::isNull(args[2])) {
        if (!ev::isNumber(args[2])) {
            return ev::throwTypeError("KernelModule.knobGradGrid: gridDim must be a number in 1..3");
        }
        const double d = ev::toDouble(args[2]);
        if (d < 1.0 || d > 3.0) {
            return ev::throwRangeError("KernelModule.knobGradGrid: gridDim must be 1, 2, or 3");
        }
        gridDim = static_cast<uint32_t>(d);
    }

    auto gradRes = compileOff([&] {
        return h->cell->result->compile_knob_grid_grads(fnName, knobNames, gridDim);
    });
    if (!gradRes->ok) throwCompileError(*gradRes);
    if (gradRes->kernels.empty()) {
        return ev::throwError("KernelModule.knobGradGrid: no gradient reduction kernel generated");
    }
    auto gradCell = makeModuleCell(std::move(gradRes));
    return makeKernelValue(gradCell, gradCell->result->kernels[0].name, "KernelModule.knobGradGrid");
}

Value js_kernelVerify(Value self, std::span<const Value>) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) {
        return ev::throwTypeError("Kernel.verify: invalid receiver");
    }
    KernelVerifyReport rep = h->kernel->verify();
    ev::Persistent violArr(ev::makeArray(static_cast<uint32_t>(rep.violations.size())));
    for (size_t i = 0; i < rep.violations.size(); ++i) {
        ev::Persistent s(ev::fromUtf8(rep.violations[i]));
        violArr.set(ev::setElement(violArr.get(), static_cast<uint32_t>(i), s.get()));
    }
    ev::ObjectBuilder out;
    out.set("spirvParsed", rep.spirv_parsed);
    out.set("memorySafe", rep.memory_safe);
    out.set("barrierUniform", rep.barrier_uniform);
    out.set("sharedAccessesProven", static_cast<double>(rep.shared_accesses_proven));
    out.set("globalAccessesProven", static_cast<double>(rep.global_accesses_proven));
    out.set("barriersProven", static_cast<double>(rep.barriers_proven));
    out.set("violations", violArr.get());
    out.set("diagnostic", rep.diagnostic);
    return out.get();
}

Value js_kernelVerifySandbox(Value self, std::span<const Value>) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) {
        return ev::throwTypeError("Kernel.verifySandbox: invalid receiver");
    }
    KernelSandboxReport rep = h->kernel->verify_sandbox();
    ev::Persistent violArr(ev::makeArray(static_cast<uint32_t>(rep.violations.size())));
    for (size_t i = 0; i < rep.violations.size(); ++i) {
        ev::Persistent s(ev::fromUtf8(rep.violations[i]));
        violArr.set(ev::setElement(violArr.get(), static_cast<uint32_t>(i), s.get()));
    }
    ev::ObjectBuilder out;
    out.set("passed", rep.passed);
    out.set("portabilityOk", rep.portability_ok);
    out.set("spirvParsed", rep.spirv_parsed);
    out.set("memorySafe", rep.memory_safe);
    out.set("barrierUniform", rep.barrier_uniform);
    out.set("sharedAccessesProven", static_cast<double>(rep.shared_accesses_proven));
    out.set("globalAccessesProven", static_cast<double>(rep.global_accesses_proven));
    out.set("barriersProven", static_cast<double>(rep.barriers_proven));
    out.set("violations", violArr.get());
    out.set("diagnostic", rep.diagnostic);
    return out.get();
}

Value js_kernelSetTelemetry(Value self, std::span<const Value> args) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) {
        return ev::throwTypeError("Kernel.setTelemetry: invalid receiver");
    }
    if (args.empty() || ev::isNull(args[0]) || ev::isUndefined(args[0])) {
        h->telemetry_buffer.reset();
        h->telemetry_typed_array.reset();
        h->kernel->set_telemetry_buffer(nullptr, 0, 0);
        return self;
    }
    if (auto* bh = static_cast<BufferHandle*>(g_bufferClass.unwrap(args[0]))) {
        if (!bh->buffer) {
            return ev::throwError("Kernel.setTelemetry: KernelBuffer has been destroyed");
        }
        h->telemetry_typed_array.reset();
        h->telemetry_buffer = bh->buffer;
        const uint32_t cap = static_cast<uint32_t>(
            std::min<size_t>(bh->buffer->bytes() / sizeof(float), 0x7fffffffu));
        h->kernel->set_telemetry_buffer(
            static_cast<float*>(bh->buffer->hostDataForCpu()),
            cap,
            bh->buffer->address());
        return self;
    }
    if (ev::isTypedArray(args[0])) {
        const auto info = ev::typedArrayInfo(args[0]);
        if (!info || info.elementKind != ev::elements::Float32) {
            return ev::throwTypeError("Kernel.setTelemetry: expects Float32Array, KernelBuffer, or null");
        }
        h->telemetry_buffer.reset();
        h->telemetry_typed_array = std::make_shared<ev::Persistent>(args[0]);
        const uint32_t cap = static_cast<uint32_t>(std::min<size_t>(info.elementCount, 0x7fffffffu));
        h->kernel->set_telemetry_buffer(reinterpret_cast<float*>(info.data), cap, 0);
        return self;
    }
    return ev::throwTypeError("Kernel.setTelemetry: expects Float32Array, KernelBuffer, or null");
}

void syncKernelTelemetry(KernelHandle& h, bool gpu) {
    if (!h.kernel || h.kernel->telemetry_slots == 0 || !h.kernel->uses_telemetry) return;
    if (h.telemetry_typed_array && !ev::isUndefined(h.telemetry_typed_array->get())) {
        const auto info = ev::typedArrayInfo(h.telemetry_typed_array->get());
        const uint32_t cap = static_cast<uint32_t>(std::min<size_t>(info.elementCount, 0x7fffffffu));
        h.kernel->set_telemetry_buffer(reinterpret_cast<float*>(info.data), cap, 0);
        return;
    }
    if (h.telemetry_buffer) {
        std::string err;
        const uint32_t cap = static_cast<uint32_t>(
            std::min<size_t>(h.telemetry_buffer->bytes() / sizeof(float), 0x7fffffffu));
        if (gpu) {
            if (!h.telemetry_buffer->prepareForGpu(&err)) {
                ev::throwError("Kernel.dispatch telemetry: " + err);
            }
            h.kernel->set_telemetry_buffer(nullptr, cap, h.telemetry_buffer->address());
        } else {
            if (!h.telemetry_buffer->prepareForCpu(&err)) {
                ev::throwError("Kernel.run telemetry: " + err);
            }
            h.kernel->set_telemetry_buffer(
                static_cast<float*>(h.telemetry_buffer->hostDataForCpu()),
                cap,
                h.telemetry_buffer->address());
        }
    }
}

void finishKernelTelemetry(KernelHandle& h, bool gpu) {
    if (!h.kernel || h.kernel->telemetry_slots == 0 || !h.kernel->uses_telemetry) return;
    if (!h.telemetry_buffer) return;
    if (gpu) {
        h.telemetry_buffer->markGpuWritten();
    } else {
        h.telemetry_buffer->markHostWritten();
    }
}

void decorateKernelTargetExtras(ev::ObjectBuilder& b) {
    b.def("verify", 0, js_kernelVerify);
    b.def("verifySandbox", 0, js_kernelVerifySandbox);
    b.def("setTelemetry", 1, js_kernelSetTelemetry);
    b.accessor("telemetrySlots", [](Value self, std::span<const Value>) -> Value {
        KernelHandle* h = kernelHandle(self);
        return h && h->kernel ? ev::fromDouble(static_cast<double>(h->kernel->telemetry_slots)) : ev::undefined();
    });
    b.accessor("msl", [](Value self, std::span<const Value>) -> Value {
        KernelHandle* h = kernelHandle(self);
        return h && h->kernel ? ev::fromUtf8(h->kernel->msl) : ev::undefined();
    });
    b.accessor("mslEntry", [](Value self, std::span<const Value>) -> Value {
        KernelHandle* h = kernelHandle(self);
        return h && h->kernel ? ev::fromUtf8(h->kernel->msl_entry) : ev::undefined();
    });
    b.accessor("ptx", [](Value self, std::span<const Value>) -> Value {
        KernelHandle* h = kernelHandle(self);
        return h && h->kernel ? ev::fromUtf8(h->kernel->ptx) : ev::undefined();
    });
    b.accessor("spirvDisassembly", [](Value self, std::span<const Value>) -> Value {
        KernelHandle* h = kernelHandle(self);
        return h && h->kernel ? ev::fromUtf8(h->kernel->spirv_disassembly) : ev::undefined();
    });
}

} // namespace brokernel::api
