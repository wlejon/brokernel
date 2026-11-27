// JS API surface for Phase 3 diagnostics & constant/enum helpers:
//   - Kernel.compareCpuGpu(grid, ...args, opts?)
//   - Kernel.check(grid, ...args, opts?)
//   - Kernel.probeIf(grid, ...args, opts?)
//   - KernelModule.enums
//   - bro.kernel.registerConstants(name, decls)
#include "api_internal.h"
#include "../compile_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace brokernel::api {

namespace {

bool isJsArray(Value v) {
    if (!ev::isObject(v)) return false;
    auto arrCtor = ev::globalValue("Array");
    if (!arrCtor.found) return false;
    ev::Persistent isArrayFn(ev::getProperty(arrCtor.value, "isArray"));
    if (!ev::isFunction(isArrayFn.get())) return false;
    auto res = ev::call(isArrayFn.get(), arrCtor.value, std::span<const Value>(&v, 1));
    return !res.thrown && ev::toBool(res.value);
}

bool isPlainObject(Value v) {
    if (!ev::isObject(v) || ev::isFunction(v) || ev::isTypedArray(v) || ev::isArrayBuffer(v)) return false;
    if (g_bufferClass.unwrap(v) || g_kernelClass.unwrap(v) || g_moduleClass.unwrap(v) || g_fnClass.unwrap(v)) {
        return false;
    }
    return !isJsArray(v);
}

std::vector<std::string> objectOwnKeys(Value obj, const std::string& what) {
    ev::Persistent held(obj);
    auto object = ev::globalValue("Object");
    if (!object.found) ev::throwError(what + ": no Object global");
    ev::Persistent keysFn(ev::getProperty(object.value, "keys"));
    const Value arg = held.get();
    auto keys = ev::call(keysFn.get(), ev::undefined(), std::span<const Value>(&arg, 1));
    if (keys.thrown) ev::throwValue(keys.value);
    ev::Persistent names(keys.value);
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(names.get(), "length")));
    std::vector<std::string> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        out.push_back(ev::toUtf8(ev::getElement(names.get(), i)));
    }
    return out;
}

size_t countPassedParams(const CompiledKernel& k) noexcept {
    size_t count = 0;
    for (const auto& p : k.params) {
        if (isPassed(p)) ++count;
    }
    return count;
}

void splitGridAndOpts(
    const CompiledKernel& k,
    std::span<const Value> args,
    const std::string& what,
    GridShape& outShape,
    std::span<const Value>& outKargs,
    Value& outOpts) {
    const size_t passed = countPassedParams(k);
    outOpts = ev::undefined();
    if (k.grid_dim > 0) {
        if (args.empty()) ev::throwTypeError(what + ": the first argument is the grid");
        outShape = shapeArg(args[0], k, what);
        std::span<const Value> rem = args.subspan(1);
        if (rem.size() == passed + 1 && (isPlainObject(rem.back()) || ev::isUndefined(rem.back()) || ev::isNull(rem.back()))) {
            if (isPlainObject(rem.back())) outOpts = rem.back();
            outKargs = rem.subspan(0, passed);
        } else {
            outKargs = rem;
        }
        return;
    }

    // Point kernel (grid_dim == 0): grid may be omitted or passed explicitly.
    if (args.size() == passed + 2 && (isPlainObject(args.back()) || ev::isUndefined(args.back()) || ev::isNull(args.back()))) {
        outShape = shapeArg(args[0], k, what);
        if (isPlainObject(args.back())) outOpts = args.back();
        outKargs = args.subspan(1, passed);
    } else if (args.size() == passed + 1) {
        if (isPlainObject(args.back()) || ev::isUndefined(args.back()) || ev::isNull(args.back())) {
            outShape = GridShape(1, 1, 1);
            if (isPlainObject(args.back())) outOpts = args.back();
            outKargs = args.subspan(0, passed);
        } else {
            outShape = shapeArg(args[0], k, what);
            outKargs = args.subspan(1);
        }
    } else {
        outShape = GridShape(1, 1, 1);
        outKargs = args;
    }
}

bool hasKernelBufferArg(std::span<const Value> kargs) {
    for (Value v : kargs) {
        if (g_bufferClass.unwrap(v) != nullptr) return true;
    }
    return false;
}

Value coordToJsArray(const int32_t coord[3]) {
    ev::ArrayBuilder arr(3);
    arr.set(0, coord[0]);
    arr.set(1, coord[1]);
    arr.set(2, coord[2]);
    return arr.get();
}

CheckReport checkOnDevice(
    GpuDevice& dev,
    const CompiledKernel& k,
    GridShape shape,
    std::span<const KernelArg> gpuArgs,
    bool trapNonFinite,
    std::string* err) {
    CheckReport rep;
    std::shared_ptr<const CompiledKernel> variant = k.ensure_checked_variant(trapNonFinite);
    const CompiledKernel* ck = variant ? variant.get() : &k;
    if (!ck->checks_compiled && !ck->trap_non_finite_compiled) {
        if (err) *err = "kernel '" + k.name + "' has no checked variant available";
        rep.ok = false;
        rep.message = err ? *err : "kernel has no checked variant available";
        return rep;
    }
    std::shared_ptr<GpuProgram> prog = dev.createProgram(*ck, err);
    if (!prog) {
        rep.ok = false;
        rep.message = err ? *err : "failed to build checked GPU program";
        return rep;
    }
    std::shared_ptr<GpuBuffer> returns;
    if (const size_t rbytes = ck->gpu_return_bytes(shape)) {
        returns = allocateBuffer(rbytes, err);
        if (!returns) {
            rep.ok = false;
            rep.message = err ? *err : "failed to allocate return buffer";
            return rep;
        }
    }
    std::shared_ptr<GpuBuffer> checkBuf = allocateBuffer(32, err);
    if (!checkBuf) {
        rep.ok = false;
        rep.message = err ? *err : "failed to allocate check buffer";
        return rep;
    }
    GpuLaunch launch;
    if (!ck->gpu_launch(shape, gpuArgs, returns ? returns->address() : 0, launch, err)) {
        rep.ok = false;
        rep.message = err ? *err : "gpu_launch failed";
        return rep;
    }
    if (launch.check_arg_index < launch.args.size()) {
        launch.args[launch.check_arg_index] = brass::gpu::VulkanArg(checkBuf->address());
    }
    std::shared_ptr<GpuBuffer> dummyProbe;
    if (launch.probe_arg_index < launch.args.size()) {
        dummyProbe = allocateBuffer(16, err);
        if (!dummyProbe) {
            rep.ok = false;
            rep.message = err ? *err : "failed to allocate probe buffer";
            return rep;
        }
        const int32_t inactive[4] = {-1, -1, -1, 0};
        if (!dummyProbe->write(0, inactive, sizeof(inactive), err)) {
            rep.ok = false;
            rep.message = err ? *err : "failed to write probe buffer";
            return rep;
        }
        launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(dummyProbe->address());
    }
    std::shared_ptr<GpuBuffer> dummyProbeIf;
    if (launch.probe_if_arg_index < launch.args.size()) {
        dummyProbeIf = allocateBuffer(16, err);
        if (!dummyProbeIf) {
            rep.ok = false;
            rep.message = err ? *err : "failed to allocate probe_if buffer";
            return rep;
        }
        launch.args[launch.probe_if_arg_index] = brass::gpu::VulkanArg(dummyProbeIf->address());
    }
    if (!dev.dispatch(*prog, *ck, launch, err)) {
        rep.ok = false;
        rep.message = err ? *err : "GPU check dispatch failed";
        return rep;
    }
    uint32_t checkWords[8] = {};
    if (!checkBuf->read(0, checkWords, sizeof(checkWords), err)) {
        rep.ok = false;
        rep.message = err ? *err : "failed to read check buffer";
        return rep;
    }
    return detail::decode_check_report(*ck, checkWords);
}

bool probeIfOnDevice(
    GpuDevice& dev,
    const CompiledKernel& k,
    GridShape shape,
    std::span<const KernelArg> gpuArgs,
    uint32_t maxSamples,
    std::vector<ConditionalProbeSample>& out,
    std::string* err) {
    out.clear();
    std::shared_ptr<const CompiledKernel> variant = k.ensure_probed_variant();
    const CompiledKernel* pk = variant ? variant.get() : &k;
    if (!pk->has_conditional_probes()) return true;

    std::shared_ptr<GpuProgram> prog = dev.createProgram(*pk, err);
    if (!prog) return false;

    std::shared_ptr<GpuBuffer> returns;
    if (const size_t rbytes = pk->gpu_return_bytes(shape)) {
        returns = allocateBuffer(rbytes, err);
        if (!returns) return false;
    }

    const size_t bufWords = 2 + static_cast<size_t>(maxSamples) * 9;
    std::vector<uint32_t> hostBuf(bufWords, 0);
    hostBuf[0] = 0;
    hostBuf[1] = maxSamples;
    std::shared_ptr<GpuBuffer> pifBuf = allocateBuffer(std::max<size_t>(bufWords * sizeof(uint32_t), 16), err);
    if (!pifBuf) return false;
    if (!pifBuf->write(0, hostBuf.data(), bufWords * sizeof(uint32_t), err)) return false;

    GpuLaunch launch;
    if (!pk->gpu_launch(shape, gpuArgs, returns ? returns->address() : 0, launch, err)) return false;
    if (launch.probe_if_arg_index < launch.args.size()) {
        launch.args[launch.probe_if_arg_index] = brass::gpu::VulkanArg(pifBuf->address());
    }
    std::shared_ptr<GpuBuffer> dummyProbe;
    if (launch.probe_arg_index < launch.args.size()) {
        dummyProbe = allocateBuffer(16, err);
        if (!dummyProbe) return false;
        const int32_t inactive[4] = {-1, -1, -1, 0};
        if (!dummyProbe->write(0, inactive, sizeof(inactive), err)) return false;
        launch.args[launch.probe_arg_index] = brass::gpu::VulkanArg(dummyProbe->address());
    }
    std::shared_ptr<GpuBuffer> dummyCheck;
    if (launch.check_arg_index < launch.args.size()) {
        dummyCheck = allocateBuffer(32, err);
        if (!dummyCheck) return false;
        launch.args[launch.check_arg_index] = brass::gpu::VulkanArg(dummyCheck->address());
    }

    if (!dev.dispatch(*prog, *pk, launch, err)) return false;
    if (!pifBuf->read(0, hostBuf.data(), bufWords * sizeof(uint32_t), err)) return false;
    detail::decode_probe_if_samples(*pk, hostBuf, maxSamples, out);
    return true;
}

Value js_kernelCompareCpuGpu(Value self, std::span<const Value> args) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) return ev::throwTypeError("Kernel.compareCpuGpu: invalid receiver");
    const CompiledKernel& k = *h->kernel;

    GridShape shape;
    std::span<const Value> kargs;
    Value opts = ev::undefined();
    splitGridAndOpts(k, args, "Kernel.compareCpuGpu", shape, kargs, opts);

    double relTol = 1e-5;
    double absTol = 1e-6;
    bool hasUlpTol = false;
    uint64_t ulpTol = 0;
    if (ev::isObject(opts)) {
        ev::Persistent optsP(opts);
        const Value rt = ev::getProperty(optsP.get(), "relTol");
        if (!ev::isUndefined(rt)) {
            if (!ev::isNumber(rt)) return ev::throwTypeError("Kernel.compareCpuGpu: relTol must be a number");
            relTol = ev::toDouble(rt);
        }
        const Value at = ev::getProperty(optsP.get(), "absTol");
        if (!ev::isUndefined(at)) {
            if (!ev::isNumber(at)) return ev::throwTypeError("Kernel.compareCpuGpu: absTol must be a number");
            absTol = ev::toDouble(at);
        }
        const Value ut = ev::getProperty(optsP.get(), "ulpTol");
        if (!ev::isUndefined(ut) && !ev::isNull(ut)) {
            if (ev::isBigInt(ut)) {
                hasUlpTol = true;
                ulpTol = ev::toUint64(ut);
            } else if (ev::isNumber(ut)) {
                const double d = ev::toDouble(ut);
                if (d < 0.0) return ev::throwRangeError("Kernel.compareCpuGpu: ulpTol must be non-negative");
                hasUlpTol = true;
                ulpTol = static_cast<uint64_t>(d);
            } else {
                return ev::throwTypeError("Kernel.compareCpuGpu: ulpTol must be a non-negative integer or null");
            }
        }
    }

    ArgPack pack(k, kargs, "Kernel.compareCpuGpu", /*gpu=*/false);
    ensureCpuBuilt(k);
    ParityReport rep = k.compare_cpu_gpu(shape, pack.bind(kargs), relTol, absTol);
    pack.finishCpu();

    if (hasUlpTol && rep.ran_gpu && rep.max_ulp_diff > ulpTol) {
        rep.ok = false;
        if (rep.message.empty()) {
            std::ostringstream oss;
            oss << "CPU/GPU ULP mismatch at (" << rep.worst_coord[0] << ", " << rep.worst_coord[1] << ", "
                << rep.worst_coord[2] << "): cpu=" << rep.cpu_val << ", gpu=" << rep.gpu_val
                << ", max_ulp_diff=" << rep.max_ulp_diff << " > ulpTol=" << ulpTol;
            rep.message = oss.str();
        }
    }

    ev::ObjectBuilder out;
    out.set("ok", rep.ok);
    out.set("ranGpu", rep.ran_gpu);
    out.set("elementsCompared", static_cast<double>(rep.elements_compared));
    out.set("maxAbsDiff", rep.max_abs_diff);
    out.set("maxRelDiff", rep.max_rel_diff);
    out.set("maxUlpDiff", static_cast<double>(rep.max_ulp_diff));
    out.set("worstCoord", coordToJsArray(rep.worst_coord));
    out.set("cpuVal", rep.cpu_val);
    out.set("gpuVal", rep.gpu_val);
    out.set("message", rep.message);
    return out.get();
}

Value js_kernelCheck(Value self, std::span<const Value> args) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) return ev::throwTypeError("Kernel.check: invalid receiver");
    const CompiledKernel& k = *h->kernel;

    GridShape shape;
    std::span<const Value> kargs;
    Value opts = ev::undefined();
    splitGridAndOpts(k, args, "Kernel.check", shape, kargs, opts);

    bool useGpu = false;
    bool trapNonFinite = true;
    if (ev::isObject(opts)) {
        ev::Persistent optsP(opts);
        const Value gv = ev::getProperty(optsP.get(), "gpu");
        if (!ev::isUndefined(gv)) {
            if (!ev::isBool(gv)) return ev::throwTypeError("Kernel.check: gpu option must be a boolean");
            useGpu = ev::toBool(gv);
        }
        const Value tv = ev::getProperty(optsP.get(), "trapNonFinite");
        if (!ev::isUndefined(tv)) {
            if (!ev::isBool(tv)) return ev::throwTypeError("Kernel.check: trapNonFinite option must be a boolean");
            trapNonFinite = ev::toBool(tv);
        }
    }

    std::string err;
    CheckReport rep;
    if (!useGpu) {
        ArgPack pack(k, kargs, "Kernel.check", /*gpu=*/false);
        ensureCpuBuilt(k);
        rep = k.check_cpu(shape, pack.bind(kargs), trapNonFinite, &err);
    } else if (hasKernelBufferArg(kargs) && currentDevice()) {
        ArgPack pack(k, kargs, "Kernel.check", /*gpu=*/true);
        rep = checkOnDevice(*currentDevice(), k, shape, pack.gpuArgs(), trapNonFinite, &err);
    } else {
        ArgPack pack(k, kargs, "Kernel.check", /*gpu=*/false);
        rep = k.check_gpu(shape, pack.bind(kargs), trapNonFinite, &err);
    }

    ev::ObjectBuilder out;
    out.set("ok", rep.ok);
    if (rep.kind == 1) out.set("kind", "check");
    else if (rep.kind == 2) out.set("kind", "non_finite");
    else out.setNull("kind");
    out.set("coord", coordToJsArray(rep.coord));
    out.set("label", rep.label);
    out.set("value", rep.value);
    out.set("message", rep.message);
    return out.get();
}

Value js_kernelProbeIf(Value self, std::span<const Value> args) {
    KernelHandle* h = kernelHandle(self);
    if (!h || !h->kernel) return ev::throwTypeError("Kernel.probeIf: invalid receiver");
    const CompiledKernel& k = *h->kernel;

    GridShape shape;
    std::span<const Value> kargs;
    Value opts = ev::undefined();
    splitGridAndOpts(k, args, "Kernel.probeIf", shape, kargs, opts);

    bool useGpu = false;
    uint32_t maxSamples = 32;
    if (ev::isObject(opts)) {
        ev::Persistent optsP(opts);
        const Value gv = ev::getProperty(optsP.get(), "gpu");
        if (!ev::isUndefined(gv)) {
            if (!ev::isBool(gv)) return ev::throwTypeError("Kernel.probeIf: gpu option must be a boolean");
            useGpu = ev::toBool(gv);
        }
        const Value mv = ev::getProperty(optsP.get(), "maxSamples");
        if (!ev::isUndefined(mv)) {
            if (!ev::isNumber(mv)) return ev::throwTypeError("Kernel.probeIf: maxSamples must be a number");
            const double d = ev::toDouble(mv);
            if (d < 0.0 || d > 1048576.0) {
                return ev::throwRangeError("Kernel.probeIf: maxSamples must be in [0, 1048576]");
            }
            maxSamples = static_cast<uint32_t>(d);
        }
    }

    std::vector<ConditionalProbeSample> samples;
    std::string err;
    bool ok = false;
    if (!useGpu) {
        ArgPack pack(k, kargs, "Kernel.probeIf", /*gpu=*/false);
        ensureCpuBuilt(k);
        ok = k.probe_if_cpu(shape, pack.bind(kargs), samples, maxSamples, &err);
    } else if (hasKernelBufferArg(kargs) && currentDevice()) {
        ArgPack pack(k, kargs, "Kernel.probeIf", /*gpu=*/true);
        ok = probeIfOnDevice(*currentDevice(), k, shape, pack.gpuArgs(), maxSamples, samples, &err);
    } else {
        ArgPack pack(k, kargs, "Kernel.probeIf", /*gpu=*/false);
        ok = k.probe_if_gpu(shape, pack.bind(kargs), samples, maxSamples, &err);
    }
    if (!ok) return ev::throwError("Kernel.probeIf: " + err);

    ev::ArrayBuilder arr(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& s = samples[i];
        ev::ObjectBuilder sb;
        sb.set("coord", coordToJsArray(s.coord));
        sb.set("label", s.label);
        if (s.values.size() == 1) {
            sb.set("value", s.values[0]);
        } else {
            ev::ArrayBuilder vals(s.values.size());
            for (size_t vi = 0; vi < s.values.size(); ++vi) {
                vals.set(static_cast<uint32_t>(vi), s.values[vi]);
            }
            sb.set("value", vals.get());
        }
        arr.set(static_cast<uint32_t>(i), sb.get());
    }
    return arr.get();
}

// --- bro.kernel.registerConstants(name, decls) ------------------------------

bool isValidIdent(std::string_view s) {
    if (s.empty()) return false;
    auto isStart = [](unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
    auto isCont = [&](unsigned char c) { return isStart(c) || (c >= '0' && c <= '9'); };
    if (!isStart(static_cast<unsigned char>(s[0]))) return false;
    for (size_t i = 1; i < s.size(); ++i) {
        if (!isCont(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

std::string formatFloatLit(double d, bool f64, const std::string& what) {
    if (!std::isfinite(d)) {
        ev::throwRangeError(what + ": floating-point constant must be finite");
    }
    std::ostringstream oss;
    oss << std::setprecision(f64 ? 17 : 9) << d;
    std::string s = oss.str();
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
        s += ".0";
    }
    if (f64) s += "f64";
    return s;
}

std::string stripSpaces(std::string_view s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r')) --b;
    return std::string(s.substr(a, b - a));
}

bool isScalarTypeName(std::string_view t) {
    return t == "f32" || t == "f64" || t == "i32" || t == "u32" || t == "i64" || t == "u64" || t == "bool";
}

std::string vectorElemType(std::string_view typeStr) {
    const size_t lt = typeStr.find('<');
    const size_t gt = typeStr.rfind('>');
    if (lt != std::string_view::npos && gt != std::string_view::npos && gt > lt + 1) {
        return stripSpaces(typeStr.substr(lt + 1, gt - lt - 1));
    }
    if (!typeStr.empty()) {
        const char tail = typeStr.back();
        if (tail == 'f') return "f32";
        if (tail == 'd') return "f64";
        if (tail == 'i') return "i32";
        if (tail == 'u') return "u32";
    }
    return "f32";
}

bool parseArrayTypeElem(std::string_view typeStr, std::string& elemOut) {
    if (typeStr.size() < 8 || typeStr.compare(0, 6, "array<") != 0 || typeStr.back() != '>') return false;
    std::string_view inner = typeStr.substr(6, typeStr.size() - 7);
    int depth = 0;
    for (size_t i = 0; i < inner.size(); ++i) {
        char c = inner[i];
        if (c == '<') ++depth;
        else if (c == '>') --depth;
        else if (c == ',' && depth == 0) {
            elemOut = stripSpaces(inner.substr(0, i));
            return !elemOut.empty();
        }
    }
    return false;
}

std::string formatTypedLiteral(const std::string& typeStr, Value v, const std::string& what) {
    if (typeStr == "f32") {
        if (!ev::isNumber(v)) ev::throwTypeError(what + ": expected number for f32");
        return formatFloatLit(ev::toDouble(v), false, what);
    }
    if (typeStr == "f64") {
        if (!ev::isNumber(v)) ev::throwTypeError(what + ": expected number for f64");
        return formatFloatLit(ev::toDouble(v), true, what);
    }
    if (typeStr == "i32") {
        if (!ev::isNumber(v) && !ev::isBigInt(v)) ev::throwTypeError(what + ": expected integer for i32");
        return std::to_string(static_cast<int32_t>(ev::toInt64(v)));
    }
    if (typeStr == "u32") {
        if (!ev::isNumber(v) && !ev::isBigInt(v)) ev::throwTypeError(what + ": expected integer for u32");
        return std::to_string(static_cast<uint32_t>(ev::toUint64(v))) + "u";
    }
    if (typeStr == "i64") {
        if (!ev::isNumber(v) && !ev::isBigInt(v)) ev::throwTypeError(what + ": expected integer for i64");
        return std::to_string(ev::toInt64(v)) + "i64";
    }
    if (typeStr == "u64") {
        if (!ev::isNumber(v) && !ev::isBigInt(v)) ev::throwTypeError(what + ": expected integer for u64");
        return std::to_string(ev::toUint64(v)) + "u64";
    }
    if (typeStr == "bool") {
        if (!ev::isBool(v)) ev::throwTypeError(what + ": expected boolean for bool");
        return ev::toBool(v) ? "true" : "false";
    }

    std::string arrElem;
    if (parseArrayTypeElem(typeStr, arrElem)) {
        if (!isJsArray(v)) ev::throwTypeError(what + ": expected array value for " + typeStr);
        ev::Persistent arrP(v);
        const uint32_t len = static_cast<uint32_t>(ev::toDouble(ev::getProperty(arrP.get(), "length")));
        std::ostringstream oss;
        oss << "[";
        for (uint32_t i = 0; i < len; ++i) {
            if (i > 0) oss << ", ";
            oss << formatTypedLiteral(arrElem, ev::getElement(arrP.get(), i), what);
        }
        oss << "]";
        return oss.str();
    }

    // Vector type: vec2<T>..vec4<T> or vec2f..vec4u
    if (!isJsArray(v)) ev::throwTypeError(what + ": expected array of components for " + typeStr);
    ev::Persistent arrP(v);
    const uint32_t len = static_cast<uint32_t>(ev::toDouble(ev::getProperty(arrP.get(), "length")));
    const std::string elem = vectorElemType(typeStr);
    std::ostringstream oss;
    oss << typeStr << "(";
    for (uint32_t i = 0; i < len; ++i) {
        if (i > 0) oss << ", ";
        oss << formatTypedLiteral(elem, ev::getElement(arrP.get(), i), what);
    }
    oss << ")";
    return oss.str();
}

void emitEnumDecl(
    std::ostringstream& src,
    const std::string& key,
    Value obj,
    const std::string& what) {
    ev::Persistent objP(obj);
    std::string underlying = "i32";
    const Value uProp = ev::getProperty(objP.get(), "__enum");
    const Value eProp = ev::getProperty(objP.get(), "enum");
    const Value kindVal = !ev::isUndefined(uProp) ? uProp : eProp;
    if (ev::isString(kindVal)) {
        underlying = ev::toUtf8(kindVal);
        if (underlying != "i32" && underlying != "u32") {
            ev::throwTypeError(what + ": enum '" + key + "' underlying type must be 'i32' or 'u32'");
        }
    }

    Value variantsObj = ev::getProperty(objP.get(), "variants");
    const bool explicitVariants = isPlainObject(variantsObj);
    ev::Persistent vHolder(explicitVariants ? variantsObj : objP.get());
    std::vector<std::string> vkeys = objectOwnKeys(vHolder.get(), what);

    src << "pub enum " << key << " : " << underlying << " {\n";
    for (const std::string& vk : vkeys) {
        if (!explicitVariants && (vk == "__enum" || vk == "enum")) continue;
        if (!isValidIdent(vk)) {
            ev::throwTypeError(what + ": invalid enum variant name '" + vk + "'");
        }
        const Value vv = ev::getProperty(vHolder.get(), vk);
        if (!ev::isNumber(vv) && !ev::isBigInt(vv)) {
            ev::throwTypeError(what + ": enum variant '" + key + "::" + vk + "' value must be an integer");
        }
        src << "    " << vk << " = " << ev::toInt64(vv) << ",\n";
    }
    src << "}\n";
}

Value js_registerConstants(Value, std::span<const Value> args) {
    const std::string what = "bro.kernel.registerConstants";
    if (args.size() < 2 || !ev::isString(args[0]) || !isPlainObject(args[1])) {
        return ev::throwTypeError(what + ": expected (name: string, decls: object)");
    }
    const std::string modName = ev::toUtf8(args[0]);
    if (modName.empty()) {
        return ev::throwTypeError(what + ": module name must not be empty");
    }

    ev::Persistent declsP(args[1]);
    const std::vector<std::string> keys = objectOwnKeys(declsP.get(), what);

    std::ostringstream src;
    src << "module " << modName << ";\n";

    for (const std::string& key : keys) {
        if (!isValidIdent(key)) {
            return ev::throwTypeError(what + ": invalid identifier '" + key + "'");
        }
        ev::Persistent valP(ev::getProperty(declsP.get(), key));
        const Value val = valP.get();

        if (ev::isNumber(val)) {
            src << "pub const " << key << ": f32 = " << formatFloatLit(ev::toDouble(val), false, what) << ";\n";
        } else if (ev::isBool(val)) {
            src << "pub const " << key << ": bool = " << (ev::toBool(val) ? "true" : "false") << ";\n";
        } else if (ev::isBigInt(val)) {
            src << "pub const " << key << ": i64 = " << ev::toInt64(val) << "i64;\n";
        } else if (isJsArray(val)) {
            const uint32_t len = static_cast<uint32_t>(ev::toDouble(ev::getProperty(val, "length")));
            if (len == 0) {
                return ev::throwTypeError(what + ": empty array constant '" + key + "'");
            }
            ev::Persistent firstP(ev::getElement(val, 0));
            if (len == 2 && ev::isString(firstP.get())) {
                const std::string typeStr = stripSpaces(ev::toUtf8(firstP.get()));
                ev::Persistent innerP(ev::getElement(val, 1));
                const Value inner = innerP.get();
                std::string unusedElem;
                if (!parseArrayTypeElem(typeStr, unusedElem) && isJsArray(inner)) {
                    const uint32_t innerLen = static_cast<uint32_t>(ev::toDouble(ev::getProperty(inner, "length")));
                    const bool nested = innerLen > 0 && isJsArray(ev::getElement(inner, 0));
                    if (nested || isScalarTypeName(typeStr)) {
                        src << "pub const " << key << ": array<" << typeStr << ", " << innerLen << "> = [\n";
                        for (uint32_t i = 0; i < innerLen; ++i) {
                            if (i > 0) src << ",\n";
                            src << "    " << formatTypedLiteral(typeStr, ev::getElement(inner, i), what);
                        }
                        src << "\n];\n";
                    } else {
                        src << "pub const " << key << ": " << typeStr << " = "
                            << formatTypedLiteral(typeStr, inner, what) << ";\n";
                    }
                } else {
                    src << "pub const " << key << ": " << typeStr << " = "
                        << formatTypedLiteral(typeStr, inner, what) << ";\n";
                }
            } else if (len >= 2 && len <= 4) {
                const std::string vtype = "vec" + std::to_string(len) + "<f32>";
                src << "pub const " << key << ": " << vtype << " = "
                    << formatTypedLiteral(vtype, val, what) << ";\n";
            } else {
                const std::string atype = "array<f32, " + std::to_string(len) + ">";
                src << "pub const " << key << ": " << atype << " = "
                    << formatTypedLiteral(atype, val, what) << ";\n";
            }
        } else if (isPlainObject(val)) {
            const Value uProp = ev::getProperty(val, "__enum");
            const Value eProp = ev::getProperty(val, "enum");
            if (ev::isUndefined(uProp) && ev::isUndefined(eProp)) {
                return ev::throwTypeError(what + ": object constant '" + key + "' must specify __enum or enum");
            }
            emitEnumDecl(src, key, val, what);
        } else {
            return ev::throwTypeError(what + ": unsupported constant value for '" + key + "'");
        }
    }

    registerNamedModuleSource(modName, src.str());
    return ev::fromUtf8(modName);
}

} // namespace

void decorateKernelDiagExtras(ev::ObjectBuilder& b) {
    b.def("compareCpuGpu", 1, js_kernelCompareCpuGpu);
    b.def("check", 0, js_kernelCheck);
    b.def("probeIf", 0, js_kernelProbeIf);
}

void decorateModuleDiagExtras(ev::ObjectBuilder& b) {
    b.accessor("enums", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->cell || !h->cell->result) return ev::undefined();
        ev::ObjectBuilder out;
        for (const auto& ei : h->cell->result->enums) {
            ev::ObjectBuilder eb;
            for (const auto& [vname, vval] : ei.variants) {
                eb.set(vname, static_cast<double>(vval));
            }
            out.set(ei.name, eb.get());
        }
        return out.get();
    });
}

void decorateBroKernelDiagExtras(ev::ObjectBuilder& b) {
    b.def("registerConstants", 2, js_registerConstants);
}

} // namespace brokernel::api
