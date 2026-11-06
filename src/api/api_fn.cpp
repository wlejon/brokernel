#include "api_internal.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace brokernel::api {

namespace {

std::mutex g_registeredModulesMu;
std::unordered_map<std::string, std::string> g_registeredModules;
std::unordered_map<std::string, std::string> g_resolverFallback;
std::unordered_map<std::string, std::string> g_hostModules;
thread_local ev::Persistent t_moduleResolver;
thread_local bool t_hasModuleResolver = false;

std::string normalizeModuleKey(std::string_view name) {
    std::string out;
    out.reserve(name.size());
    for (size_t i = 0; i < name.size(); ++i) {
        if (i + 1 < name.size() && name[i] == ':' && name[i + 1] == ':') {
            out.push_back('/');
            ++i;
        } else {
            out.push_back(name[i]);
        }
    }
    return out;
}

std::string canonicalModuleName(std::string_view name) {
    std::string out;
    out.reserve(name.size() + 4);
    for (char c : name) {
        if (c == '/') out += "::";
        else out.push_back(c);
    }
    return out;
}

Value constValueToJs(const ConstValue& cv) {
    if (!cv.elements.empty()) {
        ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(cv.elements.size())));
        for (size_t i = 0; i < cv.elements.size(); ++i) {
            ev::Persistent elem(constValueToJs(cv.elements[i]));
            arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), elem.get()));
        }
        return arr.get();
    }
    if (cv.type.is_bool()) return ev::fromBool(cv.bool_val);
    if (cv.type.is_float()) return ev::fromDouble(cv.float_val);
    return ev::fromDouble(static_cast<double>(cv.int_val));
}

ConstValue jsToConstValue(Value v, const std::string& what) {
    if (ev::isBool(v)) return ConstValue::from_bool(ev::toBool(v));
    if (ev::isNumber(v)) return ConstValue::from_f64(ev::toDouble(v));
    if (ev::isBigInt(v)) {
        ConstValue c;
        c.type = Type::i64();
        c.int_val = ev::toInt64(v);
        return c;
    }
    if (ev::isObject(v)) {
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(v, "length")));
        if (n >= 1) {
            ConstValue cv;
            cv.type = (n >= 2 && n <= 4) ? Type::vec(ScalarKind::F32, n)
                                         : Type::array_of(Type::f32(), n);
            cv.elements.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                cv.elements.push_back(jsToConstValue(ev::getElement(v, i), what));
            }
            return cv;
        }
    }
    ev::throwTypeError(what + ": knob value must be a number, boolean, or array");
    return ConstValue::from_f32(0.0f);
}

std::unordered_map<std::string, ConstValue> parseKnobOverrides(Value obj, const std::string& what) {
    std::unordered_map<std::string, ConstValue> out;
    if (!ev::isObject(obj)) {
        ev::throwTypeError(what + ": expected an object of knob overrides");
    }
    ev::Persistent heldObj(obj);
    auto object = ev::globalValue("Object");
    if (!object.found) ev::throwError(what + ": no Object global");
    ev::Persistent keysFn(ev::getProperty(object.value, "keys"));
    const Value oarg = heldObj.get();
    auto keys = ev::call(keysFn.get(), ev::undefined(), std::span<const Value>(&oarg, 1));
    if (keys.thrown) ev::throwValue(keys.value);
    ev::Persistent names(keys.value);
    const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(names.get(), "length")));
    for (uint32_t i = 0; i < n; ++i) {
        const std::string k = ev::toUtf8(ev::getElement(names.get(), i));
        const Value v = ev::getProperty(heldObj.get(), k);
        out[k] = jsToConstValue(v, what);
    }
    return out;
}

Value formatProbeSamples(const std::vector<ProbeSample>& samples) {
    ev::ObjectBuilder b;
    for (const auto& s : samples) {
        if (!s.hit) continue;
        if (s.lanes <= 1) {
            if (s.type.is_bool()) {
                b.set(s.name, ev::fromBool(s.values[0] != 0.0f));
            } else {
                b.set(s.name, ev::fromDouble(s.values[0]));
            }
        } else {
            ev::Persistent arr(ev::makeArray(s.lanes));
            for (uint32_t i = 0; i < s.lanes && i < 4; ++i) {
                ev::Persistent elem(ev::fromDouble(s.values[i]));
                arr.set(ev::setElement(arr.get(), i, elem.get()));
            }
            b.set(s.name, arr.get());
        }
    }
    return b.get();
}

void parseCellCoord(Value v, int32_t& gx, int32_t& gy, int32_t& gz, const std::string& what) {
    gx = 0;
    gy = 0;
    gz = 0;
    if (ev::isNumber(v)) {
        gx = static_cast<int32_t>(ev::toDouble(v));
    } else if (ev::isObject(v)) {
        const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(v, "length")));
        if (n < 1 || n > 3) ev::throwTypeError(what + ": cell coordinate must be a number or [x, y, z]");
        gx = static_cast<int32_t>(ev::toDouble(ev::getElement(v, 0)));
        if (n >= 2) gy = static_cast<int32_t>(ev::toDouble(ev::getElement(v, 1)));
        if (n >= 3) gz = static_cast<int32_t>(ev::toDouble(ev::getElement(v, 2)));
    } else {
        ev::throwTypeError(what + ": cell coordinate must be a number or [x, y, z]");
    }
}

bool probeOnDevice(
    GpuDevice& dev,
    const CompiledKernel& k,
    int32_t gx, int32_t gy, int32_t gz,
    GridShape shape,
    std::span<const KernelArg> gpu_args,
    std::vector<ProbeSample>& out,
    std::string* err) {
    out.clear();
    std::shared_ptr<const CompiledKernel> probed_holder;
    const CompiledKernel* pk = &k;
    if (!k.probes_compiled && !k.probes.empty() && k.build_probed_variant) {
        probed_holder = k.build_probed_variant();
        if (probed_holder) pk = probed_holder.get();
    }
    if (pk->probes.empty()) return true;

    std::shared_ptr<GpuProgram> prog = dev.createProgram(*pk, err);
    if (!prog) return false;

    std::shared_ptr<GpuBuffer> returns;
    if (const size_t rbytes = pk->gpu_return_bytes(shape)) {
        returns = allocateBuffer(rbytes, err);
        if (!returns) return false;
    }

    uint32_t max_idx = 0;
    for (const auto& p : pk->probes) max_idx = std::max(max_idx, p.index + 1);
    const size_t probe_words = 4 + static_cast<size_t>(max_idx) * 5;
    std::vector<uint32_t> host_probe(probe_words, 0);
    host_probe[0] = static_cast<uint32_t>(gx);
    host_probe[1] = static_cast<uint32_t>(gy);
    host_probe[2] = static_cast<uint32_t>(gz);

    std::shared_ptr<GpuBuffer> probe_buf = allocateBuffer(probe_words * sizeof(uint32_t), err);
    if (!probe_buf) return false;
    if (!probe_buf->write(0, host_probe.data(), probe_words * sizeof(uint32_t), err)) return false;

    GpuLaunch launch;
    if (!pk->gpu_launch(shape, gpu_args, returns ? returns->address() : 0, launch, err)) return false;
    if (pk->probes_compiled && pk->stage == Stage::Compute && !pk->probes.empty() && !launch.args.empty()) {
        launch.args.back() = brass::gpu::VulkanArg(probe_buf->address());
    }
    if (!dev.dispatch(*prog, *pk, launch, err)) return false;
    if (!probe_buf->read(0, host_probe.data(), probe_words * sizeof(uint32_t), err)) return false;

    out.reserve(pk->probes.size());
    for (const auto& p : pk->probes) {
        ProbeSample s;
        s.name = p.name;
        s.type = p.type;
        s.lanes = p.type.is_vector() ? p.type.dimension : 1;
        const size_t base = 4 + static_cast<size_t>(p.index) * 5;
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
}

Value js_kernelProbe(Value self, std::span<const Value> args) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(self));
    if (!h || !h->kernel) return ev::throwTypeError("Kernel.probe: invalid receiver");
    const CompiledKernel& k = *h->kernel;

    if (k.grid_dim == 0) {
        ArgPack pack(k, args, "Kernel.probe");
        std::vector<ProbeSample> samples;
        std::string err;
        if (!k.probe_cpu(0, 0, 0, GridShape(1, 1, 1), pack.bind(args), samples, &err)) {
            return ev::throwError("Kernel.probe: " + err);
        }
        return formatProbeSamples(samples);
    }

    if (args.size() < 2) {
        return ev::throwTypeError("Kernel.probe: grid kernel requires (cellCoord, gridShape, ...args)");
    }
    int32_t gx = 0, gy = 0, gz = 0;
    parseCellCoord(args[0], gx, gy, gz, "Kernel.probe");
    const GridShape shape = shapeArg(args[1], k, "Kernel.probe");
    const std::span<const Value> kargs = args.subspan(2);

    bool has_gpu_buffers = false;
    for (Value v : kargs) {
        if (g_bufferClass.unwrap(v) != nullptr) {
            has_gpu_buffers = true;
            break;
        }
    }

    std::vector<ProbeSample> samples;
    std::string err;
    if (has_gpu_buffers || (!k.has_cpu_build() && currentDevice())) {
        ArgPack pack(k, kargs, "Kernel.probe", /*gpu=*/true);
        if (GpuDevice* dev = currentDevice()) {
            if (!probeOnDevice(*dev, k, gx, gy, gz, shape, pack.gpuArgs(), samples, &err)) {
                return ev::throwError("Kernel.probe: " + err);
            }
        } else {
            if (!k.probe_gpu(gx, gy, gz, shape, pack.gpuArgs(), samples, &err)) {
                return ev::throwError("Kernel.probe: " + err);
            }
        }
    } else {
        ArgPack pack(k, kargs, "Kernel.probe", /*gpu=*/false);
        if (!k.probe_cpu(gx, gy, gz, shape, pack.bind(kargs), samples, &err)) {
            return ev::throwError("Kernel.probe: " + err);
        }
    }
    return formatProbeSamples(samples);
}

Value js_kernelKnobs(Value self, std::span<const Value> args) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(self));
    if (!h || !h->kernel || !h->module_result) {
        return ev::throwTypeError("Kernel.knobs: invalid receiver");
    }
    if (args.empty()) return self;
    auto overrides = parseKnobOverrides(args[0], "Kernel.knobs");
    auto next_res = std::make_shared<CompileResult>(h->module_result->with_knobs(overrides));
    if (!next_res->ok) throwCompileError(*next_res);
    const CompiledKernel* nk = next_res->find_kernel(h->kernel->name);
    if (!nk) return ev::throwError("Kernel.knobs: kernel '" + h->kernel->name + "' not found after recompilation");
    h->module_result = next_res;
    h->kernel = std::shared_ptr<const CompiledKernel>(next_res, nk);
    h->program.reset();
    if (nk->grid_dim > 0 && !nk->spirv.words.empty()) {
        if (GpuDevice* dev = currentDevice()) {
            std::string err;
            h->program = dev->createProgram(*nk, &err);
        }
    }
    return self;
}

Value js_kernelStats(Value self, std::span<const Value>) {
    auto* h = static_cast<KernelHandle*>(g_kernelClass.unwrap(self));
    if (!h || !h->kernel) return ev::throwTypeError("Kernel.stats: invalid receiver");
    const auto& words = h->kernel->spirv.words;
    uint32_t instructions = 0, arithmetic = 0, samples = 0, loads = 0, stores = 0, branches = 0, loops = 0, functions = 0;
    if (words.size() >= 5 && words[0] == 0x07230203u) {
        size_t i = 5;
        while (i < words.size()) {
            const uint32_t word0 = words[i];
            const uint16_t wc = static_cast<uint16_t>(word0 >> 16);
            const uint16_t op = static_cast<uint16_t>(word0 & 0xFFFFu);
            if (wc == 0 || i + wc > words.size()) break;
            ++instructions;
            if (op == 54) ++functions;
            else if (op == 61 || op == 227) ++loads;
            else if (op == 62 || op == 228 || op == 99) ++stores;
            else if (op >= 87 && op <= 98) ++samples;
            else if ((op >= 126 && op <= 175) || (op >= 179 && op <= 209) || op == 12) ++arithmetic;
            else if (op == 246) ++loops;
            else if (op >= 249 && op <= 251) ++branches;
            i += wc;
        }
    }
    ev::ObjectBuilder out;
    out.set("spirvBytes", static_cast<double>(words.size() * 4));
    out.set("instructions", static_cast<double>(instructions));
    out.set("arithmetic", static_cast<double>(arithmetic));
    out.set("samples", static_cast<double>(samples));
    out.set("loads", static_cast<double>(loads));
    out.set("stores", static_cast<double>(stores));
    out.set("branches", static_cast<double>(branches));
    out.set("loops", static_cast<double>(loops));
    out.set("functions", static_cast<double>(functions));

    std::vector<ExecutableStats> execs;
    if (h->program) execs = h->program->executableStats();
    ev::Persistent execArr(ev::makeArray(static_cast<uint32_t>(execs.size())));
    for (size_t ei = 0; ei < execs.size(); ++ei) {
        ev::ObjectBuilder eb;
        eb.set("name", execs[ei].name);
        eb.set("description", execs[ei].description);
        eb.set("subgroupSize", static_cast<double>(execs[ei].subgroupSize));
        ev::ObjectBuilder sb;
        for (const auto& st : execs[ei].statistics) {
            sb.set(st.name, st.value);
        }
        ev::Persistent sv(sb.get());
        eb.set("statistics", sv.get());
        ev::Persistent evObj(eb.get());
        execArr.set(ev::setElement(execArr.get(), static_cast<uint32_t>(ei), evObj.get()));
    }
    out.set("executables", execArr.get());
    return out.get();
}

// ---------------------------------------------------------------------------
// KernelFn (`mod.fn(name)`)
// ---------------------------------------------------------------------------

struct BatchVariant {
    std::shared_ptr<const CompileResult> result;
    const CompiledKernel* kernel = nullptr;
    std::shared_ptr<GpuProgram> program;
};

struct FnHandle {
    std::shared_ptr<const CompileResult> module_result;
    std::string fn_name;
    std::shared_ptr<const CompileResult> point_result;
    const CompiledKernel* point_kernel = nullptr;
    std::unordered_map<uint64_t, BatchVariant> batch_variants;
};

bool buildPointKernelForFn(FnHandle& h, std::string* err) {
    if (const CompiledKernel* existing = h.module_result->find_kernel(h.fn_name)) {
        if (existing->grid_dim == 0) {
            h.point_result = h.module_result;
            h.point_kernel = existing;
            return true;
        }
    }
    Compiler compiler;
    for (const auto& sp : h.module_result->search_paths) compiler.add_search_path(sp);
    for (const auto& [mn, ms] : h.module_result->loaded_modules) compiler.register_module(mn, ms);
    if (h.module_result->module_resolver) compiler.set_module_resolver(h.module_result->module_resolver);

    CompileOptions opts = h.module_result->options;
    opts.entry_name = h.fn_name;
    opts.emit_spirv = false;
    opts.compile_cpu = true;
    auto res = std::make_shared<CompileResult>(compiler.compile_source(h.module_result->source, h.module_result->file_name, opts));
    if (!res->ok) {
        if (err) *err = res->error_message();
        return false;
    }
    const CompiledKernel* pk = res->find_kernel(h.fn_name);
    if (!pk) {
        if (err) *err = "no function '" + h.fn_name + "'";
        return false;
    }
    h.point_result = std::move(res);
    h.point_kernel = pk;
    return true;
}

const BatchVariant* getOrBuildBatchVariant(FnHandle& h, uint64_t batch_mask, bool need_gpu, std::string* err) {
    const uint64_t key = batch_mask | (need_gpu ? (1ull << 63) : 0ull);
    auto it = h.batch_variants.find(key);
    if (it != h.batch_variants.end()) return &it->second;

    const CompiledKernel& pk = *h.point_kernel;
    const std::string wrapper_name = "__bro_fn_batch_" + h.fn_name + "_" + std::to_string(batch_mask);

    std::ostringstream src;
    src << h.module_result->source << "\n";
    if (!need_gpu && pk.uses_f64) {
        src << "@cpu ";
    }
    src << "@kernel @grid(1)\n";
    src << "fn " << wrapper_name << "(__g: grid<1>";

    size_t passed_idx = 0;
    for (const auto& p : pk.params) {
        if (!isPassed(p)) continue;
        const bool batched = ((batch_mask >> passed_idx) & 1ull) != 0;
        src << ", __p" << passed_idx << ": ";
        if (isStore(p)) {
            src << p.type.to_string();
        } else if (batched) {
            src << "read_store<" << p.type.to_string() << ", 1>";
        } else {
            src << p.type.to_string();
        }
        ++passed_idx;
    }
    src << ")";
    if (pk.has_return_value()) {
        src << " -> " << pk.return_type.to_string();
    }
    src << " {\n    ";
    if (pk.has_return_value()) src << "return ";
    src << h.fn_name << "(";
    passed_idx = 0;
    bool first = true;
    for (const auto& p : pk.params) {
        if (!isPassed(p)) continue;
        if (!first) src << ", ";
        first = false;
        const bool batched = ((batch_mask >> passed_idx) & 1ull) != 0;
        if (!isStore(p) && batched) {
            src << "__p" << passed_idx << ".load(__g.x())";
        } else {
            src << "__p" << passed_idx;
        }
        ++passed_idx;
    }
    src << ");\n}\n";

    Compiler compiler;
    for (const auto& sp : h.module_result->search_paths) compiler.add_search_path(sp);
    for (const auto& [mn, ms] : h.module_result->loaded_modules) compiler.register_module(mn, ms);
    if (h.module_result->module_resolver) compiler.set_module_resolver(h.module_result->module_resolver);

    CompileOptions opts = h.module_result->options;
    opts.entry_name = wrapper_name;
    opts.emit_spirv = need_gpu;
    opts.compile_cpu = !need_gpu;

    auto res = std::make_shared<CompileResult>(compiler.compile_source(src.str(), h.module_result->file_name, opts));
    if (!res->ok) {
        if (err) *err = res->error_message();
        return nullptr;
    }
    const CompiledKernel* bk = res->find_kernel(wrapper_name);
    if (!bk) {
        if (err) *err = "failed to compile batch wrapper for '" + h.fn_name + "'";
        return nullptr;
    }

    BatchVariant bv;
    bv.result = std::move(res);
    bv.kernel = bk;
    if (need_gpu && !bk->spirv.words.empty()) {
        if (GpuDevice* dev = currentDevice()) {
            bv.program = dev->createProgram(*bk, err);
            if (!bv.program) return nullptr;
        }
    }
    auto [ins_it, _] = h.batch_variants.emplace(key, std::move(bv));
    return &ins_it->second;
}

Value js_fnEval(Value self, std::span<const Value> args) {
    auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
    if (!h || !h->point_kernel) return ev::throwTypeError("KernelFn.eval: invalid receiver");
    const CompiledKernel& k = *h->point_kernel;
    if (!k.cpu_point_fn().is_valid()) {
        return ev::throwTypeError("KernelFn.eval: '" + h->fn_name + "' has no CPU build");
    }
    ArgPack pack(k, args, "KernelFn.eval");
    alignas(16) uint8_t ret[256] = {};
    std::string err;
    const bool ok = k.eval_cpu(pack.bind(args), k.return_byte_size <= sizeof ret ? ret : nullptr, &err);
    if (!ok) return ev::throwError("KernelFn.eval: " + err);
    if (!k.has_return_value() || k.return_byte_size > sizeof ret) return ev::undefined();
    return decodeLeaves(k, ret);
}

Value js_fnProbe(Value self, std::span<const Value> args) {
    auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
    if (!h || !h->point_kernel) return ev::throwTypeError("KernelFn.probe: invalid receiver");
    const CompiledKernel& k = *h->point_kernel;
    ArgPack pack(k, args, "KernelFn.probe");
    std::vector<ProbeSample> samples;
    std::string err;
    if (!k.probe_cpu(0, 0, 0, GridShape(1, 1, 1), pack.bind(args), samples, &err)) {
        return ev::throwError("KernelFn.probe: " + err);
    }
    return formatProbeSamples(samples);
}

Value js_fnRun(Value self, std::span<const Value> args) {
    auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
    if (!h || !h->point_kernel) return ev::throwTypeError("KernelFn.run: invalid receiver");
    if (args.empty()) return ev::throwTypeError("KernelFn.run: first argument must be the count");
    const GridShape shape = shapeArg(args[0], *h->point_kernel, "KernelFn.run");
    const std::span<const Value> fargs = args.subspan(1);
    const CompiledKernel& pk = *h->point_kernel;

    uint64_t batch_mask = 0;
    size_t passed_idx = 0;
    for (const auto& p : pk.params) {
        if (!isPassed(p)) continue;
        if (passed_idx < fargs.size() && !isStore(p)) {
            const Value v = fargs[passed_idx];
            if (ev::isTypedArray(v)) {
                const size_t param_bytes = p.mir_param_count * scalarBytes(p.type.scalar);
                const auto info = ev::typedArrayInfo(v);
                if (p.role == ParamRole::Scalar || shape.width == 1 || info.byteLength != param_bytes) {
                    batch_mask |= (1ull << passed_idx);
                }
            }
        }
        ++passed_idx;
    }

    std::string err;
    const BatchVariant* bv = getOrBuildBatchVariant(*h, batch_mask, /*need_gpu=*/false, &err);
    if (!bv || !bv->kernel) return ev::throwError("KernelFn.run: " + err);
    const CompiledKernel& k = *bv->kernel;
    ArgPack pack(k, fargs, "KernelFn.run");

    if (k.has_return_value()) {
        const auto& kinds = k.return_leaf_kinds;
        bool uniform = !kinds.empty();
        for (ScalarKind sk : kinds) uniform = uniform && sk == kinds[0];
        const auto kind = uniform ? elementKindOf(kinds[0]) : std::nullopt;
        const size_t cells = shape.total_points();
        const size_t count = kind ? cells * kinds.size() : cells * k.return_byte_size;
        if (count > 0xffffffffull) return ev::throwRangeError("KernelFn.run: the returns do not fit a typed array");
        HeldArgs held(fargs);
        ev::Persistent out(ev::createTypedArray(kind ? *kind : ev::elements::Uint8, static_cast<uint32_t>(count)));
        const std::vector<Value> current = held.current();
        std::span<const KernelArg> bound = pack.bind(current);
        const auto info = ev::typedArrayInfo(out.get());
        if (!k.dispatch_cpu(shape, bound, info.data, &err)) return ev::throwError("KernelFn.run: " + err);
        return out.get();
    }
    if (!k.dispatch_cpu(shape, pack.bind(fargs), nullptr, &err)) return ev::throwError("KernelFn.run: " + err);
    return ev::undefined();
}

Value js_fnDispatch(Value self, std::span<const Value> args) {
    auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
    if (!h || !h->point_kernel) return ev::throwTypeError("KernelFn.dispatch: invalid receiver");
    GpuDevice* dev = currentDevice();
    if (!dev) return ev::throwError("KernelFn.dispatch: this realm has no GPU device (bro.kernel.gpu is false)");
    if (args.empty()) return ev::throwTypeError("KernelFn.dispatch: first argument must be the count");
    const GridShape shape = shapeArg(args[0], *h->point_kernel, "KernelFn.dispatch");
    const std::span<const Value> fargs = args.subspan(1);
    const CompiledKernel& pk = *h->point_kernel;

    uint64_t batch_mask = 0;
    size_t passed_idx = 0;
    for (const auto& p : pk.params) {
        if (!isPassed(p)) continue;
        if (passed_idx < fargs.size() && !isStore(p)) {
            if (g_bufferClass.unwrap(fargs[passed_idx]) != nullptr) {
                batch_mask |= (1ull << passed_idx);
            }
        }
        ++passed_idx;
    }

    std::string err;
    const BatchVariant* bv = getOrBuildBatchVariant(*h, batch_mask, /*need_gpu=*/true, &err);
    if (!bv || !bv->kernel || !bv->program) return ev::throwError("KernelFn.dispatch: " + err);
    const CompiledKernel& k = *bv->kernel;
    ArgPack pack(k, fargs, "KernelFn.dispatch", /*gpu=*/true);

    std::shared_ptr<GpuBuffer> returns;
    if (const size_t bytes = k.gpu_return_bytes(shape)) {
        returns = allocateBuffer(bytes, &err);
        if (!returns) return ev::throwError("KernelFn.dispatch: " + err);
    }
    GpuLaunch launch;
    if (!k.gpu_launch(shape, pack.gpuArgs(), returns ? returns->address() : 0, launch, &err) ||
        !dev->dispatch(*bv->program, k, launch, &err)) {
        return ev::throwError("KernelFn.dispatch: " + err);
    }
    return returns ? makeBuffer(std::move(returns)) : ev::undefined();
}

Value js_fnKnobs(Value self, std::span<const Value> args) {
    auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
    if (!h || !h->module_result) return ev::throwTypeError("KernelFn.knobs: invalid receiver");
    if (args.empty()) return self;
    auto overrides = parseKnobOverrides(args[0], "KernelFn.knobs");
    auto next_res = std::make_shared<CompileResult>(h->module_result->with_knobs(overrides));
    if (!next_res->ok) throwCompileError(*next_res);
    h->module_result = std::move(next_res);
    h->batch_variants.clear();
    std::string err;
    if (!buildPointKernelForFn(*h, &err)) return ev::throwError("KernelFn.knobs: " + err);
    return self;
}

} // namespace

std::optional<std::string> resolveRegisteredModule(std::string_view name) {
    const std::string exact(name);
    const std::string norm = normalizeModuleKey(name);
    const std::string canon = canonicalModuleName(name);
    {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        if (auto it = g_registeredModules.find(exact); it != g_registeredModules.end()) return it->second;
        if (auto it = g_registeredModules.find(norm); it != g_registeredModules.end()) return it->second;
        if (auto it = g_registeredModules.find(canon); it != g_registeredModules.end()) return it->second;
    }

    if (t_hasModuleResolver && ev::isFunction(t_moduleResolver.get())) {
        ev::Persistent fn(t_moduleResolver.get());
        auto tryCall = [&](const std::string& argStr) -> std::optional<std::string> {
            ev::Persistent arg(ev::fromUtf8(argStr));
            const Value a = arg.get();
            auto res = ev::call(fn.get(), ev::undefined(), std::span<const Value>(&a, 1));
            if (res.thrown) ev::throwValue(res.value);
            if (ev::isString(res.value)) return ev::toUtf8(res.value);
            return std::nullopt;
        };
        auto cacheAndReturn = [&](std::string src) -> std::optional<std::string> {
            std::lock_guard<std::mutex> lock(g_registeredModulesMu);
            g_resolverFallback[exact] = src;
            g_resolverFallback[norm] = src;
            g_resolverFallback[canon] = src;
            return src;
        };
        if (auto r = tryCall(canon)) return cacheAndReturn(std::move(*r));
        if (norm != canon) {
            if (auto r = tryCall(norm)) return cacheAndReturn(std::move(*r));
        }
    } else {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        if (auto it = g_resolverFallback.find(exact); it != g_resolverFallback.end()) return it->second;
        if (auto it = g_resolverFallback.find(norm); it != g_resolverFallback.end()) return it->second;
        if (auto it = g_resolverFallback.find(canon); it != g_resolverFallback.end()) return it->second;
    }
    {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        if (auto it = g_hostModules.find(exact); it != g_hostModules.end()) return it->second;
        if (auto it = g_hostModules.find(norm); it != g_hostModules.end()) return it->second;
        if (auto it = g_hostModules.find(canon); it != g_hostModules.end()) return it->second;
    }
    return std::nullopt;
}

void registerHostModule(std::string_view name, std::string_view source) {
    const std::string src(source);
    std::lock_guard<std::mutex> lock(g_registeredModulesMu);
    g_hostModules[std::string(name)] = src;
    g_hostModules[normalizeModuleKey(name)] = src;
    g_hostModules[canonicalModuleName(name)] = src;
}

void decorateFnClass(ev::ObjectBuilder& b) {
    b.def("eval", 0, js_fnEval);
    b.def("probe", 0, js_fnProbe);
    b.def("run", 1, js_fnRun);
    b.def("dispatch", 1, js_fnDispatch);
    b.def("knobs", 1, js_fnKnobs);
    b.accessor("name", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
        return h ? ev::fromUtf8(h->fn_name) : ev::undefined();
    });
    b.accessor("returns", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<FnHandle*>(g_fnClass.unwrap(self));
        if (!h || !h->point_kernel) return ev::undefined();
        return h->point_kernel->has_return_value()
                   ? ev::fromUtf8(h->point_kernel->return_type.to_string())
                   : ev::null();
    });
}

void decorateKernelExtras(ev::ObjectBuilder& b) {
    b.def("probe", 0, js_kernelProbe);
    b.def("knobs", 1, js_kernelKnobs);
    b.def("stats", 0, js_kernelStats);
}

void decorateModuleExtras(ev::ObjectBuilder& b) {
    b.accessor("consts", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->result) return ev::undefined();
        ev::ObjectBuilder obj;
        for (const auto& c : h->result->constants) {
            ev::Persistent val(constValueToJs(c.value));
            obj.set(c.name, val.get());
        }
        return obj.get();
    });
    b.accessor("knobs", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->result) return ev::undefined();
        ev::ObjectBuilder obj;
        for (const auto& c : h->result->constants) {
            if (!c.is_knob) continue;
            ev::Persistent val(constValueToJs(c.value));
            obj.set(c.name, val.get());
        }
        return obj.get();
    });
    b.def("withKnobs", 1, [](Value self, std::span<const Value> args) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->result) return ev::throwTypeError("KernelModule.withKnobs: invalid receiver");
        std::unordered_map<std::string, ConstValue> overrides;
        if (!args.empty() && !ev::isUndefined(args[0]) && !ev::isNull(args[0])) {
            overrides = parseKnobOverrides(args[0], "KernelModule.withKnobs");
        }
        auto next_res = std::make_shared<CompileResult>(h->result->with_knobs(overrides));
        if (!next_res->ok) throwCompileError(*next_res);
        auto next_handle = std::make_unique<ModuleHandle>();
        next_handle->result = std::move(next_res);
        return g_moduleClass.createInstance(std::move(next_handle));
    });
    b.def("structLayout", 1, [](Value self, std::span<const Value> args) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->result) return ev::throwTypeError("KernelModule.structLayout: invalid receiver");
        if (args.empty() || !ev::isString(args[0])) {
            return ev::throwTypeError("KernelModule.structLayout: requires a struct name string");
        }
        const std::string name = ev::toUtf8(args[0]);
        const TypeRegistry& types = h->result->types;
        const StructDef* sdef = types.find_struct(name);
        if (!sdef) {
            for (const auto& st : types.structs()) {
                auto pos = st.name.rfind("::");
                if (pos != std::string::npos && st.name.substr(pos + 2) == name) {
                    sdef = &st;
                    break;
                }
            }
        }
        if (!sdef) return ev::null();
        const Type st_type = Type::make_struct(sdef->id, sdef->name);
        const auto l140 = types.block_layout(st_type, true);
        const auto l430 = types.block_layout(st_type, false);

        ev::Persistent fieldsArr(ev::makeArray(static_cast<uint32_t>(sdef->fields.size())));
        uint32_t byte_off = 0;
        for (size_t i = 0; i < sdef->fields.size(); ++i) {
            const auto& f = sdef->fields[i];
            ev::ObjectBuilder fb;
            fb.set("name", f.name);
            fb.set("type", f.type.to_string());
            fb.set("byteOffset", static_cast<double>(byte_off));
            fb.set("std140Offset", static_cast<double>(types.block_field_offset(*sdef, f.name, true)));
            fb.set("std430Offset", static_cast<double>(types.block_field_offset(*sdef, f.name, false)));
            ev::Persistent fv(fb.get());
            fieldsArr.set(ev::setElement(fieldsArr.get(), static_cast<uint32_t>(i), fv.get()));
            byte_off += types.byte_size(f.type);
        }

        ev::ObjectBuilder out;
        out.set("name", name);
        out.set("byteSize", static_cast<double>(types.byte_size(st_type)));
        out.set("std140Size", static_cast<double>(l140.size));
        out.set("std140Align", static_cast<double>(l140.align));
        out.set("std430Size", static_cast<double>(l430.size));
        out.set("std430Align", static_cast<double>(l430.align));
        out.set("fields", fieldsArr.get());
        return out.get();
    });
    b.accessor("functions", [](Value self, std::span<const Value>) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->result) return ev::undefined();
        const auto& fns = h->result->exported_functions;
        ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(fns.size())));
        for (size_t i = 0; i < fns.size(); ++i) {
            ev::Persistent s(ev::fromUtf8(fns[i]));
            arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), s.get()));
        }
        return arr.get();
    });
    b.def("fn", 1, [](Value self, std::span<const Value> args) -> Value {
        auto* h = static_cast<ModuleHandle*>(g_moduleClass.unwrap(self));
        if (!h || !h->result) return ev::throwTypeError("KernelModule.fn: invalid receiver");
        if (args.empty() || !ev::isString(args[0])) {
            return ev::throwTypeError("KernelModule.fn: requires a function name string");
        }
        auto fn_handle = std::make_unique<FnHandle>();
        fn_handle->module_result = h->result;
        fn_handle->fn_name = ev::toUtf8(args[0]);
        std::string err;
        if (!buildPointKernelForFn(*fn_handle, &err)) {
            return ev::throwError("KernelModule.fn: " + err);
        }
        return g_fnClass.createInstance(std::move(fn_handle));
    });
}

void decorateBroKernelExtras(ev::ObjectBuilder& b) {
    b.def("resolve", 1, [](Value, std::span<const Value> args) -> Value {
        if (args.empty() || ev::isNull(args[0]) || ev::isUndefined(args[0])) {
            t_moduleResolver.set(ev::undefined());
            t_hasModuleResolver = false;
            return ev::undefined();
        }
        if (!ev::isFunction(args[0])) {
            return ev::throwTypeError("bro.kernel.resolve: expected a resolver function or null");
        }
        t_moduleResolver.set(args[0]);
        t_hasModuleResolver = true;
        return ev::undefined();
    });
    b.def("registerModule", 2, [](Value, std::span<const Value> args) -> Value {
        if (args.size() < 2 || !ev::isString(args[0]) || !ev::isString(args[1])) {
            return ev::throwTypeError("bro.kernel.registerModule: expected (name: string, source: string)");
        }
        const std::string name = ev::toUtf8(args[0]);
        const std::string source = ev::toUtf8(args[1]);
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        g_registeredModules[name] = source;
        g_registeredModules[normalizeModuleKey(name)] = source;
        g_registeredModules[canonicalModuleName(name)] = source;
        return ev::undefined();
    });
    b.def("clearModules", 0, [](Value, std::span<const Value>) -> Value {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        g_registeredModules.clear();
        return ev::undefined();
    });
}

void configureCompilerModulesAndOptions(
    std::span<const Value> args,
    Compiler& compiler,
    std::string& file,
    CompileOptions& compile_opts) {
    {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        for (const auto& [name, src] : g_registeredModules) {
            compiler.register_module(name, src);
        }
    }
    compiler.set_module_resolver([](std::string_view name) {
        return resolveRegisteredModule(name);
    });

    if (args.size() > 1 && ev::isObject(args[1])) {
        ev::Persistent opts(args[1]);
        const Value f = ev::getProperty(opts.get(), "file");
        if (ev::isString(f)) file = std::filesystem::path(ev::toUtf8(f)).filename().string();

        ev::Persistent mods(ev::getProperty(opts.get(), "modules"));
        if (ev::isObject(mods.get())) {
            auto object = ev::globalValue("Object");
            if (!object.found) ev::throwError("bro.kernel.compile: no Object global");
            ev::Persistent keysFn(ev::getProperty(object.value, "keys"));
            const Value marg = mods.get();
            auto keys = ev::call(keysFn.get(), ev::undefined(), std::span<const Value>(&marg, 1));
            if (keys.thrown) ev::throwValue(keys.value);
            ev::Persistent names(keys.value);
            const uint32_t n = static_cast<uint32_t>(ev::toDouble(ev::getProperty(names.get(), "length")));
            for (uint32_t i = 0; i < n; ++i) {
                const std::string name = ev::toUtf8(ev::getElement(names.get(), i));
                const Value src = ev::getProperty(mods.get(), name);
                if (!ev::isString(src)) ev::throwTypeError("bro.kernel.compile: module '" + name + "' is not a string");
                compiler.register_module(name, ev::toUtf8(src));
            }
        }

        const Value knobs = ev::getProperty(opts.get(), "knobs");
        if (ev::isObject(knobs)) {
            compile_opts.knob_overrides = parseKnobOverrides(knobs, "bro.kernel.compile");
        }
        const Value ep = ev::getProperty(opts.get(), "enableProbes");
        if (ev::isBool(ep)) compile_opts.enable_probes = ev::toBool(ep);
        const Value efp = ev::getProperty(opts.get(), "enableFragmentProbes");
        if (ev::isBool(efp)) compile_opts.enable_fragment_probes = ev::toBool(efp);
    }
}

} // namespace brokernel::api
