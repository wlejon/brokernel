// bro.kernel.reload: new sources for modules already compiled, swapped into
// every live module that reads them.
#include "api_internal.h"

#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace brokernel::api {

namespace {

// { name: source } from a JS object of strings.
std::unordered_map<std::string, std::string> parseSources(Value obj, const std::string& what) {
    std::unordered_map<std::string, std::string> out;
    if (!ev::isObject(obj)) ev::throwTypeError(what + ": expected an object of { name: source }");
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
        if (!ev::isString(v)) ev::throwTypeError(what + ": the source of '" + k + "' is not a string");
        out[k] = ev::toUtf8(v);
    }
    return out;
}

// Whether two compiles of a kernel are the same code: the same SPIR-V, or
// for a kernel with no GPU build, the same lowered CPU functions.
bool sameCode(const CompiledKernel& a, const CompiledKernel& b) {
    if (!a.spirv.words.empty() || !b.spirv.words.empty()) return a.spirv.words == b.spirv.words;
    return a.code_hash == b.code_hash;
}

Value stringArray(const std::vector<std::string>& v) {
    ev::Persistent arr(ev::makeArray(static_cast<uint32_t>(v.size())));
    for (size_t i = 0; i < v.size(); ++i) {
        ev::Persistent s(ev::fromUtf8(v[i]));
        arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), s.get()));
    }
    return arr.get();
}

// bro.kernel.reload({ name: source }): every live module whose file is one
// of the names, or that imported one, recompiles with the new sources: all
// of them or none (the first error throws and nothing changes). Its kernels
// follow at their next use. A registered or resolved module of a name takes
// the new source too, so later compiles see it. Returns what recompiled,
// [{ file, kernels }], `kernels` the names whose code changed.
Value js_reload(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("bro.kernel.reload: expected { name: source }");
    const auto sources = parseSources(args[0], "bro.kernel.reload");
    struct Next {
        std::shared_ptr<ModuleCell> cell;
        std::shared_ptr<const CompileResult> result;
    };
    std::vector<Next> next;
    for (auto& cell : liveModuleCells()) {
        const CompileResult& r = *cell->result;
        bool touched = false;
        for (const auto& [k, _] : sources) touched = touched || r.depends_on(k);
        if (touched) next.push_back({std::move(cell), nullptr});
    }
    // The modules compile together, a thread each, from the sources they
    // hold (the resolver runs JS, on this thread alone); one that needs a
    // module it did not hold (a new import) compiles again here with it.
    {
        std::vector<std::thread> threads;
        std::vector<std::shared_ptr<CompileResult>> results(next.size());
        for (size_t i = 0; i < next.size(); ++i) {
            threads.emplace_back([&, i] {
                try {
                    results[i] = std::make_shared<CompileResult>(next[i].cell->result->with_sources(sources, false));
                } catch (...) {
                    results[i] = std::make_shared<CompileResult>();   // not ok: compiled again on this thread
                }
            });
        }
        for (auto& t : threads) t.join();
        for (size_t i = 0; i < next.size(); ++i) {
            if (!results[i]->ok) results[i] = std::make_shared<CompileResult>(next[i].cell->result->with_sources(sources));
            if (!results[i]->ok) throwCompileError(*results[i]);
            next[i].result = std::move(results[i]);
        }
    }
    updateRegisteredSources(sources);

    ev::Persistent out(ev::makeArray(static_cast<uint32_t>(next.size())));
    for (size_t i = 0; i < next.size(); ++i) {
        const CompileResult& old = *next[i].cell->result;
        std::vector<std::string> changed;
        for (const auto& k : next[i].result->kernels) {
            const CompiledKernel* was = old.find_kernel(k.name);
            if (!was || !sameCode(*was, k)) changed.push_back(k.name);
        }
        releaseLater(std::exchange(next[i].cell->result, next[i].result));
        ++next[i].cell->generation;
        ev::Persistent names(stringArray(changed));
        ev::ObjectBuilder entry;
        entry.set("file", next[i].result->file_name);
        entry.set("kernels", names.get());
        ev::Persistent e(entry.get());
        out.set(ev::setElement(out.get(), static_cast<uint32_t>(i), e.get()));
    }
    return out.get();
}

} // namespace

void decorateReload(ev::ObjectBuilder& b) {
    b.def("reload", 1, js_reload);
}

} // namespace brokernel::api
