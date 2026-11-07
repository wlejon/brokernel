// Modules by name for bro.kernel's compiles: registered ones
// (bro.kernel.registerModule), the JS resolver (bro.kernel.resolve) and what it
// answered, and the host's built-in modules (registerHostModule). On a compile
// worker (api_async.cpp) a name the tables do not hold is asked of the thread
// that owns the JS resolver.
#include "api_internal.h"

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

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

struct Keys {
    std::string exact, norm, canon;
    explicit Keys(std::string_view name) : exact(name), norm(normalizeModuleKey(name)), canon(canonicalModuleName(name)) {}
};

// `table`'s source under any spelling of the name (the lock held).
std::optional<std::string> lookup(const std::unordered_map<std::string, std::string>& table, const Keys& k) {
    for (const std::string* key : {&k.exact, &k.norm, &k.canon}) {
        if (auto it = table.find(*key); it != table.end()) return it->second;
    }
    return std::nullopt;
}

} // namespace

bool threadHasJsResolver() {
    return t_hasModuleResolver && ev::isFunction(t_moduleResolver.get());
}

std::optional<std::string> resolveFromTables(std::string_view name, bool withHostModules) {
    const Keys k(name);
    std::lock_guard<std::mutex> lock(g_registeredModulesMu);
    if (auto s = lookup(g_registeredModules, k)) return s;
    if (auto s = lookup(g_resolverFallback, k)) return s;
    if (withHostModules) return lookup(g_hostModules, k);
    return std::nullopt;
}

std::optional<std::string> resolveWithJs(std::string_view name, ev::Persistent& thrown, bool& threw) {
    threw = false;
    const Keys k(name);
    {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        if (auto s = lookup(g_registeredModules, k)) return s;
    }
    if (threadHasJsResolver()) {
        ev::Persistent fn(t_moduleResolver.get());
        auto tryCall = [&](const std::string& argStr) -> std::optional<std::string> {
            ev::Persistent arg(ev::fromUtf8(argStr));
            const Value a = arg.get();
            auto res = ev::call(fn.get(), ev::undefined(), std::span<const Value>(&a, 1));
            if (res.thrown) {
                thrown.set(res.value);
                threw = true;
                return std::nullopt;
            }
            if (ev::isString(res.value)) return ev::toUtf8(res.value);
            return std::nullopt;
        };
        auto cacheAndReturn = [&](std::string src) -> std::optional<std::string> {
            std::lock_guard<std::mutex> lock(g_registeredModulesMu);
            g_resolverFallback[k.exact] = src;
            g_resolverFallback[k.norm] = src;
            g_resolverFallback[k.canon] = src;
            return src;
        };
        if (auto r = tryCall(k.canon)) return cacheAndReturn(std::move(*r));
        if (threw) return std::nullopt;
        if (k.norm != k.canon) {
            if (auto r = tryCall(k.norm)) return cacheAndReturn(std::move(*r));
            if (threw) return std::nullopt;
        }
    } else {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        if (auto s = lookup(g_resolverFallback, k)) return s;
    }
    std::lock_guard<std::mutex> lock(g_registeredModulesMu);
    return lookup(g_hostModules, k);
}

std::optional<std::string> resolveRegisteredModule(std::string_view name) {
    if (onCompileWorker()) return workerResolve(name);
    ev::Persistent thrown;
    bool threw = false;
    auto r = resolveWithJs(name, thrown, threw);
    if (threw) ev::throwValue(thrown.get());
    return r;
}

void updateRegisteredSources(const std::unordered_map<std::string, std::string>& sources) {
    std::lock_guard<std::mutex> lock(g_registeredModulesMu);
    for (const auto& [name, src] : sources) {
        const Keys k(name);
        for (auto* table : {&g_registeredModules, &g_resolverFallback}) {
            for (const std::string* key : {&k.exact, &k.norm, &k.canon}) {
                if (auto it = table->find(*key); it != table->end()) it->second = src;
            }
        }
    }
}

void registerHostModule(std::string_view name, std::string_view source) {
    const std::string src(source);
    const Keys k(name);
    std::lock_guard<std::mutex> lock(g_registeredModulesMu);
    g_hostModules[k.exact] = src;
    g_hostModules[k.norm] = src;
    g_hostModules[k.canon] = src;
}

void registerModulesInto(Compiler& compiler) {
    std::lock_guard<std::mutex> lock(g_registeredModulesMu);
    for (const auto& [name, src] : g_registeredModules) compiler.register_module(name, src);
}

void decorateModuleRegistry(ev::ObjectBuilder& b) {
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
        const std::string source = ev::toUtf8(args[1]);
        const Keys k(ev::toUtf8(args[0]));
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        g_registeredModules[k.exact] = source;
        g_registeredModules[k.norm] = source;
        g_registeredModules[k.canon] = source;
        return ev::undefined();
    });
    b.def("clearModules", 0, [](Value, std::span<const Value>) -> Value {
        std::lock_guard<std::mutex> lock(g_registeredModulesMu);
        g_registeredModules.clear();
        return ev::undefined();
    });
}

} // namespace brokernel::api
