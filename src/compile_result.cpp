// A compile made again from what it was compiled from: with other knobs, or
// with new sources for some of its modules.
#include "brokernel/brokernel.h"

#include <algorithm>

namespace brokernel {

CompileResult CompileResult::with_knobs(const std::unordered_map<std::string, ConstValue>& overrides) const {
    Compiler compiler;
    for (const auto& sp : search_paths) compiler.add_search_path(sp);
    for (const auto& [mname, msrc] : loaded_modules) compiler.register_module(mname, msrc);
    if (module_resolver) compiler.set_module_resolver(module_resolver);
    CompileOptions new_opts = options;
    for (const auto& [k, v] : overrides) {
        new_opts.knob_overrides[k] = v;
    }
    return compiler.compile_source(source, file_name, new_opts);
}

bool CompileResult::depends_on(const std::string& key) const {
    return key == file_name || std::binary_search(imports.begin(), imports.end(), key);
}

CompileResult CompileResult::with_sources(const std::unordered_map<std::string, std::string>& sources,
                                          bool use_resolver) const {
    Compiler compiler;
    for (const auto& sp : search_paths) compiler.add_search_path(sp);
    for (const auto& [mname, msrc] : loaded_modules) {
        auto it = sources.find(mname);
        compiler.register_module(mname, it != sources.end() ? it->second : msrc);
    }
    // A changed module this one did not hold yet (a new import of a changed
    // module): registered too, so it wins over the resolver.
    for (const auto& [mname, msrc] : sources) {
        if (mname != file_name && !loaded_modules.count(mname)) compiler.register_module(mname, msrc);
    }
    compiler.set_module_resolver(use_resolver ? module_resolver : ModuleRegistry::ModuleResolver{});
    auto root = sources.find(file_name);
    return compiler.compile_source(root != sources.end() ? root->second : source, file_name, options);
}

} // namespace brokernel
