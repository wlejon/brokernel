#pragma once
// Module registry and standard library modules for brokernel:
//   - Built-in modules: `math` (`std::math`), `noise` (`std::noise`),
//     `filter` (`std::filter`), `sphere` (`std::sphere`), `color` (`std::color`)
//   - In-memory module registration and filesystem search paths (.bk files)
//   - Parsed AST caching and import cycle detection

#include "brokernel/ast.h"
#include "brokernel/diagnostics.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brokernel {

class ModuleRegistry {
public:
    ModuleRegistry();

    /// Registers a built-in or user module by canonical name (e.g. "noise" or "world::climate").
    void register_module(std::string name, std::string source, std::string virtual_path = {});

    /// Adds a directory to search when resolving `import foo;` or `import "foo.bk";`.
    void add_search_path(std::filesystem::path dir);

    /// Returns true if a module with the given name or path is registered or on disk.
    bool has_module(std::string_view name) const;

    /// Parses and caches a module by canonical name or relative path.
    /// Reports diagnostics to `diags` if the module is missing or has syntax errors.
    const ModuleAst* load_module(
        std::string_view name_or_path,
        const SourceSpan& import_site,
        DiagnosticBag& diags,
        const std::filesystem::path& importing_file_dir = {});

    /// Formats a module path vector like `{"std", "noise"}` to `"std::noise"`.
    static std::string format_module_path(const std::vector<std::string>& path);

private:
    struct Entry {
        std::string name;
        std::string path;
        std::string source;
        std::unique_ptr<ModuleAst> ast;
        bool loading = false;
        bool failed = false;
    };

    std::unordered_map<std::string, Entry> modules_;
    std::unordered_map<std::string, std::string> aliases_; // e.g. "std::noise" -> "noise"
    std::vector<std::filesystem::path> search_paths_;

    void register_builtin_modules();
    Entry* find_or_discover(std::string_view key, const std::filesystem::path& importing_dir);
};

} // namespace brokernel
