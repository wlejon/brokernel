#pragma once
// Process-wide thread-safe parsed AST cache for brokernel.
#include "brokernel/ast.h"
#include "brokernel/sema.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace brokernel::detail {

struct AstCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    size_t entries = 0;
};

/// Controls whether `parse_source_cached` consults/populates the AST cache.
void set_ast_cache_enabled(bool enabled);
bool ast_cache_enabled();

/// Returns a cached immutable `ModuleAst` matching `(file_name, fallback_module_name, hash128(source))`,
/// or lexes + parses `source` and caches the result on success (no parse errors).
std::shared_ptr<const ModuleAst> parse_source_cached(
    std::string_view source,
    const std::string& file_name,
    DiagnosticBag& diags,
    std::string fallback_module_name = {});

/// Caches evaluated non-knob module constants keyed by immutable `ModuleAst*`.
bool lookup_cached_module_consts(const ModuleAst* ast, std::vector<std::pair<std::string, ConstValue>>& out);
void store_cached_module_consts(const ModuleAst* ast, std::vector<std::pair<std::string, ConstValue>> consts);

AstCacheStats ast_cache_stats();
void clear_ast_cache();

} // namespace brokernel::detail
