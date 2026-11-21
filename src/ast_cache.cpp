#include "ast_cache.h"

#include <brass/core/hash128.hpp>

#include <atomic>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace brokernel::detail {

namespace {

constexpr size_t kMaxAstCacheEntries = 512;
std::atomic<bool> g_ast_cache_enabled{true};

struct AstCacheKey {
    std::string file_name;
    std::string fallback_module_name;
    uint64_t hash_lo = 0;
    uint64_t hash_hi = 0;

    bool operator==(const AstCacheKey& o) const noexcept {
        return hash_lo == o.hash_lo && hash_hi == o.hash_hi &&
               file_name == o.file_name && fallback_module_name == o.fallback_module_name;
    }
};

struct AstCacheKeyHash {
    size_t operator()(const AstCacheKey& k) const noexcept {
        uint64_t h = k.hash_lo ^ (k.hash_hi * 0x9e3779b97f4a7c15ULL);
        for (unsigned char c : k.file_name) h = (h * 1099511628211ULL) ^ c;
        h ^= 0xff51afd7ed558ccdULL;
        for (unsigned char c : k.fallback_module_name) h = (h * 1099511628211ULL) ^ c;
        return static_cast<size_t>(h);
    }
};

struct GlobalAstCache {
    std::mutex mu;
    std::unordered_map<AstCacheKey, std::shared_ptr<const ModuleAst>, AstCacheKeyHash> map;
    std::deque<AstCacheKey> fifo;
    uint64_t hits = 0;
    uint64_t misses = 0;
};

GlobalAstCache& global_ast_cache() {
    static auto* cache = new GlobalAstCache();
    return *cache;
}

} // namespace

void set_ast_cache_enabled(bool enabled) {
    g_ast_cache_enabled.store(enabled, std::memory_order_release);
}

bool ast_cache_enabled() {
    return g_ast_cache_enabled.load(std::memory_order_acquire);
}

std::shared_ptr<const ModuleAst> parse_source_cached(
    std::string_view source,
    const std::string& file_name,
    DiagnosticBag& diags,
    std::string fallback_module_name) {
    const bool use_cache = g_ast_cache_enabled.load(std::memory_order_acquire);
    AstCacheKey key;
    if (use_cache) {
        const brass::Digest128 digest = brass::hash128(source.data(), source.size());
        key = AstCacheKey{file_name, fallback_module_name, digest.lo, digest.hi};
        auto& c = global_ast_cache();
        std::lock_guard<std::mutex> lock(c.mu);
        auto it = c.map.find(key);
        if (it != c.map.end()) {
            ++c.hits;
            return it->second;
        }
        ++c.misses;
    }

    DiagnosticBag local_diags;
    std::unique_ptr<ModuleAst> parsed = parse_source(source, file_name, local_diags);
    if (local_diags.has_errors() || !parsed) {
        diags.merge_from(local_diags);
        return nullptr;
    }
    diags.merge_from(local_diags);
    if (parsed->module_name.empty() && !fallback_module_name.empty()) {
        parsed->module_name = std::move(fallback_module_name);
    }

    std::shared_ptr<const ModuleAst> shared = std::move(parsed);
    if (use_cache) {
        auto& c = global_ast_cache();
        std::lock_guard<std::mutex> lock(c.mu);
        auto [it, inserted] = c.map.emplace(std::move(key), shared);
        if (inserted) {
            c.fifo.push_back(it->first);
            while (c.fifo.size() > kMaxAstCacheEntries) {
                c.map.erase(c.fifo.front());
                c.fifo.pop_front();
            }
        } else {
            shared = it->second;
        }
    }
    return shared;
}

AstCacheStats ast_cache_stats() {
    auto& c = global_ast_cache();
    std::lock_guard<std::mutex> lock(c.mu);
    return AstCacheStats{c.hits, c.misses, c.map.size()};
}

void clear_ast_cache() {
    auto& c = global_ast_cache();
    std::lock_guard<std::mutex> lock(c.mu);
    c.map.clear();
    c.fifo.clear();
    c.hits = 0;
    c.misses = 0;
}

} // namespace brokernel::detail
