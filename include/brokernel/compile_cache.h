#pragma once
// The on-disk compile cache: Compiler::compile_source loads a result it has
// compiled before (in this process or another) instead of compiling it again.
//
// Off until set_compile_cache names a directory. An entry is keyed by the
// compiler build, brokernel's version, the source, its file name and every
// CompileOptions field (knob overrides in any order); it records each module
// the compile read (as it was asked for, where it resolved and a hash of its
// source), and a lookup resolves each again as the compiler would now
// (registered modules, resolver, search paths; nothing is parsed): a changed
// or missing module is a miss. Only successful compiles are stored.
//
// A hit is the compile's result, every field: what the compile produced from
// the entry, and source, file name, options, loaded_modules, imports,
// search_paths and module_resolver from the Compiler, as a compile takes
// them. A kernel's CPU build is lowered on its first CPU run by compiling
// the source again for that kernel alone (a @cpu kernel's included, which a
// compile lowers at once), and a probed variant is compiled as a compile's
// is, when first asked for.
//
// One file per entry, `<32 hex digits>.bkc`, written to a temporary file and
// renamed, so compiles on any number of threads and processes share a
// directory; a truncated, corrupt or foreign entry is a miss. A hit touches
// its entry's mtime; a store that takes the directory past max_bytes (whole
// 4 KiB units) has a background thread delete the least recently used
// entries down to 7/8 of it. Deleting the directory is always safe.

#include <cstdint>
#include <filesystem>
#include <string>

namespace brokernel {

struct CompileCacheConfig {
    std::filesystem::path dir;              // empty: no cache
    uint64_t max_bytes = 512ull << 20;
    // What identifies the compiler build in the key. Empty: the binary that
    // holds brokernel's code (its path, size and mtime), which covers brass
    // and the built-in modules linked into it.
    std::string build_stamp;
};

/// Sets the process-wide cache (thread-safe; compiles already running keep
/// the configuration they started with).
void set_compile_cache(CompileCacheConfig config);
CompileCacheConfig compile_cache_config();

/// Waits for a trim of the directory in progress (or asked for) to finish.
void flush_compile_cache();

/// Whether a compile came from the cache (CompileResult::cache_status).
enum class CacheStatus : uint8_t {
    Off = 0,    // no cache configured, or the Compiler's use of it is off
    Hit = 1,
    Miss = 2,   // compiled; CompileResult::cache_note says why it missed
};

/// Counts since the process started.
struct CompileCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t stores = 0;
    uint64_t store_failures = 0;
};
CompileCacheStats compile_cache_stats();

} // namespace brokernel
