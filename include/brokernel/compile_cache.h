#pragma once
// The on-disk compile cache: Compiler::compile_source loads a result it has
// compiled before (in this process or another) instead of compiling it again.
//
// Off until set_compile_cache names a directory. The directory is a brass
// artifact cache (brass docs/artifact_cache.md): one `<32 hex digits>.bac`
// file per entry, written to a temporary file and renamed, checked on every
// read against both digests of its key and a checksum, so compiles on any
// number of threads and processes share a directory and a missing, torn,
// damaged or foreign file is a miss. A hit marks its entry used; stores past
// max_bytes (whole 4 KiB units) have a background thread delete the least
// recently used entries. Deleting the directory is always safe. Files of the
// cache brokernel kept before (`*.bkc`) are never read, and are deleted when
// a cache is set on their directory.
//
// An entry is keyed by brass's build (KeyBuilder::add_brass_build: its
// version, its binary, the target and the CPU features codegen reads),
// brokernel's version and binary (or build_stamp), the source, its file name,
// every CompileOptions field (knob overrides in any order) and the modules
// given by name. It records each module the compile read otherwise (as it was
// asked for, where it resolved and a hash of its source), and a lookup
// resolves each again as the compiler would now (registered modules,
// resolver, search paths; nothing is parsed): a changed or missing module is
// a miss. Only successful compiles are stored.
//
// A hit is the compile's result, every field: what the compile produced from
// the entry, and source, file name, options, loaded_modules, imports,
// search_paths and module_resolver from the Compiler, as a compile takes
// them. A probed variant is compiled as a compile's is, when first asked for.
//
// A kernel's CPU build, made on its first CPU run whether the compile hit or
// missed, is kept in the same directory as native code (brass's
// NativeModuleCache): a later run of that kernel, in this process or
// another, loads it instead of compiling. Only when it is not there does a
// hit's kernel compile the source again for its CPU build (once: the code is
// stored then). A CPU build that cannot be stored as an image (one calling
// a function it does not define) is compiled in memory as without a cache.

#include <cstdint>
#include <filesystem>
#include <string>

namespace brokernel {

struct CompileCacheConfig {
    std::filesystem::path dir;              // empty: no cache
    uint64_t max_bytes = 512ull << 20;      // 0: no bound
    // What identifies brokernel's build in the key, beside brass's own.
    // Empty: the binary that holds brokernel's code (its path, size and
    // mtime), which covers the built-in modules linked into it.
    std::string build_stamp;
};

/// Sets the process-wide cache (thread-safe; compiles already running, and
/// the CPU builds of results already made, keep the cache they started with).
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
    // CPU builds of kernels compiled through a cache: loaded as stored
    // native code, compiled and stored, or compiled in memory (not storable).
    uint64_t cpu_loads = 0;
    uint64_t cpu_stores = 0;
    uint64_t cpu_in_memory = 0;
};
CompileCacheStats compile_cache_stats();

} // namespace brokernel
