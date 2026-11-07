#pragma once
// The compile cache's parts: hashing, the byte encoding of an entry, and the
// directory (files, trims). compile_cache.cpp puts them together around
// Compiler::compile_source.
#include "brokernel/brokernel.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace brokernel::detail {

struct Digest {
    uint64_t lo = 0, hi = 0;
    std::string hex() const;   // 32 hex digits
    bool operator==(const Digest& o) const noexcept { return lo == o.lo && hi == o.hi; }
};
Digest hash128(std::string_view bytes, uint64_t seed = 0x626b63616368ull);

// Little-endian fields appended to a string.
class ByteWriter {
public:
    explicit ByteWriter(std::string& out) : out_(out) {}
    void u8(uint8_t v) { out_.push_back(static_cast<char>(v)); }
    void boolean(bool v) { u8(v ? 1 : 0); }
    void u32(uint32_t v) { raw(&v, 4); }
    void i32(int32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    void f64(double v) { raw(&v, 8); }
    void str(std::string_view s) { u64(s.size()); out_.append(s.data(), s.size()); }
    void raw(const void* p, size_t n) { out_.append(static_cast<const char*>(p), n); }

private:
    std::string& out_;
};

// Reads what ByteWriter wrote. Reading past the end, or a count longer than
// what is left, marks it bad and yields zeros: a damaged entry decodes to
// garbage that ok() rejects, never out of bounds.
class ByteReader {
public:
    ByteReader(const char* p, size_t n) : p_(p), end_(p + n) {}
    bool ok() const noexcept { return ok_; }
    bool at_end() const noexcept { return p_ == end_; }
    void fail() noexcept { ok_ = false; }
    uint8_t u8() { uint8_t v = 0; raw(&v, 1); return v; }
    bool boolean() { return u8() != 0; }
    uint32_t u32() { uint32_t v = 0; raw(&v, 4); return v; }
    int32_t i32() { int32_t v = 0; raw(&v, 4); return v; }
    uint64_t u64() { uint64_t v = 0; raw(&v, 8); return v; }
    int64_t i64() { int64_t v = 0; raw(&v, 8); return v; }
    double f64() { double v = 0; raw(&v, 8); return v; }
    std::string str() {
        const uint64_t n = u64();
        if (!ok_ || n > left()) { ok_ = false; return {}; }
        std::string s(p_, static_cast<size_t>(n));
        p_ += n;
        return s;
    }
    // An element count, each element at least `min_bytes` long.
    size_t count(size_t min_bytes = 1) {
        const uint64_t n = u64();
        if (!ok_ || n > left() / (min_bytes ? min_bytes : 1)) { ok_ = false; return 0; }
        return static_cast<size_t>(n);
    }
    void raw(void* dst, size_t n) {
        if (!ok_ || n > left()) { ok_ = false; std::memset(dst, 0, n); return; }
        std::memcpy(dst, p_, n);
        p_ += n;
    }

private:
    size_t left() const noexcept { return static_cast<size_t>(end_ - p_); }
    const char* p_;
    const char* end_;
    bool ok_ = true;
};

// A module a compile read: the import as load_module was asked for it, the
// module it resolved to, where (its registry path), and its source's hash.
struct ModuleRecord {
    std::string name;
    std::string importing_dir;
    std::string key;
    std::string path;
    Digest source;
};
// Every CompileOptions field, knob overrides by name.
void encode_options(const CompileOptions& o, ByteWriter& w);

void encode_modules(const std::vector<ModuleRecord>& mods, ByteWriter& w);
bool decode_modules(ByteReader& r, std::vector<ModuleRecord>& mods);

// What a compile produced (CompileResult less what a Compiler supplies: the
// source, file name, options, modules, search paths and resolver), with each
// kernel's CPU build reduced to whether it had one (`had_cpu`).
void encode_result(const CompileResult& res, ByteWriter& w);
bool decode_result(ByteReader& r, CompileResult& res, std::vector<uint8_t>& had_cpu);

// The binary holding brokernel's code: "path|size|mtime", or empty.
std::string build_identity();

// One entry file: a header (magic, format version, key, payload size and
// checksum) and the payload. read_entry fails with `why` for a missing,
// short, foreign or damaged file.
bool read_entry(const std::filesystem::path& file, const Digest& key, std::string& payload, std::string& why);
bool write_entry(const std::filesystem::path& file, const Digest& key, std::string_view payload);
void touch_entry(const std::filesystem::path& file);
std::filesystem::path entry_path(const std::filesystem::path& dir, const Digest& key);
constexpr uint32_t kEntryFormat = 1;

// A store of `bytes` under `keep` in `dir`: asks the trimmer thread for a walk
// when the directory may be over `max_bytes`.
void note_store(const std::filesystem::path& dir, uint64_t max_bytes, const Digest& keep, uint64_t bytes);
void wait_for_trims();

} // namespace brokernel::detail
