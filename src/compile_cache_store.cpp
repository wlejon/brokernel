// The compile cache's directory: entry files written whole and renamed into
// place, read back against their header, and held to a size by a thread of
// its own that deletes the least recently used (compile_cache.h).
#include "compile_cache_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace brokernel::detail {

namespace {

uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

uint64_t fmix(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

constexpr uint32_t kMagic = 0x31434B42u;   // "BKC1"
constexpr const char* kExt = ".bkc";
constexpr size_t kHeaderBytes = 4 + 4 + 16 + 8 + 16;

uint64_t process_id() {
#ifdef _WIN32
    return static_cast<uint64_t>(_getpid());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

// What a file of `size` bytes takes on disk: whole 4 KiB units.
constexpr uint64_t kAllocUnit = 4096;
uint64_t disk_bytes(uint64_t size) { return (size + kAllocUnit - 1) / kAllocUnit * kAllocUnit; }

using KeyName = std::array<char, 32>;

// The key of a file named `<32 hex digits>.bkc`, or false for a name the
// cache does not write.
bool key_of_name(const std::string& name, KeyName& key) {
    if (name.size() != key.size() + 4 || name.compare(key.size(), 4, kExt) != 0) return false;
    for (size_t i = 0; i < key.size(); ++i) {
        const char c = name[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        key[i] = c;
    }
    return true;
}

// Walks `dir` and, when its entries (and writes in progress) take more than
// `max_bytes`, deletes the least recently used down to 7/8 of it, sparing
// the one `kept` names. Returns what the directory takes afterwards.
uint64_t trim_dir(const std::filesystem::path& dir, uint64_t max_bytes,
                  const std::function<bool(const KeyName&)>& kept) {
    struct Item {
        KeyName key;
        uint64_t bytes = 0;
        std::filesystem::file_time_type mtime;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    std::error_code ec;
    const auto now = std::filesystem::file_time_type::clock::now();
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        std::error_code e2;
        if (!entry.is_regular_file(e2)) continue;
        const auto mtime = entry.last_write_time(e2);
        if (e2) continue;
        const uint64_t size = entry.file_size(e2);
        if (e2) continue;
        const std::string name = entry.path().filename().string();
        if (name.find(".tmp") != std::string::npos) {
            // A writer that died between write and rename; a live one's
            // bytes are about to be an entry.
            if (now - mtime > std::chrono::minutes(10)) {
                std::filesystem::remove(entry.path(), e2);
            } else {
                total += disk_bytes(size);
            }
            continue;
        }
        KeyName key;
        if (key_of_name(name, key)) {
            items.push_back({key, disk_bytes(size), mtime});
            total += disk_bytes(size);
        }
    }
    if (total <= max_bytes) return total;
    const uint64_t target = max_bytes - max_bytes / 8;
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.mtime < b.mtime; });
    for (const Item& item : items) {
        if (total <= target) break;
        if (kept(item.key)) continue;
        std::error_code e2;
        if (!std::filesystem::remove(dir / (std::string(item.key.data(), item.key.size()) + kExt), e2)) continue;
        total -= item.bytes;
    }
    return total;
}

// Per directory: what the last walk found plus what was stored since. A
// store asks for a walk only when that passes the limit or the directory has
// not been walked yet; other processes' stores show at the next walk.
struct TrimState {
    uint64_t estimate = 0;
    uint64_t max_bytes = 0;
    KeyName keep{};
    bool walked = false;
    bool pending = false;
};

struct Trimmer {
    std::mutex mu;
    std::condition_variable cv;   // work asked for, or a walk finished
    std::map<std::filesystem::path, TrimState> dirs;
    bool started = false;
    bool busy = false;

    bool any_pending() const {
        for (const auto& [d, s] : dirs) {
            if (s.pending) return true;
        }
        return false;
    }

    void run() {
        std::unique_lock<std::mutex> lock(mu);
        for (;;) {
            cv.wait(lock, [&] { return any_pending(); });
            for (auto& [dir, s] : dirs) {
                if (!s.pending) continue;
                s.pending = false;
                const std::filesystem::path d = dir;
                const uint64_t max_bytes = s.max_bytes;
                const uint64_t before = s.estimate;
                busy = true;
                lock.unlock();
                const uint64_t total = trim_dir(d, max_bytes, [&](const KeyName& key) {
                    std::lock_guard<std::mutex> guard(mu);
                    return dirs[d].keep == key;
                });
                lock.lock();
                busy = false;
                TrimState& after = dirs[d];
                // Stores made during the walk count on top (some the walk
                // saw: an over-estimate only asks for a walk sooner).
                after.estimate = total + (after.estimate > before ? after.estimate - before : 0);
                after.walked = true;
                cv.notify_all();
                break;   // `dirs` may have changed: look again
            }
        }
    }
};

Trimmer& trimmer() {
    // Never destroyed: its thread may be mid-walk at exit.
    static auto* t = new Trimmer();
    return *t;
}

} // namespace

std::string Digest::hex() const {
    static const char* kDigits = "0123456789abcdef";
    std::string s(32, '0');
    for (int i = 0; i < 16; ++i) {
        s[15 - i] = kDigits[(hi >> (i * 4)) & 15];
        s[31 - i] = kDigits[(lo >> (i * 4)) & 15];
    }
    return s;
}

// MurmurHash3 x64 128.
Digest hash128(std::string_view bytes, uint64_t seed) {
    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    const size_t len = bytes.size();
    const size_t nblocks = len / 16;
    uint64_t h1 = seed, h2 = seed;
    constexpr uint64_t c1 = 0x87c37b91114253d5ull, c2 = 0x4cf5ad432745937full;
    for (size_t i = 0; i < nblocks; ++i) {
        uint64_t k1, k2;
        std::memcpy(&k1, data + i * 16, 8);
        std::memcpy(&k2, data + i * 16 + 8, 8);
        k1 *= c1; k1 = rotl(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = rotl(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;
        k2 *= c2; k2 = rotl(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = rotl(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }
    const uint8_t* tail = data + nblocks * 16;
    const size_t rem = len & 15;
    uint64_t k1 = 0, k2 = 0;
    for (size_t i = rem; i > 8; --i) k2 ^= uint64_t(tail[i - 1]) << ((i - 9) * 8);
    if (rem > 8) { k2 *= c2; k2 = rotl(k2, 33); k2 *= c1; h2 ^= k2; }
    for (size_t i = std::min<size_t>(rem, 8); i > 0; --i) k1 ^= uint64_t(tail[i - 1]) << ((i - 1) * 8);
    if (rem > 0) { k1 *= c1; k1 = rotl(k1, 31); k1 *= c2; h1 ^= k1; }
    h1 ^= len; h2 ^= len;
    h1 += h2; h2 += h1;
    h1 = fmix(h1); h2 = fmix(h2);
    h1 += h2; h2 += h1;
    return {h1, h2};
}

std::string build_identity() {
    std::filesystem::path self;
#ifdef _WIN32
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&build_identity), &module)) {
        std::wstring buf(1024, L'\0');
        for (;;) {
            const DWORD n = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()));
            if (n == 0) break;
            if (n < buf.size()) {
                buf.resize(n);
                self = std::filesystem::path(buf);
                break;
            }
            buf.resize(buf.size() * 2);
        }
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&build_identity), &info) && info.dli_fname) self = info.dli_fname;
#endif
    if (self.empty()) return {};
    std::error_code ec;
    const auto size = std::filesystem::file_size(self, ec);
    if (ec) return {};
    const auto mtime = std::filesystem::last_write_time(self, ec);
    if (ec) return {};
    return self.generic_string() + "|" + std::to_string(size) + "|" +
           std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

std::filesystem::path entry_path(const std::filesystem::path& dir, const Digest& key) {
    return dir / (key.hex() + kExt);
}

bool read_entry(const std::filesystem::path& file, const Digest& key, std::string& payload, std::string& why) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        why = "no entry";
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    in.close();
    const std::string bytes = std::move(ss).str();
    if (bytes.size() < kHeaderBytes) {
        why = "entry truncated";
        return false;
    }
    ByteReader r(bytes.data(), kHeaderBytes);
    if (r.u32() != kMagic || r.u32() != kEntryFormat) {
        why = "entry of another format";
        return false;
    }
    Digest stored;
    stored.lo = r.u64();
    stored.hi = r.u64();
    const uint64_t size = r.u64();
    Digest sum;
    sum.lo = r.u64();
    sum.hi = r.u64();
    if (!(stored == key)) {
        why = "entry of another key";
        return false;
    }
    if (size != bytes.size() - kHeaderBytes) {
        why = "entry truncated";
        return false;
    }
    payload.assign(bytes, kHeaderBytes, std::string::npos);
    if (!(hash128(payload) == sum)) {
        why = "entry damaged";
        return false;
    }
    return true;
}

bool write_entry(const std::filesystem::path& file, const Digest& key, std::string_view payload) {
    std::string header;
    ByteWriter w(header);
    w.u32(kMagic);
    w.u32(kEntryFormat);
    w.u64(key.lo);
    w.u64(key.hi);
    w.u64(payload.size());
    const Digest sum = hash128(payload);
    w.u64(sum.lo);
    w.u64(sum.hi);

    std::error_code ec;
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path(), ec);
    // Unique per process and thread: writers of one key never share a file.
    static std::atomic<uint64_t> serial{0};
    std::filesystem::path tmp = file;
    tmp += ".tmp" + std::to_string(process_id()) + "_" + std::to_string(serial.fetch_add(1));
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        out.close();
        if (!out) {
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, file, ec);
    if (ec) {
        std::error_code rm;
        std::filesystem::remove(tmp, rm);
        return false;
    }
    return true;
}

void touch_entry(const std::filesystem::path& file) {
    std::error_code ec;
    std::filesystem::last_write_time(file, std::filesystem::file_time_type::clock::now(), ec);
}

void note_store(const std::filesystem::path& dir, uint64_t max_bytes, const Digest& keep, uint64_t bytes) {
    Trimmer& t = trimmer();
    std::lock_guard<std::mutex> lock(t.mu);
    TrimState& s = t.dirs[dir];
    s.estimate += disk_bytes(bytes);
    s.max_bytes = max_bytes;
    const std::string hex = keep.hex();
    std::copy(hex.begin(), hex.end(), s.keep.begin());
    if (s.walked && s.estimate <= max_bytes) return;
    s.pending = true;
    if (!t.started) {
        t.started = true;
        std::thread([&t] { t.run(); }).detach();
    }
    t.cv.notify_all();
}

void wait_for_trims() {
    Trimmer& t = trimmer();
    std::unique_lock<std::mutex> lock(t.mu);
    t.cv.wait(lock, [&] { return !t.busy && !t.any_pending(); });
}

} // namespace brokernel::detail
