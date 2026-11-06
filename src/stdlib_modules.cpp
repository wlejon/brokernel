#include "brokernel/module_loader.h"
#include "stdlib_noise_parts.h"

#include <fstream>
#include <mutex>
#include <sstream>

namespace brokernel {

namespace {

constexpr const char* kStdMathSource = R"BK(
module math;

pub const PI: f64 = 3.14159265358979323846;
pub const TAU: f64 = 6.28318530717958647692;
pub const HALF_PI: f64 = 1.57079632679489661923;
pub const E: f64 = 2.71828182845904523536;
pub const SQRT2: f64 = 1.41421356237309504880;
pub const LN2: f64 = 0.69314718055994530942;
pub const LOG2E: f64 = 1.44269504088896340736;

pub fn square<T>(x: T) -> T {
    return x * x;
}

pub fn cube<T>(x: T) -> T {
    return x * x * x;
}

pub fn saturate<T>(x: T) -> T {
    return clamp(x, 0.0, 1.0);
}

pub fn lerp<T>(a: T, b: T, t: T) -> T {
    return a + (b - a) * t;
}

pub fn smootherstep<T>(edge0: T, edge1: T, x: T) -> T {
    let t: T = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

pub fn remap<T>(x: T, in_lo: T, in_hi: T, out_lo: T, out_hi: T) -> T {
    let t: T = (x - in_lo) / (in_hi - in_lo);
    return out_lo + (out_hi - out_lo) * t;
}

pub fn radians<T>(deg: T) -> T {
    return deg * 0.01745329251994329577;
}

pub fn degrees<T>(rad: T) -> T {
    return rad * 57.2957795130823208768;
}
)BK";

constexpr const char* kStdFilterSource = R"BK(
module filter;

pub fn laplacian1d<T>(s: stencil<T, 1, 1>) -> T {
    return s.at(-1) + s.at(1) - 2.0 * s.at(0);
}

pub fn laplacian2d<T>(s: stencil<T, 2, 1>) -> T {
    return s.at(-1, 0) + s.at(1, 0) + s.at(0, -1) + s.at(0, 1) - 4.0 * s.at(0, 0);
}

pub fn laplacian3d<T>(s: stencil<T, 3, 1>) -> T {
    return s.at(-1, 0, 0) + s.at(1, 0, 0) +
           s.at(0, -1, 0) + s.at(0, 1, 0) +
           s.at(0, 0, -1) + s.at(0, 0, 1) -
           6.0 * s.at(0, 0, 0);
}

pub fn laplacian<T, const D: i32>(s: stencil<T, D, 1>) -> T {
    if D == 1 {
        return laplacian1d<T>(s);
    } else if D == 2 {
        return laplacian2d<T>(s);
    } else {
        return laplacian3d<T>(s);
    }
}

pub fn gradient2d<T>(s: stencil<T, 2, 1>, inv_step: T) -> vec2<T> {
    let gx: T = (s.at(1, 0) - s.at(-1, 0)) * (0.5 * inv_step);
    let gy: T = (s.at(0, 1) - s.at(0, -1)) * (0.5 * inv_step);
    return vec2<T>(gx, gy);
}

pub fn gradient<T, const D: i32>(s: stencil<T, D, 1>, inv_step: T) -> vec<T, D> {
    if D == 1 {
        return (s.at(1) - s.at(-1)) * (0.5 * inv_step);
    } else if D == 2 {
        return gradient2d<T>(s, inv_step);
    } else {
        let gx: T = (s.at(1, 0, 0) - s.at(-1, 0, 0)) * (0.5 * inv_step);
        let gy: T = (s.at(0, 1, 0) - s.at(0, -1, 0)) * (0.5 * inv_step);
        let gz: T = (s.at(0, 0, 1) - s.at(0, 0, -1)) * (0.5 * inv_step);
        return vec3<T>(gx, gy, gz);
    }
}

pub fn normal2d<T>(s: stencil<T, 2, 1>, cell_size: T) -> vec3<T> {
    let inv_two_step: T = 0.5 / cell_size;
    let dhdx: T = (s.at(1, 0) - s.at(-1, 0)) * inv_two_step;
    let dhdz: T = (s.at(0, 1) - s.at(0, -1)) * inv_two_step;
    return normalize(vec3<T>(-dhdx, 1.0, -dhdz));
}

pub fn tent2d<T>(s: stencil<T, 2, 1>) -> T {
    let corners: T = s.at(-1, -1) + s.at(1, -1) + s.at(-1, 1) + s.at(1, 1);
    let edges: T = (s.at(0, -1) + s.at(-1, 0) + s.at(1, 0) + s.at(0, 1)) * 2.0;
    let center: T = s.at(0, 0) * 4.0;
    return (corners + edges + center) * 0.0625;
}

pub fn sobel2d<T>(s: stencil<T, 2, 1>) -> vec2<T> {
    let gx: T = (s.at(1, -1) + 2.0 * s.at(1, 0) + s.at(1, 1)) -
                (s.at(-1, -1) + 2.0 * s.at(-1, 0) + s.at(-1, 1));
    let gy: T = (s.at(-1, 1) + 2.0 * s.at(0, 1) + s.at(1, 1)) -
                (s.at(-1, -1) + 2.0 * s.at(0, -1) + s.at(1, -1));
    return vec2<T>(gx, gy);
}

pub fn diffuse<T, const D: i32>(s: stencil<T, D, 1>, rate: T) -> T {
    if D == 1 {
        return s.at(0) + rate * laplacian1d<T>(s);
    } else if D == 2 {
        return s.at(0, 0) + rate * laplacian2d<T>(s);
    } else {
        return s.at(0, 0, 0) + rate * laplacian3d<T>(s);
    }
}

pub fn box_blur<T, const D: i32, const R: i32>(s: stencil<T, D, R>) -> T {
    var sum: T = 0.0;
    let span: i32 = 2 * R + 1;
    if D == 1 {
        @unroll for ix in -R..=R {
            sum += s.at(ix);
        }
        return sum / (span as T);
    } else if D == 2 {
        @unroll for iy in -R..=R {
            @unroll for ix in -R..=R {
                sum += s.at(ix, iy);
            }
        }
        let count: i32 = span * span;
        return sum / (count as T);
    } else {
        @unroll for iz in -R..=R {
            @unroll for iy in -R..=R {
                @unroll for ix in -R..=R {
                    sum += s.at(ix, iy, iz);
                }
            }
        }
        let count: i32 = span * span * span;
        return sum / (count as T);
    }
}

pub fn box_blur1d<T, const R: i32>(s: stencil<T, 1, R>) -> T {
    return box_blur<T, 1, R>(s);
}

pub fn box_blur2d<T, const R: i32>(s: stencil<T, 2, R>) -> T {
    return box_blur<T, 2, R>(s);
}

pub fn box_blur3d<T, const R: i32>(s: stencil<T, 3, R>) -> T {
    return box_blur<T, 3, R>(s);
}
)BK";

constexpr const char* kStdSphereSource = R"BK(
module sphere;

// Maps a cube face (0..5: +X, -X, +Y, -Y, +Z, -Z) and [-1, 1] UV coordinates
// to a unit direction vector on the sphere, using Everitt's tangent-warp reduction.
pub fn cube_to_unit<T>(face: i32, u: T, v: T) -> vec3<T> {
    var dir: vec3<T> = vec3<T>(0.0, 0.0, 1.0);
    if face == 0 {
        dir = vec3<T>(1.0, v, -u);
    } else if face == 1 {
        dir = vec3<T>(-1.0, v, u);
    } else if face == 2 {
        dir = vec3<T>(u, 1.0, -v);
    } else if face == 3 {
        dir = vec3<T>(u, -1.0, v);
    } else if face == 4 {
        dir = vec3<T>(u, v, 1.0);
    } else {
        dir = vec3<T>(-u, v, -1.0);
    }
    return normalize(dir);
}

pub fn cube_to_sphere<T>(face: i32, u: T, v: T) -> vec3<T> {
    return cube_to_unit<T>(face, u, v);
}

// Computes camera-relative or cell-anchor-relative f32 offset from f64 planetary positions.
pub fn relative_pos(world_pos: vec3<f64>, anchor_origin: vec3<f64>) -> vec3<f32> {
    let delta: vec3<f64> = world_pos - anchor_origin;
    return vec3<f32>(delta.x as f32, delta.y as f32, delta.z as f32);
}

// Converts latitude [-pi/2, pi/2] and longitude [-pi, pi] in radians to a unit sphere direction.
pub fn latlon_to_unit<T>(lat: T, lon: T) -> vec3<T> {
    let clat: T = cos(lat);
    return vec3<T>(clat * cos(lon), sin(lat), clat * sin(lon));
}
)BK";

constexpr const char* kStdColorSource = R"BK(
module color;

pub fn luminance<T>(rgb: vec3<T>) -> T {
    return rgb.x * 0.2126 + rgb.y * 0.7152 + rgb.z * 0.0722;
}

pub fn aces_tonemap<T>(x: vec3<T>) -> vec3<T> {
    let a: T = 2.51;
    let b: T = 0.03;
    let c: T = 2.43;
    let d: T = 0.59;
    let e: T = 0.14;
    let num: vec3<T> = x * (x * a + vec3<T>(b));
    let den: vec3<T> = x * (x * c + vec3<T>(d)) + vec3<T>(e);
    return clamp(num / den, vec3<T>(0.0), vec3<T>(1.0));
}
)BK";

std::mutex g_default_resolver_mu;
ModuleRegistry::ModuleResolver g_default_resolver;

} // namespace

void ModuleRegistry::set_default_resolver(ModuleResolver resolver) {
    std::lock_guard<std::mutex> lock(g_default_resolver_mu);
    g_default_resolver = std::move(resolver);
}

std::optional<std::string> ModuleRegistry::resolve_default(std::string_view name) {
    ModuleResolver fn;
    {
        std::lock_guard<std::mutex> lock(g_default_resolver_mu);
        fn = g_default_resolver;
    }
    return fn ? fn(name) : std::nullopt;
}

ModuleRegistry::ModuleRegistry() {
    register_builtin_modules();
    {
        std::lock_guard<std::mutex> lock(g_default_resolver_mu);
        resolver_ = g_default_resolver;
    }
}

void ModuleRegistry::register_builtin_modules() {
    register_module("math", kStdMathSource, "<builtin:math.bk>");
    aliases_["std::math"] = "math";

    std::string noise_src;
    noise_src.reserve(64000);
    append_std_noise_core(noise_src);
    append_std_noise_3d(noise_src);
    append_std_noise_fractal_warp(noise_src);

    register_module("noise", std::move(noise_src), "<builtin:noise.bk>");
    aliases_["std::noise"] = "noise";
    aliases_["fastnoise"] = "noise";
    aliases_["std::fastnoise"] = "noise";

    register_module("filter", kStdFilterSource, "<builtin:filter.bk>");
    aliases_["std::filter"] = "filter";

    register_module("sphere", kStdSphereSource, "<builtin:sphere.bk>");
    aliases_["std::sphere"] = "sphere";

    register_module("color", kStdColorSource, "<builtin:color.bk>");
    aliases_["std::color"] = "color";
}

void ModuleRegistry::register_module(std::string name, std::string source, std::string virtual_path) {
    if (virtual_path.empty()) {
        virtual_path = name + ".bk";
    }
    Entry e;
    e.name = name;
    e.path = std::move(virtual_path);
    e.source = std::move(source);
    modules_[name] = std::move(e);
}

void ModuleRegistry::add_search_path(std::filesystem::path dir) {
    search_paths_.push_back(std::move(dir));
}

void ModuleRegistry::set_resolver(ModuleResolver resolver) {
    resolver_ = std::move(resolver);
}

std::unordered_map<std::string, std::string> ModuleRegistry::custom_modules() const {
    std::unordered_map<std::string, std::string> out;
    for (const auto& [k, e] : modules_) {
        if (!e.path.starts_with("<builtin:")) {
            out.emplace(k, e.source);
        }
    }
    return out;
}

std::vector<std::string> ModuleRegistry::imported_modules() const {
    std::vector<std::string> out;
    for (const auto& [k, e] : modules_) {
        if (e.ast && !e.path.starts_with("<builtin:")) out.push_back(k);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::string ModuleRegistry::format_module_path(const std::vector<std::string>& path) {
    std::string out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i > 0) out += "::";
        out += path[i];
    }
    return out;
}

ModuleRegistry::Entry* ModuleRegistry::find_or_discover(
    std::string_view key,
    const std::filesystem::path& importing_dir) {
    std::string k(key);
    auto alias_it = aliases_.find(k);
    if (alias_it != aliases_.end()) {
        k = alias_it->second;
    }
    auto it = modules_.find(k);
    if (it != modules_.end()) {
        return &it->second;
    }

    if (resolver_) {
        std::optional<std::string> resolved = resolver_(key);
        if (!resolved && k != key) {
            resolved = resolver_(k);
        }
        if (!resolved) {
            std::string canon = k;
            if (canon.ends_with(".bk")) canon.resize(canon.size() - 3);
            for (size_t pos = 0; pos < canon.size(); ++pos) {
                if (canon[pos] == '/' || canon[pos] == '\\') {
                    canon.replace(pos, 1, "::");
                    pos += 1;
                }
            }
            if (canon != key && canon != k) {
                resolved = resolver_(canon);
            }
        }
        if (resolved.has_value()) {
            register_module(k, std::move(*resolved));
            return &modules_[k];
        }
    }

    // Convert `foo::bar` to `foo/bar.bk` when searching on disk
    std::string rel_path = k;
    for (size_t pos = 0; (pos = rel_path.find("::", pos)) != std::string::npos;) {
        rel_path.replace(pos, 2, "/");
        pos += 1;
    }
    if (!rel_path.ends_with(".bk")) {
        rel_path += ".bk";
    }

    std::vector<std::filesystem::path> dirs;
    if (!importing_dir.empty()) dirs.push_back(importing_dir);
    for (const auto& sp : search_paths_) dirs.push_back(sp);

    for (const auto& dir : dirs) {
        std::filesystem::path candidate = dir / rel_path;
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && std::filesystem::is_regular_file(candidate, ec)) {
            std::ifstream ifs(candidate, std::ios::binary);
            if (ifs) {
                std::ostringstream oss;
                oss << ifs.rdbuf();
                std::string canon = candidate.stem().string();
                register_module(k, oss.str(), candidate.generic_string());
                return &modules_[k];
            }
        }
    }
    return nullptr;
}

bool ModuleRegistry::has_module(std::string_view name) const {
    std::string k(name);
    if (aliases_.count(k)) k = aliases_.at(k);
    return modules_.count(k) > 0;
}

const ModuleAst* ModuleRegistry::load_module(
    std::string_view name_or_path,
    const SourceSpan& import_site,
    DiagnosticBag& diags,
    const std::filesystem::path& importing_file_dir) {
    Entry* entry = find_or_discover(name_or_path, importing_file_dir);
    if (!entry) {
        diags.error(import_site, "module '" + std::string(name_or_path) + "' not found");
        return nullptr;
    }
    if (entry->loading) {
        diags.error(import_site, "circular module import involving '" + entry->name + "'");
        return nullptr;
    }
    if (entry->failed) {
        return nullptr;
    }
    if (entry->ast) {
        return entry->ast.get();
    }

    entry->loading = true;
    DiagnosticBag mod_diags;
    entry->ast = parse_source(entry->source, entry->path, mod_diags);
    entry->loading = false;

    if (mod_diags.has_errors() || !entry->ast) {
        entry->failed = true;
        diags.merge_from(mod_diags);
        return nullptr;
    }
    if (entry->ast->module_name.empty()) {
        entry->ast->module_name = entry->name;
    }
    return entry->ast.get();
}

} // namespace brokernel
