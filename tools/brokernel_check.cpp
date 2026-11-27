#include "brokernel/brokernel.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ResolveMapping {
    std::string prefix;
    std::filesystem::path dir;
};

struct VerifyEntry {
    std::string kernel;
    bool ok = false;
    brokernel::KernelVerifyReport report;
};

struct ModuleCheckOutcome {
    std::string file;
    bool ok = false;
    std::vector<std::string> kernels;
    std::vector<brokernel::Diagnostic> diagnostics;
    std::vector<VerifyEntry> verify;
};

void printUsage(std::ostream& os) {
    os << "Usage: brokernel-check [options] <file-or-dir>...\n"
       << "Options:\n"
       << "  -h, --help                 Show this help message\n"
       << "  -I <dir>                   Add module search path (repeatable)\n"
       << "  --resolve <prefix>=<dir>   Map <prefix>::name to <dir>/<name>.bk (repeatable)\n"
       << "  --verify                   Verify Compute SPIR-V safety & barrier uniformity\n"
       << "  --no-spirv                 Lex/parse/sema check only (skip SPIR-V & CPU codegen)\n"
       << "  --enable-checks            Compile with runtime check() instrumentation\n"
       << "  --json                     Output machine-readable JSON report\n";
}

std::string jsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

const char* severityName(brokernel::Severity sev) {
    switch (sev) {
        case brokernel::Severity::Error: return "error";
        case brokernel::Severity::Warning: return "warning";
        default: return "note";
    }
}

bool parseResolveSpec(std::string_view spec, ResolveMapping& out) {
    const size_t eq = spec.find('=');
    if (eq == std::string_view::npos || eq == 0 || eq + 1 >= spec.size()) {
        return false;
    }
    out.prefix = std::string(spec.substr(0, eq));
    out.dir = std::filesystem::path(spec.substr(eq + 1));
    return true;
}

std::optional<std::string> resolvePrefixedModule(
    std::string_view name,
    const std::vector<ResolveMapping>& resolvers) {
    for (const auto& r : resolvers) {
        std::string_view rest;
        if (name.size() > r.prefix.size() + 2 &&
            name.compare(0, r.prefix.size(), r.prefix) == 0 &&
            name[r.prefix.size()] == ':' && name[r.prefix.size() + 1] == ':') {
            rest = name.substr(r.prefix.size() + 2);
        } else if (name.size() > r.prefix.size() + 1 &&
                   name.compare(0, r.prefix.size(), r.prefix) == 0 &&
                   name[r.prefix.size()] == '/') {
            rest = name.substr(r.prefix.size() + 1);
        } else {
            continue;
        }
        std::string rel;
        rel.reserve(rest.size());
        for (size_t i = 0; i < rest.size(); ++i) {
            if (i + 1 < rest.size() && rest[i] == ':' && rest[i + 1] == ':') {
                rel.push_back('/');
                ++i;
            } else {
                rel.push_back(rest[i]);
            }
        }
        const std::filesystem::path candidate = r.dir / (rel + ".bk");
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec)) {
            std::ifstream ifs(candidate, std::ios::binary);
            if (ifs) {
                std::ostringstream ss;
                ss << ifs.rdbuf();
                return ss.str();
            }
        }
    }
    return std::nullopt;
}

void printJsonReport(
    std::ostream& os,
    size_t total_files,
    size_t passed_files,
    size_t failed_files,
    const std::vector<ModuleCheckOutcome>& outcomes) {
    os << "{\n"
       << "  \"files\": " << total_files << ",\n"
       << "  \"passed\": " << passed_files << ",\n"
       << "  \"failed\": " << failed_files << ",\n"
       << "  \"modules\": [\n";
    for (size_t i = 0; i < outcomes.size(); ++i) {
        const auto& m = outcomes[i];
        os << "    {\n"
           << "      \"file\": \"" << jsonEscape(m.file) << "\",\n"
           << "      \"ok\": " << (m.ok ? "true" : "false") << ",\n"
           << "      \"kernels\": [";
        for (size_t ki = 0; ki < m.kernels.size(); ++ki) {
            if (ki > 0) os << ", ";
            os << "\"" << jsonEscape(m.kernels[ki]) << "\"";
        }
        os << "],\n"
           << "      \"diagnostics\": [";
        if (!m.diagnostics.empty()) os << "\n";
        for (size_t di = 0; di < m.diagnostics.size(); ++di) {
            const auto& d = m.diagnostics[di];
            os << "        {\"severity\": \"" << severityName(d.severity) << "\""
               << ", \"file\": \"" << jsonEscape(d.span.file) << "\""
               << ", \"line\": " << d.span.start.line
               << ", \"column\": " << d.span.start.column
               << ", \"message\": \"" << jsonEscape(d.message) << "\"}";
            if (di + 1 < m.diagnostics.size()) os << ",";
            os << "\n";
        }
        os << (m.diagnostics.empty() ? "" : "      ") << "],\n"
           << "      \"verify\": [";
        if (!m.verify.empty()) os << "\n";
        for (size_t vi = 0; vi < m.verify.size(); ++vi) {
            const auto& v = m.verify[vi];
            os << "        {\"kernel\": \"" << jsonEscape(v.kernel) << "\""
               << ", \"ok\": " << (v.ok ? "true" : "false")
               << ", \"spirvParsed\": " << (v.report.spirv_parsed ? "true" : "false")
               << ", \"memorySafe\": " << (v.report.memory_safe ? "true" : "false")
               << ", \"barrierUniform\": " << (v.report.barrier_uniform ? "true" : "false")
               << ", \"sharedAccessesProven\": " << v.report.shared_accesses_proven
               << ", \"globalAccessesProven\": " << v.report.global_accesses_proven
               << ", \"barriersProven\": " << v.report.barriers_proven
               << ", \"violations\": [";
            for (size_t ui = 0; ui < v.report.violations.size(); ++ui) {
                if (ui > 0) os << ", ";
                os << "\"" << jsonEscape(v.report.violations[ui]) << "\"";
            }
            os << "], \"diagnostic\": \"" << jsonEscape(v.report.diagnostic) << "\"}";
            if (vi + 1 < m.verify.size()) os << ",";
            os << "\n";
        }
        os << (m.verify.empty() ? "" : "      ") << "]\n"
           << "    }";
        if (i + 1 < outcomes.size()) os << ",";
        os << "\n";
    }
    os << "  ]\n"
       << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::filesystem::path> search_paths;
    std::vector<ResolveMapping> resolvers;
    std::vector<std::string> inputs;
    bool verify = false;
    bool no_spirv = false;
    bool enable_checks = false;
    bool json = false;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "-h" || arg == "--help") {
            printUsage(std::cout);
            return 0;
        }
        if (arg == "--verify") {
            verify = true;
            continue;
        }
        if (arg == "--no-spirv") {
            no_spirv = true;
            continue;
        }
        if (arg == "--enable-checks") {
            enable_checks = true;
            continue;
        }
        if (arg == "--json") {
            json = true;
            continue;
        }
        if (arg == "-I") {
            if (i + 1 >= argc) {
                std::cerr << "brokernel-check: missing directory after -I\n";
                return 1;
            }
            search_paths.emplace_back(argv[++i]);
            continue;
        }
        if (arg.size() > 2 && arg.substr(0, 2) == "-I") {
            search_paths.emplace_back(arg.substr(2));
            continue;
        }
        if (arg == "--resolve") {
            if (i + 1 >= argc) {
                std::cerr << "brokernel-check: missing <prefix>=<dir> after --resolve\n";
                return 1;
            }
            ResolveMapping m;
            if (!parseResolveSpec(argv[++i], m)) {
                std::cerr << "brokernel-check: invalid --resolve argument (expected <prefix>=<dir>)\n";
                return 1;
            }
            resolvers.push_back(std::move(m));
            continue;
        }
        if (arg.size() > 10 && arg.substr(0, 10) == "--resolve=") {
            ResolveMapping m;
            if (!parseResolveSpec(arg.substr(10), m)) {
                std::cerr << "brokernel-check: invalid --resolve argument (expected <prefix>=<dir>)\n";
                return 1;
            }
            resolvers.push_back(std::move(m));
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "brokernel-check: unknown option '" << arg << "'\n";
            printUsage(std::cerr);
            return 1;
        }
        inputs.emplace_back(arg);
    }

    if (inputs.empty()) {
        std::cerr << "brokernel-check: no input files or directories specified\n";
        printUsage(std::cerr);
        return 1;
    }

    std::vector<std::filesystem::path> files;
    for (const std::string& raw : inputs) {
        const std::filesystem::path p(raw);
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) {
            std::cerr << "brokernel-check: path not found: " << raw << "\n";
            return 1;
        }
        if (std::filesystem::is_regular_file(p, ec)) {
            files.push_back(p.lexically_normal());
        } else if (std::filesystem::is_directory(p, ec)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(p, ec)) {
                if (ec) {
                    std::cerr << "brokernel-check: failed to scan directory '" << raw << "': " << ec.message() << "\n";
                    return 1;
                }
                if (entry.is_regular_file(ec) && entry.path().extension() == ".bk") {
                    files.push_back(entry.path().lexically_normal());
                }
            }
        } else {
            std::cerr << "brokernel-check: unsupported path type: " << raw << "\n";
            return 1;
        }
    }

    std::sort(files.begin(), files.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
        return a.string() < b.string();
    });
    files.erase(std::unique(files.begin(), files.end()), files.end());

    std::vector<ModuleCheckOutcome> outcomes;
    outcomes.reserve(files.size());
    size_t passed = 0;
    size_t failed = 0;

    for (const auto& file : files) {
        brokernel::Compiler compiler;
        for (const auto& sp : search_paths) {
            compiler.add_search_path(sp);
        }
        if (!resolvers.empty()) {
            compiler.set_module_resolver([&resolvers](std::string_view name) {
                return resolvePrefixedModule(name, resolvers);
            });
        }

        brokernel::CompileOptions opts;
        opts.compile_cpu = false;
        if (no_spirv) {
            opts.emit_spirv = false;
        }
        if (enable_checks) {
            opts.enable_checks = true;
        }

        brokernel::CompileResult res = compiler.compile_file(file, opts);
        ModuleCheckOutcome out;
        out.file = file.string();
        out.ok = res.ok;
        out.diagnostics = res.diagnostics.items();
        for (const auto& k : res.kernels) {
            out.kernels.push_back(k.name);
        }

        if (out.ok && verify && !no_spirv) {
            for (const auto& k : res.kernels) {
                if (k.stage != brokernel::Stage::Compute || k.spirv.words.empty()) continue;
                VerifyEntry ve;
                ve.kernel = k.name;
                ve.report = k.verify();
                ve.ok = ve.report.spirv_parsed && ve.report.memory_safe && ve.report.barrier_uniform;
                if (!ve.ok) out.ok = false;
                out.verify.push_back(std::move(ve));
            }
        }

        if (out.ok) {
            ++passed;
        } else {
            ++failed;
            if (!json) {
                if (!res.ok) {
                    std::cerr << res.error_message() << "\n";
                }
                for (const auto& ve : out.verify) {
                    if (!ve.ok) {
                        std::cerr << out.file << ": verify failed for kernel '" << ve.kernel << "': "
                                  << (ve.report.diagnostic.empty() ? "verification violation" : ve.report.diagnostic)
                                  << "\n";
                        for (const auto& v : ve.report.violations) {
                            std::cerr << "  - " << v << "\n";
                        }
                    }
                }
            }
        }
        outcomes.push_back(std::move(out));
    }

    if (json) {
        printJsonReport(std::cout, files.size(), passed, failed, outcomes);
    } else {
        std::cout << "brokernel-check: " << files.size() << " file(s), "
                  << passed << " passed, " << failed << " failed\n";
    }
    return failed == 0 ? 0 : 1;
}
