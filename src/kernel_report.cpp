#include "kernel_report.h"

#include <brass/mir/printer.hpp>

#include <algorithm>
#include <unordered_map>

namespace brokernel::detail {

uint64_t code_hash(const CompiledKernel& ck) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const void* data, size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
    };
    if (!ck.spirv.words.empty()) {
        mix(ck.spirv.words.data(), ck.spirv.words.size() * sizeof(uint32_t));
    } else if (ck.cpu_) {
        if (ck.cpu_->point) { const std::string s = brass::to_string(*ck.cpu_->point); mix(s.data(), s.size()); }
        if (ck.cpu_->grid) { const std::string s = brass::to_string(*ck.cpu_->grid); mix(s.data(), s.size()); }
    }
    return h;
}

std::vector<CompiledKernel::FunctionCost> function_costs(const brass::Function& fn, const brass::DebugContext& dc,
                                                         const std::string& entry) {
    std::unordered_map<std::string, CompiledKernel::FunctionCost> by;
    auto at = [&](const std::string& n) -> CompiledKernel::FunctionCost& {
        auto& c = by[n];
        c.name = n;
        return c;
    };
    std::vector<std::string> chain;
    for (const brass::BasicBlock* bb : fn.blocks()) {
        for (const brass::Instruction* inst : *bb) {
            chain.clear();
            for (uint32_t id = inst->loc().inlined_at_id; id != 0;) {
                const brass::InlinedScope* s = dc.get_inlined_scope(id);
                if (!s) break;
                chain.push_back(s->callee_name);
                id = s->parent_inlined_at_id;
            }
            chain.push_back(entry);
            at(chain.front()).self += 1;
            // A function counts once however often it recurs up the chain.
            for (size_t i = 0; i < chain.size(); ++i) {
                if (std::find(chain.begin(), chain.begin() + i, chain[i]) == chain.begin() + i) at(chain[i]).total += 1;
            }
        }
    }
    std::vector<CompiledKernel::FunctionCost> out;
    out.reserve(by.size());
    for (auto& [n, c] : by) out.push_back(std::move(c));
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.total != b.total ? a.total > b.total : a.name < b.name;
    });
    return out;
}

} // namespace brokernel::detail
