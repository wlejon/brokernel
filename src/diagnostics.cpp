#include "brokernel/diagnostics.h"

#include <algorithm>
#include <sstream>

namespace brokernel {

SourceMap::SourceMap(std::string file, std::string source)
    : file_(std::move(file)), source_(std::move(source)) {
    line_starts_.push_back(0);
    for (uint32_t i = 0; i < static_cast<uint32_t>(source_.size()); ++i) {
        if (source_[i] == '\n') {
            line_starts_.push_back(i + 1);
        }
    }
}

SourcePos SourceMap::pos_at(uint32_t offset) const noexcept {
    if (line_starts_.empty()) {
        return SourcePos{1, 1, offset};
    }
    uint32_t clamped = std::min(offset, static_cast<uint32_t>(source_.size()));
    auto it = std::upper_bound(line_starts_.begin(), line_starts_.end(), clamped);
    uint32_t line_idx = static_cast<uint32_t>(std::distance(line_starts_.begin(), it) - 1);
    uint32_t col = clamped - line_starts_[line_idx] + 1;
    return SourcePos{line_idx + 1, col, clamped};
}

SourceSpan SourceMap::span(uint32_t start_offset, uint32_t end_offset) const {
    return SourceSpan{file_, pos_at(start_offset), pos_at(std::max(start_offset, end_offset))};
}

std::string_view SourceMap::line_text(uint32_t line_1based) const noexcept {
    if (line_1based == 0 || line_1based > line_starts_.size()) {
        return {};
    }
    uint32_t start = line_starts_[line_1based - 1];
    uint32_t end = (line_1based < line_starts_.size())
                       ? line_starts_[line_1based]
                       : static_cast<uint32_t>(source_.size());
    while (end > start && (source_[end - 1] == '\n' || source_[end - 1] == '\r')) {
        --end;
    }
    return std::string_view(source_).substr(start, end - start);
}

std::string Diagnostic::format() const {
    std::ostringstream oss;
    const char* sev = "error";
    if (severity == Severity::Warning) sev = "warning";
    else if (severity == Severity::Note) sev = "note";

    if (!span.file.empty()) {
        oss << span.file << ":" << span.start.line << ":" << span.start.column << ": ";
    }
    oss << sev << ": " << message;
    if (!line_text.empty()) {
        oss << "\n  " << line_text << "\n  ";
        uint32_t col = span.start.column > 0 ? span.start.column - 1 : 0;
        for (uint32_t i = 0; i < col && i < line_text.size(); ++i) {
            oss << (line_text[i] == '\t' ? '\t' : ' ');
        }
        uint32_t len = 1;
        if (span.end.line == span.start.line && span.end.column > span.start.column) {
            len = std::min<uint32_t>(span.end.column - span.start.column, 64u);
        }
        for (uint32_t i = 0; i < len; ++i) oss << '^';
    }
    return oss.str();
}

void DiagnosticBag::add(Severity severity, SourceSpan span, std::string message, std::string line_text) {
    if (severity == Severity::Error) {
        ++error_count_;
    }
    items_.push_back(Diagnostic{severity, std::move(span), std::move(message), std::move(line_text)});
}

void DiagnosticBag::error(SourceSpan span, std::string message, std::string line_text) {
    add(Severity::Error, std::move(span), std::move(message), std::move(line_text));
}

void DiagnosticBag::warning(SourceSpan span, std::string message, std::string line_text) {
    add(Severity::Warning, std::move(span), std::move(message), std::move(line_text));
}

void DiagnosticBag::note(SourceSpan span, std::string message, std::string line_text) {
    add(Severity::Note, std::move(span), std::move(message), std::move(line_text));
}

std::vector<std::string> DiagnosticBag::messages() const {
    std::vector<std::string> out;
    out.reserve(items_.size());
    for (const auto& d : items_) {
        out.push_back(d.format());
    }
    return out;
}

std::string DiagnosticBag::format_all() const {
    std::ostringstream oss;
    for (size_t i = 0; i < items_.size(); ++i) {
        if (i > 0) oss << "\n";
        oss << items_[i].format();
    }
    return oss.str();
}

void DiagnosticBag::merge_from(const DiagnosticBag& other) {
    for (const auto& item : other.items_) {
        add(item.severity, item.span, item.message, item.line_text);
    }
}

void DiagnosticBag::clear() {
    items_.clear();
    error_count_ = 0;
}

} // namespace brokernel
