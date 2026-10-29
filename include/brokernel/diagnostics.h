#pragma once
// Diagnostics: source locations, spans, and structured compiler messages.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brokernel {

/// 1-based line and column position within a source file.
struct SourcePos {
    uint32_t line = 1;
    uint32_t column = 1;
    uint32_t offset = 0;
};

/// Half-open byte span [start.offset, end.offset) in a named source unit.
struct SourceSpan {
    std::string file;
    SourcePos start;
    SourcePos end;

    static SourceSpan point(std::string file, uint32_t line, uint32_t col, uint32_t offset = 0) {
        SourcePos p{line, col, offset};
        return SourceSpan{std::move(file), p, p};
    }

    static SourceSpan merge(const SourceSpan& a, const SourceSpan& b) {
        if (a.file.empty()) return b;
        if (b.file.empty()) return a;
        return SourceSpan{a.file, a.start, b.end};
    }
};

enum class Severity : uint8_t {
    Error = 0,
    Warning = 1,
    Note = 2,
};

struct Diagnostic {
    Severity severity = Severity::Error;
    SourceSpan span;
    std::string message;
    std::string line_text; // Optional source line for caret rendering

    std::string format() const;
};

/// Maps byte offsets in a source buffer to 1-based (line, column) coordinates
/// and extracts individual source lines for diagnostic formatting.
class SourceMap {
public:
    SourceMap() = default;
    SourceMap(std::string file, std::string source);

    const std::string& file() const noexcept { return file_; }
    const std::string& source() const noexcept { return source_; }

    SourcePos pos_at(uint32_t offset) const noexcept;
    SourceSpan span(uint32_t start_offset, uint32_t end_offset) const;
    std::string_view line_text(uint32_t line_1based) const noexcept;

private:
    std::string file_;
    std::string source_;
    std::vector<uint32_t> line_starts_;
};

/// Accumulates diagnostics across lexing, parsing, type checking and lowering.
class DiagnosticBag {
public:
    void add(Severity severity, SourceSpan span, std::string message, std::string line_text = {});
    void error(SourceSpan span, std::string message, std::string line_text = {});
    void warning(SourceSpan span, std::string message, std::string line_text = {});
    void note(SourceSpan span, std::string message, std::string line_text = {});

    bool empty() const noexcept { return items_.empty(); }
    bool has_errors() const noexcept { return error_count_ > 0; }
    size_t error_count() const noexcept { return error_count_; }
    size_t size() const noexcept { return items_.size(); }

    const std::vector<Diagnostic>& items() const noexcept { return items_; }
    std::vector<std::string> messages() const;
    std::string format_all() const;

    void merge_from(const DiagnosticBag& other);
    void clear();

private:
    std::vector<Diagnostic> items_;
    size_t error_count_ = 0;
};

} // namespace brokernel
