#include "brokernel/ast.h"

#include <cctype>
#include <cstdlib>
#include <cstring>

namespace brokernel {

namespace {

bool is_ident_start(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool is_ident_cont(char c) noexcept {
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

bool is_digit(char c) noexcept {
    return c >= '0' && c <= '9';
}

bool is_hex_digit(char c) noexcept {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

TokenKind classify_keyword(std::string_view text) noexcept {
    if (text == "module")   return TokenKind::KwModule;
    if (text == "import")   return TokenKind::KwImport;
    if (text == "from")     return TokenKind::KwFrom;
    if (text == "as")       return TokenKind::KwAs;
    if (text == "pub")      return TokenKind::KwPub;
    if (text == "fn")       return TokenKind::KwFn;
    if (text == "struct")   return TokenKind::KwStruct;
    if (text == "const")    return TokenKind::KwConst;
    if (text == "let")      return TokenKind::KwLet;
    if (text == "var")      return TokenKind::KwVar;
    if (text == "if")       return TokenKind::KwIf;
    if (text == "else")     return TokenKind::KwElse;
    if (text == "for")      return TokenKind::KwFor;
    if (text == "in")       return TokenKind::KwIn;
    if (text == "while")    return TokenKind::KwWhile;
    if (text == "break")    return TokenKind::KwBreak;
    if (text == "continue") return TokenKind::KwContinue;
    if (text == "return")   return TokenKind::KwReturn;
    if (text == "true")     return TokenKind::KwTrue;
    if (text == "false")    return TokenKind::KwFalse;
    return TokenKind::Ident;
}

} // namespace

std::vector<Token> lex_source(const SourceMap& src, DiagnosticBag& diags) {
    std::vector<Token> tokens;
    const std::string& s = src.source();
    const uint32_t n = static_cast<uint32_t>(s.size());
    uint32_t i = 0;

    auto peek = [&](uint32_t delta = 0) -> char {
        return (i + delta < n) ? s[i + delta] : '\0';
    };

    auto emit_err = [&](uint32_t start, uint32_t end, std::string msg) {
        SourceSpan sp = src.span(start, end);
        diags.error(sp, std::move(msg), std::string(src.line_text(sp.start.line)));
    };

    while (i < n) {
        char c = peek();

        // Whitespace
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            ++i;
            continue;
        }

        // Line comment
        if (c == '/' && peek(1) == '/') {
            i += 2;
            while (i < n && peek() != '\n') ++i;
            continue;
        }

        // Block comment (supports nesting)
        if (c == '/' && peek(1) == '*') {
            uint32_t start = i;
            i += 2;
            int depth = 1;
            while (i < n && depth > 0) {
                if (peek() == '/' && peek(1) == '*') {
                    depth++;
                    i += 2;
                } else if (peek() == '*' && peek(1) == '/') {
                    depth--;
                    i += 2;
                } else {
                    ++i;
                }
            }
            if (depth > 0) {
                emit_err(start, i, "unterminated block comment");
            }
            continue;
        }

        uint32_t start = i;

        // Identifiers and keywords
        if (is_ident_start(c)) {
            while (is_ident_cont(peek())) ++i;
            std::string text = s.substr(start, i - start);
            Token tok;
            tok.kind = classify_keyword(text);
            tok.span = src.span(start, i);
            tok.text = std::move(text);
            tokens.push_back(std::move(tok));
            continue;
        }

        // Numbers (integer and floating-point literals)
        if (is_digit(c)) {
            bool is_float = false;
            int base = 10;
            std::string clean;

            if (c == '0' && (peek(1) == 'x' || peek(1) == 'X')) {
                base = 16;
                i += 2;
                while (is_hex_digit(peek()) || peek() == '_') {
                    if (peek() != '_') clean.push_back(peek());
                    ++i;
                }
                if (clean.empty()) {
                    emit_err(start, i, "invalid hexadecimal literal");
                }
            } else if (c == '0' && (peek(1) == 'b' || peek(1) == 'B')) {
                base = 2;
                i += 2;
                while (peek() == '0' || peek() == '1' || peek() == '_') {
                    if (peek() != '_') clean.push_back(peek());
                    ++i;
                }
                if (clean.empty()) {
                    emit_err(start, i, "invalid binary literal");
                }
            } else {
                while (is_digit(peek()) || peek() == '_') {
                    if (peek() != '_') clean.push_back(peek());
                    ++i;
                }
                // Fractional part: `.` followed by digit (not `..` range operator or `.method`)
                if (peek() == '.' && peek(1) != '.' && !is_ident_start(peek(1))) {
                    is_float = true;
                    clean.push_back('.');
                    ++i;
                    while (is_digit(peek()) || peek() == '_') {
                        if (peek() != '_') clean.push_back(peek());
                        ++i;
                    }
                }
                // Exponent part
                if (peek() == 'e' || peek() == 'E') {
                    is_float = true;
                    clean.push_back(peek());
                    ++i;
                    if (peek() == '+' || peek() == '-') {
                        clean.push_back(peek());
                        ++i;
                    }
                    bool has_exp_digits = false;
                    while (is_digit(peek()) || peek() == '_') {
                        if (peek() != '_') {
                            clean.push_back(peek());
                            has_exp_digits = true;
                        }
                        ++i;
                    }
                    if (!has_exp_digits) {
                        emit_err(start, i, "malformed floating-point exponent");
                    }
                }
            }

            // Optional type suffix: f32, f, f64, d, i32, i, u32, u, i64, u64
            ScalarKind suffix = ScalarKind::Void;
            if (is_ident_start(peek())) {
                uint32_t suf_start = i;
                while (is_ident_cont(peek())) ++i;
                std::string_view suf(s.data() + suf_start, i - suf_start);
                if (suf == "f32" || suf == "f") {
                    suffix = ScalarKind::F32;
                    is_float = true;
                } else if (suf == "f64" || suf == "d") {
                    suffix = ScalarKind::F64;
                    is_float = true;
                } else if (suf == "i32" || suf == "i") {
                    suffix = ScalarKind::I32;
                } else if (suf == "u32" || suf == "u") {
                    suffix = ScalarKind::U32;
                } else if (suf == "i64") {
                    suffix = ScalarKind::I64;
                } else if (suf == "u64") {
                    suffix = ScalarKind::U64;
                } else {
                    emit_err(suf_start, i, "unknown numeric literal suffix '" + std::string(suf) + "'");
                }
            }

            Token tok;
            tok.span = src.span(start, i);
            tok.text = s.substr(start, i - start);
            tok.literal_suffix = suffix;
            if (is_float) {
                tok.kind = TokenKind::FloatLit;
                tok.float_val = std::strtod(clean.c_str(), nullptr);
            } else {
                tok.kind = TokenKind::IntLit;
                uint64_t uval = std::strtoull(clean.c_str(), nullptr, base);
                tok.int_val = static_cast<int64_t>(uval);
            }
            tokens.push_back(std::move(tok));
            continue;
        }

        // String literal ("...")
        if (c == '"') {
            ++i;
            std::string val;
            bool closed = false;
            while (i < n) {
                char ch = peek();
                if (ch == '"') {
                    ++i;
                    closed = true;
                    break;
                }
                if (ch == '\\' && i + 1 < n) {
                    ++i;
                    char esc = peek();
                    switch (esc) {
                        case 'n':  val.push_back('\n'); break;
                        case 'r':  val.push_back('\r'); break;
                        case 't':  val.push_back('\t'); break;
                        case '\\': val.push_back('\\'); break;
                        case '"':  val.push_back('"'); break;
                        default:   val.push_back(esc); break;
                    }
                    ++i;
                } else {
                    val.push_back(ch);
                    ++i;
                }
            }
            if (!closed) {
                emit_err(start, i, "unterminated string literal");
            }
            Token tok;
            tok.kind = TokenKind::StringLit;
            tok.span = src.span(start, i);
            tok.text = std::move(val);
            tokens.push_back(std::move(tok));
            continue;
        }

        // Multi-character operators and punctuation
        auto push_tok = [&](TokenKind k, uint32_t len) {
            Token tok;
            tok.kind = k;
            tok.span = src.span(start, start + len);
            tok.text = s.substr(start, len);
            tokens.push_back(std::move(tok));
            i += len;
        };

        if (c == '.' && peek(1) == '.' && peek(2) == '=') { push_tok(TokenKind::DotDotEq, 3); continue; }
        if (c == '<' && peek(1) == '<' && peek(2) == '=') { push_tok(TokenKind::ShlEq, 3); continue; }
        if (c == '>' && peek(1) == '>' && peek(2) == '=') { push_tok(TokenKind::ShrEq, 3); continue; }

        if (c == ':' && peek(1) == ':') { push_tok(TokenKind::ColonColon, 2); continue; }
        if (c == '.' && peek(1) == '.') { push_tok(TokenKind::DotDot, 2); continue; }
        if (c == '-' && peek(1) == '>') { push_tok(TokenKind::Arrow, 2); continue; }
        if (c == '&' && peek(1) == '&') { push_tok(TokenKind::AmpAmp, 2); continue; }
        if (c == '|' && peek(1) == '|') { push_tok(TokenKind::PipePipe, 2); continue; }
        if (c == '=' && peek(1) == '=') { push_tok(TokenKind::EqEq, 2); continue; }
        if (c == '!' && peek(1) == '=') { push_tok(TokenKind::BangEq, 2); continue; }
        if (c == '<' && peek(1) == '=') { push_tok(TokenKind::Le, 2); continue; }
        if (c == '>' && peek(1) == '=') { push_tok(TokenKind::Ge, 2); continue; }
        if (c == '<' && peek(1) == '<') { push_tok(TokenKind::Shl, 2); continue; }
        if (c == '>' && peek(1) == '>') { push_tok(TokenKind::Shr, 2); continue; }
        if (c == '+' && peek(1) == '=') { push_tok(TokenKind::PlusEq, 2); continue; }
        if (c == '-' && peek(1) == '=') { push_tok(TokenKind::MinusEq, 2); continue; }
        if (c == '*' && peek(1) == '=') { push_tok(TokenKind::StarEq, 2); continue; }
        if (c == '/' && peek(1) == '=') { push_tok(TokenKind::SlashEq, 2); continue; }
        if (c == '%' && peek(1) == '=') { push_tok(TokenKind::PercentEq, 2); continue; }
        if (c == '&' && peek(1) == '=') { push_tok(TokenKind::AmpEq, 2); continue; }
        if (c == '|' && peek(1) == '=') { push_tok(TokenKind::PipeEq, 2); continue; }
        if (c == '^' && peek(1) == '=') { push_tok(TokenKind::CaretEq, 2); continue; }

        switch (c) {
            case '@': push_tok(TokenKind::At, 1); break;
            case '(': push_tok(TokenKind::LParen, 1); break;
            case ')': push_tok(TokenKind::RParen, 1); break;
            case '{': push_tok(TokenKind::LBrace, 1); break;
            case '}': push_tok(TokenKind::RBrace, 1); break;
            case '[': push_tok(TokenKind::LBracket, 1); break;
            case ']': push_tok(TokenKind::RBracket, 1); break;
            case ',': push_tok(TokenKind::Comma, 1); break;
            case ';': push_tok(TokenKind::Semi, 1); break;
            case ':': push_tok(TokenKind::Colon, 1); break;
            case '.': push_tok(TokenKind::Dot, 1); break;
            case '?': push_tok(TokenKind::Question, 1); break;
            case '+': push_tok(TokenKind::Plus, 1); break;
            case '-': push_tok(TokenKind::Minus, 1); break;
            case '*': push_tok(TokenKind::Star, 1); break;
            case '/': push_tok(TokenKind::Slash, 1); break;
            case '%': push_tok(TokenKind::Percent, 1); break;
            case '&': push_tok(TokenKind::Amp, 1); break;
            case '|': push_tok(TokenKind::Pipe, 1); break;
            case '^': push_tok(TokenKind::Caret, 1); break;
            case '~': push_tok(TokenKind::Tilde, 1); break;
            case '!': push_tok(TokenKind::Bang, 1); break;
            case '<': push_tok(TokenKind::Lt, 1); break;
            case '>': push_tok(TokenKind::Gt, 1); break;
            case '=': push_tok(TokenKind::Eq, 1); break;
            default:
                emit_err(start, start + 1, std::string("unexpected character '") + c + "'");
                push_tok(TokenKind::Invalid, 1);
                break;
        }
    }

    Token eof;
    eof.kind = TokenKind::Eof;
    eof.span = src.span(n, n);
    tokens.push_back(std::move(eof));
    return tokens;
}

} // namespace brokernel
