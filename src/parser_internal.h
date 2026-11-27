#pragma once
// Internal parser class shared between src/parser_decls.cpp and src/parser.cpp.

#include "brokernel/ast.h"

namespace brokernel::detail {

class Parser {
public:
    Parser(const SourceMap& src, const std::vector<Token>& tokens, DiagnosticBag& diags)
        : src_(src), tokens_(tokens), diags_(diags) {}

    std::unique_ptr<ModuleAst> parse();

private:
    const SourceMap& src_;
    const std::vector<Token>& tokens_;
    DiagnosticBag& diags_;
    size_t pos_ = 0;

    const Token& peek(size_t offset = 0) const noexcept {
        size_t idx = pos_ + offset;
        return idx < tokens_.size() ? tokens_[idx] : tokens_.back();
    }
    const Token& previous() const noexcept { return pos_ > 0 ? tokens_[pos_ - 1] : tokens_.front(); }
    bool is_at_end() const noexcept { return peek().kind == TokenKind::Eof; }
    bool check(TokenKind k, size_t offset = 0) const noexcept { return peek(offset).kind == k; }
    const Token& advance() noexcept {
        if (!is_at_end()) ++pos_;
        return previous();
    }
    bool match(TokenKind k) noexcept {
        if (check(k)) {
            advance();
            return true;
        }
        return false;
    }
    const Token& consume(TokenKind k, const char* what) {
        if (check(k)) return advance();
        error(peek().span, std::string("expected ") + what + ", got '" + peek().text + "'");
        return peek();
    }
    void error(const SourceSpan& sp, std::string msg) {
        diags_.error(sp, std::move(msg), std::string(src_.line_text(sp.start.line)));
    }
    void synchronize() {
        advance();
        while (!is_at_end()) {
            if (previous().kind == TokenKind::Semi || previous().kind == TokenKind::RBrace) return;
            switch (peek().kind) {
                case TokenKind::At:
                case TokenKind::KwPub:
                case TokenKind::KwFn:
                case TokenKind::KwStruct:
                case TokenKind::KwEnum:
                case TokenKind::KwConst:
                case TokenKind::KwImport:
                case TokenKind::KwFrom:
                case TokenKind::KwShared:
                case TokenKind::KwVar:
                    return;
                default:
                    if (check(TokenKind::Ident) && peek().text == "newtype") return;
                    advance();
                    break;
            }
        }
    }

    // Declarations (src/parser_decls.cpp)
    std::vector<std::string> parse_ident_path(const char* what);
    std::vector<Attribute> parse_attributes();
    ImportDecl parse_import(bool is_pub = false);
    std::vector<GenericParamDecl> parse_generic_params();
    GenericArg parse_generic_arg();
    TypeExprPtr parse_type();
    StructDecl parse_struct(bool is_pub);
    EnumDecl parse_enum(bool is_pub);
    NewtypeDecl parse_newtype(bool is_pub);
    ConstDecl parse_const(bool is_pub, std::vector<Attribute> attrs = {});
    FnDecl parse_fn(bool is_pub, std::vector<Attribute> attrs);

    // Statements & expressions (src/parser.cpp)
    std::vector<StmtPtr> parse_block_stmts();
    StmtPtr parse_stmt();
    std::vector<StmtPtr> parse_stmt_or_block();
    StmtPtr parse_let_or_var(bool is_var, const SourceSpan& start_sp);
    StmtPtr parse_if_stmt(const SourceSpan& start_sp, std::vector<Attribute> attrs);
    StmtPtr parse_for_stmt(const SourceSpan& start_sp, std::vector<Attribute> attrs);
    std::optional<std::optional<BinaryOp>> match_assign_op();

    ExprPtr parse_expr();
    ExprPtr parse_ternary();
    static int binary_prec(TokenKind k, BinaryOp* out_op) noexcept;
    ExprPtr parse_binary(int min_prec);
    ExprPtr parse_cast();
    ExprPtr parse_unary();
    bool try_parse_generic_call_args(std::vector<GenericArg>& out_args);
    bool try_parse_nested_type_args(std::vector<GenericArg>& out_args);
    ExprPtr parse_postfix();
    ExprPtr parse_primary();
};

} // namespace brokernel::detail
