#include "brokernel/ast.h"
#include <utility>
namespace brokernel {
namespace {
class Parser {
public:
    Parser(const SourceMap& src, const std::vector<Token>& tokens, DiagnosticBag& diags)
        : src_(src), tokens_(tokens), diags_(diags) {}
    std::unique_ptr<ModuleAst> parse() {
        auto mod = std::make_unique<ModuleAst>();
        mod->file_path = src_.file();
        if (!tokens_.empty()) {
            mod->span = tokens_.front().span;
        }
        // Optional `module foo::bar;`
        if (check(TokenKind::KwModule)) {
            advance();
            auto path = parse_ident_path("module name");
            std::string mname;
            for (size_t i = 0; i < path.size(); ++i) {
                if (i > 0) mname += "::";
                mname += path[i];
            }
            mod->module_name = std::move(mname);
            match(TokenKind::Semi);
        }
        while (!is_at_end()) {
            std::vector<Attribute> attrs = parse_attributes();
            bool is_pub = match(TokenKind::KwPub);
            if (check(TokenKind::KwImport) || check(TokenKind::KwFrom)) {
                if (!attrs.empty()) {
                    error(attrs.front().span, "attributes are not allowed on import declarations");
                }
                mod->imports.push_back(parse_import());
            } else if (check(TokenKind::KwStruct)) {
                if (!attrs.empty()) {
                    error(attrs.front().span, "attributes are not allowed on struct declarations");
                }
                mod->structs.push_back(parse_struct(is_pub));
            } else if (check(TokenKind::KwConst)) {
                if (!attrs.empty()) {
                    error(attrs.front().span, "attributes are not allowed on const declarations");
                }
                mod->consts.push_back(parse_const(is_pub));
            } else if (check(TokenKind::KwFn)) {
                mod->functions.push_back(parse_fn(is_pub, std::move(attrs)));
            } else if (check(TokenKind::Semi)) {
                advance();
            } else {
                error(peek().span, "expected declaration ('import', 'struct', 'const', or 'fn'), got '" + peek().text + "'");
                synchronize();
            }
        }
        return mod;
    }
private:
    const SourceMap& src_;
    const std::vector<Token>& tokens_;
    DiagnosticBag& diags_;
    size_t pos_ = 0;
    const Token& peek(size_t offset = 0) const noexcept {
        size_t idx = pos_ + offset;
        return idx < tokens_.size() ? tokens_[idx] : tokens_.back();
    }
    const Token& previous() const noexcept {
        return pos_ > 0 ? tokens_[pos_ - 1] : tokens_.front();
    }
    bool is_at_end() const noexcept {
        return peek().kind == TokenKind::Eof;
    }
    bool check(TokenKind k, size_t offset = 0) const noexcept {
        return peek(offset).kind == k;
    }
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
                case TokenKind::KwConst:
                case TokenKind::KwImport:
                case TokenKind::KwFrom:
                    return;
                default:
                    advance();
                    break;
            }
        }
    }
    std::vector<std::string> parse_ident_path(const char* what) {
        std::vector<std::string> parts;
        const Token& first = consume(TokenKind::Ident, what);
        if (!first.text.empty()) parts.push_back(first.text);
        while (match(TokenKind::ColonColon)) {
            if (check(TokenKind::Ident)) {
                parts.push_back(advance().text);
            } else {
                break;
            }
        }
        return parts;
    }
    // ---- Attributes (@kernel, @grid(2), @workgroup_size(64, 1, 1), @binding(0, 1)) ----
    std::vector<Attribute> parse_attributes() {
        std::vector<Attribute> attrs;
        while (match(TokenKind::At)) {
            Attribute a;
            SourceSpan start_sp = previous().span;
            const Token& name_tok = consume(TokenKind::Ident, "attribute name");
            a.name = name_tok.text;
            a.span = SourceSpan::merge(start_sp, name_tok.span);
            if (match(TokenKind::LParen)) {
                while (!check(TokenKind::RParen) && !is_at_end()) {
                    bool neg = match(TokenKind::Minus);
                    if (check(TokenKind::IntLit)) {
                        int64_t v = advance().int_val;
                        a.int_args.push_back(neg ? -v : v);
                    } else if (check(TokenKind::Ident) || check(TokenKind::StringLit)) {
                        a.str_args.push_back(advance().text);
                    } else {
                        error(peek().span, "expected attribute argument");
                        advance();
                    }
                    if (!match(TokenKind::Comma)) break;
                }
                const Token& rp = consume(TokenKind::RParen, "')' after attribute arguments");
                a.span = SourceSpan::merge(start_sp, rp.span);
            }
            attrs.push_back(std::move(a));
        }
        return attrs;
    }
    // ---- Imports --------------------------------------------------------------
    ImportDecl parse_import() {
        ImportDecl imp;
        imp.span = peek().span;
        if (match(TokenKind::KwFrom)) {
            // `from std::noise import fbm, simplex2d;` or `from "path.bk" import *;`
            if (check(TokenKind::StringLit)) {
                imp.file_path = advance().text;
            } else {
                imp.module_path = parse_ident_path("module name");
            }
            consume(TokenKind::KwImport, "'import' in from-import declaration");
            if (match(TokenKind::Star)) {
                imp.wildcard = true;
            } else {
                bool braced = match(TokenKind::LBrace);
                do {
                    const Token& sym = consume(TokenKind::Ident, "imported symbol name");
                    imp.symbols.push_back(sym.text);
                } while (match(TokenKind::Comma) && !check(TokenKind::RBrace));
                if (braced) consume(TokenKind::RBrace, "'}'");
            }
            match(TokenKind::Semi);
            imp.span = SourceSpan::merge(imp.span, previous().span);
            return imp;
        }
        consume(TokenKind::KwImport, "'import'");
        if (check(TokenKind::StringLit)) {
            imp.file_path = advance().text;
            if (match(TokenKind::KwAs)) {
                imp.alias = consume(TokenKind::Ident, "module alias").text;
            } else {
                imp.wildcard = true;
            }
            match(TokenKind::Semi);
            imp.span = SourceSpan::merge(imp.span, previous().span);
            return imp;
        }
        // `import std::noise;`, `import noise::{fbm, perlin};`, `import noise::*;`, `import noise as n;`
        const Token& first = consume(TokenKind::Ident, "module name");
        imp.module_path.push_back(first.text);
        while (match(TokenKind::ColonColon)) {
            if (match(TokenKind::Star)) {
                imp.wildcard = true;
                break;
            }
            if (match(TokenKind::LBrace)) {
                while (!check(TokenKind::RBrace) && !is_at_end()) {
                    if (match(TokenKind::Star)) {
                        imp.wildcard = true;
                    } else {
                        imp.symbols.push_back(consume(TokenKind::Ident, "imported symbol").text);
                    }
                    if (!match(TokenKind::Comma)) break;
                }
                consume(TokenKind::RBrace, "'}' after import list");
                break;
            }
            imp.module_path.push_back(consume(TokenKind::Ident, "module path segment").text);
        }
        if (match(TokenKind::KwAs)) {
            imp.alias = consume(TokenKind::Ident, "module alias").text;
        }
        match(TokenKind::Semi);
        imp.span = SourceSpan::merge(imp.span, previous().span);
        return imp;
    }
    // ---- Generic Parameters `<T, const D: i32>` -------------------------------
    std::vector<GenericParamDecl> parse_generic_params() {
        std::vector<GenericParamDecl> params;
        if (!match(TokenKind::Lt)) return params;
        while (!check(TokenKind::Gt) && !is_at_end()) {
            GenericParamDecl gp;
            gp.span = peek().span;
            gp.is_const = match(TokenKind::KwConst);
            gp.name = consume(TokenKind::Ident, "generic parameter name").text;
            if (match(TokenKind::Colon)) {
                gp.constraint_or_type = parse_type();
                // `D: i32` or `D: u32` without `const` keyword is also treated as a const integer generic parameter
                if (gp.constraint_or_type &&
                    (gp.constraint_or_type->name == "i32" || gp.constraint_or_type->name == "u32" ||
                     gp.constraint_or_type->name == "int" || gp.constraint_or_type->name == "uint")) {
                    gp.is_const = true;
                }
            }
            if (match(TokenKind::Eq)) {
                bool neg = match(TokenKind::Minus);
                if (check(TokenKind::IntLit)) {
                    int64_t v = advance().int_val;
                    gp.default_int = neg ? -v : v;
                } else {
                    gp.default_type = parse_type();
                }
            }
            gp.span = SourceSpan::merge(gp.span, previous().span);
            params.push_back(std::move(gp));
            if (!match(TokenKind::Comma)) break;
        }
        consume(TokenKind::Gt, "'>' after generic parameters");
        return params;
    }
    // ---- Types (`f32`, `f64`, `vec<T, D>`, `stencil<f32, 2, 1>`, `[f32; 4]`) --
    GenericArg parse_generic_arg() {
        GenericArg arg;
        arg.span = peek().span;
        if (check(TokenKind::Minus) && check(TokenKind::IntLit, 1)) {
            advance();
            int64_t v = advance().int_val;
            arg.int_arg = -v;
            arg.span = SourceSpan::merge(arg.span, previous().span);
            return arg;
        }
        if (check(TokenKind::IntLit)) {
            arg.int_arg = advance().int_val;
            arg.span = previous().span;
            return arg;
        }
        arg.type_arg = parse_type();
        if (arg.type_arg && arg.type_arg->args.empty()) {
            arg.ident_or_const_name = arg.type_arg->name;
        }
        arg.span = SourceSpan::merge(arg.span, previous().span);
        return arg;
    }
    TypeExprPtr parse_type() {
        auto t = std::make_unique<TypeExpr>();
        t->span = peek().span;
        // Fixed array shorthand `[T; N]`
        if (match(TokenKind::LBracket)) {
            t->name = "array";
            GenericArg elem_arg;
            elem_arg.span = peek().span;
            elem_arg.type_arg = parse_type();
            t->args.push_back(std::move(elem_arg));
            consume(TokenKind::Semi, "';' in array type '[T; N]'");
            t->args.push_back(parse_generic_arg());
            const Token& rb = consume(TokenKind::RBracket, "']' after array type");
            t->span = SourceSpan::merge(t->span, rb.span);
            return t;
        }
        const Token& name_tok = consume(TokenKind::Ident, "type name");
        t->name = name_tok.text;
        while (match(TokenKind::ColonColon)) {
            t->name += "::" + consume(TokenKind::Ident, "type name").text;
        }
        if (match(TokenKind::Lt)) {
            while (!check(TokenKind::Gt) && !is_at_end()) {
                t->args.push_back(parse_generic_arg());
                if (!match(TokenKind::Comma)) break;
            }
            consume(TokenKind::Gt, "'>' after type arguments");
        }
        t->span = SourceSpan::merge(t->span, previous().span);
        return t;
    }
    // ---- Top-Level Struct / Const / Fn ----------------------------------------
    StructDecl parse_struct(bool is_pub) {
        StructDecl s;
        s.is_pub = is_pub;
        s.span = consume(TokenKind::KwStruct, "'struct'").span;
        s.name = consume(TokenKind::Ident, "struct name").text;
        s.generic_params = parse_generic_params();
        consume(TokenKind::LBrace, "'{' in struct declaration");
        while (!check(TokenKind::RBrace) && !is_at_end()) {
            StructFieldDecl f;
            f.span = peek().span;
            f.name = consume(TokenKind::Ident, "struct field name").text;
            consume(TokenKind::Colon, "':' after struct field name");
            f.type = parse_type();
            f.span = SourceSpan::merge(f.span, previous().span);
            s.fields.push_back(std::move(f));
            if (!match(TokenKind::Comma)) {
                match(TokenKind::Semi);
            }
        }
        const Token& rb = consume(TokenKind::RBrace, "'}' after struct fields");
        match(TokenKind::Semi);
        s.span = SourceSpan::merge(s.span, rb.span);
        return s;
    }
    ConstDecl parse_const(bool is_pub) {
        ConstDecl c;
        c.is_pub = is_pub;
        c.span = consume(TokenKind::KwConst, "'const'").span;
        c.name = consume(TokenKind::Ident, "constant name").text;
        if (match(TokenKind::Colon)) {
            c.type = parse_type();
        }
        consume(TokenKind::Eq, "'=' in const declaration");
        c.value = parse_expr();
        const Token& semi = consume(TokenKind::Semi, "';' after const declaration");
        c.span = SourceSpan::merge(c.span, semi.span);
        return c;
    }
    FnDecl parse_fn(bool is_pub, std::vector<Attribute> attrs) {
        FnDecl fn;
        fn.is_pub = is_pub;
        fn.attributes = std::move(attrs);
        fn.span = consume(TokenKind::KwFn, "'fn'").span;
        if (!fn.attributes.empty()) {
            fn.span = SourceSpan::merge(fn.attributes.front().span, fn.span);
        }
        fn.name = consume(TokenKind::Ident, "function name").text;
        fn.generic_params = parse_generic_params();
        consume(TokenKind::LParen, "'(' after function name");
        while (!check(TokenKind::RParen) && !is_at_end()) {
            ParamDecl p;
            p.attributes = parse_attributes();
            p.span = peek().span;
            p.name = consume(TokenKind::Ident, "parameter name").text;
            consume(TokenKind::Colon, "':' after parameter name");
            p.type = parse_type();
            if (match(TokenKind::Eq)) {
                p.default_value = parse_expr();
            }
            p.span = SourceSpan::merge(p.span, previous().span);
            fn.params.push_back(std::move(p));
            if (!match(TokenKind::Comma)) break;
        }
        consume(TokenKind::RParen, "')' after function parameters");
        if (match(TokenKind::Arrow) || match(TokenKind::Colon)) {
            fn.return_type = parse_type();
        }
        fn.body = parse_block_stmts();
        fn.span = SourceSpan::merge(fn.span, previous().span);
        return fn;
    }
    // ---- Statements -----------------------------------------------------------
    std::vector<StmtPtr> parse_block_stmts() {
        consume(TokenKind::LBrace, "'{'");
        std::vector<StmtPtr> stmts;
        while (!check(TokenKind::RBrace) && !is_at_end()) {
            if (match(TokenKind::Semi)) continue;
            StmtPtr st = parse_stmt();
            if (st) stmts.push_back(std::move(st));
        }
        consume(TokenKind::RBrace, "'}'");
        return stmts;
    }
    StmtPtr parse_stmt() {
        std::vector<Attribute> attrs = parse_attributes();
        SourceSpan start_sp = attrs.empty() ? peek().span : attrs.front().span;
        if (check(TokenKind::LBrace)) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::Block;
            st->attributes = std::move(attrs);
            st->body = parse_block_stmts();
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        if (match(TokenKind::KwLet) || match(TokenKind::KwVar)) {
            bool is_var = previous().kind == TokenKind::KwVar;
            // Allow `let mut x = ...` as `var x = ...`
            if (!is_var && check(TokenKind::Ident) && peek().text == "mut") {
                advance();
                is_var = true;
            }
            auto st = parse_let_or_var(is_var, start_sp);
            st->attributes = std::move(attrs);
            consume(TokenKind::Semi, "';' after variable declaration");
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        if (match(TokenKind::KwIf)) {
            return parse_if_stmt(start_sp, std::move(attrs));
        }
        if (match(TokenKind::KwFor)) {
            return parse_for_stmt(start_sp, std::move(attrs));
        }
        if (match(TokenKind::KwWhile)) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::While;
            st->attributes = std::move(attrs);
            st->cond = parse_expr();
            st->body = parse_stmt_or_block();
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        if (match(TokenKind::KwBreak)) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::Break;
            consume(TokenKind::Semi, "';' after 'break'");
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        if (match(TokenKind::KwContinue)) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::Continue;
            consume(TokenKind::Semi, "';' after 'continue'");
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        if (match(TokenKind::KwReturn)) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::Return;
            if (!check(TokenKind::Semi) && !check(TokenKind::RBrace)) {
                st->expr = parse_expr();
            }
            consume(TokenKind::Semi, "';' after 'return'");
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        // Expression or assignment statement
        ExprPtr lhs = parse_expr();
        if (auto assign_op = match_assign_op()) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::Assign;
            st->assign_target = std::move(lhs);
            st->compound_op = *assign_op;
            st->assign_value = parse_expr();
            consume(TokenKind::Semi, "';' after assignment");
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        auto st = std::make_unique<Stmt>();
        st->kind = Stmt::Kind::Expr;
        st->expr = std::move(lhs);
        consume(TokenKind::Semi, "';' after expression");
        st->span = SourceSpan::merge(start_sp, previous().span);
        return st;
    }
    std::vector<StmtPtr> parse_stmt_or_block() {
        if (check(TokenKind::LBrace)) {
            return parse_block_stmts();
        }
        std::vector<StmtPtr> out;
        if (StmtPtr s = parse_stmt()) out.push_back(std::move(s));
        return out;
    }
    StmtPtr parse_let_or_var(bool is_var, const SourceSpan& start_sp) {
        auto st = std::make_unique<Stmt>();
        st->kind = is_var ? Stmt::Kind::Var : Stmt::Kind::Let;
        st->span = start_sp;
        st->var_name = consume(TokenKind::Ident, "variable name").text;
        if (match(TokenKind::Colon)) {
            st->var_type = parse_type();
        }
        if (match(TokenKind::Eq)) {
            st->init_expr = parse_expr();
        } else if (!is_var) {
            error(peek().span, "'let' binding requires an initializer");
        }
        return st;
    }
    StmtPtr parse_if_stmt(const SourceSpan& start_sp, std::vector<Attribute> attrs) {
        auto st = std::make_unique<Stmt>();
        st->kind = Stmt::Kind::If;
        st->attributes = std::move(attrs);
        st->cond = parse_expr();
        st->body = parse_stmt_or_block();
        if (match(TokenKind::KwElse)) {
            if (match(TokenKind::KwIf)) {
                st->else_body.push_back(parse_if_stmt(previous().span, {}));
            } else {
                st->else_body = parse_stmt_or_block();
            }
        }
        st->span = SourceSpan::merge(start_sp, previous().span);
        return st;
    }
    StmtPtr parse_for_stmt(const SourceSpan& start_sp, std::vector<Attribute> attrs) {
        bool has_lparen = match(TokenKind::LParen);
        // Range-based for loop: `for i in start..end` or `for (i in start..=end)`
        if (check(TokenKind::Ident) &&
            (check(TokenKind::KwIn, 1) || (check(TokenKind::Colon, 1) && check(TokenKind::Ident, 2) && check(TokenKind::KwIn, 3)))) {
            auto st = std::make_unique<Stmt>();
            st->kind = Stmt::Kind::ForRange;
            st->attributes = std::move(attrs);
            st->var_name = advance().text;
            if (match(TokenKind::Colon)) {
                st->var_type = parse_type();
            }
            consume(TokenKind::KwIn, "'in' in range for-loop");
            st->range_start = parse_expr();
            if (match(TokenKind::DotDotEq)) {
                st->inclusive_range = true;
            } else {
                consume(TokenKind::DotDot, "'..' or '..=' in range for-loop");
                st->inclusive_range = false;
            }
            st->range_end = parse_expr();
            if (has_lparen) consume(TokenKind::RParen, "')' after for range");
            st->body = parse_stmt_or_block();
            st->span = SourceSpan::merge(start_sp, previous().span);
            return st;
        }
        // C-style for loop: `for (var i: i32 = 0; i < n; i += 1) { ... }`
        auto st = std::make_unique<Stmt>();
        st->kind = Stmt::Kind::ForC;
        st->attributes = std::move(attrs);
        if (!check(TokenKind::Semi)) {
            if (match(TokenKind::KwLet) || match(TokenKind::KwVar)) {
                bool is_var = previous().kind == TokenKind::KwVar;
                st->for_init = parse_let_or_var(is_var, previous().span);
            } else {
                ExprPtr e = parse_expr();
                if (auto op = match_assign_op()) {
                    auto init_st = std::make_unique<Stmt>();
                    init_st->kind = Stmt::Kind::Assign;
                    init_st->assign_target = std::move(e);
                    init_st->compound_op = *op;
                    init_st->assign_value = parse_expr();
                    init_st->span = init_st->assign_target->span;
                    st->for_init = std::move(init_st);
                }
            }
        }
        consume(TokenKind::Semi, "';' after for-loop initializer");
        if (!check(TokenKind::Semi)) {
            st->cond = parse_expr();
        }
        consume(TokenKind::Semi, "';' after for-loop condition");
        if (!check(TokenKind::RParen) && !check(TokenKind::LBrace)) {
            ExprPtr step_lhs = parse_expr();
            if (auto op = match_assign_op()) {
                auto step_st = std::make_unique<Stmt>();
                step_st->kind = Stmt::Kind::Assign;
                step_st->assign_target = std::move(step_lhs);
                step_st->compound_op = *op;
                step_st->assign_value = parse_expr();
                step_st->span = step_st->assign_target->span;
                st->for_step = std::move(step_st);
            } else {
                auto step_st = std::make_unique<Stmt>();
                step_st->kind = Stmt::Kind::Expr;
                step_st->expr = std::move(step_lhs);
                step_st->span = step_st->expr->span;
                st->for_step = std::move(step_st);
            }
        }
        if (has_lparen) consume(TokenKind::RParen, "')' after for-loop clauses");
        st->body = parse_stmt_or_block();
        st->span = SourceSpan::merge(start_sp, previous().span);
        return st;
    }
    std::optional<std::optional<BinaryOp>> match_assign_op() {
        if (match(TokenKind::Eq))        return std::optional<BinaryOp>{std::nullopt};
        if (match(TokenKind::PlusEq))    return std::optional<BinaryOp>{BinaryOp::Add};
        if (match(TokenKind::MinusEq))   return std::optional<BinaryOp>{BinaryOp::Sub};
        if (match(TokenKind::StarEq))    return std::optional<BinaryOp>{BinaryOp::Mul};
        if (match(TokenKind::SlashEq))   return std::optional<BinaryOp>{BinaryOp::Div};
        if (match(TokenKind::PercentEq)) return std::optional<BinaryOp>{BinaryOp::Mod};
        if (match(TokenKind::AmpEq))     return std::optional<BinaryOp>{BinaryOp::BitAnd};
        if (match(TokenKind::PipeEq))    return std::optional<BinaryOp>{BinaryOp::BitOr};
        if (match(TokenKind::CaretEq))   return std::optional<BinaryOp>{BinaryOp::BitXor};
        if (match(TokenKind::ShlEq))     return std::optional<BinaryOp>{BinaryOp::Shl};
        if (match(TokenKind::ShrEq))     return std::optional<BinaryOp>{BinaryOp::Shr};
        return std::nullopt;
    }
    // ---- Expressions (Pratt / Precedence Climbing) ----------------------------
    ExprPtr parse_expr() {
        return parse_ternary();
    }
    ExprPtr parse_ternary() {
        ExprPtr c = parse_binary(1);
        if (match(TokenKind::Question)) {
            ExprPtr t = parse_expr();
            consume(TokenKind::Colon, "':' in ternary conditional expression");
            ExprPtr f = parse_ternary();
            auto sel = std::make_unique<Expr>();
            sel->kind = Expr::Kind::Select;
            sel->span = SourceSpan::merge(c->span, f->span);
            sel->cond = std::move(c);
            sel->lhs = std::move(t);
            sel->rhs = std::move(f);
            return sel;
        }
        return c;
    }
    static int binary_prec(TokenKind k, BinaryOp* out_op) noexcept {
        switch (k) {
            case TokenKind::PipePipe: *out_op = BinaryOp::LogOr;  return 1;
            case TokenKind::AmpAmp:   *out_op = BinaryOp::LogAnd; return 2;
            case TokenKind::Pipe:     *out_op = BinaryOp::BitOr;  return 3;
            case TokenKind::Caret:    *out_op = BinaryOp::BitXor; return 4;
            case TokenKind::Amp:      *out_op = BinaryOp::BitAnd; return 5;
            case TokenKind::EqEq:     *out_op = BinaryOp::Eq;     return 6;
            case TokenKind::BangEq:   *out_op = BinaryOp::Ne;     return 6;
            case TokenKind::Lt:       *out_op = BinaryOp::Lt;     return 7;
            case TokenKind::Le:       *out_op = BinaryOp::Le;     return 7;
            case TokenKind::Gt:       *out_op = BinaryOp::Gt;     return 7;
            case TokenKind::Ge:       *out_op = BinaryOp::Ge;     return 7;
            case TokenKind::Shl:      *out_op = BinaryOp::Shl;    return 8;
            case TokenKind::Plus:     *out_op = BinaryOp::Add;    return 9;
            case TokenKind::Minus:    *out_op = BinaryOp::Sub;    return 9;
            case TokenKind::Star:     *out_op = BinaryOp::Mul;    return 10;
            case TokenKind::Slash:    *out_op = BinaryOp::Div;    return 10;
            case TokenKind::Percent:  *out_op = BinaryOp::Mod;    return 10;
            default: return 0;
        }
    }
    ExprPtr parse_binary(int min_prec) {
        ExprPtr left = parse_cast();
        while (true) {
            BinaryOp op{};
            int prec = binary_prec(peek().kind, &op);
            // Two adjacent `>` (the lexer splits `>>` for type arguments) shift.
            bool shr = check(TokenKind::Gt) && check(TokenKind::Gt, 1) &&
                       peek(1).span.start.offset == peek().span.start.offset + 1;
            if (shr) {
                op = BinaryOp::Shr;
                prec = 8;
            }
            if (prec < min_prec) break;
            advance();
            if (shr) advance();
            ExprPtr right = parse_binary(prec + 1);
            auto bin = std::make_unique<Expr>();
            bin->kind = Expr::Kind::Binary;
            bin->binary_op = op;
            bin->span = SourceSpan::merge(left->span, right->span);
            bin->lhs = std::move(left);
            bin->rhs = std::move(right);
            left = std::move(bin);
        }
        return left;
    }
    ExprPtr parse_cast() {
        ExprPtr e = parse_unary();
        while (match(TokenKind::KwAs)) {
            TypeExprPtr target = parse_type();
            auto cast = std::make_unique<Expr>();
            cast->kind = Expr::Kind::Cast;
            cast->span = SourceSpan::merge(e->span, target->span);
            cast->lhs = std::move(e);
            cast->target_type = std::move(target);
            e = std::move(cast);
        }
        return e;
    }
    ExprPtr parse_unary() {
        if (match(TokenKind::Minus) || match(TokenKind::Plus) ||
            match(TokenKind::Bang) || match(TokenKind::Tilde)) {
            Token op_tok = previous();
            UnaryOp uop = UnaryOp::Neg;
            if (op_tok.kind == TokenKind::Plus) uop = UnaryOp::Pos;
            else if (op_tok.kind == TokenKind::Bang) uop = UnaryOp::Not;
            else if (op_tok.kind == TokenKind::Tilde) uop = UnaryOp::BitNot;
            ExprPtr sub = parse_unary();
            auto u = std::make_unique<Expr>();
            u->kind = Expr::Kind::Unary;
            u->unary_op = uop;
            u->span = SourceSpan::merge(op_tok.span, sub->span);
            u->lhs = std::move(sub);
            return u;
        }
        return parse_postfix();
    }
    // Speculatively checks if `< ... >` at current `pos_` is a generic argument
    // list immediately followed by `(` or `{`. Never emits diagnostics on failure.
    bool try_parse_generic_call_args(std::vector<GenericArg>& out_args) {
        size_t saved = pos_;
        if (!match(TokenKind::Lt)) return false;
        std::vector<GenericArg> parsed;
        while (!check(TokenKind::Gt) && !is_at_end()) {
            GenericArg ga;
            ga.span = peek().span;
            if (check(TokenKind::Minus) && check(TokenKind::IntLit, 1)) {
                advance();
                ga.int_arg = -advance().int_val;
            } else if (check(TokenKind::IntLit)) {
                ga.int_arg = advance().int_val;
            } else if (check(TokenKind::Ident)) {
                std::string tname = advance().text;
                while (match(TokenKind::ColonColon)) {
                    if (!check(TokenKind::Ident)) { pos_ = saved; return false; }
                    tname += "::" + advance().text;
                }
                auto te = std::make_unique<TypeExpr>();
                te->span = ga.span;
                te->name = tname;
                if (check(TokenKind::Lt)) {
                    std::vector<GenericArg> nested;
                    if (!try_parse_nested_type_args(nested)) {
                        pos_ = saved;
                        return false;
                    }
                    te->args = std::move(nested);
                } else {
                    ga.ident_or_const_name = tname;
                }
                ga.type_arg = std::move(te);
            } else {
                pos_ = saved;
                return false;
            }
            ga.span = SourceSpan::merge(ga.span, previous().span);
            parsed.push_back(std::move(ga));
            if (!match(TokenKind::Comma)) break;
        }
        if (!match(TokenKind::Gt) || parsed.empty()) {
            pos_ = saved;
            return false;
        }
        if (!check(TokenKind::LParen) && !check(TokenKind::LBrace)) {
            pos_ = saved;
            return false;
        }
        out_args = std::move(parsed);
        return true;
    }
    bool try_parse_nested_type_args(std::vector<GenericArg>& out_args) {
        if (!match(TokenKind::Lt)) return false;
        while (!check(TokenKind::Gt) && !is_at_end()) {
            GenericArg ga;
            ga.span = peek().span;
            if (check(TokenKind::IntLit)) {
                ga.int_arg = advance().int_val;
            } else if (check(TokenKind::Ident)) {
                auto te = std::make_unique<TypeExpr>();
                te->span = peek().span;
                te->name = advance().text;
                if (check(TokenKind::Lt)) {
                    if (!try_parse_nested_type_args(te->args)) return false;
                } else {
                    ga.ident_or_const_name = te->name;
                }
                ga.type_arg = std::move(te);
            } else {
                return false;
            }
            out_args.push_back(std::move(ga));
            if (!match(TokenKind::Comma)) break;
        }
        return match(TokenKind::Gt);
    }
    ExprPtr parse_postfix() {
        ExprPtr e = parse_primary();
        while (true) {
            if (match(TokenKind::LParen)) {
                auto call = std::make_unique<Expr>();
                call->kind = Expr::Kind::Call;
                call->span = e->span;
                call->name = e->name;
                call->path = e->path;
                call->generic_args = std::move(e->generic_args);
                call->lhs = std::move(e);
                while (!check(TokenKind::RParen) && !is_at_end()) {
                    call->args.push_back(parse_expr());
                    if (!match(TokenKind::Comma)) break;
                }
                const Token& rp = consume(TokenKind::RParen, "')' after call arguments");
                call->span = SourceSpan::merge(call->span, rp.span);
                e = std::move(call);
            } else if (match(TokenKind::Dot)) {
                const Token& member = consume(TokenKind::Ident, "field or method name after '.'");
                std::vector<GenericArg> gargs;
                if (match(TokenKind::ColonColon) && check(TokenKind::Lt)) {
                    try_parse_generic_call_args(gargs);
                } else if (check(TokenKind::Lt)) {
                    try_parse_generic_call_args(gargs);
                }
                if (match(TokenKind::LParen)) {
                    auto mc = std::make_unique<Expr>();
                    mc->kind = Expr::Kind::MethodCall;
                    mc->name = member.text;
                    mc->generic_args = std::move(gargs);
                    mc->lhs = std::move(e);
                    while (!check(TokenKind::RParen) && !is_at_end()) {
                        mc->args.push_back(parse_expr());
                        if (!match(TokenKind::Comma)) break;
                    }
                    const Token& rp = consume(TokenKind::RParen, "')' after method arguments");
                    mc->span = SourceSpan::merge(mc->lhs->span, rp.span);
                    e = std::move(mc);
                } else {
                    auto fld = std::make_unique<Expr>();
                    fld->kind = Expr::Kind::Field;
                    fld->name = member.text;
                    fld->span = SourceSpan::merge(e->span, member.span);
                    fld->lhs = std::move(e);
                    e = std::move(fld);
                }
            } else if (match(TokenKind::LBracket)) {
                auto idx = std::make_unique<Expr>();
                idx->kind = Expr::Kind::Index;
                idx->span = e->span;
                idx->lhs = std::move(e);
                while (!check(TokenKind::RBracket) && !is_at_end()) {
                    idx->args.push_back(parse_expr());
                    if (!match(TokenKind::Comma)) break;
                }
                const Token& rb = consume(TokenKind::RBracket, "']' after index");
                idx->span = SourceSpan::merge(idx->span, rb.span);
                e = std::move(idx);
            } else {
                break;
            }
        }
        return e;
    }
    ExprPtr parse_primary() {
        if (match(TokenKind::IntLit)) {
            const Token& tok = previous();
            auto e = std::make_unique<Expr>();
            e->kind = Expr::Kind::IntLit;
            e->span = tok.span;
            e->int_val = tok.int_val;
            e->literal_suffix = tok.literal_suffix;
            return e;
        }
        if (match(TokenKind::FloatLit)) {
            const Token& tok = previous();
            auto e = std::make_unique<Expr>();
            e->kind = Expr::Kind::FloatLit;
            e->span = tok.span;
            e->float_val = tok.float_val;
            e->literal_suffix = tok.literal_suffix;
            return e;
        }
        if (match(TokenKind::KwTrue) || match(TokenKind::KwFalse)) {
            const Token& tok = previous();
            auto e = std::make_unique<Expr>();
            e->kind = Expr::Kind::BoolLit;
            e->span = tok.span;
            e->bool_val = (tok.kind == TokenKind::KwTrue);
            return e;
        }
        if (match(TokenKind::LParen)) {
            ExprPtr inner = parse_expr();
            consume(TokenKind::RParen, "')'");
            return inner;
        }
        if (match(TokenKind::LBracket)) {
            const Token& lb = previous();
            auto call = std::make_unique<Expr>();
            call->kind = Expr::Kind::Call;
            call->name = "array";
            while (!check(TokenKind::RBracket) && !is_at_end()) {
                call->args.push_back(parse_expr());
                if (!match(TokenKind::Comma)) break;
            }
            const Token& rb = consume(TokenKind::RBracket, "']' after array literal");
            call->span = SourceSpan::merge(lb.span, rb.span);
            return call;
        }
        if (match(TokenKind::Ident)) {
            Token first = previous();
            std::vector<std::string> path = {first.text};
            std::vector<GenericArg> generic_args;
            while (match(TokenKind::ColonColon)) {
                if (check(TokenKind::Lt)) {
                    try_parse_generic_call_args(generic_args);
                    break;
                }
                const Token& seg = consume(TokenKind::Ident, "identifier after '::'");
                path.push_back(seg.text);
            }
            if (generic_args.empty() && check(TokenKind::Lt)) {
                // Only attempt `<...>` generic arguments on a call/struct construct
                try_parse_generic_call_args(generic_args);
            }
            // Struct literal `StructName { field: expr, ... }`
            // Guard: only treat `Ident { ... }` as a struct literal when `{` is followed by
            // `Ident :` (so `if cond { ... }` is never misparsed as a struct literal!).
            if (check(TokenKind::LBrace) &&
                ((check(TokenKind::Ident, 1) && check(TokenKind::Colon, 2)) ||
                 (!generic_args.empty() && check(TokenKind::RBrace, 1)))) {
                advance(); // consume '{'
                auto sl = std::make_unique<Expr>();
                sl->kind = Expr::Kind::StructLit;
                sl->name = path.back();
                sl->path = path;
                sl->target_type = std::make_unique<TypeExpr>();
                sl->target_type->span = first.span;
                sl->target_type->name = path.empty() ? "" : path[0];
                for (size_t i = 1; i < path.size(); ++i) {
                    sl->target_type->name += "::" + path[i];
                }
                sl->target_type->args = std::move(generic_args);
                sl->span = first.span;
                while (!check(TokenKind::RBrace) && !is_at_end()) {
                    FieldInit fi;
                    fi.span = peek().span;
                    fi.name = consume(TokenKind::Ident, "field name in struct literal").text;
                    consume(TokenKind::Colon, "':' after field name in struct literal");
                    fi.value = parse_expr();
                    fi.span = SourceSpan::merge(fi.span, previous().span);
                    sl->fields.push_back(std::move(fi));
                    if (!match(TokenKind::Comma)) break;
                }
                const Token& rb = consume(TokenKind::RBrace, "'}' after struct literal");
                sl->span = SourceSpan::merge(first.span, rb.span);
                return sl;
            }
            // Generic call `foo<T, D>(...)` or `vec3<f64>(...)`
            if (!generic_args.empty() && match(TokenKind::LParen)) {
                auto callee = std::make_unique<Expr>();
                callee->span = first.span;
                if (path.size() == 1) {
                    callee->kind = Expr::Kind::Ident;
                    callee->name = path[0];
                } else {
                    callee->kind = Expr::Kind::Path;
                    callee->name = path.back();
                    callee->path = path;
                }
                auto call = std::make_unique<Expr>();
                call->kind = Expr::Kind::Call;
                call->name = path.back();
                if (path.size() > 1) call->path = std::move(path);
                call->lhs = std::move(callee);
                call->generic_args = std::move(generic_args);
                while (!check(TokenKind::RParen) && !is_at_end()) {
                    call->args.push_back(parse_expr());
                    if (!match(TokenKind::Comma)) break;
                }
                const Token& rp = consume(TokenKind::RParen, "')' after generic call arguments");
                call->span = SourceSpan::merge(first.span, rp.span);
                return call;
            }
            auto id = std::make_unique<Expr>();
            id->span = SourceSpan::merge(first.span, previous().span);
            if (path.size() == 1) {
                id->kind = Expr::Kind::Ident;
                id->name = std::move(path[0]);
            } else {
                id->kind = Expr::Kind::Path;
                id->name = path.back();
                id->path = std::move(path);
            }
            return id;
        }
        error(peek().span, "expected expression, got '" + peek().text + "'");
        Token bad = advance();
        auto fallback = std::make_unique<Expr>();
        fallback->kind = Expr::Kind::IntLit;
        fallback->span = bad.span;
        return fallback;
    }
};
} // namespace
std::unique_ptr<ModuleAst> parse_module(
    const SourceMap& src,
    const std::vector<Token>& tokens,
    DiagnosticBag& diags) {
    Parser p(src, tokens, diags);
    return p.parse();
}
std::unique_ptr<ModuleAst> parse_source(
    std::string_view source,
    std::string file_name,
    DiagnosticBag& diags) {
    SourceMap sm(std::move(file_name), std::string(source));
    auto tokens = lex_source(sm, diags);
    if (diags.has_errors()) {
        return nullptr;
    }
    return parse_module(sm, tokens, diags);
}
} // namespace brokernel
