#include "parser_internal.h"

#include <utility>

namespace brokernel {

namespace detail {

// ---- Statements -----------------------------------------------------------
std::vector<StmtPtr> Parser::parse_block_stmts() {
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

StmtPtr Parser::parse_stmt() {
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
    if (match(TokenKind::KwShared)) {
        match(TokenKind::KwVar);
        auto st = parse_let_or_var(true, start_sp);
        st->is_shared = true;
        st->attributes = std::move(attrs);
        consume(TokenKind::Semi, "';' after shared variable declaration");
        st->span = SourceSpan::merge(start_sp, previous().span);
        return st;
    }
    if (match(TokenKind::KwLet) || match(TokenKind::KwVar)) {
        bool is_var = previous().kind == TokenKind::KwVar;
        if (!is_var && check(TokenKind::Ident) && peek().text == "mut") { advance(); is_var = true; }
        auto st = parse_let_or_var(is_var, start_sp);
        st->attributes = std::move(attrs);
        consume(TokenKind::Semi, "';' after variable declaration");
        st->span = SourceSpan::merge(start_sp, previous().span);
        return st;
    }
    if (match(TokenKind::KwIf)) return parse_if_stmt(start_sp, std::move(attrs));
    if (match(TokenKind::KwFor)) return parse_for_stmt(start_sp, std::move(attrs));
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
        if (!check(TokenKind::Semi) && !check(TokenKind::RBrace)) st->expr = parse_expr();
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

std::vector<StmtPtr> Parser::parse_stmt_or_block() {
    if (check(TokenKind::LBrace)) {
        return parse_block_stmts();
    }
    std::vector<StmtPtr> out;
    if (StmtPtr s = parse_stmt()) out.push_back(std::move(s));
    return out;
}

StmtPtr Parser::parse_let_or_var(bool is_var, const SourceSpan& start_sp) {
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

StmtPtr Parser::parse_if_stmt(const SourceSpan& start_sp, std::vector<Attribute> attrs) {
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

StmtPtr Parser::parse_for_stmt(const SourceSpan& start_sp, std::vector<Attribute> attrs) {
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

std::optional<std::optional<BinaryOp>> Parser::match_assign_op() {
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
ExprPtr Parser::parse_expr() {
    return parse_ternary();
}

ExprPtr Parser::parse_ternary() {
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

int Parser::binary_prec(TokenKind k, BinaryOp* out_op) noexcept {
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

ExprPtr Parser::parse_binary(int min_prec) {
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

ExprPtr Parser::parse_cast() {
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

ExprPtr Parser::parse_unary() {
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
bool Parser::try_parse_generic_call_args(std::vector<GenericArg>& out_args) {
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
        } else if (match(TokenKind::KwTrue)) {
            ga.int_arg = 1;
        } else if (match(TokenKind::KwFalse)) {
            ga.int_arg = 0;
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

bool Parser::try_parse_nested_type_args(std::vector<GenericArg>& out_args) {
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

ExprPtr Parser::parse_postfix() {
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
            if (check(TokenKind::IntLit) && peek().text == "0") {
                const Token& z = advance();
                auto fld = std::make_unique<Expr>();
                fld->kind = Expr::Kind::Field;
                fld->name = "0";
                fld->span = SourceSpan::merge(e->span, z.span);
                fld->lhs = std::move(e);
                e = std::move(fld);
                continue;
            }
            if (check(TokenKind::FloatLit) && peek().text == "0.0") {
                const Token& zz = advance();
                auto fld1 = std::make_unique<Expr>();
                fld1->kind = Expr::Kind::Field;
                fld1->name = "0";
                fld1->span = SourceSpan::merge(e->span, zz.span);
                fld1->lhs = std::move(e);
                auto fld2 = std::make_unique<Expr>();
                fld2->kind = Expr::Kind::Field;
                fld2->name = "0";
                fld2->span = fld1->span;
                fld2->lhs = std::move(fld1);
                e = std::move(fld2);
                continue;
            }
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

ExprPtr Parser::parse_primary() {
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
    if (match(TokenKind::StringLit)) {
        const Token& tok = previous();
        auto e = std::make_unique<Expr>();
        e->kind = Expr::Kind::StringLit;
        e->span = tok.span;
        e->name = tok.text;
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
            try_parse_generic_call_args(generic_args);
        }
        // Compile-time sizeof / std430_sizeof / std140_sizeof (type arg)
        if (path.size() == 1 && generic_args.empty() && check(TokenKind::LParen) &&
            (first.text == "sizeof" || first.text == "std430_sizeof" || first.text == "std140_sizeof")) {
            advance(); // '('
            auto call = std::make_unique<Expr>();
            call->kind = Expr::Kind::Call;
            call->name = first.text;
            GenericArg ga;
            ga.span = peek().span;
            ga.type_arg = parse_type();
            if (ga.type_arg && ga.type_arg->args.empty()) {
                ga.ident_or_const_name = ga.type_arg->name;
            }
            ga.span = SourceSpan::merge(ga.span, previous().span);
            call->generic_args.push_back(std::move(ga));
            const Token& rp = consume(TokenKind::RParen, "')' after type argument");
            call->span = SourceSpan::merge(first.span, rp.span);
            return call;
        }
        // Compile-time offsetof / std430_offsetof / std140_offsetof (Struct, field.path)
        if (path.size() == 1 && generic_args.empty() && check(TokenKind::LParen) &&
            (first.text == "offsetof" || first.text == "std430_offsetof" || first.text == "std140_offsetof")) {
            advance(); // '('
            auto call = std::make_unique<Expr>();
            call->kind = Expr::Kind::Call;
            call->name = first.text;
            GenericArg ga;
            ga.span = peek().span;
            ga.type_arg = parse_type();
            if (ga.type_arg && ga.type_arg->args.empty()) {
                ga.ident_or_const_name = ga.type_arg->name;
            }
            ga.span = SourceSpan::merge(ga.span, previous().span);
            call->generic_args.push_back(std::move(ga));
            consume(TokenKind::Comma, "',' after struct type");
            auto push_seg = [&](const Token& seg_tok) {
                auto seg = std::make_unique<Expr>();
                seg->kind = Expr::Kind::StringLit;
                seg->span = seg_tok.span;
                seg->name = seg_tok.text;
                call->args.push_back(std::move(seg));
            };
            push_seg(consume(TokenKind::Ident, "field name"));
            while (match(TokenKind::Dot)) {
                if (check(TokenKind::IntLit) && peek().text == "0") {
                    push_seg(advance());
                } else {
                    push_seg(consume(TokenKind::Ident, "nested field name after '.'"));
                }
            }
            const Token& rp = consume(TokenKind::RParen, "')' after field path");
            call->span = SourceSpan::merge(first.span, rp.span);
            return call;
        }
        // Struct literal `StructName { field: expr, ... }`
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
        if (!generic_args.empty() && match(TokenKind::LParen)) {
            auto callee = std::make_unique<Expr>();
            callee->span = first.span;
            callee->kind = path.size() == 1 ? Expr::Kind::Ident : Expr::Kind::Path;
            callee->name = path.back();
            if (path.size() > 1) callee->path = path;
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
        id->kind = path.size() == 1 ? Expr::Kind::Ident : Expr::Kind::Path;
        id->name = path.back();
        if (path.size() > 1) id->path = std::move(path);
        return id;
    }
    error(peek().span, "expected expression, got '" + peek().text + "'");
    Token bad = advance();
    auto fallback = std::make_unique<Expr>();
    fallback->kind = Expr::Kind::IntLit;
    fallback->span = bad.span;
    return fallback;
}

} // namespace detail

std::unique_ptr<ModuleAst> parse_module(const SourceMap& src, const std::vector<Token>& tokens, DiagnosticBag& diags) {
    detail::Parser p(src, tokens, diags);
    return p.parse();
}

std::unique_ptr<ModuleAst> parse_source(std::string_view source, std::string file_name, DiagnosticBag& diags) {
    SourceMap sm(std::move(file_name), std::string(source));
    auto tokens = lex_source(sm, diags);
    return diags.has_errors() ? nullptr : parse_module(sm, tokens, diags);
}

} // namespace brokernel
