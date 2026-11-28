#include "parser_internal.h"

#include <iterator>
#include <utility>

namespace brokernel::detail {

std::unique_ptr<ModuleAst> Parser::parse() {
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
        if (check(TokenKind::At)) {
            auto more = parse_attributes();
            attrs.insert(attrs.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
        }
        if (check(TokenKind::KwImport) || check(TokenKind::KwFrom)) {
            if (!attrs.empty()) error(attrs.front().span, "attributes are not allowed on import declarations");
            mod->imports.push_back(parse_import(is_pub));
        } else if (check(TokenKind::KwStruct)) {
            if (!attrs.empty()) error(attrs.front().span, "attributes are not allowed on struct declarations");
            mod->structs.push_back(parse_struct(is_pub));
        } else if (check(TokenKind::KwEnum)) {
            if (!attrs.empty()) error(attrs.front().span, "attributes are not allowed on enum declarations");
            mod->enums.push_back(parse_enum(is_pub));
        } else if (check(TokenKind::Ident) && peek().text == "newtype") {
            if (!attrs.empty()) error(attrs.front().span, "attributes are not allowed on newtype declarations");
            mod->newtypes.push_back(parse_newtype(is_pub));
        } else if (check(TokenKind::KwConst)) {
            mod->consts.push_back(parse_const(is_pub, std::move(attrs)));
        } else if (check(TokenKind::KwFn)) {
            mod->functions.push_back(parse_fn(is_pub, std::move(attrs)));
        } else if (check(TokenKind::KwVar) || check(TokenKind::KwShared)) {
            BindingDecl d;
            d.is_pub = is_pub;
            d.attributes = std::move(attrs);
            d.span = advance().span;
            if (previous().kind == TokenKind::KwShared) {
                d.is_shared = true;
                match(TokenKind::KwVar);
            }
            d.name = consume(TokenKind::Ident, "resource name").text;
            consume(TokenKind::Colon, "':' and the resource's type");
            d.type = parse_type();
            d.span = SourceSpan::merge(d.span, consume(TokenKind::Semi, "';' after a resource").span);
            mod->bindings.push_back(std::move(d));
        } else if (check(TokenKind::Semi)) {
            advance();
        } else {
            error(peek().span, "expected declaration ('import', 'struct', 'enum', 'newtype', 'const', or 'fn'), got '" + peek().text + "'");
            synchronize();
        }
    }
    return mod;
}

std::vector<std::string> Parser::parse_ident_path(const char* what) {
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

std::vector<Attribute> Parser::parse_attributes() {
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

ImportDecl Parser::parse_import(bool is_pub) {
    ImportDecl imp;
    imp.is_pub = is_pub;
    imp.span = peek().span;
    if (match(TokenKind::KwFrom)) {
        if (check(TokenKind::StringLit)) imp.file_path = advance().text;
        else imp.module_path = parse_ident_path("module name");
        consume(TokenKind::KwImport, "'import' in from-import declaration");
        if (match(TokenKind::Star)) {
            imp.wildcard = true;
        } else {
            bool braced = match(TokenKind::LBrace);
            do {
                imp.symbols.push_back(consume(TokenKind::Ident, "imported symbol name").text);
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
        if (match(TokenKind::KwAs)) imp.alias = consume(TokenKind::Ident, "module alias").text;
        else imp.wildcard = true;
        match(TokenKind::Semi);
        imp.span = SourceSpan::merge(imp.span, previous().span);
        return imp;
    }
    imp.module_path.push_back(consume(TokenKind::Ident, "module name").text);
    while (match(TokenKind::ColonColon)) {
        if (match(TokenKind::Star)) { imp.wildcard = true; break; }
        if (match(TokenKind::LBrace)) {
            while (!check(TokenKind::RBrace) && !is_at_end()) {
                if (match(TokenKind::Star)) imp.wildcard = true;
                else imp.symbols.push_back(consume(TokenKind::Ident, "imported symbol").text);
                if (!match(TokenKind::Comma)) break;
            }
            consume(TokenKind::RBrace, "'}' after import list");
            break;
        }
        imp.module_path.push_back(consume(TokenKind::Ident, "module path segment").text);
    }
    if (match(TokenKind::KwAs)) imp.alias = consume(TokenKind::Ident, "module alias").text;
    match(TokenKind::Semi);
    imp.span = SourceSpan::merge(imp.span, previous().span);
    return imp;
}

std::vector<GenericParamDecl> Parser::parse_generic_params() {
    std::vector<GenericParamDecl> params;
    if (!match(TokenKind::Lt)) return params;
    while (!check(TokenKind::Gt) && !is_at_end()) {
        GenericParamDecl gp;
        gp.span = peek().span;
        gp.is_const = match(TokenKind::KwConst);
        gp.name = consume(TokenKind::Ident, "generic parameter name").text;
        if (match(TokenKind::Colon)) {
            gp.constraint_or_type = parse_type();
            if (gp.constraint_or_type &&
                (gp.constraint_or_type->name == "i32" || gp.constraint_or_type->name == "u32" ||
                 gp.constraint_or_type->name == "int" || gp.constraint_or_type->name == "uint" ||
                 gp.constraint_or_type->name == "bool")) {
                gp.is_const = true;
            }
        }
        if (match(TokenKind::Eq)) {
            bool neg = match(TokenKind::Minus);
            if (check(TokenKind::IntLit)) {
                int64_t v = advance().int_val;
                gp.default_int = neg ? -v : v;
            } else if (match(TokenKind::KwTrue)) {
                gp.default_int = 1;
            } else if (match(TokenKind::KwFalse)) {
                gp.default_int = 0;
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

GenericArg Parser::parse_generic_arg() {
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
    if (match(TokenKind::KwTrue)) {
        arg.int_arg = 1;
        arg.span = previous().span;
        return arg;
    }
    if (match(TokenKind::KwFalse)) {
        arg.int_arg = 0;
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

TypeExprPtr Parser::parse_type() {
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

StructDecl Parser::parse_struct(bool is_pub) {
    StructDecl s;
    s.is_pub = is_pub;
    s.span = consume(TokenKind::KwStruct, "'struct'").span;
    s.name = consume(TokenKind::Ident, "struct name").text;
    s.generic_params = parse_generic_params();
    consume(TokenKind::LBrace, "'{' in struct declaration");
    while (!check(TokenKind::RBrace) && !is_at_end()) {
        const size_t before = pos_;
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
        if (pos_ == before) advance();
    }
    const Token& rb = consume(TokenKind::RBrace, "'}' after struct fields");
    match(TokenKind::Semi);
    s.span = SourceSpan::merge(s.span, rb.span);
    return s;
}

EnumDecl Parser::parse_enum(bool is_pub) {
    EnumDecl e;
    e.is_pub = is_pub;
    e.span = consume(TokenKind::KwEnum, "'enum'").span;
    e.name = consume(TokenKind::Ident, "enum name").text;
    if (match(TokenKind::Colon)) {
        e.underlying = parse_type();
    }
    consume(TokenKind::LBrace, "'{' in enum declaration");
    while (!check(TokenKind::RBrace) && !is_at_end()) {
        EnumVariantDecl v;
        v.span = peek().span;
        v.name = consume(TokenKind::Ident, "enum variant name").text;
        if (match(TokenKind::Eq)) {
            v.value = parse_expr();
        }
        v.span = SourceSpan::merge(v.span, previous().span);
        e.variants.push_back(std::move(v));
        if (!match(TokenKind::Comma)) {
            match(TokenKind::Semi);
        }
    }
    const Token& rb = consume(TokenKind::RBrace, "'}' after enum variants");
    match(TokenKind::Semi);
    e.span = SourceSpan::merge(e.span, rb.span);
    return e;
}

NewtypeDecl Parser::parse_newtype(bool is_pub) {
    NewtypeDecl nt;
    nt.is_pub = is_pub;
    nt.span = advance().span; // contextual 'newtype'
    nt.name = consume(TokenKind::Ident, "newtype name").text;
    consume(TokenKind::Eq, "'=' in newtype declaration");
    nt.target = parse_type();
    const Token& semi = consume(TokenKind::Semi, "';' after newtype declaration");
    nt.span = SourceSpan::merge(nt.span, semi.span);
    return nt;
}

ConstDecl Parser::parse_const(bool is_pub, std::vector<Attribute> attrs) {
    ConstDecl c;
    c.is_pub = is_pub;
    for (const auto& a : attrs) {
        if (a.name == "knob") c.is_knob = true;
        else error(a.span, "attributes other than '@knob' are not allowed on const declarations");
    }
    c.attributes = std::move(attrs);
    c.span = consume(TokenKind::KwConst, "'const'").span;
    if (!c.attributes.empty()) c.span = SourceSpan::merge(c.attributes.front().span, c.span);
    c.name = consume(TokenKind::Ident, "constant name").text;
    if (match(TokenKind::Colon)) c.type = parse_type();
    consume(TokenKind::Eq, "'=' in const declaration");
    c.value = parse_expr();
    const Token& semi = consume(TokenKind::Semi, "';' after const declaration");
    c.span = SourceSpan::merge(c.span, semi.span);
    return c;
}

FnDecl Parser::parse_fn(bool is_pub, std::vector<Attribute> attrs) {
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
        if (check(TokenKind::Ident) &&
            (peek().text == "inout" || peek().text == "out" || peek().text == "mut") &&
            pos_ + 1 < tokens_.size() && tokens_[pos_ + 1].kind == TokenKind::Ident) {
            advance();
        }
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

} // namespace brokernel::detail
