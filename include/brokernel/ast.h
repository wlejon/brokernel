#pragma once
// Tokens and Abstract Syntax Tree (AST) for brokernel (.bk).

#include "brokernel/diagnostics.h"
#include "brokernel/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace brokernel {

enum class TokenKind : uint8_t {
    Eof = 0,
    Invalid,

    // Identifiers and literals
    Ident,
    IntLit,
    FloatLit,
    StringLit,

    // Keywords
    KwModule,
    KwImport,
    KwFrom,
    KwAs,
    KwPub,
    KwFn,
    KwStruct,
    KwConst,
    KwLet,
    KwVar,
    KwIf,
    KwElse,
    KwFor,
    KwIn,
    KwWhile,
    KwBreak,
    KwContinue,
    KwReturn,
    KwTrue,
    KwFalse,

    // Punctuation & operators
    At,          // @
    LParen,      // (
    RParen,      // )
    LBrace,      // {
    RBrace,      // }
    LBracket,    // [
    RBracket,    // ]
    Comma,       // ,
    Semi,        // ;
    Colon,       // :
    ColonColon,  // ::
    Dot,         // .
    DotDot,      // ..
    DotDotEq,    // ..=
    Arrow,       // ->
    Question,    // ?

    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    Amp,         // &
    Pipe,        // |
    Caret,       // ^
    Tilde,       // ~
    Bang,        // !
    Shl,         // <<

    AmpAmp,      // &&
    PipePipe,    // ||
    EqEq,        // ==
    BangEq,      // !=
    Lt,          // <
    Le,          // <=
    Gt,          // >
    Ge,          // >=

    Eq,          // =
    PlusEq,      // +=
    MinusEq,     // -=
    StarEq,      // *=
    SlashEq,     // /=
    PercentEq,   // %=
    AmpEq,       // &=
    PipeEq,      // |=
    CaretEq,     // ^=
    ShlEq,       // <<=
    ShrEq,       // >>=
};

struct Token {
    TokenKind kind = TokenKind::Eof;
    SourceSpan span;
    std::string text;
    int64_t int_val = 0;
    double float_val = 0.0;
    ScalarKind literal_suffix = ScalarKind::Void; // Void = unsuffixed (contextual)
};

/// Tokenizes a .bk source buffer.
std::vector<Token> lex_source(const SourceMap& src, DiagnosticBag& diags);

// ---------------------------------------------------------------------------
// AST Nodes
// ---------------------------------------------------------------------------

struct Expr;
struct Stmt;
struct TypeExpr;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;
using TypeExprPtr = std::unique_ptr<TypeExpr>;

/// A type or compile-time integer argument inside `<...>`, e.g. `vec<f64, 3>`
/// or `stencil<f32, 2, 1>` or `fbm<f64, 3>(p)`.
struct GenericArg {
    SourceSpan span;
    TypeExprPtr type_arg;              // Non-null when this argument is a type or identifier
    std::optional<int64_t> int_arg;    // Set when this argument is a literal integer constant
    std::string ident_or_const_name;   // Name when argument could be either a type param T or const param D
};

struct TypeExpr {
    SourceSpan span;
    std::string name;                  // e.g. "f32", "f64", "vec3", "vec", "stencil", "store", "T"
    std::vector<GenericArg> args;
};

struct GenericParamDecl {
    SourceSpan span;
    std::string name;
    bool is_const = false;             // true for `const D: i32`
    TypeExprPtr constraint_or_type;    // Optional type for const param (`i32`) or trait bound
    std::optional<int64_t> default_int;
    TypeExprPtr default_type;
};

struct Attribute {
    SourceSpan span;
    std::string name;                  // e.g. "kernel", "compute", "vertex", "fragment", "grid", "workgroup_size", "binding", "unroll"
    std::vector<int64_t> int_args;
    std::vector<std::string> str_args;
};

// ---- Expressions ----------------------------------------------------------

enum class UnaryOp : uint8_t {
    Neg = 0,     // -x
    Pos,         // +x
    Not,         // !x
    BitNot,      // ~x
};

enum class BinaryOp : uint8_t {
    Add = 0,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogAnd,
    LogOr,
};

struct FieldInit {
    SourceSpan span;
    std::string name;
    ExprPtr value;
};

struct Expr {
    enum class Kind : uint8_t {
        IntLit = 0,
        FloatLit,
        BoolLit,
        Ident,
        Path,        // module::item
        Unary,
        Binary,
        Select,      // cond ? a : b
        Cast,        // expr as Type
        Call,        // fn_or_type<Args>(args...)
        MethodCall,  // receiver.method(args...)
        Field,       // receiver.field or vector swizzle (.x, .xy, .zyx)
        Index,       // receiver[i] or stencil[dx, dy]
        StructLit,   // StructName<Args> { field: val, ... }
    };

    Kind kind = Kind::IntLit;
    SourceSpan span;
    Type resolved_type; // Filled by Sema

    // Literal payloads
    int64_t int_val = 0;
    double float_val = 0.0;
    bool bool_val = false;
    ScalarKind literal_suffix = ScalarKind::Void; // Void = unsuffixed contextual literal

    // Identifier / Path
    std::string name;
    std::vector<std::string> path; // For Path (`a::b`)

    // Operators
    UnaryOp unary_op = UnaryOp::Neg;
    BinaryOp binary_op = BinaryOp::Add;

    // Sub-expressions
    ExprPtr lhs;
    ExprPtr rhs;
    ExprPtr cond;

    // Calls / Generics / Indexing / Struct literals
    std::vector<GenericArg> generic_args;
    std::vector<ExprPtr> args;
    std::vector<FieldInit> fields;
    TypeExprPtr target_type;
};

// ---- Statements -----------------------------------------------------------

struct Stmt {
    enum class Kind : uint8_t {
        Block = 0,
        Let,         // let name [: Type] = expr;
        Var,         // var name [: Type] [= expr];
        Assign,      // target = expr; (compound assignments desugar or carry op)
        If,
        ForRange,    // for i in start..end { ... } or start..=end
        ForC,        // for (init; cond; step) { ... }
        While,
        Break,
        Continue,
        Return,
        Expr,
    };

    Kind kind = Kind::Block;
    SourceSpan span;
    std::vector<Attribute> attributes;

    // Let / Var
    std::string var_name;
    TypeExprPtr var_type;
    ExprPtr init_expr;
    Type resolved_var_type;

    // Assign
    ExprPtr assign_target;
    ExprPtr assign_value;
    std::optional<BinaryOp> compound_op; // nullopt for plain `=`

    // If / While / For
    ExprPtr cond;
    std::vector<StmtPtr> body;
    std::vector<StmtPtr> else_body;

    // ForRange / ForC
    bool inclusive_range = false; // `..=` vs `..`
    ExprPtr range_start;
    ExprPtr range_end;
    ExprPtr range_step;
    StmtPtr for_init;
    StmtPtr for_step;

    // Return / ExprStmt
    ExprPtr expr;
};

// ---- Top-level Declarations -----------------------------------------------

struct StructFieldDecl {
    SourceSpan span;
    std::string name;
    TypeExprPtr type;
};

struct StructDecl {
    SourceSpan span;
    bool is_pub = true;
    std::string name;
    std::vector<GenericParamDecl> generic_params;
    std::vector<StructFieldDecl> fields;
};

struct ConstDecl {
    SourceSpan span;
    bool is_pub = true;
    std::string name;
    TypeExprPtr type;
    ExprPtr value;
};

struct ParamDecl {
    SourceSpan span;
    std::vector<Attribute> attributes;
    std::string name;
    TypeExprPtr type;
    ExprPtr default_value;

    const Attribute* find_attribute(std::string_view attr_name) const noexcept {
        for (const auto& a : attributes) {
            if (a.name == attr_name) return &a;
        }
        return nullptr;
    }
};

struct FnDecl {
    SourceSpan span;
    bool is_pub = true;
    std::vector<Attribute> attributes;
    std::string name;
    std::vector<GenericParamDecl> generic_params;
    std::vector<ParamDecl> params;
    TypeExprPtr return_type;
    std::vector<StmtPtr> body;

    bool has_attribute(std::string_view attr_name) const noexcept {
        for (const auto& a : attributes) {
            if (a.name == attr_name) return true;
        }
        return false;
    }

    const Attribute* find_attribute(std::string_view attr_name) const noexcept {
        for (const auto& a : attributes) {
            if (a.name == attr_name) return &a;
        }
        return nullptr;
    }

    bool is_entry_kernel() const noexcept {
        return has_attribute("kernel") || has_attribute("compute") ||
               has_attribute("vertex") || has_attribute("fragment");
    }
};

struct ImportDecl {
    SourceSpan span;
    std::vector<std::string> module_path; // e.g. {"std", "noise"} or {"noise"}
    std::string file_path;                // Non-empty if `import "path.bk" as alias;`
    std::string alias;                    // Optional alias (`import std::noise as n;`)
    std::vector<std::string> symbols;     // Selective imports (`import noise::{fbm, simplex};`)
    bool wildcard = false;                // `import noise::*;`
};

struct ModuleAst {
    SourceSpan span;
    std::string module_name;
    std::string file_path;
    std::vector<ImportDecl> imports;
    std::vector<ConstDecl> consts;
    std::vector<StructDecl> structs;
    std::vector<FnDecl> functions;
};

/// Parses a .bk source buffer into a ModuleAst.
std::unique_ptr<ModuleAst> parse_module(
    const SourceMap& src,
    const std::vector<Token>& tokens,
    DiagnosticBag& diags);

std::unique_ptr<ModuleAst> parse_source(
    std::string_view source,
    std::string file_name,
    DiagnosticBag& diags);

} // namespace brokernel
