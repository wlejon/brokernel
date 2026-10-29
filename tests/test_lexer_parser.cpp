// Lexer and Parser tests for brokernel (.bk).

#include "brokernel/brokernel.h"

#undef NDEBUG
#include <cassert>
#include <iostream>
#include <string>

using namespace brokernel;

static void test_lex_tokens() {
    const char* src = R"(
        module world::terrain;
        import std::noise::{fbm, simplex3d};
        from filter import laplacian2d;
        const RADIUS: f64 = 6371000.0f64;
        @kernel @grid(2) @workgroup_size(8, 8, 1)
        fn erode(h: stencil<f64, 2, 1>, out: write_store<f64, 2>, dt: f64) {
            for dy in -1..=1 {
                let v = h.at(0, dy);
            }
        }
    )";
    SourceMap sm("terrain.bk", src);
    DiagnosticBag diags;
    auto tokens = lex_source(sm, diags);
    assert(!diags.has_errors());
    assert(!tokens.empty());
    assert(tokens.front().kind == TokenKind::KwModule);
    assert(tokens.back().kind == TokenKind::Eof);

    auto ast = parse_module(sm, tokens, diags);
    assert(ast != nullptr);
    assert(!diags.has_errors());
    assert(ast->module_name == "world::terrain");
    assert(ast->imports.size() == 2);
    assert(ast->imports[0].symbols.size() == 2);
    assert(ast->imports[1].symbols.size() == 1);
    assert(ast->consts.size() == 1);
    assert(ast->functions.size() == 1);
    assert(ast->functions[0].is_entry_kernel());
    assert(ast->functions[0].has_attribute("grid"));
    assert(ast->functions[0].has_attribute("workgroup_size"));
}

static void test_parse_generics_and_expressions() {
    const char* src = R"(
        struct Sample<T, const D: i32> {
            pos: vec<T, D>,
            height: T,
        }

        fn blend<T>(a: T, b: T, w: T) -> T {
            let clamped = w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
            return a * (1.0 - clamped) + b * clamped;
        }
    )";
    DiagnosticBag diags;
    auto ast = parse_source(src, "generics.bk", diags);
    assert(ast != nullptr);
    assert(!diags.has_errors());
    assert(ast->structs.size() == 1);
    assert(ast->structs[0].generic_params.size() == 2);
    assert(!ast->structs[0].generic_params[0].is_const);
    assert(ast->structs[0].generic_params[1].is_const);
    assert(ast->functions.size() == 1);
    assert(ast->functions[0].generic_params.size() == 1);
}

static void test_parse_syntax_error_diagnostics() {
    const char* bad_src = R"(
        fn broken(x: f64 -> f64 {
            return x + ;
        }
    )";
    DiagnosticBag diags;
    auto ast = parse_source(bad_src, "broken.bk", diags);
    (void)ast;
    assert(diags.has_errors());
    std::string formatted = diags.format_all();
    assert(formatted.find("broken.bk:") != std::string::npos);
    assert(formatted.find("error:") != std::string::npos);
}

int main() {
    test_lex_tokens();
    test_parse_generics_and_expressions();
    test_parse_syntax_error_diagnostics();
    std::cout << "test_lexer_parser: ALL PASSED\n";
    return 0;
}
