#include "sema_internal.h"
#include <algorithm>
#include <cstdlib>

namespace brokernel::detail {

Type FnCheckState::check_method_call(Expr* e) {
    Type recv_t = check_expr(e->lhs.get());
    const std::string& m = e->name;
    if (recv_t.is_grid()) {
        uint32_t dim = recv_t.dimension;
        ensure_grid_dim(dim);
        if (m == "x" || m == "y" || m == "z" || m == "width" || m == "height" || m == "depth") {
            e->resolved_type = Type::i32();
            return Type::i32();
        }
        if (m == "linear_index" || m == "total_cells") {
            e->resolved_type = Type::i64();
            return Type::i64();
        }
        if (m == "coord" || m == "pos") {
            Type rt = (dim == 1) ? Type::i32() : Type::vec(ScalarKind::I32, dim);
            e->resolved_type = rt;
            return rt;
        }
        if (m == "uv") {
            Type rt = (dim == 1) ? Type::f32() : Type::vec(ScalarKind::F32, dim);
            e->resolved_type = rt;
            return rt;
        }
        if (m == "uv_f64") {
            note_f64(&e->span);
            Type rt = (dim == 1) ? Type::f64() : Type::vec(ScalarKind::F64, dim);
            e->resolved_type = rt;
            return rt;
        }
        ctx.diags.error(e->span, "unknown grid method '" + m + "'");
        return Type::void_type();
    }
    if (recv_t.is_stencil()) {
        LocalVar* lv = (e->lhs->kind == Expr::Kind::Ident) ? lookup_var(e->lhs->name) : nullptr;
        CheckedParam* cparam = (lv && lv->param_index >= 0) ? &fn.params[lv->param_index] : nullptr;
        if (m == "radius") {
            int32_t r = cparam && cparam->declared_radius >= 0 ? cparam->declared_radius : recv_t.stencil_radius;
            e->kind = Expr::Kind::IntLit;
            e->int_val = std::max(0, r);
            e->resolved_type = Type::i32();
            return Type::i32();
        }
        if (m == "write" || m == "store") {
            ctx.diags.error(e->span, "cannot write to read-only stencil");
            return Type::void_type();
        }
        if (m == "read" || m == "center") {
            if (!e->args.empty()) {
                ctx.diags.error(e->span, "stencil.read() takes 0 arguments");
            }
            e->args.clear();
            for (uint32_t d = 0; d < recv_t.dimension; ++d) {
                auto zero = std::make_unique<Expr>();
                zero->kind = Expr::Kind::IntLit;
                zero->span = e->span;
                zero->int_val = 0;
                zero->resolved_type = Type::i32();
                e->args.push_back(std::move(zero));
            }
            e->name = "at";
        }
        if (e->name == "at") {
            if (cparam) {
                validate_stencil_access(*cparam, e->args, e->span);
            }
            Type elem = recv_t.element_type();
            mark_type_usage(elem);
            e->resolved_type = elem;
            return elem;
        }
        ctx.diags.error(e->span, "unknown stencil method '" + m + "'");
        return Type::void_type();
    }
    if (recv_t.is_store()) {
        uint32_t dim = recv_t.dimension;
        Type elem = recv_t.element_type();
        mark_type_usage(elem);
        if (m == "read") {
            if (!can_Read(recv_t.access)) {
                ctx.diags.error(e->span, "cannot read from write-only store");
            }
            ensure_grid_dim(dim);
            if (!e->args.empty()) {
                ctx.diags.error(e->span, "store.read() takes 0 arguments (use store.load(...) for explicit coordinates)");
            }
            e->resolved_type = elem;
            return elem;
        }
        if (m == "write") {
            if (!can_Write(recv_t.access)) {
                ctx.diags.error(e->span, "cannot write to read-only store");
            }
            ensure_grid_dim(dim);
            if (e->args.size() != 1) {
                ctx.diags.error(e->span, "store.write(value) expects 1 argument");
            } else {
                check_expr(e->args[0].get(), elem);
                check_assignable(e->args[0].get(), elem, ctx.diags, e->args[0]->span);
            }
            e->resolved_type = Type::void_type();
            return Type::void_type();
        }
        if (m == "load") {
            if (!can_Read(recv_t.access)) {
                ctx.diags.error(e->span, "cannot read from write-only store");
            }
            if (e->args.size() != dim) {
                ctx.diags.error(e->span, "store.load expects " + std::to_string(dim) + " coordinate argument(s)");
            }
            for (auto& a : e->args) {
                Type at = check_expr(a.get(), Type::i32());
                if (!at.is_integer() || !at.is_scalar()) {
                    ctx.diags.error(a->span, "store.load coordinate must be a scalar integer, got '" + at.to_string() + "'");
                }
            }
            e->resolved_type = elem;
            return elem;
        }
        if (m == "store") {
            if (!can_Write(recv_t.access)) {
                ctx.diags.error(e->span, "cannot write to read-only store");
            }
            if (e->args.size() != dim + 1) {
                ctx.diags.error(e->span, "store.store expects " + std::to_string(dim) + " coordinate argument(s) and 1 value argument");
            } else {
                for (size_t i = 0; i < dim; ++i) {
                    Type at = check_expr(e->args[i].get(), Type::i32());
                    if (!at.is_integer() || !at.is_scalar()) {
                        ctx.diags.error(e->args[i]->span, "store.store coordinate must be a scalar integer, got '" + at.to_string() + "'");
                    }
                }
                check_expr(e->args[dim].get(), elem);
                check_assignable(e->args[dim].get(), elem, ctx.diags, e->args[dim]->span);
            }
            e->resolved_type = Type::void_type();
            return Type::void_type();
        }
        if (m == "atomic_add" || m == "atomic_min" || m == "atomic_max" || m == "atomic_exchange" ||
            m == "atomic_or" || m == "atomic_and" || m == "atomic_xor" || m == "atomic_cas" ||
            m == "add" || m == "min" || m == "max" || m == "exchange" ||
            m == "or" || m == "and" || m == "xor" || m == "cas" || m == "compare_exchange") {
            std::string canonical = (m == "compare_exchange" || m == "cas") ? "atomic_cas"
                                  : (m.rfind("atomic_", 0) == 0 ? m : "atomic_" + m);
            e->name = canonical;
            if (!can_Write(recv_t.access) || !can_Read(recv_t.access)) {
                ctx.diags.error(e->span, "atomic store operation requires a read-write store ('rw_store' or 'store')");
            }
            bool is_cas = (canonical == "atomic_cas");
            bool valid_elem = elem.is_i32() || elem.is_u32() || (canonical == "atomic_add" && elem.is_f32());
            if (!valid_elem) {
                std::string allowed = (canonical == "atomic_add") ? "'i32', 'u32', or 'f32'" : "'i32' or 'u32'";
                ctx.diags.error(e->span, canonical + " requires store element type " + allowed + ", got '" + elem.to_string() + "'");
            }
            size_t val_args = is_cas ? 2 : 1;
            if (e->args.size() != dim + val_args) {
                ctx.diags.error(e->span, canonical + " expects " + std::to_string(dim) + " coordinate argument(s) and " +
                                         std::to_string(val_args) + " value argument(s)");
            } else {
                for (size_t i = 0; i < dim; ++i) {
                    Type at = check_expr(e->args[i].get(), Type::i32());
                    if (!at.is_integer() || !at.is_scalar()) {
                        ctx.diags.error(e->args[i]->span, canonical + " coordinate must be a scalar integer, got '" + at.to_string() + "'");
                    }
                }
                for (size_t i = dim; i < dim + val_args; ++i) {
                    check_expr(e->args[i].get(), elem);
                    check_assignable(e->args[i].get(), elem, ctx.diags, e->args[i]->span);
                }
            }
            e->resolved_type = elem;
            return elem;
        }
        ctx.diags.error(e->span, "unknown store method '" + m + "'");
        return Type::void_type();
    }
    if (recv_t.is_image() || recv_t.is_sampler()) {
        return check_texture_method(e, recv_t);
    }
    // A buffer's element count (its runtime array's).
    if (recv_t.is_buffer() && m == "len") {
        if (!e->args.empty()) ctx.diags.error(e->span, "len() takes no arguments");
        e->resolved_type = Type::i32();
        return e->resolved_type;
    }
    // Atomics as methods on @binding buffer<T> and on shared variables/arrays.
    if (m == "atomic_add" || m == "atomic_min" || m == "atomic_max" || m == "atomic_exchange" ||
        m == "atomic_or" || m == "atomic_and" || m == "atomic_xor" || m == "atomic_cas" ||
        m == "add" || m == "min" || m == "max" || m == "exchange" ||
        m == "or" || m == "and" || m == "xor" || m == "cas" || m == "compare_exchange") {
        std::string canonical = (m == "compare_exchange" || m == "cas") ? "atomic_cas"
                              : (m.rfind("atomic_", 0) == 0 ? m : "atomic_" + m);
        e->name = canonical;
        const Expr* base = e->lhs.get();
        while (base && (base->kind == Expr::Kind::Field || base->kind == Expr::Kind::Index)) base = base->lhs.get();
        bool is_buf = false;
        bool is_sh = false;
        bool read_only_buf = false;
        if (base && (base->kind == Expr::Kind::Ident || base->kind == Expr::Kind::Path)) {
            if (base->kind == Expr::Kind::Ident) {
                if (LocalVar* lv = lookup_var(base->name)) {
                    if (lv->is_shared) is_sh = true;
                    if (lv->type.is_buffer()) { is_buf = true; read_only_buf = (lv->type.access == AccessMode::Read); }
                }
            }
            if (!is_sh && !is_buf && base->global_index >= 0) {
                const auto& gp = ctx.out->globals[static_cast<size_t>(base->global_index)];
                if (gp.role == ParamRole::Shared) is_sh = true;
                if (gp.role == ParamRole::Buffer) { is_buf = true; read_only_buf = (gp.access == AccessMode::Read); }
            }
        }
        if (is_buf || is_sh) {
            if (is_buf && read_only_buf) {
                ctx.diags.error(e->span, "atomic operation on read-only buffer");
            }
            if (is_sh) {
                require_stage(Stage::Compute, canonical, e->span);
                fn.uses_barriers_or_shared = true;
            }
            bool is_cas = (canonical == "atomic_cas");
            size_t val_args = is_cas ? 2 : 1;
            // If index argument(s) are given before the value argument(s), e.g. `bins.atomic_add(i, val)`
            // or `words.atomic_or(idx, mask)` or `c.add(i, j, val)`, fold them into `e->lhs` as Index nodes.
            while (e->args.size() > val_args && (recv_t.is_buffer() || recv_t.is_array() || recv_t.is_vector())) {
                ExprPtr idx_arg = std::move(e->args.front());
                e->args.erase(e->args.begin());
                Type at = check_expr(idx_arg.get(), Type::i32());
                if (!at.is_integer() || !at.is_scalar()) {
                    ctx.diags.error(idx_arg->span, canonical + " index must be a scalar integer, got '" + at.to_string() + "'");
                }
                auto idx_expr = std::make_unique<Expr>();
                idx_expr->kind = Expr::Kind::Index;
                idx_expr->span = SourceSpan::merge(e->lhs->span, idx_arg->span);
                idx_expr->lhs = std::move(e->lhs);
                idx_expr->args.push_back(std::move(idx_arg));
                recv_t = recv_t.element_type();
                idx_expr->resolved_type = recv_t;
                e->lhs = std::move(idx_expr);
            }
            bool valid_elem = recv_t.is_i32() || recv_t.is_u32() || (canonical == "atomic_add" && recv_t.is_f32());
            if (!valid_elem) {
                std::string allowed = (canonical == "atomic_add") ? "'i32', 'u32', or 'f32'" : "'i32' or 'u32'";
                ctx.diags.error(e->span, canonical + " requires scalar place of type " + allowed + ", got '" + recv_t.to_string() + "'");
            }
            if (e->args.size() != val_args) {
                ctx.diags.error(e->span, canonical + " expects " + std::to_string(val_args) + " value argument(s) after indices");
                for (auto& a : e->args) check_expr(a.get());
            } else {
                for (size_t i = 0; i < val_args; ++i) {
                    check_expr(e->args[i].get(), recv_t);
                    check_assignable(e->args[i].get(), recv_t, ctx.diags, e->args[i]->span);
                }
            }
            e->resolved_type = recv_t;
            return recv_t;
        }
    }
    ctx.diags.error(e->span, "type '" + recv_t.to_string() + "' has no method '" + m + "'");
    return Type::void_type();
}

} // namespace brokernel::detail
