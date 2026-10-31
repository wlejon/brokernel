#include "lowering_internal.h"
#include <algorithm>
#include <brass/target/host_cpu.hpp>
namespace brokernel {
namespace detail {
static brass::BasicBlock* new_block(brass::Builder& b, std::string_view name) {
    brass::BasicBlock* prev = b.current_block();
    brass::BasicBlock* bb = b.append_block(name);
    if (prev) b.position_at_end(prev);
    return bb;
}
void MirBodyLowerer::lower_stmt_list(const std::vector<StmtPtr>& stmts) {
    for (size_t i = 0; i < stmts.size(); ++i) {
        const Stmt* s = stmts[i].get();
        if (!s) continue;
        lower_stmt(s);
        if (s->kind == Stmt::Kind::Return || s->kind == Stmt::Kind::Break || s->kind == Stmt::Kind::Continue) {
            break;
        }
        // If `s` could have executed an early return inside a branch and there are more statements in this block,
        // guard the rest of the statements with `if (!__has_returned) { ... }` so control flow stays 100% structured.
        if (i + 1 < stmts.size() && stmt_has_return(s) && has_returned_var_id >= 0) {
            brass::Value* has_ret = var_values[has_returned_var_id][0];
            brass::Value* not_ret = b.build_eq(has_ret, b.build_iconst_i32(0));
            if (!loop_stack.empty()) {
                const auto& lctx = loop_stack.back();
                not_ret = b.build_and(not_ret, b.build_eq(var_values[lctx.broke_var_id][0], b.build_iconst_i32(0)));
                not_ret = b.build_and(not_ret, b.build_eq(var_values[lctx.cont_var_id][0], b.build_iconst_i32(0)));
            }
            std::vector<StmtPtr> remaining;
            for (size_t j = i + 1; j < stmts.size(); ++j) {
                remaining.push_back(clone_stmt(stmts[j].get()));
            }
            lower_branch(not_ret, [&] { lower_stmt_list(remaining); }, nullptr);
            break;
        }
    }
}
void MirBodyLowerer::lower_branch(brass::Value* cond, const std::function<void()>& then_arm,
                                  const std::function<void()>& else_arm) {
    if (const brass::Instruction* def = cond->is_instruction() ? cond->defining_instruction() : nullptr;
        def && def->opcode() == brass::Opcode::iconst_i32) {
        if (def->imm_i32() != 0) {
            then_arm();
        } else if (else_arm) {
            else_arm();
        }
        return;
    }
    auto vars_before = var_values;
    auto live_ids = sorted_live_var_ids();
    if (Ops::is_varying(cond)) {
        brass::Value* outer = b.mask;
        b.mask = outer ? b.build_and(outer, cond) : cond;
        then_arm();
        auto vars_then = var_values;
        var_values = vars_before;
        if (else_arm) {
            brass::Value* inv = b.build_not(cond);
            b.mask = outer ? b.build_and(outer, inv) : inv;
            else_arm();
        }
        auto vars_else = var_values;
        b.mask = outer;
        var_values = vars_before;
        for (int id : live_ids) {
            auto& vt = vars_then[id];
            auto& ve = vars_else[id];
            for (size_t k = 0; k < vt.size() && k < ve.size(); ++k) {
                if (vt[k] != ve[k]) var_values[id][k] = b.build_select(cond, vt[k], ve[k]);
            }
        }
        return;
    }
    brass::BasicBlock* then_bb = new_block(b, "if_then");
    brass::BasicBlock* else_bb = new_block(b, "if_else");
    brass::BasicBlock* merge_bb = new_block(b, "if_merge");
    b.build_br_if(cond, then_bb, else_bb);
    b.position_at_end(then_bb);
    var_values = vars_before;
    then_arm();
    brass::BasicBlock* then_end = b.current_block();
    auto vars_then = var_values;
    b.position_at_end(else_bb);
    var_values = vars_before;
    if (else_arm) else_arm();
    brass::BasicBlock* else_end = b.current_block();
    auto vars_else = var_values;
    var_values = vars_before;
    // A value varying on one side is varying after the merge: the uniform
    // side is broadcast at the end of its arm.
    struct Join { int id; size_t k; brass::Value* phi; };
    std::vector<Join> joins;
    for (int id : live_ids) {
        auto& vt = vars_then[id];
        auto& ve = vars_else[id];
        for (size_t k = 0; k < vt.size() && k < ve.size(); ++k) {
            if (vt[k] != ve[k]) {
                brass::Value* phi = b.add_block_param(merge_bb, Ops::merge_type(vt[k], ve[k]));
                joins.push_back({id, k, phi});
                var_values[id][k] = phi;
            }
        }
    }
    std::vector<brass::Value*> args;
    b.position_at_end(then_end);
    for (const Join& j : joins) args.push_back(b.coerce(vars_then[j.id][j.k], j.phi->type()));
    b.build_br(merge_bb, args);
    args.clear();
    b.position_at_end(else_end);
    for (const Join& j : joins) args.push_back(b.coerce(vars_else[j.id][j.k], j.phi->type()));
    b.build_br(merge_bb, args);
    b.position_at_end(merge_bb);
}
// The names of the variables `s` assigns (a store's or a buffer's elements
// are not variables).
static void collect_assigned_names(const Stmt* s, std::set<std::string>& out) {
    if (!s) return;
    if (s->kind == Stmt::Kind::Assign) {
        const Expr* t = s->assign_target.get();
        while (t && (t->kind == Expr::Kind::Field || t->kind == Expr::Kind::Index) &&
               !(t->kind == Expr::Kind::Index && t->lhs->resolved_type.is_store()) &&
               !t->lhs->resolved_type.is_block()) {
            t = t->lhs.get();
        }
        if (t && t->kind == Expr::Kind::Ident) out.insert(t->name);
    }
    for (const auto& c : s->body) collect_assigned_names(c.get(), out);
    for (const auto& c : s->else_body) collect_assigned_names(c.get(), out);
    collect_assigned_names(s->for_init.get(), out);
    collect_assigned_names(s->for_step.get(), out);
}

static std::vector<const Stmt*> loop_writers(const Stmt* s) {
    std::vector<const Stmt*> w;
    for (const auto& c : s->body) w.push_back(c.get());
    if (s->for_step) w.push_back(s->for_step.get());
    return w;
}

void MirBodyLowerer::lower_loop(int broke_var_id, int cont_var_id, int counter_var_id,
                                const std::vector<const Stmt*>& writers,
                                const std::function<brass::Value*()>& cond_fn,
                                const std::function<void()>& body_fn, const std::function<void()>& step_fn) {
    const int seq = loop_seq++;
    std::set<int> carried = {broke_var_id, cont_var_id, counter_var_id, has_returned_var_id, ret_val_var_id};
    {
        std::set<std::string> names;
        for (const Stmt* w : writers) collect_assigned_names(w, names);
        for (const auto& n : names) {
            int vid = lookup_var_id(n);
            if (vid >= 0 && !shared_places.count(vid) && !block_places.count(vid)) carried.insert(vid);
        }
    }
    std::vector<int> live_ids;
    for (int id : sorted_live_var_ids()) {
        if (carried.count(id)) live_ids.push_back(id);
    }
    if (vplan) {
        for (int id : live_ids) {
            auto& leaves = var_values[id];
            for (size_t k = 0; k < leaves.size(); ++k) {
                if (vplan->widen.count({seq, id, k})) leaves[k] = b.widen(leaves[k]);
            }
        }
    }
    std::vector<brass::Value*> init_args = pack_var_ids(live_ids);
    brass::BasicBlock* head_bb = new_block(b, "loop_head");
    brass::BasicBlock* body_bb = new_block(b, "loop_body");
    brass::BasicBlock* exit_bb = new_block(b, "loop_exit");
    b.build_br(head_bb, init_args);
    for (int id : live_ids) {
        for (auto& leaf : var_values[id]) {
            leaf = b.add_block_param(head_bb, leaf->type());
        }
    }
    auto vars_at_head = var_values;
    b.position_at_end(head_bb);
    brass::Value* cond = cond_fn();
    brass::Value* go = b.build_eq(var_values[broke_var_id][0], b.build_iconst_i32(0));
    if (has_returned_var_id >= 0) {
        go = b.build_and(go, b.build_eq(var_values[has_returned_var_id][0], b.build_iconst_i32(0)));
    }
    cond = b.build_and(go, cond);
    // A varying condition: lanes leave at different iterations. The loop runs
    // while any lane of the enclosing ones is still in it, its body under
    // those lanes, and a lane that has left keeps its values.
    const bool varying = Ops::is_varying(cond);
    brass::Value* outer = b.mask;
    brass::Value* active = (varying && outer) ? b.build_and(outer, cond) : cond;
    b.build_br_if(varying ? b.any(active) : cond, body_bb, exit_bb);
    b.position_at_end(body_bb);
    var_values[cont_var_id] = {b.build_iconst_i32(0)};
    loop_stack.push_back(LoopLoweringCtx{head_bb, exit_bb, broke_var_id, cont_var_id});
    if (varying) b.mask = active;
    body_fn();
    loop_stack.pop_back();
    step_fn();
    b.mask = outer;
    std::vector<brass::Value*> next;
    for (int id : live_ids) {
        const auto& head_leaves = vars_at_head[id];
        const auto& cur = var_values[id];
        for (size_t k = 0; k < head_leaves.size(); ++k) {
            brass::Value* h = head_leaves[k];
            brass::Value* v = k < cur.size() ? cur[k] : h;
            if (Ops::is_varying(h)) {
                v = varying ? b.build_select(active, v, h) : b.coerce(v, h->type());
            } else if (v != h && (Ops::is_varying(v) || (varying && id != counter_var_id))) {
                if (vplan) {
                    vplan->widen.insert({seq, id, k});
                    vplan->redo = true;
                }
                v = h;
            }
            next.push_back(v);
        }
    }
    b.build_br(head_bb, next);
    b.position_at_end(exit_bb);
    var_values = vars_at_head;
}
void MirBodyLowerer::lower_stmt(const Stmt* s) {
    if (!s) return;
    switch (s->kind) {
        case Stmt::Kind::Block:
            push_scope();
            lower_stmt_list(s->body);
            pop_scope();
            break;
        case Stmt::Kind::Let:
        case Stmt::Kind::Var: {
            if (s->is_shared) {
                int id = alloc_var(s->var_name, {});
                auto it = shared_vars.find("l:" + active_fn_name + ":" + s->var_name);
                if (it != shared_vars.end()) {
                    shared_places[id] = SharedPlace{&it->second, b.build_iconst_i32(0), s->resolved_var_type, {}};
                }
                break;
            }
            if (s->kind == Stmt::Kind::Let && s->init_expr &&
                (s->resolved_var_type.is_struct() || s->resolved_var_type.is_array())) {
                BlockPlace bp;
                if (is_block_path(s->init_expr.get()) && block_place(s->init_expr.get(), bp)) {
                    int id = alloc_var(s->var_name, {});
                    block_places[id] = std::move(bp);
                    break;
                }
            }
            ValueList init = s->init_expr ? lower_expr(s->init_expr.get())
                                          : make_zero_leaves(s->resolved_var_type);
            alloc_var(s->var_name, std::move(init));
            break;
        }
        case Stmt::Kind::Assign: {
            ValueList rhs = lower_expr(s->assign_value.get());
            // 0. A place in a storage buffer at a descriptor or in shared memory.
            if (try_block_assign(s, rhs) || try_shared_assign(s, rhs)) return;
            // 1. Store indexing assignment: `st[x, y] = val` or `st[x, y] += val`
            if (s->assign_target->kind == Expr::Kind::Index &&
                s->assign_target->lhs->resolved_type.is_store()) {
                const Type& st_t = s->assign_target->lhs->resolved_type;
                Type elem_t = st_t.element_type();
                ValueList base = lower_expr(s->assign_target->lhs.get());
                uint32_t dim = st_t.dimension;
                ValueList x = dim >= 1 ? lower_expr(s->assign_target->args[0].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList y = dim >= 2 ? lower_expr(s->assign_target->args[1].get()) : ValueList{b.build_iconst_i32(0)};
                ValueList z = dim >= 3 ? lower_expr(s->assign_target->args[2].get()) : ValueList{b.build_iconst_i32(0)};
                brass::Value* lin = compute_linear_index(dim, x[0], y[0], z[0]);
                base = store_access(base, dim, x[0], y[0], z[0]);
                if (s->compound_op.has_value()) {
                    ValueList old_v = load_element_from_buffer(elem_t, base, lin);
                    for (size_t k = 0; k < old_v.size() && k < rhs.size(); ++k) {
                        rhs[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], elem_t.scalar);
                    }
                }
                store_element_to_buffer(elem_t, base, lin, rhs);
                return;
            }
            // 2. Local variable (or nested field / vector swizzle / vector or array element of local variable)
            std::function<void(const Expr*, const std::function<ValueList(const ValueList&)>&)> update_lvalue;
            update_lvalue = [&](const Expr* target, const std::function<ValueList(const ValueList&)>& write_fn) {
                if (!target) return;
                if (target->kind == Expr::Kind::Ident) {
                    int id = lookup_var_id(target->name);
                    if (id >= 0) {
                        var_values[id] = write_fn(var_values[id]);
                    }
                    return;
                }
                if (target->kind == Expr::Kind::Field && target->lhs) {
                    const Type& base_t = target->lhs->resolved_type;
                    if (base_t.is_scalar()) {
                        update_lvalue(target->lhs.get(), write_fn);
                        return;
                    }
                    if (base_t.is_vector()) {
                        const std::string& swz = target->name;
                        update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                            ValueList old_sub;
                            old_sub.reserve(swz.size());
                            for (char c : swz) {
                                uint32_t idx = swizzle_index(c);
                                old_sub.push_back(idx < cur.size() ? cur[idx] : cur[0]);
                            }
                            ValueList new_sub = write_fn(old_sub);
                            ValueList next = cur;
                            for (size_t k = 0; k < swz.size() && k < new_sub.size(); ++k) {
                                uint32_t idx = swizzle_index(swz[k]);
                                if (idx < next.size()) {
                                    next[idx] = new_sub[k];
                                }
                            }
                            return next;
                        });
                        return;
                    }
                    if (base_t.is_struct()) {
                        const StructDef* sdef = mod.types.find_struct(base_t.struct_id);
                        const StructField* sf = sdef ? sdef->find_field(target->name) : nullptr;
                        if (!sf) return;
                        update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                            ValueList old_sub;
                            old_sub.reserve(sf->flat_leaf_count);
                            for (uint32_t k = 0; k < sf->flat_leaf_count; ++k) {
                                uint32_t idx = sf->flat_leaf_offset + k;
                                old_sub.push_back(idx < cur.size() ? cur[idx] : cur[0]);
                            }
                            ValueList new_sub = write_fn(old_sub);
                            ValueList next = cur;
                            for (uint32_t k = 0; k < sf->flat_leaf_count && k < new_sub.size(); ++k) {
                                uint32_t idx = sf->flat_leaf_offset + k;
                                if (idx < next.size()) {
                                    next[idx] = new_sub[k];
                                }
                            }
                            return next;
                        });
                        return;
                    }
                    return;
                }
                if (target->kind == Expr::Kind::Index && target->lhs && !target->args.empty()) {
                    const Type& base_t = target->lhs->resolved_type;
                    if (base_t.is_scalar()) {
                        update_lvalue(target->lhs.get(), write_fn);
                        return;
                    }
                    ValueList idx_v = lower_expr(target->args[0].get());
                    brass::Value* idx = idx_v.empty() ? b.build_iconst_i32(0) : idx_v[0];
                    if (idx->type() == brass::Type::i64()) {
                        idx = b.build_trunc_i32(idx);
                    }
                    Type elem_t = base_t.element_type();
                    uint32_t elem_leaves = mod.types.flat_leaf_count(elem_t);
                    uint32_t count = base_t.dimension;
                    update_lvalue(target->lhs.get(), [&](const ValueList& cur) -> ValueList {
                        if (cur.empty() || elem_leaves == 0 || count == 0) return cur;
                        ValueList old_sub(elem_leaves, cur[0]);
                        for (uint32_t k = 0; k < elem_leaves; ++k) {
                            brass::Value* acc = k < cur.size() ? cur[k] : cur[0];
                            for (uint32_t i = 1; i < count; ++i) {
                                uint32_t off = i * elem_leaves + k;
                                if (off < cur.size()) {
                                    brass::Value* eq = b.build_eq(idx, b.build_iconst_i32(static_cast<int32_t>(i)));
                                    acc = b.build_select(eq, cur[off], acc);
                                }
                            }
                            old_sub[k] = acc;
                        }
                        ValueList new_sub = write_fn(old_sub);
                        ValueList next = cur;
                        for (uint32_t i = 0; i < count; ++i) {
                            brass::Value* eq = b.build_eq(idx, b.build_iconst_i32(static_cast<int32_t>(i)));
                            for (uint32_t k = 0; k < elem_leaves && k < new_sub.size(); ++k) {
                                uint32_t off = i * elem_leaves + k;
                                if (off < next.size()) {
                                    next[off] = b.build_select(eq, new_sub[k], next[off]);
                                }
                            }
                        }
                        return next;
                    });
                    return;
                }
            };
            ScalarKind sk = s->assign_target->resolved_type.scalar;
            update_lvalue(s->assign_target.get(), [&](const ValueList& old_v) -> ValueList {
                if (!s->compound_op.has_value()) return rhs;
                ValueList res = old_v;
                for (size_t k = 0; k < res.size() && k < rhs.size(); ++k) {
                    res[k] = lower_scalar_binary(*s->compound_op, res[k], rhs[k], sk);
                }
                return res;
            });
            break;
        }
        case Stmt::Kind::If: {
            ValueList cv = lower_expr(s->cond.get());
            brass::Value* cond = cv.empty() ? b.build_iconst_i32(0) : cv[0];
            std::function<void()> else_arm;
            if (!s->else_body.empty()) {
                else_arm = [&] { push_scope(); lower_stmt_list(s->else_body); pop_scope(); };
            }
            lower_branch(cond, [&] { push_scope(); lower_stmt_list(s->body); pop_scope(); }, else_arm);
            break;
        }
        case Stmt::Kind::ForRange: {
            ValueList start_v = lower_expr(s->range_start.get());
            ValueList end_v = lower_expr(s->range_end.get());
            push_scope();
            int loop_var_id = alloc_var(s->var_name, start_v);
            int broke_var_id = alloc_var("__loop_broke_" + std::to_string(loop_var_id), {b.build_iconst_i32(0)});
            int cont_var_id = alloc_var("__loop_cont_" + std::to_string(loop_var_id), {b.build_iconst_i32(0)});
            lower_loop(
                broke_var_id, cont_var_id, loop_var_id, loop_writers(s),
                [&] {
                    brass::Value* cur_i = var_values[loop_var_id][0];
                    return s->inclusive_range ? b.build_sle(cur_i, end_v[0]) : b.build_slt(cur_i, end_v[0]);
                },
                [&] { lower_stmt_list(s->body); },
                [&] {
                    brass::Value* step_one = (s->resolved_var_type.scalar == ScalarKind::I64 ||
                                              s->resolved_var_type.scalar == ScalarKind::U64)
                                                 ? b.build_iconst_i64(1)
                                                 : b.build_iconst_i32(1);
                    var_values[loop_var_id][0] = b.build_add(var_values[loop_var_id][0], step_one);
                });
            pop_scope();
            break;
        }
        case Stmt::Kind::ForC:
        case Stmt::Kind::While: {
            push_scope();
            if (s->kind == Stmt::Kind::ForC && s->for_init) lower_stmt(s->for_init.get());
            int broke_var_id = alloc_var("__loop_broke_" + std::to_string(next_var_id), {b.build_iconst_i32(0)});
            int cont_var_id = alloc_var("__loop_cont_" + std::to_string(next_var_id), {b.build_iconst_i32(0)});
            lower_loop(
                broke_var_id, cont_var_id, -1, loop_writers(s),
                [&]() -> brass::Value* {
                    if (!s->cond) return b.true_value();
                    ValueList cv = lower_expr(s->cond.get());
                    return cv.empty() ? b.true_value() : cv[0];
                },
                [&] { lower_stmt_list(s->body); },
                [&] { if (s->kind == Stmt::Kind::ForC && s->for_step) lower_stmt(s->for_step.get()); });
            pop_scope();
            break;
        }
        case Stmt::Kind::Break:
            if (!loop_stack.empty()) var_values[loop_stack.back().broke_var_id] = {b.true_value()};
            break;
        case Stmt::Kind::Continue:
            if (!loop_stack.empty()) var_values[loop_stack.back().cont_var_id] = {b.true_value()};
            break;
        case Stmt::Kind::Return:
            if (s->expr && ret_val_var_id >= 0) var_values[ret_val_var_id] = lower_expr(s->expr.get());
            if (has_returned_var_id >= 0) var_values[has_returned_var_id] = {b.true_value()};
            break;
        case Stmt::Kind::Expr:
            lower_expr(s->expr.get());
            break;
    }
}

namespace {

// What every cell of the CPU grid function shares: the extents, the return
// store and the kernel's parameters, unpacked once in the entry block.
struct GridFrame {
    brass::Value* gw = nullptr;
    brass::Value* gh = nullptr;
    brass::Value* gd = nullptr;
    brass::Value* lsz_x = nullptr;
    brass::Value* lsz_y = nullptr;
    brass::Value* lsz_z = nullptr;
    brass::Value* nwg_x = nullptr;
    brass::Value* nwg_y = nullptr;
    brass::Value* nwg_z = nullptr;
    brass::Value* has_out_i32 = nullptr;   // the shape's fourth word: nonzero when out_ret is a store
    brass::Value* out_ret = nullptr;
    brass::Value* scratch = nullptr;       // a vector build's masked-off stores
    std::vector<ValueList> params;
    std::unordered_map<size_t, std::pair<brass::Value*, brass::Value*>> block_ptrs;
};

// One cell of the grid (lanes 0), or `lanes` cells from x on (x a uniform
// i32, the cells x .. x + lanes - 1 of row gy, slice gz).
void lower_grid_cells(brass::codegen::KernelBuilder& kb, const CheckedModule& mod, const CheckedFunction& fn,
                      DiagnosticBag& diags, const GridFrame& f, uint32_t lanes, VectorPlan* plan, brass::Value* x,
                      brass::Value* gy, brass::Value* gz, bool x_interior = false) {
    brass::Builder& b = kb.builder();
    MirBodyLowerer lowerer(mod, fn, diags, kb, /*is_gpu=*/false);
    lowerer.b.lanes = lanes;
    lowerer.b.scratch = f.scratch;
    lowerer.vplan = plan;
    lowerer.x_interior = x_interior;
    lowerer.push_scope();
    lowerer.gw = f.gw;
    lowerer.gh = f.gh;
    lowerer.gd = f.gd;
    lowerer.gx = lanes ? lowerer.b.build_add(x, lowerer.b.iota_i32()) : x;
    lowerer.gy = gy;
    lowerer.gz = gz;
    lowerer.lsz_x = f.lsz_x;
    lowerer.lsz_y = f.lsz_y;
    lowerer.lsz_z = f.lsz_z;
    lowerer.nwg_x = f.nwg_x;
    lowerer.nwg_y = f.nwg_y;
    lowerer.nwg_z = f.nwg_z;
    lowerer.wgid_x = lowerer.b.build_udiv(lowerer.gx, f.lsz_x);
    lowerer.wgid_y = b.build_udiv(gy, f.lsz_y);
    lowerer.wgid_z = b.build_udiv(gz, f.lsz_z);
    lowerer.lid_x = lowerer.b.build_sub(lowerer.gx, lowerer.b.build_mul(lowerer.wgid_x, f.lsz_x));
    lowerer.lid_y = b.build_sub(gy, b.build_mul(lowerer.wgid_y, f.lsz_y));
    lowerer.lid_z = b.build_sub(gz, b.build_mul(lowerer.wgid_z, f.lsz_z));
    lowerer.cpu_block_ptrs = f.block_ptrs;
    lowerer.active_return_type = fn.return_type;
    lowerer.has_returned_var_id = lowerer.alloc_var("__has_returned", {b.build_iconst_i32(0)});
    if (!fn.return_type.is_void()) {
        lowerer.ret_val_var_id = lowerer.alloc_var("__ret_val", lowerer.make_zero_leaves(fn.return_type));
    }
    for (size_t i = 0; i < fn.params.size(); ++i) {
        ValueList pvals = f.params[i];
        if (lanes) {
            auto kinds = mod.types.flatten_scalars(fn.params[i].type);
            for (size_t k = 0; k < pvals.size() && k < kinds.size(); ++k) {
                if (kinds[k] == ScalarKind::Bool) {
                    pvals[k] = lowerer.b.build_ne(pvals[k], b.build_iconst_i32(0));
                }
            }
        }
        lowerer.alloc_var(fn.params[i].name, pvals);
        const ParamRole role = fn.params[i].role;
        if (role == ParamRole::Store || role == ParamRole::Stencil) {
            lowerer.bind_store_param(i, f.params[i]);
        } else if (role == ParamRole::Uniform || role == ParamRole::Buffer) {
            lowerer.block_param[pvals[0]] = i;
        }
    }
    lowerer.lower_stmt_list(fn.body);
    if (lowerer.ret_val_var_id < 0) return;
    const ValueList& ret = lowerer.var_values[lowerer.ret_val_var_id];
    const uint32_t dim = std::max(1u, fn.grid_dim);
    // Without a return store the cells write the caller's scratch: element 0,
    // or elements 0 .. lanes - 1 in a vector build (an index of x's stride,
    // so a row's cells stay one contiguous store).
    Ops& ops = lowerer.b;
    brass::Value* has_out = b.build_ne(f.has_out_i32, b.build_iconst_i32(0));
    brass::Value* lin = lowerer.compute_linear_index(dim, lowerer.gx, gy, gz);
    brass::Value* zero = b.build_iconst_i32(0);
    brass::Value* first = lanes ? lowerer.compute_linear_index(dim, ops.iota_i32(), zero, zero)
                                : b.build_iconst_i64(0);
    lowerer.store_element_to_buffer(fn.return_type, {f.out_ret}, ops.build_select(has_out, lin, first), ret);
}

// The CPU grid function:
//   void __cpu_grid(const uint64_t* args_pack, const int32_t* shape7, void* out_ret)
// Nested loops over (gz, gy, gx) with 64-bit linear indexing, so a grid past
// 2^31 cells never overflows and the inner loop has no divisions. With
// `lanes`, each row runs `lanes` cells at a time up to the last whole group,
// the rest one by one.
brass::Function* build_cpu_grid(brass::Module& m, const CheckedModule& mod, const CheckedFunction& fn,
                                const std::vector<KernelParamInfo>& block_bindings,
                                DiagnosticBag& diags, uint32_t lanes, VectorPlan* plan) {
    brass::Function* grid_fn = m.create_function(fn.name + "_cpu_grid", brass::Type::void_type(),
                                                 {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr()});
    brass::codegen::KernelBuilder kb(m, grid_fn);
    brass::Builder& b = kb.builder();
    brass::BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    brass::Value* args_pack = b.add_param(brass::Type::ptr());
    brass::Value* shape_ptr = b.add_param(brass::Type::ptr());
    GridFrame f;
    f.out_ret = b.add_param(brass::Type::ptr());
    f.gw = b.build_load(brass::Type::i32(), shape_ptr, 0);
    f.gh = b.build_load(brass::Type::i32(), shape_ptr, 4);
    f.gd = b.build_load(brass::Type::i32(), shape_ptr, 8);
    f.has_out_i32 = b.build_load(brass::Type::i32(), shape_ptr, 12);
    f.lsz_x = b.build_load(brass::Type::i32(), shape_ptr, 16);
    f.lsz_y = b.build_load(brass::Type::i32(), shape_ptr, 20);
    f.lsz_z = b.build_load(brass::Type::i32(), shape_ptr, 24);
    auto ceil_div = [&](brass::Value* ext, brass::Value* lsz) {
        return b.build_udiv(b.build_sub(b.build_add(ext, lsz), b.build_iconst_i32(1)), lsz);
    };
    f.nwg_x = ceil_div(f.gw, f.lsz_x);
    f.nwg_y = ceil_div(f.gh, f.lsz_y);
    f.nwg_z = ceil_div(f.gd, f.lsz_z);
    if (lanes) f.scratch = b.build_alloca(256, 32);
    f.params.reserve(fn.params.size());
    int32_t slot = 0;
    for (const auto& cp : fn.params) {
        if (is_bound_param(cp.role)) {
            brass::Value* placeholder =
                b.build_iconst_i32(static_cast<int32_t>((cp.descriptor_set << 16) | cp.descriptor_binding));
            f.params.push_back({placeholder});
        } else if (cp.role == ParamRole::Store || cp.role == ParamRole::Stencil) {
            f.params.push_back({b.build_load(brass::Type::ptr(), args_pack, slot * 8),
                                b.build_load(brass::Type::i64(), args_pack, slot * 8 + 8)});
            slot += 2;
        } else {
            auto kinds = mod.types.flatten_scalars(cp.type);
            ValueList leaves;
            leaves.reserve(kinds.size());
            for (ScalarKind sk : kinds) {
                leaves.push_back(b.build_load(to_mir_scalar_type(sk), args_pack, slot * 8));
                slot += 1;
            }
            f.params.push_back(std::move(leaves));
        }
    }
    for (const auto& bb : block_bindings) {
        brass::Value* ptr_v = b.build_load(brass::Type::ptr(), args_pack, slot * 8);
        brass::Value* bytes_v = b.build_load(brass::Type::i64(), args_pack, slot * 8 + 8);
        f.block_ptrs[bb.mir_param_index] = {ptr_v, bytes_v};
        slot += 2;
    }
    brass::BasicBlock* z_head = new_block(b, "grid_z_head");
    brass::BasicBlock* z_body = new_block(b, "grid_z_body");
    brass::BasicBlock* y_head = new_block(b, "grid_y_head");
    brass::BasicBlock* y_body = new_block(b, "grid_y_body");
    brass::BasicBlock* y_latch = new_block(b, "grid_y_latch");
    brass::BasicBlock* z_latch = new_block(b, "grid_z_latch");
    brass::BasicBlock* loop_exit = new_block(b, "grid_exit");

    b.build_br(z_head, {b.build_iconst_i32(0)});
    brass::Value* gz = b.add_block_param(z_head, brass::Type::i32());
    b.position_at_end(z_head);
    b.build_br_if(b.build_slt(gz, f.gd), z_body, loop_exit);

    b.position_at_end(z_body);
    b.build_br(y_head, {b.build_iconst_i32(0)});
    brass::Value* gy = b.add_block_param(y_head, brass::Type::i32());
    b.position_at_end(y_head);
    b.build_br_if(b.build_slt(gy, f.gh), y_body, z_latch);

    b.position_at_end(y_body);
    brass::Value* x_start = b.build_iconst_i32(0);
    if (lanes) {
        // x in [0, gw rounded down to a multiple of lanes), lanes at a time.
        brass::Value* vec_end = b.build_and(f.gw, b.build_iconst_i32(-static_cast<int32_t>(lanes)));
        brass::BasicBlock* v_head = new_block(b, "grid_xv_head");
        brass::BasicBlock* v_body = new_block(b, "grid_xv_body");
        brass::BasicBlock* v_exit = new_block(b, "grid_xv_exit");
        b.build_br(v_head, {b.build_iconst_i32(0)});
        brass::Value* xv = b.add_block_param(v_head, brass::Type::i32());
        b.position_at_end(v_head);
        b.build_br_if(b.build_slt(xv, vec_end), v_body, v_exit);
        b.position_at_end(v_body);
        brass::Value* x_next = b.build_add(xv, b.build_iconst_i32(static_cast<int32_t>(lanes)));
        // (With no x halo every group is inside: x + 0 is in the row.)
        const int32_t halo = fn.stencil_halo[0];
        if (halo == 0) {
            lower_grid_cells(kb, mod, fn, diags, f, lanes, plan, xv, gy, gz, /*x_interior=*/true);
            b.build_br(v_head, {x_next});
        } else {
            // A group whose stencil reads all land inside the row runs a
            // body without their x clamps: xv - halo >= 0 and
            // xv + lanes - 1 + halo <= gw - 1. The rest (a halo's width at
            // either end of a row) clamp.
            brass::BasicBlock* v_inner = new_block(b, "grid_xv_inner");
            brass::BasicBlock* v_edge = new_block(b, "grid_xv_edge");
            brass::Value* hv = b.build_iconst_i32(halo);
            brass::Value* inner = b.build_and(b.build_sge(xv, hv), b.build_sle(b.build_add(x_next, hv), f.gw));
            b.build_br_if(inner, v_inner, v_edge);
            b.position_at_end(v_inner);
            lower_grid_cells(kb, mod, fn, diags, f, lanes, plan, xv, gy, gz, /*x_interior=*/true);
            b.build_br(v_head, {x_next});
            b.position_at_end(v_edge);
            lower_grid_cells(kb, mod, fn, diags, f, lanes, plan, xv, gy, gz);
            b.build_br(v_head, {x_next});
        }
        b.position_at_end(v_exit);
        x_start = vec_end;
    }
    brass::BasicBlock* x_head = new_block(b, "grid_x_head");
    brass::BasicBlock* x_body = new_block(b, "grid_x_body");
    b.build_br(x_head, {x_start});
    brass::Value* gx = b.add_block_param(x_head, brass::Type::i32());
    b.position_at_end(x_head);
    b.build_br_if(b.build_slt(gx, f.gw), x_body, y_latch);
    b.position_at_end(x_body);
    lower_grid_cells(kb, mod, fn, diags, f, 0, nullptr, gx, gy, gz);
    b.build_br(x_head, {b.build_add(gx, b.build_iconst_i32(1))});

    b.position_at_end(y_latch);
    b.build_br(y_head, {b.build_add(gy, b.build_iconst_i32(1))});

    b.position_at_end(z_latch);
    b.build_br(z_head, {b.build_add(gz, b.build_iconst_i32(1))});

    b.position_at_end(loop_exit);
    b.build_ret_void();
    return grid_fn;
}

} // namespace
} // namespace detail
using namespace detail;
LoweredKernelMir lower_kernel_to_mir(
    const CheckedModule& mod,
    const CheckedFunction& fn,
    DiagnosticBag& diags) {
    LoweredKernelMir out;
    out.gpu_module = std::make_unique<brass::Module>(fn.name + "_gpu");
    out.cpu_module = std::make_unique<brass::Module>(fn.name + "_cpu");
    // -----------------------------------------------------------------------
    // 1. Build GPU MIR Function (`gpu_function`)
    // -----------------------------------------------------------------------
    out.gpu_function = out.gpu_module->create_function(fn.name, brass::Type::void_type());
    {
        brass::codegen::KernelBuilder kb(*out.gpu_module, out.gpu_function);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        MirBodyLowerer lowerer(mod, fn, diags, kb, /*is_gpu=*/true);
        lowerer.push_scope();
        lowerer.active_return_type = fn.return_type;
        lowerer.has_returned_var_id = lowerer.alloc_var("__has_returned", {b.build_iconst_i32(0)});
        if (!fn.return_type.is_void()) {
            lowerer.ret_val_var_id = lowerer.alloc_var("__ret_val", lowerer.make_zero_leaves(fn.return_type));
        }
        uint32_t mir_idx = 0;
        for (const auto& cp : fn.params) {
            KernelParamInfo pinfo;
            pinfo.name = cp.name;
            pinfo.type = cp.type;
            pinfo.role = cp.role;
            pinfo.access = cp.access;
            pinfo.dimension = cp.type.dimension;
            pinfo.declared_radius = cp.declared_radius;
            pinfo.inferred_radius = cp.inferred_radius;
            pinfo.halo[0] = cp.halo[0];
            pinfo.halo[1] = cp.halo[1];
            pinfo.halo[2] = cp.halo[2];
            pinfo.stencil_offsets = cp.stencil_offsets;
            pinfo.descriptor_set = cp.descriptor_set;
            pinfo.descriptor_binding = cp.descriptor_binding;
            pinfo.mir_param_index = mir_idx;
            if (cp.role == ParamRole::Grid) {
                pinfo.mir_param_count = 0;
                lowerer.alloc_var(cp.name, {b.build_iconst_i32(0)});
            } else if (cp.role == ParamRole::Store || cp.role == ParamRole::Stencil) {
                // Its address and its element count (the bound of every access).
                pinfo.mir_param_count = 2;
                pinfo.element_bytes = mod.types.byte_size(cp.type.element_type());
                brass::Value* ptr_p = b.add_param(brass::Type::ptr());
                brass::Value* count_p = b.add_param(brass::Type::i64());
                lowerer.alloc_var(cp.name, {ptr_p, count_p});
                lowerer.bind_store_param(out.param_infos.size(), {ptr_p, count_p});
                mir_idx += 2;
            } else if (cp.role == ParamRole::Image || cp.role == ParamRole::Sampler) {
                pinfo.mir_param_count = 0;
                pinfo.image_format = cp.image_format;
                brass::Value* placeholder =
                    b.build_iconst_i32(static_cast<int32_t>((cp.descriptor_set << 16) | cp.descriptor_binding));
                lowerer.texture_param[placeholder] = out.param_infos.size();
                lowerer.alloc_var(cp.name, {placeholder});
            } else if (cp.role == ParamRole::Uniform || cp.role == ParamRole::Buffer) {
                pinfo.mir_param_count = 0;
                brass::Value* placeholder =
                    b.build_iconst_i32(static_cast<int32_t>((cp.descriptor_set << 16) | cp.descriptor_binding));
                lowerer.block_param[placeholder] = out.param_infos.size();
                lowerer.alloc_var(cp.name, {placeholder});
            } else if (cp.role == ParamRole::StageInput) {
                pinfo.mir_param_count = 0;
                const uint32_t lanes = cp.type.is_vector() ? cp.type.dimension : 1;
                const bool integer = cp.type.scalar != ScalarKind::F32;
                ValueList leaves;
                for (uint32_t i = 0; i < lanes; ++i) {
                    leaves.push_back(kb.stage_input(cp.location, lanes, i, integer,
                                                    cp.flat ? brass::target::kSpirvStageFlat : 0u));
                }
                lowerer.alloc_var(cp.name, std::move(leaves));
            } else {
                auto kinds = mod.types.flatten_scalars(cp.type);
                pinfo.mir_param_count = static_cast<uint32_t>(kinds.size());
                ValueList leaves;
                leaves.reserve(kinds.size());
                for (ScalarKind sk : kinds) {
                    leaves.push_back(b.add_param(to_mir_scalar_type(sk)));
                }
                lowerer.alloc_var(cp.name, std::move(leaves));
                mir_idx += pinfo.mir_param_count;
            }
            out.param_infos.push_back(std::move(pinfo));
        }
        if (fn.grid_dim > 0) {
            out.grid_dim_mir_index = mir_idx;
            lowerer.gw = b.add_param(brass::Type::i32());
            mir_idx += 1;
            lowerer.gh = fn.grid_dim >= 2 ? (mir_idx += 1, b.add_param(brass::Type::i32())) : b.build_iconst_i32(1);
            lowerer.gd = fn.grid_dim >= 3 ? (mir_idx += 1, b.add_param(brass::Type::i32())) : b.build_iconst_i32(1);
        } else {
            lowerer.gw = b.build_iconst_i32(1);
            lowerer.gh = b.build_iconst_i32(1);
            lowerer.gd = b.build_iconst_i32(1);
        }
        brass::Value* gpu_out_ptr = nullptr;
        if (!fn.return_type.is_void()) {
            out.return_out_mir_index = mir_idx;
            gpu_out_ptr = b.add_param(brass::Type::ptr());
            mir_idx += 1;
        }
        {
            std::vector<brass::Type> ptypes;
            ptypes.reserve(entry->params().size());
            for (brass::Value* p : entry->params()) ptypes.push_back(p->type());
            out.gpu_function->set_param_types(std::move(ptypes));
        }
        if (fn.stage == Stage::Compute) {
            lowerer.gx = kb.global_tid_x();
            lowerer.gy = fn.grid_dim >= 2 ? b.build_add(b.build_mul(kb.ctaid_y(), kb.ntid_y()), kb.tid_y()) : b.build_iconst_i32(0);
            lowerer.gz = fn.grid_dim >= 3 ? b.build_add(b.build_mul(kb.ctaid_z(), kb.ntid_z()), kb.tid_z()) : b.build_iconst_i32(0);
        } else {
            lowerer.gx = b.build_iconst_i32(0);
            lowerer.gy = b.build_iconst_i32(0);
            lowerer.gz = b.build_iconst_i32(0);
        }
        lowerer.init_shared_memory();
        brass::BasicBlock* done_bb = nullptr;
        if (fn.stage == Stage::Compute && fn.grid_dim > 0 && !fn.uses_barriers_or_shared) {
            brass::Value* in_bounds = b.build_ult(lowerer.gx, lowerer.gw);
            if (fn.grid_dim >= 2) in_bounds = b.build_and(in_bounds, b.build_ult(lowerer.gy, lowerer.gh));
            if (fn.grid_dim >= 3) in_bounds = b.build_and(in_bounds, b.build_ult(lowerer.gz, lowerer.gd));
            brass::BasicBlock* body_bb = new_block(b, "in_bounds");
            done_bb = new_block(b, "done");
            b.build_br_if(in_bounds, body_bb, done_bb);
            b.position_at_end(body_bb);
        }
        lowerer.lower_stmt_list(fn.body);
        for (size_t i = 0; i < out.param_infos.size(); ++i) {
            out.param_infos[i].grid_cover_dim = lowerer.grid_cover[i];
        }
        out.uses_barriers_or_shared = fn.uses_barriers_or_shared;
        auto make_block_info = [&](const CheckedParam& cp, size_t idx) {
            KernelParamInfo bi;
            bi.name = cp.name;
            bi.type = cp.type;
            bi.role = cp.role;
            bi.access = cp.access;
            bi.descriptor_set = cp.descriptor_set;
            bi.descriptor_binding = cp.descriptor_binding;
            bi.mir_param_index = static_cast<uint32_t>(idx);
            bi.element_bytes = mod.types.block_stride(cp.type.element_type(), cp.type.is_uniform());
            return bi;
        };
        for (size_t i = 0; i < fn.params.size(); ++i) {
            if (fn.params[i].role == ParamRole::Uniform || fn.params[i].role == ParamRole::Buffer) {
                out.block_bindings.push_back(make_block_info(fn.params[i], i));
            }
        }
        std::vector<size_t> used_globals;
        for (size_t idx : lowerer.used_block_params) {
            if (idx >= fn.params.size()) used_globals.push_back(idx);
        }
        std::sort(used_globals.begin(), used_globals.end(), [&](size_t a, size_t b_idx) {
            const auto& ga = mod.globals[a - fn.params.size()];
            const auto& gb = mod.globals[b_idx - fn.params.size()];
            if (ga.descriptor_set != gb.descriptor_set) return ga.descriptor_set < gb.descriptor_set;
            return ga.descriptor_binding < gb.descriptor_binding;
        });
        for (size_t idx : used_globals) {
            out.block_bindings.push_back(make_block_info(mod.globals[idx - fn.params.size()], idx));
        }
        if (gpu_out_ptr && lowerer.ret_val_var_id >= 0) {
            brass::Value* lin = (fn.grid_dim > 0)
                                    ? lowerer.compute_linear_index(fn.grid_dim, lowerer.gx, lowerer.gy, lowerer.gz)
                                    : b.build_iconst_i32(0);
            lowerer.store_element_to_buffer(fn.return_type, {gpu_out_ptr}, lin, lowerer.var_values[lowerer.ret_val_var_id]);
        }
        if (done_bb) {
            b.build_br(done_bb);
            b.position_at_end(done_bb);
        }
        b.build_ret_void();
    }
    // 2. Build CPU Point & Grid MIR Functions (`cpu_point_function`, `cpu_grid_function`)
    if (fn.stage != Stage::Compute || fn.uses_textures) return out;
    out.cpu_point_function = out.cpu_module->create_function(
        fn.name + "_cpu_point", brass::Type::void_type(),
        {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr()});
    {
        brass::codegen::KernelBuilder kb(*out.cpu_module, out.cpu_point_function);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        brass::Value* args_pack = b.add_param(brass::Type::ptr());
        brass::Value* coords_shape = b.add_param(brass::Type::ptr());
        brass::Value* out_ret = b.add_param(brass::Type::ptr());
        MirBodyLowerer lowerer(mod, fn, diags, kb, /*is_gpu=*/false);
        lowerer.push_scope();
        lowerer.active_return_type = fn.return_type;
        lowerer.has_returned_var_id = lowerer.alloc_var("__has_returned", {b.build_iconst_i32(0)});
        if (!fn.return_type.is_void()) {
            lowerer.ret_val_var_id = lowerer.alloc_var("__ret_val", lowerer.make_zero_leaves(fn.return_type));
        }
        lowerer.gx = b.build_load(brass::Type::i32(), coords_shape, 0);
        lowerer.gy = b.build_load(brass::Type::i32(), coords_shape, 4);
        lowerer.gz = b.build_load(brass::Type::i32(), coords_shape, 8);
        lowerer.gw = b.build_load(brass::Type::i32(), coords_shape, 12);
        lowerer.gh = b.build_load(brass::Type::i32(), coords_shape, 16);
        lowerer.gd = b.build_load(brass::Type::i32(), coords_shape, 20);
        lowerer.lid_x = b.build_load(brass::Type::i32(), coords_shape, 24);
        lowerer.lid_y = b.build_load(brass::Type::i32(), coords_shape, 28);
        lowerer.lid_z = b.build_load(brass::Type::i32(), coords_shape, 32);
        lowerer.wgid_x = b.build_load(brass::Type::i32(), coords_shape, 36);
        lowerer.wgid_y = b.build_load(brass::Type::i32(), coords_shape, 40);
        lowerer.wgid_z = b.build_load(brass::Type::i32(), coords_shape, 44);
        lowerer.lsz_x = b.build_load(brass::Type::i32(), coords_shape, 48);
        lowerer.lsz_y = b.build_load(brass::Type::i32(), coords_shape, 52);
        lowerer.lsz_z = b.build_load(brass::Type::i32(), coords_shape, 56);
        lowerer.nwg_x = b.build_load(brass::Type::i32(), coords_shape, 60);
        lowerer.nwg_y = b.build_load(brass::Type::i32(), coords_shape, 64);
        lowerer.nwg_z = b.build_load(brass::Type::i32(), coords_shape, 68);
        lowerer.cpu_shared_base = b.build_load(brass::Type::ptr(), coords_shape, 72);
        lowerer.cpu_barrier_fn = b.build_load(brass::Type::ptr(), coords_shape, 80);
        lowerer.cpu_barrier_ctx = b.build_load(brass::Type::ptr(), coords_shape, 88);
        lowerer.init_shared_memory();
        int32_t slot = 0;
        for (size_t i = 0; i < fn.params.size(); ++i) {
            const auto& cp = fn.params[i];
            if (is_bound_param(cp.role)) {
                brass::Value* placeholder =
                    b.build_iconst_i32(static_cast<int32_t>((cp.descriptor_set << 16) | cp.descriptor_binding));
                if (cp.role == ParamRole::Uniform || cp.role == ParamRole::Buffer) {
                    lowerer.block_param[placeholder] = i;
                }
                lowerer.alloc_var(cp.name, {placeholder});
            } else if (cp.role == ParamRole::Store || cp.role == ParamRole::Stencil) {
                brass::Value* ptr_v = b.build_load(brass::Type::ptr(), args_pack, slot * 8);
                brass::Value* count_v = b.build_load(brass::Type::i64(), args_pack, slot * 8 + 8);
                lowerer.alloc_var(cp.name, {ptr_v, count_v});
                lowerer.bind_store_param(i, {ptr_v, count_v});
                slot += 2;
            } else {
                auto kinds = mod.types.flatten_scalars(cp.type);
                ValueList leaves;
                leaves.reserve(kinds.size());
                for (ScalarKind sk : kinds) {
                    leaves.push_back(b.build_load(to_mir_scalar_type(sk), args_pack, slot * 8));
                    slot += 1;
                }
                lowerer.alloc_var(cp.name, std::move(leaves));
            }
        }
        for (const auto& bb : out.block_bindings) {
            brass::Value* ptr_v = b.build_load(brass::Type::ptr(), args_pack, slot * 8);
            brass::Value* bytes_v = b.build_load(brass::Type::i64(), args_pack, slot * 8 + 8);
            lowerer.cpu_block_ptrs[bb.mir_param_index] = {ptr_v, bytes_v};
            slot += 2;
        }
        lowerer.lower_stmt_list(fn.body);
        out.cpu_shared_bytes = lowerer.total_cpu_shared_bytes;
        if (lowerer.ret_val_var_id >= 0) {
            lowerer.store_element_to_buffer(fn.return_type, {out_ret}, b.build_iconst_i32(0),
                                            lowerer.var_values[lowerer.ret_val_var_id]);
        }
        b.build_ret_void();
    }
    if (fn.uses_barriers_or_shared || (fn.grid_dim == 0 && fn.return_type.is_void())) return out;
    for (uint32_t lanes : {8u, 4u}) {
        if (!brass::host_has_avx2()) break;
        VectorPlan plan;
        bool narrow = false;
        bool scalar_only = false;
        for (int attempt = 0; attempt < 64 && !out.cpu_grid_function; ++attempt) {
            DiagnosticBag attempt_diags;
            brass::Function* grid = nullptr;
            try {
                grid = build_cpu_grid(*out.cpu_module, mod, fn, out.block_bindings, attempt_diags, lanes, &plan);
            } catch (const NeedNarrowLanes&) {
                narrow = true;
            } catch (const VectorUnsupported&) {
                scalar_only = true;
            }
            if (grid && !plan.redo) {
                out.cpu_grid_function = grid;
                out.cpu_grid_lanes = lanes;
                break;
            }
            if (!grid) {
                out.cpu_module->remove_function(out.cpu_module->functions().back());
            } else {
                out.cpu_module->remove_function(grid);
            }
            plan.redo = false;
            if (narrow || scalar_only) break;
        }
        if (out.cpu_grid_function || scalar_only) break;
    }
    if (!out.cpu_grid_function) {
        out.cpu_grid_function = build_cpu_grid(*out.cpu_module, mod, fn, out.block_bindings, diags, 0, nullptr);
    }
    return out;
}
} // namespace brokernel
