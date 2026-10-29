#include "lowering_internal.h"
#include <algorithm>
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
            auto vars_before = var_values;
            auto live_ids = sorted_live_var_ids();
            brass::BasicBlock* cont_bb = new_block(b, "after_ret_check");
            brass::BasicBlock* skip_bb = new_block(b, "skip_after_ret");
            brass::BasicBlock* merge_bb = new_block(b, "ret_check_merge");
            b.build_br_if(not_ret, cont_bb, skip_bb);
            b.position_at_end(cont_bb);
            std::vector<StmtPtr> remaining;
            for (size_t j = i + 1; j < stmts.size(); ++j) {
                remaining.push_back(clone_stmt(stmts[j].get()));
            }
            lower_stmt_list(remaining);
            brass::BasicBlock* cont_end = b.current_block();
            auto vars_cont = var_values;
            b.position_at_end(skip_bb);
            brass::BasicBlock* skip_end = b.current_block();
            std::vector<brass::Value*> cont_args;
            std::vector<brass::Value*> skip_args;
            var_values = vars_before;
            for (int id : live_ids) {
                auto& v_cont = vars_cont[id];
                auto& v_skip = vars_before[id];
                for (size_t k = 0; k < v_skip.size() && k < v_cont.size(); ++k) {
                    if (v_cont[k] != v_skip[k]) {
                        brass::Value* phi = b.add_block_param(merge_bb, v_skip[k]->type());
                        cont_args.push_back(v_cont[k]);
                        skip_args.push_back(v_skip[k]);
                        var_values[id][k] = phi;
                    }
                }
            }
            b.position_at_end(cont_end);
            b.build_br(merge_bb, cont_args);
            b.position_at_end(skip_end);
            b.build_br(merge_bb, skip_args);
            b.position_at_end(merge_bb);
            break;
        }
    }
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
            ValueList init = s->init_expr ? lower_expr(s->init_expr.get())
                                          : make_zero_leaves(s->resolved_var_type);
            alloc_var(s->var_name, std::move(init));
            break;
        }
        case Stmt::Kind::Assign: {
            ValueList rhs = lower_expr(s->assign_value.get());
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
                if (s->compound_op.has_value()) {
                    ValueList old_v = load_element_from_buffer(elem_t, base[0], lin);
                    for (size_t k = 0; k < old_v.size() && k < rhs.size(); ++k) {
                        rhs[k] = lower_scalar_binary(*s->compound_op, old_v[k], rhs[k], elem_t.scalar);
                    }
                }
                store_element_to_buffer(elem_t, base[0], lin, rhs);
                return;
            }
            // 2. Local variable (or vector swizzle / struct field / array element of local variable)
            if (s->assign_target->kind == Expr::Kind::Ident) {
                int id = lookup_var_id(s->assign_target->name);
                if (id >= 0) {
                    if (s->compound_op.has_value()) {
                        auto& cur = var_values[id];
                        ScalarKind sk = s->assign_target->resolved_type.scalar;
                        for (size_t k = 0; k < cur.size() && k < rhs.size(); ++k) {
                            cur[k] = lower_scalar_binary(*s->compound_op, cur[k], rhs[k], sk);
                        }
                    } else {
                        var_values[id] = std::move(rhs);
                    }
                }
                return;
            }
            if (s->assign_target->kind == Expr::Kind::Field &&
                s->assign_target->lhs->kind == Expr::Kind::Ident) {
                int id = lookup_var_id(s->assign_target->lhs->name);
                if (id < 0) return;
                auto& cur = var_values[id];
                const Type& base_t = s->assign_target->lhs->resolved_type;
                if (base_t.is_vector()) {
                    const std::string& swz = s->assign_target->name;
                    for (size_t k = 0; k < swz.size() && k < rhs.size(); ++k) {
                        uint32_t idx = swizzle_index(swz[k]);
                        if (idx < cur.size()) {
                            cur[idx] = s->compound_op.has_value()
                                           ? lower_scalar_binary(*s->compound_op, cur[idx], rhs[k], base_t.scalar)
                                           : rhs[k];
                        }
                    }
                } else if (base_t.is_struct()) {
                    const StructDef* sdef = mod.types.find_struct(base_t.struct_id);
                    if (sdef) {
                        if (const StructField* sf = sdef->find_field(s->assign_target->name)) {
                            for (uint32_t k = 0; k < sf->flat_leaf_count && k < rhs.size(); ++k) {
                                uint32_t idx = sf->flat_leaf_offset + k;
                                if (idx < cur.size()) {
                                    cur[idx] = s->compound_op.has_value()
                                                   ? lower_scalar_binary(*s->compound_op, cur[idx], rhs[k], sf->type.scalar)
                                                   : rhs[k];
                                }
                            }
                        }
                    }
                }
                return;
            }
            break;
        }
        case Stmt::Kind::If: {
            ValueList cv = lower_expr(s->cond.get());
            brass::Value* cond = cv.empty() ? b.build_iconst_i32(0) : cv[0];
            auto vars_before = var_values;
            auto live_ids = sorted_live_var_ids();
            brass::BasicBlock* then_bb = new_block(b, "if_then");
            brass::BasicBlock* else_bb = new_block(b, "if_else");
            brass::BasicBlock* merge_bb = new_block(b, "if_merge");
            b.build_br_if(cond, then_bb, else_bb);
            b.position_at_end(then_bb);
            var_values = vars_before;
            push_scope();
            lower_stmt_list(s->body);
            pop_scope();
            brass::BasicBlock* then_end = b.current_block();
            auto vars_then = var_values;
            b.position_at_end(else_bb);
            var_values = vars_before;
            if (!s->else_body.empty()) {
                push_scope();
                lower_stmt_list(s->else_body);
                pop_scope();
            }
            brass::BasicBlock* else_end = b.current_block();
            auto vars_else = var_values;
            std::vector<brass::Value*> then_args;
            std::vector<brass::Value*> else_args;
            var_values = vars_before;
            for (int id : live_ids) {
                auto& vt = vars_then[id];
                auto& ve = vars_else[id];
                for (size_t k = 0; k < vt.size() && k < ve.size(); ++k) {
                    if (vt[k] != ve[k]) {
                        brass::Value* phi = b.add_block_param(merge_bb, vt[k]->type());
                        then_args.push_back(vt[k]);
                        else_args.push_back(ve[k]);
                        var_values[id][k] = phi;
                    }
                }
            }
            b.position_at_end(then_end);
            b.build_br(merge_bb, then_args);
            b.position_at_end(else_end);
            b.build_br(merge_bb, else_args);
            b.position_at_end(merge_bb);
            break;
        }
        case Stmt::Kind::ForRange: {
            ValueList start_v = lower_expr(s->range_start.get());
            ValueList end_v = lower_expr(s->range_end.get());
            push_scope();
            int loop_var_id = alloc_var(s->var_name, start_v);
            int broke_var_id = alloc_var("__loop_broke_" + std::to_string(loop_var_id), {b.build_iconst_i32(0)});
            int cont_var_id = alloc_var("__loop_cont_" + std::to_string(loop_var_id), {b.build_iconst_i32(0)});
            auto live_ids = sorted_live_var_ids();
            std::vector<brass::Value*> init_args = pack_var_ids(live_ids);
            brass::BasicBlock* head_bb = new_block(b, "for_head");
            brass::BasicBlock* body_bb = new_block(b, "for_body");
            brass::BasicBlock* exit_bb = new_block(b, "for_exit");
            b.build_br(head_bb, init_args);
            for (int id : live_ids) {
                for (auto& leaf : var_values[id]) {
                    leaf = b.add_block_param(head_bb, leaf->type());
                }
            }
            auto vars_at_head = var_values;
            b.position_at_end(head_bb);
            brass::Value* cur_i = var_values[loop_var_id][0];
            brass::Value* cond = s->inclusive_range ? b.build_sle(cur_i, end_v[0])
                                                    : b.build_slt(cur_i, end_v[0]);
            brass::Value* not_broke = b.build_eq(var_values[broke_var_id][0], b.build_iconst_i32(0));
            if (has_returned_var_id >= 0) {
                not_broke = b.build_and(not_broke, b.build_eq(var_values[has_returned_var_id][0], b.build_iconst_i32(0)));
            }
            cond = b.build_and(not_broke, cond);
            b.build_br_if(cond, body_bb, exit_bb);
            b.position_at_end(body_bb);
            var_values[cont_var_id] = {b.build_iconst_i32(0)};
            LoopLoweringCtx lctx{head_bb, exit_bb, broke_var_id, cont_var_id};
            loop_stack.push_back(lctx);
            lower_stmt_list(s->body);
            loop_stack.pop_back();
            // Step loop variable and branch back to head_bb
            brass::Value* step_one = (s->resolved_var_type.scalar == ScalarKind::I64 ||
                                      s->resolved_var_type.scalar == ScalarKind::U64)
                                         ? b.build_iconst_i64(1)
                                         : b.build_iconst_i32(1);
            var_values[loop_var_id][0] = b.build_add(var_values[loop_var_id][0], step_one);
            b.build_br(head_bb, pack_var_ids(live_ids));
            b.position_at_end(exit_bb);
            var_values = vars_at_head;
            pop_scope();
            break;
        }
        case Stmt::Kind::ForC:
        case Stmt::Kind::While: {
            push_scope();
            if (s->kind == Stmt::Kind::ForC && s->for_init) {
                lower_stmt(s->for_init.get());
            }
            int broke_var_id = alloc_var("__loop_broke_" + std::to_string(next_var_id), {b.build_iconst_i32(0)});
            int cont_var_id = alloc_var("__loop_cont_" + std::to_string(next_var_id), {b.build_iconst_i32(0)});
            auto live_ids = sorted_live_var_ids();
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
            brass::Value* cond = b.build_iconst_i32(1);
            if (s->cond) {
                ValueList cv = lower_expr(s->cond.get());
                if (!cv.empty()) cond = cv[0];
            }
            brass::Value* not_broke = b.build_eq(var_values[broke_var_id][0], b.build_iconst_i32(0));
            if (has_returned_var_id >= 0) {
                not_broke = b.build_and(not_broke, b.build_eq(var_values[has_returned_var_id][0], b.build_iconst_i32(0)));
            }
            cond = b.build_and(not_broke, cond);
            b.build_br_if(cond, body_bb, exit_bb);
            b.position_at_end(body_bb);
            var_values[cont_var_id] = {b.build_iconst_i32(0)};
            LoopLoweringCtx lctx{head_bb, exit_bb, broke_var_id, cont_var_id};
            loop_stack.push_back(lctx);
            lower_stmt_list(s->body);
            loop_stack.pop_back();
            if (s->kind == Stmt::Kind::ForC && s->for_step) {
                lower_stmt(s->for_step.get());
            }
            b.build_br(head_bb, pack_var_ids(live_ids));
            b.position_at_end(exit_bb);
            var_values = vars_at_head;
            pop_scope();
            break;
        }
        case Stmt::Kind::Break: {
            if (!loop_stack.empty()) {
                var_values[loop_stack.back().broke_var_id] = {b.build_iconst_i32(1)};
            }
            break;
        }
        case Stmt::Kind::Continue: {
            if (!loop_stack.empty()) {
                var_values[loop_stack.back().cont_var_id] = {b.build_iconst_i32(1)};
            }
            break;
        }
        case Stmt::Kind::Return: {
            if (s->expr && ret_val_var_id >= 0) {
                var_values[ret_val_var_id] = lower_expr(s->expr.get());
            }
            if (has_returned_var_id >= 0) {
                var_values[has_returned_var_id] = {b.build_iconst_i32(1)};
            }
            break;
        }
        case Stmt::Kind::Expr:
            lower_expr(s->expr.get());
            break;
    }
}
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
                pinfo.mir_param_count = 1;
                brass::Value* ptr_p = b.add_param(brass::Type::ptr());
                lowerer.alloc_var(cp.name, {ptr_p});
                mir_idx += 1;
            } else if (cp.role == ParamRole::Image || cp.role == ParamRole::Sampler) {
                pinfo.mir_param_count = 0;
                lowerer.alloc_var(cp.name, {b.build_iconst_i32(static_cast<int32_t>(cp.descriptor_binding))});
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
        // Implicit grid extent parameters (__grid_w, __grid_h, __grid_d) when grid_dim > 0
        if (fn.grid_dim > 0) {
            out.grid_dim_mir_index = mir_idx;
            lowerer.gw = b.add_param(brass::Type::i32());
            mir_idx += 1;
            if (fn.grid_dim >= 2) {
                lowerer.gh = b.add_param(brass::Type::i32());
                mir_idx += 1;
            } else {
                lowerer.gh = b.build_iconst_i32(1);
            }
            if (fn.grid_dim >= 3) {
                lowerer.gd = b.add_param(brass::Type::i32());
                mir_idx += 1;
            } else {
                lowerer.gd = b.build_iconst_i32(1);
            }
        } else {
            lowerer.gw = b.build_iconst_i32(1);
            lowerer.gh = b.build_iconst_i32(1);
            lowerer.gd = b.build_iconst_i32(1);
        }
        // Implicit output buffer parameter (__out) when return_type != void
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
        // Compute global grid coordinates (gx, gy, gz)
        if (fn.stage == Stage::Compute) {
            lowerer.gx = kb.global_tid_x();
            if (fn.grid_dim >= 2) {
                lowerer.gy = b.build_add(b.build_mul(kb.ctaid_y(), kb.ntid_y()), kb.tid_y());
            } else {
                lowerer.gy = b.build_iconst_i32(0);
            }
            if (fn.grid_dim >= 3) {
                lowerer.gz = b.build_add(b.build_mul(kb.ctaid_z(), kb.ntid_z()), kb.tid_z());
            } else {
                lowerer.gz = b.build_iconst_i32(0);
            }
        } else {
            lowerer.gx = b.build_iconst_i32(0);
            lowerer.gy = b.build_iconst_i32(0);
            lowerer.gz = b.build_iconst_i32(0);
        }
        // Bounds guard for grid compute kernels (skipped if kernel uses workgroup barriers)
        brass::BasicBlock* done_bb = nullptr;
        if (fn.stage == Stage::Compute && fn.grid_dim > 0 && !fn.uses_barriers_or_shared) {
            brass::Value* in_bounds = b.build_ult(lowerer.gx, lowerer.gw);
            if (fn.grid_dim >= 2) {
                in_bounds = b.build_and(in_bounds, b.build_ult(lowerer.gy, lowerer.gh));
            }
            if (fn.grid_dim >= 3) {
                in_bounds = b.build_and(in_bounds, b.build_ult(lowerer.gz, lowerer.gd));
            }
            brass::BasicBlock* body_bb = new_block(b, "in_bounds");
            done_bb = new_block(b, "done");
            b.build_br_if(in_bounds, body_bb, done_bb);
            b.position_at_end(body_bb);
        }
        lowerer.lower_stmt_list(fn.body);
        if (gpu_out_ptr && lowerer.ret_val_var_id >= 0) {
            brass::Value* lin = (fn.grid_dim > 0)
                                    ? lowerer.compute_linear_index(fn.grid_dim, lowerer.gx, lowerer.gy, lowerer.gz)
                                    : b.build_iconst_i32(0);
            lowerer.store_element_to_buffer(fn.return_type, gpu_out_ptr, lin, lowerer.var_values[lowerer.ret_val_var_id]);
        }
        if (done_bb) {
            b.build_br(done_bb);
            b.position_at_end(done_bb);
        }
        b.build_ret_void();
    }
    // -----------------------------------------------------------------------
    // 2. Build CPU Point & Grid MIR Functions (`cpu_point_function`, `cpu_grid_function`)
    // -----------------------------------------------------------------------
    // Signature of `cpu_point_function`:
    //   void __cpu_point(const uint64_t* args_pack, const int32_t* coords_and_shape6, void* out_ret)
    // Signature of `cpu_grid_function`:
    //   void __cpu_grid(const uint64_t* args_pack, const int32_t* shape3, void* out_ret)
    //
    // The CPU runs compute kernels cell by cell, a subgroup and workgroup of
    // one. Vertex and fragment stages (rasterizer inputs, derivatives,
    // discard) and kernels that cooperate through barriers or shared memory
    // have no meaning there, so they get no CPU build.
    if (fn.stage != Stage::Compute || fn.uses_barriers_or_shared) return out;
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
        int32_t slot = 0;
        for (const auto& cp : fn.params) {
            if (cp.role == ParamRole::Grid || cp.role == ParamRole::Image || cp.role == ParamRole::Sampler) {
                lowerer.alloc_var(cp.name, {b.build_iconst_i32(0)});
            } else if (cp.role == ParamRole::Store || cp.role == ParamRole::Stencil) {
                brass::Value* ptr_v = b.build_load(brass::Type::ptr(), args_pack, slot * 8);
                lowerer.alloc_var(cp.name, {ptr_v});
                slot += 1;
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
        lowerer.lower_stmt_list(fn.body);
        if (lowerer.ret_val_var_id >= 0) {
            lowerer.store_element_to_buffer(fn.return_type, out_ret, b.build_iconst_i32(0),
                                            lowerer.var_values[lowerer.ret_val_var_id]);
        }
        b.build_ret_void();
    }
    out.cpu_grid_function = out.cpu_module->create_function(
        fn.name + "_cpu_grid", brass::Type::void_type(),
        {brass::Type::ptr(), brass::Type::ptr(), brass::Type::ptr()});
    {
        brass::codegen::KernelBuilder kb(*out.cpu_module, out.cpu_grid_function);
        brass::Builder& b = kb.builder();
        brass::BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        brass::Value* args_pack = b.add_param(brass::Type::ptr());
        brass::Value* shape_ptr = b.add_param(brass::Type::ptr());
        brass::Value* out_ret = b.add_param(brass::Type::ptr());
        brass::Value* gw = b.build_load(brass::Type::i32(), shape_ptr, 0);
        brass::Value* gh = b.build_load(brass::Type::i32(), shape_ptr, 4);
        brass::Value* gd = b.build_load(brass::Type::i32(), shape_ptr, 8);
        brass::Value* total = b.build_mul(b.build_mul(gw, gh), gd);
        brass::Value* gh_gw = b.build_mul(gh, gw);
        // Unpack kernel parameters once in the entry block outside the loop
        std::vector<ValueList> unpacked_params;
        unpacked_params.reserve(fn.params.size());
        int32_t slot = 0;
        for (const auto& cp : fn.params) {
            if (cp.role == ParamRole::Grid || cp.role == ParamRole::Image || cp.role == ParamRole::Sampler) {
                unpacked_params.push_back({b.build_iconst_i32(0)});
            } else if (cp.role == ParamRole::Store || cp.role == ParamRole::Stencil) {
                unpacked_params.push_back({b.build_load(brass::Type::ptr(), args_pack, slot * 8)});
                slot += 1;
            } else {
                auto kinds = mod.types.flatten_scalars(cp.type);
                ValueList leaves;
                leaves.reserve(kinds.size());
                for (ScalarKind sk : kinds) {
                    leaves.push_back(b.build_load(to_mir_scalar_type(sk), args_pack, slot * 8));
                    slot += 1;
                }
                unpacked_params.push_back(std::move(leaves));
            }
        }
        // Single flat loop `idx in 0..total` recovering (gx, gy, gz)
        brass::BasicBlock* loop_head = new_block(b, "grid_head");
        brass::BasicBlock* loop_body = new_block(b, "grid_body");
        brass::BasicBlock* loop_exit = new_block(b, "grid_exit");
        b.build_br(loop_head, {b.build_iconst_i32(0)});
        brass::Value* idx = b.add_block_param(loop_head, brass::Type::i32());
        b.position_at_end(loop_head);
        b.build_br_if(b.build_slt(idx, total), loop_body, loop_exit);
        b.position_at_end(loop_body);
        MirBodyLowerer lowerer(mod, fn, diags, kb, /*is_gpu=*/false);
        lowerer.push_scope();
        lowerer.gw = gw;
        lowerer.gh = gh;
        lowerer.gd = gd;
        if (fn.grid_dim <= 1) {
            lowerer.gx = idx;
            lowerer.gy = b.build_iconst_i32(0);
            lowerer.gz = b.build_iconst_i32(0);
        } else if (fn.grid_dim == 2) {
            lowerer.gx = b.build_smod(idx, gw);
            lowerer.gy = b.build_sdiv(idx, gw);
            lowerer.gz = b.build_iconst_i32(0);
        } else {
            lowerer.gz = b.build_sdiv(idx, gh_gw);
            brass::Value* rem = b.build_smod(idx, gh_gw);
            lowerer.gy = b.build_sdiv(rem, gw);
            lowerer.gx = b.build_smod(rem, gw);
        }
        lowerer.active_return_type = fn.return_type;
        lowerer.has_returned_var_id = lowerer.alloc_var("__has_returned", {b.build_iconst_i32(0)});
        if (!fn.return_type.is_void()) {
            lowerer.ret_val_var_id = lowerer.alloc_var("__ret_val", lowerer.make_zero_leaves(fn.return_type));
        }
        for (size_t i = 0; i < fn.params.size(); ++i) {
            lowerer.alloc_var(fn.params[i].name, unpacked_params[i]);
        }
        lowerer.lower_stmt_list(fn.body);
        if (lowerer.ret_val_var_id >= 0) {
            // Only write if shape4[3] != 0 (i.e. return_store_out != nullptr)
            brass::Value* has_out_i32 = b.build_load(brass::Type::i32(), shape_ptr, 12);
            brass::Value* has_out = b.build_ne(has_out_i32, b.build_iconst_i32(0));
            brass::BasicBlock* write_out_bb = new_block(b, "write_ret");
            brass::BasicBlock* next_iter_bb = new_block(b, "next_iter");
            b.build_br_if(has_out, write_out_bb, next_iter_bb);
            b.position_at_end(write_out_bb);
            lowerer.store_element_to_buffer(fn.return_type, out_ret, idx,
                                            lowerer.var_values[lowerer.ret_val_var_id]);
            b.build_br(next_iter_bb);
            b.position_at_end(next_iter_bb);
        }
        brass::Value* next_idx = b.build_add(idx, b.build_iconst_i32(1));
        b.build_br(loop_head, {next_idx});
        b.position_at_end(loop_exit);
        b.build_ret_void();
    }
    return out;
}
} // namespace brokernel
