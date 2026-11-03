#pragma once
// ============================================================
//  petsc_solver_impl.hpp
//
//  Two responsibilities, kept in one file because they are
//  tightly coupled:
//
//  1.  IJacobianStrategy  — how the Jacobian is approximated.
//      Concrete strategies own their own buffers, shell matrices,
//      and PETSc callbacks.  PetscSolver calls them polymorphically.
//
//  2.  PetscSolver<Traits>  — owns all PETSc objects (TS, DM, Vecs).
//      All PETSc callbacks (fun_implicit, rhs_func, post_step_func,
//      primitive_to_conservative) are *static methods here*, not
//      free functions in residual_functs.hpp.
//
//  What disappeared entirely:
//    • The `NeedsChangeOfVariable` bool template parameter — it is
//      now a plain runtime 'if' inside fun_implicit.
//    • The free functions in residual_functs.hpp — replaced by static
//      methods, so the circular include with problem_def.hpp is gone.
//    • PetscSovrastructures — its fields are now first-class members
//      of PetscSolver with clear ownership semantics.
// ============================================================

#include "physics_evaluator.hpp"
#include "vec_dof_view.hpp"

namespace numPDE
{

    // Forward declaration needed by the strategy interface
    template <class Traits>
    class PetscSolver;

    // ================================================================
    //  Jacobian Strategy hierarchy
    // ================================================================

    template <class Traits>
    struct IJacobianStrategy
    {
        // ── Preconditioner state (managed by base class) ──
        Mat           P_mat{};
        MatFDColoring fdcolor{};

        virtual ~IJacobianStrategy()
        {
            if (P_mat) MatDestroy(&P_mat);
            if (fdcolor) MatFDColoringDestroy(&fdcolor);
        }

        virtual PetscErrorCode setup(TS ts, PetscSolver<Traits>& solver) = 0;

      protected:
        // I here pass dummy_ts when ud shouòld actually be void* snes from PETSc doc.
        static PetscErrorCode ts_coloring_wrapper(void* dummy_ts, Vec U, Vec F, void* /*ctx*/)
        {
            PetscFunctionBeginUser;
            TS   ts = static_cast<TS>(dummy_ts);
            SNES snes;

            // Fetch the internal SNES and let it evaluate the full, properly shifted
            // TS residual. This ensures the Preconditioner matches the true Jacobian!
            PetscCall(TSGetSNES(ts, &snes));
            PetscCall(SNESComputeFunction(snes, U, F));

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // Helper to generate the Preconditioner using DMDA coloring
        PetscErrorCode setup_colored_preconditioner(PetscSolver<Traits>& solver)
        {
            PetscFunctionBeginUser;

            // Let DMDA preallocate the exact sparsity pattern
            PetscCall(DMCreateMatrix(solver.dmda, &P_mat));

            // Generate the graph coloring
            ISColoring iscoloring;
            PetscCall(DMCreateColoring(solver.dmda, IS_COLORING_GLOBAL, &iscoloring));

            // Setup the FD Coloring context
            PetscCall(MatFDColoringCreate(P_mat, iscoloring, &fdcolor));

            // Wire it to wrapper!
            PetscCall(MatFDColoringSetFunction(
                fdcolor, &IJacobianStrategy<Traits>::ts_coloring_wrapper, nullptr));

            PetscCall(MatFDColoringSetFromOptions(fdcolor));
            PetscCall(MatFDColoringSetUp(P_mat, iscoloring, fdcolor));

            PetscCall(ISColoringDestroy(&iscoloring));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    };

    // ── No Jacobian (explicit time stepping) ──────────────────────────
    template <class Traits>
    struct NoJacobianStrategy final : IJacobianStrategy<Traits>
    {
        PetscErrorCode setup(TS, PetscSolver<Traits>&) override
        {
            // Nothing to register — RK/SSP methods never call TSSetIJacobian
            return PETSC_SUCCESS;
        }
    };

    // ── Finite-difference Jacobian via -snes_fd ────────────────────────
#if false
    template <class Traits>
    struct FDJacobianStrategy final : IJacobianStrategy<Traits>
    {
        PetscErrorCode setup(TS ts, PetscSolver<Traits>&) override
        {
            PetscFunctionBeginUser;
            SNES snes;
            PetscCall(TSGetSNES(ts, &snes));
            // Let PETSc build the FD Jacobian automatically
            PetscCall(SNESSetJacobian(snes, NULL, NULL, SNESComputeJacobianDefault, NULL));
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    };
#else 
// ── Finite-difference Jacobian with coloring ───────────────────────
    template <class Traits>
    struct FDJacobianStrategy final : IJacobianStrategy<Traits>
    {
        PetscErrorCode setup(TS ts, PetscSolver<Traits>& solver) override
        {
            PetscFunctionBeginUser;

            // 1. Setup the sparse preconditioner matrix and coloring graph using DMDA
            PetscCall(this->setup_colored_preconditioner(solver));

            // 2. Register with TS using our TS-compatible callback
            PetscCall(TSSetIJacobian(ts, this->P_mat, this->P_mat, compute_fd_jac, this));

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        static PetscErrorCode compute_fd_jac(TS ts, PetscReal /*t*/, Vec U, Vec /*U_dot*/,
                                             PetscReal /*sigma*/, Mat A, Mat P, void* ctx)
        {
            PetscFunctionBeginUser;
            auto* self = static_cast<FDJacobianStrategy*>(ctx);

            // Assemble P using the DMDA coloring (passes 'ts' to ts_coloring_wrapper!)
            PetscCall(MatFDColoringApply(P, self->fdcolor, U, ts));

            // If -snes_mf is used on the CLI, A != P and needs a final assembly call
            if (A != P)
            {
                PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
                PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }
    };
#endif

    // ── Complex-step Jacobian ──────────────────────────────────────────
    // Owns the complex buffers, the shell matrix, and its own callbacks.
    // The context pointer registered with MatShellSetContext and
    // TSSetIJacobian is 'this' (the strategy itself), so it is fully
    // self-contained — no navigation through Problem* needed.
    template <class Traits>
    struct ComplexStepJacobianStrategy final : IJacobianStrategy<Traits>
    {
        // ── Owned state ───────────────────────────────────────────────
        std::vector<CplxType> u_pl_iv; // u + i*v*eps  (ghosted, size set in setup)
        std::vector<CplxType> jv_buff; // residual in complex space  (same size)
        Mat                   j_shell{};

        // Scalars updated before each MatMult via pre_jv_eval
        PetscReal                  sigma{1.0};
        PetscReal                  t_curr{0.0}; // Time needed for the residual evaluation!
        static constexpr PetscReal csd_eps{1e-7};

        // Non-owning snapshot of the current solution, set by pre_jv_eval
        Vec u_curr{};

        // Back-pointer set in setup() — non-owning
        PetscSolver<Traits>* m_solver{nullptr};

        ~ComplexStepJacobianStrategy() { MatDestroy(&j_shell); }

        // ── Setup ─────────────────────────────────────────────────────
        PetscErrorCode setup(TS ts, PetscSolver<Traits>& solver) override
        {
            PetscFunctionBeginUser;
            m_solver = &solver;

            PetscCall(this->setup_colored_preconditioner(solver));

            const auto& phys = solver.r_phys;
            auto        n_elems_ghost{1uz};
            for (std::size_t i{}; i < Traits::n_dims; ++i)
                n_elems_ghost *= phys.m_dm_info.ghost_size[i];
            const auto n_ghosted = n_elems_ghost * phys.get_n_dof_per_element();
            u_pl_iv.resize(n_ghosted);
            jv_buff.resize(n_ghosted);

            PetscInt n_local, n_global;
            PetscCall(VecGetLocalSize(solver.solution, &n_local));
            PetscCall(VecGetSize(solver.solution, &n_global));

            // Shell context is 'this' — the strategy knows everything it needs
            PetscCall(MatCreateShell(PETSC_COMM_WORLD, n_local, n_local, n_global, n_global, this,
                                     &j_shell));
            PetscCall(MatShellSetOperation(j_shell, MATOP_MULT, (void (*)(void)) matmult));

            // Provide the Preconditioner matrix (FD + COLOR)
            PetscCall(TSSetIJacobian(ts, j_shell, this->P_mat, pre_jv_eval, this));

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // ── Static callbacks ──────────────────────────────────────────

        // Called by PETSc before each MatMult to snapshot sigma and u_curr
        static PetscErrorCode pre_jv_eval(TS ts, PetscReal, Vec U, Vec /*U_dot*/,
                                          PetscReal sigma_val, Mat /*J*/, Mat P, void* ctx)
        {
            PetscFunctionBeginUser;
            auto* self   = static_cast<ComplexStepJacobianStrategy*>(ctx);
            self->sigma  = sigma_val;
            self->u_curr = U;
            PetscCall(TSGetTime(ts, std::addressof(self->t_curr)));

            SNES   snes;
            KSP    ksp;
            PC     pc;
            PCType pc_type;
            PetscCall(TSGetSNES(ts, &snes));
            PetscCall(SNESGetKSP(snes, &ksp));
            PetscCall(KSPGetPC(ksp, &pc));
            PetscCall(PCGetType(pc, &pc_type));

            // Only assemble the colored preconditioning matrix if a PC is active
            if (std::strcmp(pc_type, PCNONE) != 0)
            {
                PetscCall(MatFDColoringApply(P, self->fdcolor, U, ts));
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }
        // J*v via complex step: Im( sigma*(u + i*v*eps) + R(u + i*v*eps) ) / eps
        static PetscErrorCode matmult(Mat J, Vec v_vec, Vec out_v)
        {
            PetscFunctionBeginUser;
            ComplexStepJacobianStrategy* self;
            PetscCall(MatShellGetContext(J, &self));

            // Compile-time branching completely drops the unchosen block from compilation
            if constexpr (Traits::n_dims == 1)
            {
                PetscCall(self->matmult_1d_impl(self, v_vec, out_v));
            }
            else if constexpr (Traits::n_dims == 2)
            {
                PetscCall(self->matmult_2d_impl(self, v_vec, out_v));
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }

      private:
        static PetscErrorCode matmult_2d_impl(ComplexStepJacobianStrategy* self, const Vec v_vec,
                                              Vec out_v)
        {
            PetscFunctionBeginUser;
            auto&       solver     = *self->m_solver;
            auto&       phys       = solver.r_phys;
            const auto& dminfo     = phys.m_dm_info;
            const auto  n_dof_elem = phys.get_n_dof_per_element();
            const auto  x_nds      = phys.m_gll_x.get_n_nodes();
            const auto  y_nds      = phys.m_gll_y.get_n_nodes();
            const auto  h          = self->csd_eps;
            const auto  one_over_h = 1.0 / h;

            WriteCplxView<2> u_iv_view{self->u_pl_iv, dminfo};
            ReadDOFView<2>   v_view{solver.dmda, v_vec, VecType::Local};
            ReadDOFView<2>   sol_view{solver.dmda, self->u_curr, VecType::Local};
            WriteDOFView<2>  jv_view{solver.dmda, out_v, VecType::Global};

            // ── Step 1: Build u + i*v*eps ──
            for (const auto j : dminfo.get_ghosted_elem_range(1uz))
            {
                for (const auto i : dminfo.get_ghosted_elem_range(0uz))
                {
                    auto       cplx = u_iv_view.as_span(i, j, n_dof_elem);
                    const auto v    = v_view.as_span(i, j, n_dof_elem);
                    const auto u    = sol_view.as_span(i, j, n_dof_elem);

                    if (!phys.m_opts.get_cond_sys())
                    {
                        for (auto&& [u_val, v_val, out] : std::views::zip(u, v, cplx))
                            out = CplxType(u_val, v_val * h);
                    }
                    else
                    {
                        const Indexer<const PetscReal> u_idx{u, x_nds, y_nds};
                        const Indexer<const PetscReal> v_idx{v, x_nds, y_nds};
                        Indexer<CplxType>              cplx_idx{cplx, x_nds, y_nds};

                        for (std::size_t l = 0; l < y_nds; ++l)
                        {
                            for (std::size_t k = 0; k < x_nds; ++k)
                            {
                                // Pointwise expansion over multi-dimensional node coordinates
                                const auto tilde_rho_c =
                                    CplxType{u_idx(0, k, l), v_idx(0, k, l) * h};
                                const auto u_vel_c = CplxType{u_idx(1, k, l), v_idx(1, k, l) * h};
                                const auto v_vel_c = CplxType{u_idx(2, k, l), v_idx(2, k, l) * h};
                                const auto tilde_ie_c =
                                    CplxType{u_idx(3, k, l), v_idx(3, k, l) * h};

                                // Generalized tuple expansion handles 2D velocities perfectly
                                std::tie(cplx_idx(0, k, l), cplx_idx(1, k, l), cplx_idx(2, k, l),
                                         cplx_idx(3, k, l)) =
                                    prim_to_cons_elem(tilde_rho_c, tilde_ie_c, u_vel_c, v_vel_c);
                            }
                        }
                    }
                }
            }

            // ── Step 2: Seed jv_view with sigma * Im(u_iv) / h ──
            for (const auto j : dminfo.get_global_elem_range(1))
            {
                for (const auto i : dminfo.get_global_elem_range(0))
                {
                    auto       u_iv_sp = u_iv_view.as_span(i, j, n_dof_elem);
                    const auto jv_sp   = jv_view.as_span(i, j, n_dof_elem);
                    for (auto&& [u_iv_val, jv_out] : std::views::zip(u_iv_sp, jv_sp))
                        jv_out = self->sigma * std::imag(u_iv_val) * one_over_h;
                }
            }

            // ── Step 4: Complex Multi-D Residual Evaluation ──
            WriteCplxView<2> f_cplx{self->jv_buff, dminfo};
            PetscCall(phys.template residual_kernel<CplxType>(u_iv_view, f_cplx, self->t_curr));

            // ── Step 5: Extract Im(R(u + i*v*eps)) / h and accumulate ──
            for (const auto j : dminfo.get_global_elem_range(1))
            {
                for (const auto i : dminfo.get_global_elem_range(0uz))
                {
                    auto       f_sp  = f_cplx.as_span(i, j, n_dof_elem);
                    const auto jv_sp = jv_view.as_span(i, j, n_dof_elem);
                    for (auto&& [f_c, jv_out] : std::views::zip(f_sp, jv_sp))
                        jv_out += std::imag(f_c) * one_over_h;
                }
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        static PetscErrorCode matmult_1d_impl(ComplexStepJacobianStrategy* self, const Vec v_vec,
                                              Vec out_v)
        {
            PetscFunctionBeginUser;

            auto&       solver     = *self->m_solver;
            auto&       phys       = solver.r_phys;
            const auto& dminfo     = phys.m_dm_info;
            const auto  n_dof_elem = phys.get_n_dof_per_element();
            const auto  nds        = phys.m_gll.get_n_nodes();
            const auto  h          = self->csd_eps;
            const auto  one_over_h = 1.0 / h;

            WriteCplxView<Traits::n_dims> u_iv_view{self->u_pl_iv, dminfo};
            ReadDOFView<Traits::n_dims>   v_view{solver.dmda, v_vec, VecType::Local};
            ReadDOFView<Traits::n_dims>   sol_view{solver.dmda, self->u_curr, VecType::Local};
            WriteDOFView<Traits::n_dims>  jv_view{solver.dmda, out_v, VecType::Global};

            // ── Step 1: Build u + i*v*eps (with optional change-of-variable) ──
            for (const auto elem : dminfo.get_ghosted_elem_range())
            {
                auto       cplx = u_iv_view.as_span(elem, n_dof_elem);
                const auto v    = v_view.as_span(elem, n_dof_elem);
                const auto u    = sol_view.as_span(elem, n_dof_elem);

                if (!phys.m_opts.get_cond_sys())
                {
                    for (auto&& [u_i, v_i, out] : std::views::zip(u, v, cplx))
                        out = CplxType{u_i, v_i * h};
                }
                else
                {
                    // Q = M(u + i*v*eps) : prim-to-cons in complex arithmetic
                    for (const auto node : std::views::iota(0uz, nds))
                    {
                        const auto exprho_c = CplxType{u[0 * nds + node], v[0 * nds + node] * h};
                        const auto vel_c    = CplxType{u[1 * nds + node], v[1 * nds + node] * h};
                        const auto expe_c   = CplxType{u[2 * nds + node], v[2 * nds + node] * h};
                        std::tie(cplx[0 * nds + node], cplx[1 * nds + node], cplx[2 * nds + node]) =
                            prim_to_cons_elem(exprho_c, expe_c, vel_c);
                    }
                }
            }

            // ── Step 2: Seed jv_view with sigma * Im(u_iv) / h ────────────────
            for (const auto elem : dminfo.get_global_elem_range())
            {
                auto       u_iv_sp = u_iv_view.as_span(elem, n_dof_elem);
                const auto jv_sp   = jv_view.as_span(elem, n_dof_elem);
                for (auto&& [u_iv, jv_out] : std::views::zip(u_iv_sp, jv_sp))
                    jv_out = self->sigma * std::imag(u_iv) * one_over_h;
            }

            // ── Step 4: Complex residual evaluation ───────────────────────────
            WriteCplxView<Traits::n_dims> f_cplx{self->jv_buff, dminfo};
            // Time = 0 is safe here: residual_kernel only uses time for BC lambdas,
            // and Jacobian-vector products are typically evaluated at the same time
            // as the IFunction call. Thread 'curr_time' through if your BCs are
            // time-dependent and you need exact linearisation of them.
            PetscCall(phys.template residual_kernel<CplxType>(u_iv_view, f_cplx, self->t_curr));

            // ── Step 5: Extract Im(R(u + i*v*eps)) / h and accumulate ─────────
            for (const auto elem : dminfo.get_global_elem_range())
            {
                auto       f_sp  = f_cplx.as_span(elem, n_dof_elem);
                const auto jv_sp = jv_view.as_span(elem, n_dof_elem);
                for (auto&& [f_c, jv_out] : std::views::zip(f_sp, jv_sp))
                    jv_out += std::imag(f_c) * one_over_h;
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        };
    };

    // ================================================================
    //  PetscSolver<Traits>
    //
    //  Owns all PETSc objects and the chosen Jacobian strategy.
    //  All PETSc-registered callbacks are static methods of this class.
    //  They all cast their void* context to PetscSolver<Traits>* and
    //  delegate to r_phys — the separation is enforced at the type level.
    // ================================================================

    template <class Traits>
    class PetscSolver
    {
      public:
        using T = typename Traits::T;

        // ── PETSc objects (all owned, destroyed in destructor) ────────
        DM  dmda{};
        TS  ts{};
        Vec solution{};

        // Optional buffers — allocated lazily in allocate_optional_buffers()
        Vec u_phys{};     // change-of-variable workspace (prim → cons)
        Vec u_filtered{}; // workspace for filters

        // ── Jacobian strategy ─────────────────────────────────────────
        // Selected at construction time based on ConfigOptions.
        // The virtual call overhead is negligible (called O(Newton iters), not O(DOF)).
        std::unique_ptr<IJacobianStrategy<Traits>> jac_strategy;

        // ── Reference to physics (non-owning) ──────────────────────
        PhysicsEvaluator<Traits>& r_phys;

        explicit PetscSolver(PhysicsEvaluator<Traits>& phys_ev) : r_phys{phys_ev}
        {
            // 1. Build the DMDA. This may adjust n_elems in m_physics.m_opts
            //    to match what MPI actually distributed, and it writes the
            //    resolved DMInfos back into r_phy.m_dm_info.
            this->setup_dmda(r_phys.m_opts);

            // 2. Create TS, allocate Vecs, select and wire the Jacobian strategy,
            //    register all PETSc callbacks, impose the IC.
            this->setup_ts();
        }

        ~PetscSolver()
        {
            TSDestroy(&ts); // also destroys the internal SNES
            VecDestroy(&solution);
            if (u_phys) VecDestroy(&u_phys);
            if (u_filtered) VecDestroy(&u_filtered);
            // dmda: lifetime is coupled to TS via TSSetDM, so NOT destroyed here.
            // If you ever decouple them, add DMDestroy(&dmda).
        }

        // ── Setup (called in order from Problem constructor) ──────────

        // 1. Build the DMDA and update r_phys.m_dm_info
        PetscErrorCode setup_dmda(const ConfigOptions<T>& opts)
        {
            PetscFunctionBeginUser;

            const PetscInt stencil_width = 1;
            const auto     dofs_per_elem = r_phys.get_n_dof_per_element();

            if constexpr (Traits::n_dims == 1)
            {
                DMBoundaryType bnd =
                    r_phys.is_periodic ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_GHOSTED;
                PetscCall(DMDACreate1d(PETSC_COMM_WORLD, bnd, opts.get_n_elems(0), dofs_per_elem,
                                       stencil_width, NULL, &dmda));
            }
            else if constexpr (Traits::n_dims == 2)
            {
                DMBoundaryType bnd_x =
                    r_phys.is_periodic_x ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_GHOSTED;
                DMBoundaryType bnd_y =
                    r_phys.is_periodic_y ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_GHOSTED;
                PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bnd_x, bnd_y, DMDA_STENCIL_BOX,
                                       opts.get_n_elems(0), opts.get_n_elems(1), PETSC_DECIDE,
                                       PETSC_DECIDE, dofs_per_elem, stencil_width, NULL, NULL,
                                       &dmda));
            }
            else
            {
                SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Only 1D and 2D supported");
            }

            PetscCall(DMSetFromOptions(dmda));
            PetscCall(DMSetUp(dmda));

            // Push the resolved layout back to the physics engine
            r_phys.m_dm_info = DMInfos<Traits::n_dims>(dmda);

            // Consistency-check and possibly update n_elems if DMDA redistributed
            std::array<PetscInt, 3> actuals{};
            PetscCall(DMDAGetInfo(dmda, NULL, &actuals[0], &actuals[1], &actuals[2], NULL, NULL,
                                  NULL, NULL, NULL, NULL, NULL, NULL, NULL));

            for (auto i{0uz}; i < Traits::n_dims; ++i)
            {
                if (static_cast<std::size_t>(actuals[i]) != opts.get_n_elems(i))
                {
                    PetscCall(PetscPrintf(
                        PETSC_COMM_WORLD,
                        "WARNING: DMDA mismatch in %zu -th direction, updating n_elems %zu → %d\n",
                        i, opts.get_n_elems(i), actuals[i]));
                    r_phys.m_opts.set_n_elems(static_cast<std::size_t>(actuals[i]), i);
                }
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // 2. Create TS, allocate vectors, register callbacks
        PetscErrorCode setup_ts()
        {
            PetscFunctionBeginUser;

            PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
            PetscCall(TSSetTimeStep(ts, 0.01));
            PetscCall(TSSetMaxTime(ts, 1.0));
            PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
            PetscCall(TSSetFromOptions(ts)); // CLI options override defaults

            PetscCall(TSSetApplicationContext(ts, this));

            PetscCall(DMCreateGlobalVector(dmda, &solution));
            PetscCall(impose_ic_on_sol_vec());

            PetscCall(TSSetDM(ts, dmda)); // SNES + DM are linked here

            PetscCall(allocate_optional_buffers());

            const bool impl = is_ts_implicit(ts);

            if (impl)
            {
                PetscCall(TSSetIFunction(ts, NULL, fun_implicit, this));

                if (r_phys.m_opts.get_cond_sys())
                    PetscCall(TSSetTransientVariable(ts, primitive_to_conservative, this));

                // Strategy is constructed, then wired into 'ts' by its own setup()
                jac_strategy = make_jac_strategy(/*implicit=*/true);
                PetscCall(jac_strategy->setup(ts, *this));

                // Evaluated AFTER time step is checked and accepted
                if (r_phys.m_opts.get_do_post_ts())
                {

                    if (r_phys.m_stab.filters_active())
                        PetscCall(TSSetPostStep(this->ts, log_t_and_appl_stab));
                    else
                        PetscCall(TSSetPostStep(this->ts, log_time));
                }
                else
                {
                    if (r_phys.m_stab.filters_active())
                        PetscCall(TSSetPostEvaluate(this->ts, apply_stabilization));

                    PetscCall(TSSetPostStep(this->ts, log_time));
                }
            }
            else
            {
                PetscCall(TSSetRHSFunction(ts, NULL, rhs_func, this));
                // Explicit methods need no Jacobian
                jac_strategy = std::make_unique<NoJacobianStrategy<Traits>>();

                if (r_phys.m_stab.filters_active())
                    PetscCall(TSSetPostStep(this->ts, log_t_and_appl_stab));
                else
                    PetscCall(TSSetPostStep(this->ts, log_time));
            }

            PetscCall(TSSetUp(ts));
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // 3. Fill solution with the initial condition
        PetscErrorCode impose_ic_on_sol_vec()
        {
            PetscFunctionBeginUser;
            PetscCall(r_phys.impose_ic(solution, dmda));
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // ── Static PETSc callbacks ─────────────────────────────────────
        // All cast ctx to PetscSolver<Traits>* — no navigation through Problem* —
        // and delegate entirely to r_phys.

        static PetscErrorCode fun_implicit(TS ts, PetscReal /*t*/, Vec U, Vec U_t, Vec F, void* ctx)
        {
            PetscFunctionBeginUser;
            auto* self = static_cast<PetscSolver<Traits>*>(ctx);

            PetscReal time{};
            PetscCall(TSGetTime(ts, &time));

            Vec active = U;

            // Change of variable: prim → cons before the residual evaluation
            // Plain runtime 'if' replaces the old NeedsChangeOfVariable<bool> template param
            if (self->r_phys.m_opts.get_cond_sys())
            {
                ReadDOFView<Traits::n_dims>  prim{self->dmda, active, VecType::Global};
                WriteDOFView<Traits::n_dims> cons{self->dmda, self->u_phys, VecType::Global};
                PetscCall(self->r_phys.template prim_to_cons_kernel<T>(prim, cons));
                active = self->u_phys;
            }

            // Physical residual
            {
                ReadDOFView<Traits::n_dims>  x{self->dmda, active, VecType::Local};
                WriteDOFView<Traits::n_dims> f{self->dmda, F, VecType::Global};
                PetscCall(self->r_phys.template residual_kernel<T>(x, f, static_cast<T>(time)));
            }

            PetscCall(VecAYPX(F, +1.0, U_t)); // F ← U_t + F
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        static PetscErrorCode rhs_func(TS /*ts*/, PetscReal time, Vec U, Vec rhs, void* ctx)
        {
            PetscFunctionBeginUser;
            auto* self = static_cast<PetscSolver<Traits>*>(ctx);

            {
                ReadDOFView<Traits::n_dims>  x{self->dmda, U, VecType::Local};
                WriteDOFView<Traits::n_dims> f{self->dmda, rhs, VecType::Global};
                PetscCall(self->r_phys.template residual_kernel<T>(x, f, static_cast<T>(time)));
            }

            PetscCall(VecScale(rhs, -1.0)); // rhs = -R(U)
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        static PetscErrorCode log_t_and_appl_stab(TS ts)
        {
            PetscFunctionBeginUser;

            apply_stabilization(ts);
            log_time(ts);

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // Work using the u_filtered workspace
        static PetscErrorCode apply_stabilization(TS ts)
        {
            PetscFunctionBeginUser;
            PetscSolver<Traits>* self{nullptr};
            PetscCall(TSGetApplicationContext(ts, &self));

            // Copy instead of std::swap because petsc may give troubles otherwise
            // // Vec holds just a pointer, swapping them may end up in issues since
            // // TS stores and works just with the pointers of solution vec
            PetscCall(VecCopy(self->solution, self->u_filtered));

            {
                ReadDOFView<Traits::n_dims>  sol{self->dmda, self->u_filtered, VecType::Local};
                WriteDOFView<Traits::n_dims> out{self->dmda, self->solution, VecType::Global};

                self->r_phys.template limit_solution<T>(sol, out);
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }
        static PetscErrorCode log_time(TS ts)
        {
            PetscFunctionBeginUser;
            /*
             *  void* ctx;
             *  PetscCall(TSGetApplicationContext(ts, &ctx));
             *  auto* self = static_cast<PetscSolver<Traits>*>(ctx);
             */

            PetscReal time{};
            PetscCall(TSGetTime(ts, &time));

            numPDE::println("Time : {:}", time);
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        static PetscErrorCode primitive_to_conservative(TS, Vec in, Vec out, void* ctx)
        {
            PetscFunctionBeginUser;
            auto*                        self = static_cast<PetscSolver<Traits>*>(ctx);
            ReadDOFView<Traits::n_dims>  i_view{self->dmda, in, VecType::Global};
            WriteDOFView<Traits::n_dims> o_view{self->dmda, out, VecType::Global};
            PetscCall(self->r_phys.template prim_to_cons_kernel<T>(i_view, o_view));
            PetscFunctionReturn(PETSC_SUCCESS);
        }

      private:
        // ── Internal helpers ──────────────────────────────────────────

        PetscErrorCode allocate_optional_buffers()
        {
            PetscFunctionBeginUser;
            if (r_phys.m_opts.get_cond_sys()) PetscCall(VecDuplicate(solution, &u_phys));
            if (r_phys.m_stab.filters_active()) PetscCall(VecDuplicate(solution, &u_filtered));
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // Single place where the Jacobian type is decided.
        // Adding a new approximation = adding a new case here + a new Strategy class.
        std::unique_ptr<IJacobianStrategy<Traits>> make_jac_strategy(bool implicit)
        {
            if (!implicit) return std::make_unique<NoJacobianStrategy<Traits>>();

            switch (r_phys.m_opts.get_jac_type())
            {
                case JacobianType::ComplexFD:
                    return std::make_unique<ComplexStepJacobianStrategy<Traits>>();
                case JacobianType::FD:
                default:
                    return std::make_unique<FDJacobianStrategy<Traits>>();
            }
        }
    };

} // namespace numPDE
