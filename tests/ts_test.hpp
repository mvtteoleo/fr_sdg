#pragma once
#include "../include/macro_and_libc.hpp"
#include "vec_view.hpp"
#include <exception>
#include <print>
#include <ranges>

namespace numPDE
{

    static constexpr const char* help_message =
        "\n\n=========================================================\n"
        "Test to check the PETSc machinery for the TS \n"
        "*  BDF_2 \n./build/test/ts_test  -ts_monitor_error -ts_dt 0.01 "
        "-ts_type bdf -ts_bdf_order 2 -ts_adapt_type none -pc_type none"
        "\n* RK \n./build/test/ts_test  "
        "-ts_monitor_error -ts_dt 0.1 -ts_type rk  -ts_adapt_type none"
        "\n\n=========================================================";

    // Transformation function
    PetscErrorCode tilde_to_u(const Vec u_tilde, Vec u)
    {
        PetscFunctionBeginUser;
        VecReadSpan  u_t_sp(u_tilde);
        VecWriteSpan u_sp(u);
        std::ranges::transform(u_t_sp.span(), u_sp.span().begin(),
                               [](auto u_ti) { return std::exp(u_ti); });

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode u_to_tilde(const Vec u, Vec u_tilde)
    {
        PetscFunctionBeginUser;
        VecWriteSpan u_t_sp(u_tilde);
        VecReadSpan  u_sp(u);
        std::ranges::transform(u_sp.span(), u_t_sp.span().begin(),
                               [](auto u_i) { return std::log(u_i); });
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode primitive_to_conservative([[maybe_unused]] TS ts, Vec in, Vec out,
                                             [[maybe_unused]] void* ctx)
    {
        PetscFunctionBeginUser;
        PetscCall(tilde_to_u(in, out));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // The constant variable is needed to allow easy switch from implicit to explicit
    enum class TimeInt
    {
        Implicit = 0,
        Explicit
    };
    template <typename T, TimeInt time_int>
    void my_rhs_impl(std::span<const T> u, std::span<T> f)
    {
        T c_type{};
        if constexpr (time_int == TimeInt::Explicit) c_type = T{-1.0};
        if constexpr (time_int == TimeInt::Implicit) c_type = T{1.0};

        f[0] = c_type * (u[0] - u[1]);
        f[1] = c_type * (u[1] - u[2]);
        f[2] = c_type * (u[2]);
        return;
    }

    PetscErrorCode rhs_func(TS /*ts*/, PetscReal /*t*/, Vec v_in, Vec v_out, void* /*ctx*/)
    {
        PetscFunctionBeginUser;
        VecReadSpan  u_view{v_in};
        VecWriteSpan g_view{v_out};
        my_rhs_impl<PetscReal, TimeInt::Explicit>(u_view.span(), g_view.span());
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscErrorCode petsc_sol_fun(TS /*ts*/, PetscReal t, Vec u, void* /*ctx*/)
    {
        PetscFunctionBeginUser;
        const auto et = std::exp(-t);

        PetscReal* u_ptr;
        PetscCall(VecGetArray(u, &u_ptr));

        u_ptr[0] = (1 + t + 0.5 * t * t) * et;
        u_ptr[1] = (1 + t) * et;
        u_ptr[2] = et;

        PetscCall(VecRestoreArray(u, &u_ptr));
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscErrorCode petsc_sol_fun_tilde_var(TS ts, PetscReal t, Vec u, void* ctx)
    {
        PetscFunctionBeginUser;
        PetscCall(petsc_sol_fun(ts, t, u, ctx));
        PetscCall(u_to_tilde(u, u));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    struct AppCtx
    {
        bool      ts_implicit{false};
        bool      condition_system{false};
        bool      complex_fd{false};
        PetscReal shift{};        // The 'sigma' shift from TS
        PetscReal epsilon{1e-30}; // Finite difference step size
    };

    struct JacShellCtx
    {
        std::vector<numPDE::CplxType> u_plus_iv{};
        std::vector<numPDE::CplxType> complex_fun_eval{};
    };

    // Forward declaration, definition is after to have a more consistent approach
    template <bool NeedsChangeOfVariable = false>
    static PetscErrorCode comp_step_deriv(Mat J, Vec v, Vec w);
    template <bool NeedsChangeOfVariable = false>
    static PetscErrorCode pre_jv_eval(TS /*ts*/, PetscReal /*t*/, Vec U, Vec /*Udot*/,
                                      PetscReal shift, Mat /*J*/, Mat /*P*/, void* ctx_void);
    template <bool NeedsChangeOfVariable = false>
    static PetscErrorCode fun_implicit(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void* ctx);
    struct PETScData
    {
        TS       ts{};
        Vec      u_sol{};
        Vec      u_curr{};
        Vec      u_phys{};
        Mat      j_shell{};
        Mat      j_precond{};
        PetscInt prob_size{3}; // Problem size

        // Buffer to store the F(u + iv) in case the complex_fd option is active
        JacShellCtx j_shell_ctx{};
        // Ctx with all the options
        AppCtx ctx{};

        PETScData() = delete;
        PETScData(int* argc, char*** argv, const char* message) : ctx{}
        {
            PetscInitialize(argc, argv, NULL, message);
            // Parse Options
            PetscCallVoid(TSCreate(PETSC_COMM_WORLD, &ts));
            PetscCallVoid(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, prob_size, &this->u_sol));
            PetscCallVoid(
                PetscOptionsGetBool(NULL, NULL, "-condition_system", &ctx.condition_system, NULL));
            PetscCallVoid(PetscOptionsGetBool(NULL, NULL, "-complex_fd", &ctx.complex_fd, NULL));
            // INITIAL CONDITION
            PetscCallVoid(this->initialize_problem(ctx));

            std::println("Time int is implicit : {} ", ctx.ts_implicit);
            std::println("Complex step deriv   : {} ", ctx.complex_fd);
            std::println("Change if variable   : {} ", ctx.condition_system);
        }

        PETScData(int* argc, char*** argv) : PETScData(argc, argv, help_message) {};
        ~PETScData()
        {
            VecDestroy(&this->u_sol);
            VecDestroy(&this->u_curr);
            if (ctx.condition_system) VecDestroy(&this->u_phys);
            TSDestroy(&ts);
            PetscFinalize();
        };
        PetscErrorCode solve(bool write_to_cli = true)
        {

            PetscFunctionBeginUser;
            PetscCall(TSSolve(this->ts, this->u_sol));

            if (ctx.condition_system) PetscCall(numPDE::tilde_to_u(this->u_sol, this->u_sol));

            if (write_to_cli)
            {

                PetscReal final_time{};
                PetscCall(TSGetTime(ts, &final_time));

                if (is_rank_0()) std::println("Final Solution at t = {:}:", final_time);

                PetscCall(VecView(this->u_sol, PETSC_VIEWER_STDOUT_WORLD));
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        };

      private:
        PetscErrorCode set_initial_conditions()
        {
            PetscFunctionBeginUser;
            PetscFunctionReturn(PETSC_SUCCESS);
        };

        PetscErrorCode initialize_problem(AppCtx& ctx)
        {
            PetscFunctionBeginUser;
            // I lie, the problem is linear in reality
            PetscCall(TSSetProblemType(this->ts, TS_NONLINEAR));

            // Set default time stepping parameters
            PetscCall(TSSetTimeStep(this->ts, 0.01));
            PetscCall(TSSetMaxTime(this->ts, 0.1));
            PetscCall(TSSetExactFinalTime(this->ts, TS_EXACTFINALTIME_MATCHSTEP));
            PetscCall(TSSetFromOptions(this->ts));

            // Set the exact solution and initial condition
            PetscCall(VecSet(this->u_sol, 1.0));
            PetscCall(TSSetSolutionFunction(this->ts, petsc_sol_fun, &ctx));

            PetscCall(this->check_ts_implicit());

            // Approach the problem as F(U_dot, U, t) = 0
            if (ctx.ts_implicit)
            {
                /*The residual vector, I let petsc handle it with NULL*/
                PetscCall(TSSetIFunction(this->ts, NULL, fun_implicit<false>, this));
                // Tell the internal DMShell what global vector looks like
                DM dm;
                PetscCall(TSGetDM(this->ts, &dm));
                PetscCall(DMShellSetGlobalVector(dm, this->u_sol));

                PetscCall(VecDuplicate(u_sol, &this->u_curr));

                // Handle JFunc for both cases
                if (ctx.complex_fd) PetscCall(this->handle_Jv_eval(ctx));

                // Handle IFunc for the condition variable case
                if (ctx.condition_system) PetscCall(this->handle_change_of_var(ctx));

                // if (ctx.complex_fd and ctx.condition_system) std::terminate();
            }
            // Approach the problem as U_dot = G(U, t)
            else
            {
                PetscCall(TSSetRHSFunction(this->ts, NULL, rhs_func, &ctx));

                if (ctx.complex_fd)
                    std::println("The complex_fd option is IGNORED due to explicit timestepping");

                if (ctx.condition_system)
                    std::println(
                        "The change of space option is IGNORED due to explicit timestepping");
            }

            PetscCall(TSSetSolution(this->ts, this->u_sol));
            // Called by default, but is good practice
            PetscCall(TSSetUp(ts));
            PetscFunctionReturn(PETSC_SUCCESS);
        };
        PetscErrorCode check_ts_implicit()
        {
            PetscFunctionBeginUser;

            // Check what the CLI options applied
            TSType ts_type;
            PetscCall(TSGetType(this->ts, &ts_type));
            std::string_view type_str(ts_type);

            // List of PETSc implicit/IMEX integrators
            if (type_str == TSBDF || type_str == TSTHETA || type_str == TSALPHA ||
                type_str == TSBEULER || type_str == TSCN || type_str == TSARKIMEX ||
                type_str == TSROSW)
            {
                ctx.ts_implicit = true;
            }
            else
            {
                // Defaults to explicit for TSRK, TSEULER, TSSSP, etc.
                ctx.ts_implicit = false;
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        PetscErrorCode handle_change_of_var(AppCtx& ctx)
        {
            PetscFunctionBeginUser;
            // Case in which the solver solves in primitive variables (u_tilde)
            PetscCall(TSSetTransientVariable(this->ts, primitive_to_conservative, &ctx));
            std::println("================================");
            std::println(" Condition system activated.");
            std::println(" The Newton solver is now working in tilde space");
            std::println("================================");

            PetscCall(u_to_tilde(this->u_sol, this->u_sol));
            PetscCall(VecDuplicate(u_sol, &u_phys));

            PetscCall(TSSetSolutionFunction(this->ts, petsc_sol_fun_tilde_var, &ctx));

            // Provide new IFunc, PETSc does not call the change of variable by himself
            PetscCall(TSSetIFunction(this->ts, NULL, fun_implicit<true>, this));

            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // Define the j_shell to perform the CSD
        PetscErrorCode handle_Jv_eval(AppCtx& ctx)
        {
            PetscFunctionBeginUser;
            std::println("================================");
            std::println(" Complex FD activated.");
            std::println(
                " The jacobian is now evaluated as Jv = sigma*v + Im( R(u + i*v) / norm(v))");
            std::println("================================");

            //  Explain to PETSc how the shell matrix needs
            //  to act to perform the Jv evaluation
            const auto glob_size = prob_size;
            const auto loc_size  = PETSC_DECIDE;
            this->j_shell_ctx.u_plus_iv.resize(prob_size);
            this->j_shell_ctx.complex_fun_eval.resize(prob_size);

            PetscCall(VecDuplicate(this->u_sol, &this->u_curr));

            PetscCall(MatCreateShell(PETSC_COMM_WORLD, loc_size, loc_size, glob_size, glob_size,
                                     this, &this->j_shell));

            if (!ctx.condition_system)
            {
                PetscCall(MatShellSetOperation(this->j_shell, MATOP_MULT,
                                               (void (*)(void)) comp_step_deriv<true>));
                PetscCall(TSSetIJacobian(this->ts, this->j_shell, NULL, pre_jv_eval<true>, this));
            }
            else
            {
                PetscCall(MatShellSetOperation(this->j_shell, MATOP_MULT,
                                               (void (*)(void)) comp_step_deriv<false>));
                PetscCall(TSSetIJacobian(this->ts, this->j_shell, NULL, pre_jv_eval<false>, this));
            }

            PetscFunctionReturn(PETSC_SUCCESS);
        }
    };

    // MatMult definition for MatShell: w = J(u) * v
    template <bool NeedsChangeOfVariable>
    PetscErrorCode comp_step_deriv(Mat J, Vec v, Vec w)
    {
        PetscFunctionBeginUser;

        PETScData* petsc_data = nullptr;
        PetscCall(MatShellGetContext(J, &petsc_data));

        VecReadSpan  v_view{v};
        VecReadSpan  u_view{petsc_data->u_curr};
        VecWriteSpan jv_res{w};

        std::span<numPDE::CplxType> u_pl_iv{petsc_data->j_shell_ctx.u_plus_iv};
        std::span<numPDE::CplxType> f_u_iv{petsc_data->j_shell_ctx.complex_fun_eval};

        const auto sigma      = petsc_data->ctx.shift;
        const auto h          = petsc_data->ctx.epsilon;
        const auto one_over_h = 1.0 / h;

        auto u_span  = u_view.span();
        auto v_span  = v_view.span();
        auto jv_span = jv_res.span();

        // --- STEP 1: Construct u + iv*h ---
        for (auto&& [u, v_i, cplx_out] : std::views::zip(u_span, v_span, u_pl_iv))
        {
            if constexpr (NeedsChangeOfVariable)
                cplx_out = std::complex{u, u * v_i * h};
            else
                cplx_out = std::complex{u, v_i * h};
        }

        // --- STEP 2: Evaluate the complex function ---
        my_rhs_impl<numPDE::CplxType, TimeInt::Explicit>(u_pl_iv, f_u_iv);

        // --- STEP 3: Compute the Jacobian-vector product ---
        for (auto&& [f_cplx, u, v_i, jv_out] : std::views::zip(f_u_iv, u_span, v_span, jv_span))
        {
            if constexpr (NeedsChangeOfVariable)
                jv_out = std::imag(f_cplx) * one_over_h + sigma * u * v_i;
            else
                jv_out = std::imag(f_cplx) * one_over_h + sigma * v_i;
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    // This function is called before every Newton iteration step.
    // Then the solver will perform the matmul(J,v) step
    template <bool NeedsChangeOfVariable>
    static PetscErrorCode pre_jv_eval(TS /*ts*/, PetscReal /*t*/, Vec U, Vec /*Udot*/,
                                      PetscReal shift, Mat /*J*/, Mat /*P*/, void* ctx_void)
    {
        PetscFunctionBeginUser;
        auto data = static_cast<PETScData*>(ctx_void);

        if constexpr (NeedsChangeOfVariable)
            PetscCall(tilde_to_u(U, data->u_curr));
        else if constexpr (!NeedsChangeOfVariable)
            PetscCall(VecCopy(U, data->u_curr));

        data->ctx.shift = shift;

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <bool NeedsChangeOfVariable>
    static PetscErrorCode fun_implicit(TS /*ts*/, PetscReal /*t*/, Vec U, Vec U_t, Vec F, void* ctx)
    {
        PetscFunctionBeginUser;
        auto data = static_cast<PETScData*>(ctx);
        if constexpr (NeedsChangeOfVariable)
        {
            PetscCall(tilde_to_u(U, data->u_curr)); // Convert U_tilde to U_phys
        }
        else
        {
            PetscCall(VecCopy(U, data->u_curr));
        }
        VecReadSpan  u_view{data->u_curr};
        VecWriteSpan f_view{F};

        my_rhs_impl<PetscReal, TimeInt::Implicit>(u_view.span(), f_view.span());

        /*
        VecReadSpan u_t_view{U_t};
        std::ranges::transform(u_t_view.span(), f_view.span(), f_view.span().begin(),
                              [](auto du_dt, auto fu){return du_dt + fu;});
        */

        PetscCall(VecAXPY(F, 1.0, U_t));
        PetscFunctionReturn(PETSC_SUCCESS);
    }

} // namespace numPDE
