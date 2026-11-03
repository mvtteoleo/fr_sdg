#pragma once
// ============================================================
//  problem_def.hpp  (new version)
//
//  Problem<Traits> is now a thin orchestrator.
//  Its entire job is to wire PhysicsEvaluator to PetscSolver
//  in the right order, then expose solve() and compute_error().
//
//  Include graph (no more cycles):
//
//    gas_models.hpp
//    riemann_solver.hpp
//    boundary_conditions.hpp
//    parser.hpp
//    reference_element.hpp
//    cached_computations.hpp
//    filter.hpp
//    vec_dof_view.hpp
//    indexer.hpp
//         └──> physics_evaluator.hpp   (pure numerics, PETSc-free)
//                  └──> petsc_solver_impl.hpp  (PETSc wiring + strategies)
//                            └──> problem_def.hpp  (this file)
//
//  residual_functs.hpp is DELETED.  Its contents live as static
//  methods of PetscSolver<Traits> and ComplexStepJacobianStrategy<Traits>.
// ============================================================

#include "petsc_solver_impl.hpp"

namespace numPDE
{

    template <class Traits>
    class Problem
    {
      public:
        using T = typename Traits::T;

        // ── Two first-class sub-objects ───────────────────────────────
        PhysicsEvaluator<Traits> m_physics;
        PetscSolver<Traits>      m_solver;

        // ── Construction in four explicit steps ───────────────────────
        explicit Problem(const ConfigOptions<T>&                      opts,
                         const BoundaryConditions<T, Traits::n_dims>& bcs,
                         const typename Traits::GasModel&             gmod = {})
            : m_physics{opts, bcs, gmod}, m_solver{m_physics}
        {
        }

        // ── Public interface ──────────────────────────────────────────

        PetscErrorCode solve()
        {
            PetscFunctionBeginUser;
            if (is_rank_0()) std::println("Solving...");
            PetscCall(TSSolve(m_solver.ts, m_solver.solution));
            // Force one final post-step to print the error at t_final
            PetscFunctionReturn(PETSC_SUCCESS);
        }

        // Delegates to PhysicsEvaluator — the solution Vec is passed through,
        // no direct access to PetscSovrastructures required.
        decltype(auto) compute_error() const
        {
            PetscReal time;
            TSGetTime(m_solver.ts, &time);
            auto err = m_physics.compute_error(time, m_solver.solution, m_solver.dmda);
            return err;
        }
    };

} // namespace numPDE
