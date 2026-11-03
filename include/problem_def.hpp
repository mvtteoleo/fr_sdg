#pragma once

#include "boundary_conditions.hpp"
#include "filter.hpp"
#include "gas_models.hpp"
#include "indexer.hpp"
#include "macro_and_libc.hpp"
#include "parser.hpp"
#include "riemann_solver.hpp"
#include "vec_dof_view.hpp"
#include <concepts>

// Simple bool to handle the transition and allow rollback.
bool is_ts_implicit(const TS ts)
{
    // Check what the CLI options applied
    TSType ts_type;
    PetscCall(TSGetType(ts, &ts_type));
    std::string_view type_str(ts_type);

    // List of PETSc implicit/IMEX integrators
    if (type_str == TSBDF || type_str == TSTHETA || type_str == TSALPHA || type_str == TSBEULER ||
        type_str == TSCN || type_str == TSARKIMEX || type_str == TSROSW)
        return true;
    else
        // Defaults to explicit for TSRK, TSEULER, TSSSP, etc.
        return false;
};

namespace numPDE
{

    // ====================================================================
    // 1. PRIMITIVE TO CONSERVATIVE (Arbitrary Dimensions)
    // ====================================================================
    template <real_or_complex_floating_point U, std::same_as<U>... Vels>
    constexpr auto prim_to_cons_elem(U tilde_rho, U tilde_int_en, Vels... vels)
    {
        // Compute density: rho = exp(tilde_rho)
        const auto rho = std::exp(tilde_rho);

        // Unroll sum of squares for kinetic energy via unary right fold expression: sum(v_i^2)
        const auto sum_v_sq = (U{0.0} + ... + (vels * vels));
        const auto kin_en   = U{0.5} * sum_v_sq;

        // Total Energy: rho * (exp(tilde_e) + e_kin)
        const auto rhoE = rho * (std::exp(tilde_int_en) + kin_en);

        // Returns tuple: {rho, rho*v_1, rho*v_2, ..., rho*v_D, rhoE}
        return std::make_tuple(rho, (rho * vels)..., rhoE);
    };

    // ====================================================================
    // 2. CONSERVATIVE TO PRIMITIVE (Arbitrary Dimensions)
    // ====================================================================
    template <real_or_complex_floating_point U, std::same_as<U>... RhoVels>
    constexpr auto cons_to_prim_elem(U rho, U rhoE, RhoVels... rhoVels)
    {
        const auto inv_rho = U{1.0} / rho;

        // Compute kinetic energy by unrolling velocity conversions: sum((rhoV_i / rho)^2)
        const auto sum_v_sq = (U{0.0} + ... + ((rhoVels * inv_rho) * (rhoVels * inv_rho)));
        const auto kin_en   = U{0.5} * sum_v_sq;

        // Internal energy: e_int = E - e_kin
        const auto int_en = (rhoE * inv_rho) - kin_en;

        // Returns tuple: {tilde_rho, v_1, v_2, ..., v_D, tilde_e}
        return std::make_tuple(std::log(rho), (rhoVels * inv_rho)..., std::log(int_en));
    };

    /*
     template <numPDE::real_or_complex_floating_point U>
    constexpr auto prim_to_cons_elem(U tilde_rho, U vel, U tilde_int_en)
    {
        const auto rho = std::exp(tilde_rho);
        return std::make_tuple(rho, rho * vel, rho * (std::exp(tilde_int_en) + 0.5 * vel * vel));
    }

    template <numPDE::real_or_complex_floating_point U>
    constexpr auto cons_to_prim_elem(U rho, U rhoU, U rhoE)
    {
        const auto vel    = rhoU / rho;
        const auto int_en = rhoE / rho - 0.5 * vel * vel;
        return std::make_tuple(std::log(rho), vel, std::log(int_en));
    }
    */

} // namespace numPDE

#include "include/problem_def_new.hpp"
