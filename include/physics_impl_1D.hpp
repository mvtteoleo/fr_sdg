#pragma once

#include "gas_models.hpp"
#include "macro_and_libc.hpp"
namespace numPDE
{

    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 1>::residual_kernel(const RV& x_view, WV& f_view,
                                                                T time)
    {
        PetscFunctionBeginUser;

        const auto& dminfos = m_dm_info;
        const auto  nds     = m_gll.get_n_nodes();
        const auto  ndofs   = get_n_dof_per_element();

        auto as_node = [&](auto span, auto idx) -> ComprFD::NodeState<U, 1>
        {
            return {.rho  = span[0 * nds + idx],
                    .rhoU = {span[1 * nds + idx]},
                    .rhoE = span[2 * nds + idx]};
        };

        PetscInt first = dminfos.start[0];
        PetscInt last  = dminfos.max[0];

        // ── Physical boundaries ───────────────────────────────────────────
        if (!is_periodic)
        {
            // Right boundary
            if (static_cast<PetscInt>(m_opts.get_n_elems(0)) == dminfos.max[0])
            {
                --last;
                const auto         elem = dminfos.max[0] - 1;
                const auto         sp   = x_view.as_span(elem, ndofs);
                const auto         res  = f_view.as_span(elem, ndofs);
                const auto         prev = x_view.as_span(elem - 1, ndofs);
                const NeighView<U> nv{
                    .prev = as_node(prev, nds - 1),
                    .next = m_bc.template get_right_bc<U>(time, as_node(sp, nds - 1))};
                PetscCall(local_steady_res<U>(sp, nv, res));
            }

            // Left boundary
            if (PetscInt{0} == dminfos.start[0])
            {
                ++first;
                const auto         elem = 0;
                const auto         sp   = x_view.as_span(elem, ndofs);
                const auto         res  = f_view.as_span(elem, ndofs);
                const auto         next = x_view.as_span(elem + 1, ndofs);
                const NeighView<U> nv{.prev = m_bc.template get_left_bc<U>(time, as_node(sp, 0)),
                                      .next = as_node(next, 0)};
                PetscCall(local_steady_res<U>(sp, nv, res));
            }
        }

        // ────────── Interior loop (periodic ghosts included) ──────────────
        for (const auto elem : std::views::iota(first, last))
        {
            const auto         sp   = x_view.as_span(elem, ndofs);
            const auto         res  = f_view.as_span(elem, ndofs);
            const auto         prev = x_view.as_span(elem - 1, ndofs);
            const auto         next = x_view.as_span(elem + 1, ndofs);
            const NeighView<U> nv{.prev = as_node(prev, nds - 1), .next = as_node(next, 0)};
            PetscCall(local_steady_res<U>(sp, nv, res));

            // Add the radial source term
            if constexpr (Traits::coord_sys == CoordinateSystem::Radial)
            {
                for (const auto node : std::views::iota(0uz, nds))
                {
                    const auto  r     = this->get_r(elem, node);
                    const auto  inv_r = (r > 1e-8) ? 1.0 / r : 0.0;
                    const auto& rho   = sp[0 * nds + node];
                    const auto& rhoU  = sp[1 * nds + node];
                    const auto& rhoE  = sp[2 * nds + node];

                    const auto vel = rhoU / rho;

                    const auto pres = this->m_gas_mod.get_pressure(ComprFD::ThermodynamicState<U>{
                        .density = rho, .energy = rhoE / rho - U(0.5) * vel * vel});

                    res[0 * nds + node] += inv_r * rhoU;
                    res[1 * nds + node] += inv_r * rhoU * vel;
                    res[2 * nds + node] += inv_r * vel * (rhoE + pres);
                }
            }
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 1>::limit_solution(const RV& sol, WV& fil)
    {
        PetscFunctionBeginUser;
        const auto ndofs = get_n_dof_per_element();
        const auto nds   = m_gll.get_n_nodes();

        const auto elem_range = m_dm_info.get_global_elem_range();

        for (const auto elem : elem_range)
        {
            const auto sp_prev = sol.as_span(elem - 1, ndofs);
            const auto sp_next = sol.as_span(elem + 1, ndofs);
            const auto sp      = sol.as_span(elem, ndofs);
            auto       fsp     = fil.as_span(elem, ndofs);

            for (const auto eq : std::views::iota(0, 3))
            {
                m_stab.apply_element_stabilization(
                    sp_prev.subspan(eq * nds, nds), sp.subspan(eq * nds, nds),
                    sp_next.subspan(eq * nds, nds), fsp.subspan(eq * nds, nds));
            }
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <class Traits>
    template <real_or_complex_floating_point U>
    PetscErrorCode PhysicsEvaluator<Traits, 1>::local_steady_res(std::span<const U>  elem_u,
                                                                 const NeighView<U>& nv,
                                                                 std::span<U>        res)
    {
        PetscFunctionBeginUser;
        const std::size_t nds     = m_gll.get_n_nodes();
        const auto&       D       = m_cached_data.get_diff_mat();
        const auto&       dL      = m_cached_data.get_dL();
        const auto&       dR      = m_cached_data.get_dR();
        const auto        inv_jac = this->compute_inv_jacobian();

        constexpr std::size_t n_eqs = 3;

        const auto rho_s  = elem_u.subspan(0 * nds, nds);
        const auto rhou_s = elem_u.subspan(1 * nds, nds);
        const auto rhoE_s = elem_u.subspan(2 * nds, nds);

        std::ranges::fill(res, U{});

        // Volume term: R[eq,i] += D[i,j] * F^eq_j
        for (const auto j : std::views::iota(0uz, nds))
        {
            const ComprFD::NodeState<U, 1> node{rho_s[j], {rhou_s[j]}, rhoE_s[j]};
            const auto                     F_j = ComprFD::euler_flux(node, m_gas_mod);
            for (const auto i : std::views::iota(0uz, nds))
            {
                const U D_ij = inv_jac * D(i, j);
                res[0 * nds + i] += D_ij * F_j[0];
                res[1 * nds + i] += D_ij * F_j[1];
                res[2 * nds + i] += D_ij * F_j[2];
            }
        }

        const ComprFD::NodeState<U, 1> s_L{rho_s[0], {rhou_s[0]}, rhoE_s[0]};
        const ComprFD::NodeState<U, 1> s_R{rho_s[nds - 1], {rhou_s[nds - 1]}, rhoE_s[nds - 1]};
        const auto                     F_disc_L = ComprFD::euler_flux(s_L, m_gas_mod);
        const auto                     F_disc_R = ComprFD::euler_flux(s_R, m_gas_mod);

        auto F_star_L = m_riemann_solver.compute_flux(
            std::array<U, n_eqs>{nv.prev.rho, nv.prev.rhoU[0], nv.prev.rhoE},
            std::array<U, n_eqs>{rho_s[0], rhou_s[0], rhoE_s[0]}, m_gas_mod);

        auto F_star_R = m_riemann_solver.compute_flux(
            std::array<U, n_eqs>{rho_s[nds - 1], rhou_s[nds - 1], rhoE_s[nds - 1]},
            std::array<U, n_eqs>{nv.next.rho, nv.next.rhoU[0], nv.next.rhoE}, m_gas_mod);

        // Correction: R[eq,i] += ΔL^eq * g'L_i + ΔR^eq * g'R_i
        for (const auto eq : std::views::iota(0uz, n_eqs))
        {
            const U delta_L = F_star_L[eq] - F_disc_L[eq];
            const U delta_R = F_star_R[eq] - F_disc_R[eq];
            for (const auto i : std::views::iota(0uz, nds))
                res[eq * nds + i] += inv_jac * (delta_L * dL(i) + delta_R * dR(i));
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 1>::prim_to_cons_kernel(const RV& prim, WV& cons)
    {
        // ... identical to the existing implementation, just moved here ...
        PetscFunctionBeginUser;
        const auto nds   = m_gll.get_n_nodes();
        const auto ndofs = get_n_dof_per_element();
        for (const auto elem : m_dm_info.get_global_elem_range())
        {
            const auto p_sp = prim.as_span(elem, ndofs);
            auto       c_sp = cons.as_span(elem, ndofs);
            for (const auto node : std::views::iota(0uz, nds))
            {
                const auto rho = p_sp[0 * nds + node];
                const auto vel = p_sp[1 * nds + node];
                const auto ie  = p_sp[2 * nds + node];
                std::tie(c_sp[0 * nds + node], c_sp[1 * nds + node], c_sp[2 * nds + node]) =
                    prim_to_cons_elem(rho, ie, vel);
            }
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <class Traits>

    PetscErrorCode PhysicsEvaluator<Traits, 1>::impose_ic(Vec solution, DM dmda)
    {
        // Identical to the existing impose_ic_on_sol_vec, but Vec and DM are params.
        PetscFunctionBeginUser;
        WriteDOFView x{dmda, solution, VecType::Local};

        const auto  elem_range = m_dm_info.get_global_elem_range();
        const auto  nds        = m_gll.get_n_nodes();
        const auto  h          = m_opts.get_box_length(0) / m_opts.get_n_elems(0);
        const auto& csi        = m_gll.get_csi();

        // Select functional space once via a lambda; avoids a branch per node
        auto to_storage_space = [&](const ComprFD::NodeState<T, 1>& cons)
        {
            if (m_opts.get_cond_sys())
            {
                const auto [rho, vel, ie] = cons_to_prim_elem(cons.rho, cons.rhoE, cons.rhoU[0]);
                return ComprFD::NodeState<T, 1>{rho, {vel}, ie};
            }
            return cons;
        };

        for (const auto elem : elem_range)
        {
            auto sp = x.as_span(elem, get_n_dof_per_element());
            for (const auto node : std::views::iota(0uz, nds))
            {
                const T    pos     = h * (elem + 0.5 * (1.0 + csi[node]));
                const auto v       = to_storage_space(m_bc.initial_cond(pos));
                sp[0 * nds + node] = v.rho;
                sp[1 * nds + node] = v.rhoU[0];
                sp[2 * nds + node] = v.rhoE;
            }
        }
        PetscFunctionReturn(PETSC_SUCCESS);
    }
    template <class Traits>

    std::array<typename Traits::T, Traits::n_dims + 2>
    PhysicsEvaluator<Traits, 1>::compute_error(T time_curr, Vec sol, DM dmda) const
    {
        const auto& da = dmda;
        PetscReal** x;
        DMDAVecGetArrayDOF(da, sol, (void*) &x);
        const auto& dminfos{this->m_dm_info};

        const auto elem_range = dminfos.get_global_elem_range();
        const int  nds        = m_gll.get_n_nodes();
        const auto node_range = std::views::iota(0, nds);

        const auto  h       = this->m_opts.get_box_length(0) / this->m_opts.get_n_elems(0);
        const auto  Jac     = 0.5 * h;
        const auto& csi_ref = this->m_gll.get_csi();
        const auto& w_ref   = this->m_gll.get_ws();

        constexpr auto               n_eqs = 2 + n_dims;
        std::array<PetscReal, n_eqs> errs{};
        std::array<PetscReal, n_eqs> glob_errs{};
        std::ranges::fill(errs, T{0.0});
        std::ranges::fill(glob_errs, T{0.0});

        auto get_node_value_cons = [&nds](auto arr_like, auto node)
        {
            return std::make_tuple(arr_like[0 * nds + node], arr_like[1 * nds + node],
                                   arr_like[2 * nds + node]);
        };

        [[maybe_unused]] auto get_node_value_pos_pres =
            [&get_node_value_cons, &nds](auto arr_like, auto node)
        {
            const auto [rho, vel, int_en] = get_node_value_cons(arr_like, node);
            return prim_to_cons_elem(rho, int_en, vel);
        };

        auto get_node_val = [&](auto&&... args)
        {
            if (this->m_opts.get_cond_sys())
            {
                return get_node_value_pos_pres(std::forward<decltype(args)>(args)...);
            }
            else
            {
                return get_node_value_cons(std::forward<decltype(args)>(args)...);
            }
        };

        for (const auto elem : elem_range)
        {
            auto elem_sp =
                std::span<PetscReal>(std::addressof(x[elem][0]), this->get_n_dof_per_element());
            for (const auto node : node_range)
            {
                // GLL nodes are defined on [-1,1] so they need to be mapped

                const PetscReal pos          = h * (elem + 0.5 * (1.0 + csi_ref[node]));
                const auto      node_vals    = m_bc.exact_sol(time_curr, pos);
                const auto [rho, rhoU, rhoE] = get_node_val(elem_sp, node);
                const auto dRho              = rho - node_vals.rho;
                const auto dRhoU             = rhoU - node_vals.rhoU[0];
                const auto dRhoE             = rhoE - node_vals.rhoE;

                errs[0] += static_cast<PetscReal>(w_ref[node] * Jac * dRho * dRho);
                errs[1] += static_cast<PetscReal>(w_ref[node] * Jac * dRhoU * dRhoU);
                errs[2] += static_cast<PetscReal>(w_ref[node] * Jac * dRhoE * dRhoE);
            }
        }

        DMDAVecRestoreArrayDOF(da, sol, (void*) &x);
        const int root = 0;
        MPI_Reduce(errs.data(), glob_errs.data(), errs.size(), MPIU_REAL, MPIU_SUM, root,
                   MPI_COMM_WORLD);

        std::ranges::transform(glob_errs, glob_errs.begin(), [](auto v) { return std::sqrt(v); });

        return glob_errs;
    }

} // namespace numPDE
