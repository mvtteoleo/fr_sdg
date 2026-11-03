#pragma once
#include "boundary_conditions.hpp"
#include "macro_and_libc.hpp"
#include "physics_evaluator.hpp"
#include "riemann_solver.hpp"
#include "vec_dof_view.hpp"
#include <algorithm>
#include <optional>
#include <ranges>

namespace numPDE
{
    template <real_or_complex_floating_point U>
    struct Indexer
    {
        std::span<U> m_region;
        std::size_t  n_eqs{4};
        std::size_t  x_nodes{};
        std::size_t  y_nodes{};

        explicit Indexer(std::span<U> span, std::integral auto x_nodes, std::integral auto y_nodes)
            : m_region(span), x_nodes{x_nodes}, y_nodes{y_nodes}
        {
            assert(m_region.size() == n_eqs * x_nodes * y_nodes);
        }

        const U& operator()(std::integral auto eq, std::integral auto n_x,
                            std::integral auto n_y) const
        {
            return m_region[(n_y * x_nodes + n_x) * n_eqs + eq];
        }
        U& operator()(std::integral auto eq, std::integral auto n_x, std::integral auto n_y)
        {
            return m_region[(n_y * x_nodes + n_x) * n_eqs + eq];
        }
        // ── X-AXIS VIEW (Varying n_x, fixed n_y and eq) ─────────────────────
        // Elements along an X-row are separated by exactly n_eqs (stride of 4)
        auto view_x(std::integral auto y_idx, std::integral auto eq)
        {
            auto         start_ptr = &m_region[(y_idx * x_nodes) * n_eqs + eq];
            std::span<U> sub_span(start_ptr, x_nodes * n_eqs);
            return sub_span | std::views::stride(n_eqs);
        }

        auto view_x(std::integral auto y_idx, std::integral auto eq) const
        {
            auto               start_ptr = &m_region[(y_idx * x_nodes) * n_eqs + eq];
            std::span<const U> sub_span(start_ptr, x_nodes * n_eqs);
            return sub_span | std::views::stride(n_eqs);
        }

        // ── Y-AXIS VIEW (Varying n_y, fixed n_x and eq) ─────────────────────
        // Elements along a Y-column are separated by an entire row: x_nodes * n_eqs
        auto view_y(std::integral auto x_idx, std::integral auto eq)
        {
            auto         start_ptr = &m_region[x_idx * n_eqs + eq];
            std::span<U> sub_span(start_ptr, m_region.size() - (x_idx * n_eqs + eq));
            return sub_span | std::views::stride(x_nodes * n_eqs);
        }

        auto view_y(std::integral auto x_idx, std::integral auto eq) const
        {
            auto               start_ptr = &m_region[x_idx * n_eqs + eq];
            std::span<const U> sub_span(start_ptr, m_region.size() - (x_idx * n_eqs + eq));
            return sub_span | std::views::stride(x_nodes * n_eqs);
        }

        auto extract_state(std::integral auto k, std::integral auto l) const
        {
            return std::array<U, 4>{(*this)(0, k, l), (*this)(1, k, l), (*this)(2, k, l),
                                    (*this)(3, k, l)};
        };
    };

    // Apply weno limiter
    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 2>::limit_solution(const RV& sol, WV& fil)
    {

        PetscFunctionBeginUser;
        const auto x_nds = this->m_gll_x.get_n_nodes();
        const auto y_nds = this->m_gll_y.get_n_nodes();
        const auto ndofs = this->get_n_dof_per_element();
        for (const auto j : this->m_dm_info.get_global_elem_range(1))
        {
            for (const auto i : this->m_dm_info.get_global_elem_range(0))
            {
                const Indexer<const U> u_L(sol.as_span(i - 1, j, ndofs), x_nds, y_nds);
                const Indexer<const U> u_C(sol.as_span(i, j, ndofs), x_nds, y_nds);
                const Indexer<const U> u_R(sol.as_span(i + 1, j, ndofs), x_nds, y_nds);
                const Indexer<const U> u_B(sol.as_span(i, j - 1, ndofs), x_nds, y_nds);
                const Indexer<const U> u_T(sol.as_span(i, j + 1, ndofs), x_nds, y_nds);

                auto       fil_span = fil.as_span(i, j, ndofs);
                Indexer<U> u_new(fil_span, x_nds, y_nds);

                std::vector<U> scratch(ndofs);
                Indexer<U>     scr(std::span{scratch}, x_nds, y_nds);

                for (std::size_t eq = 0; eq < 4; ++eq)
                    for (std::size_t l = 0; l < y_nds; ++l)
                    {
                        // Clean, automated compile-time range type deduction!
                        this->m_stab.x.apply_element_stabilization(
                            u_L.view_x(l, eq), u_C.view_x(l, eq), u_R.view_x(l, eq),
                            scr.view_x(l, eq));
                    }
                for (std::size_t eq = 0; eq < 4; ++eq)
                    for (std::size_t k = 0; k < x_nds; ++k)
                    {
                        // Clean, automated compile-time range type deduction!
                        this->m_stab.y.apply_element_stabilization(
                            u_B.view_y(k, eq), u_C.view_y(k, eq), u_T.view_y(k, eq),
                            u_new.view_y(k, eq));
                    }

                std::ranges::transform(scratch, fil_span, fil_span.begin(),
                                       [](auto a, auto b) { return 0.5 * (a + b); });
            }
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    };

    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 2>::prim_to_cons_kernel(const RV& prim, WV& cons)
    {
        PetscFunctionBeginUser;
        const auto x_nds = this->m_gll_x.get_n_nodes();
        const auto y_nds = this->m_gll_y.get_n_nodes();
        const auto ndofs = this->get_n_dof_per_element();
        for (const auto j : this->m_dm_info.get_global_elem_range(1))
            for (const auto i : this->m_dm_info.get_global_elem_range(0))
            {
                const auto prim_idxer =
                    Indexer<const PetscReal>{prim.as_span(i, j, ndofs), x_nds, y_nds};
                auto cons_idxer = Indexer<PetscReal>{cons.as_span(i, j, ndofs), x_nds, y_nds};
                for (const auto k : std::views::iota(0uz, x_nds))
                    for (const auto l : std::views::iota(0uz, y_nds))
                    {

                        const auto& rho_tilde = prim_idxer(0, k, l);
                        const auto& U_vel     = prim_idxer(1, k, l);
                        const auto& V_vel     = prim_idxer(2, k, l);
                        const auto& e_tilde   = prim_idxer(3, k, l);
                        std::tie(cons_idxer(0, k, l), cons_idxer(1, k, l), cons_idxer(2, k, l),
                                 cons_idxer(3, k, l)) =
                            prim_to_cons_elem(rho_tilde, e_tilde, U_vel, V_vel);
                    }
            }
        PetscFunctionReturn(PETSC_SUCCESS);
    };

    template <class Traits>
    PetscErrorCode PhysicsEvaluator<Traits, 2>::impose_ic(Vec solution, DM dmda) const
    {
        PetscFunctionBeginUser;
        const auto      x_nds = this->m_gll_x.get_n_nodes();
        const auto      y_nds = this->m_gll_y.get_n_nodes();
        const auto      ndofs = this->get_n_dof_per_element();
        const auto      h_x   = this->m_opts.get_box_length(0) / this->m_opts.get_n_elems(0);
        const auto      h_y   = this->m_opts.get_box_length(1) / this->m_opts.get_n_elems(1);
        const auto&     csi_x = this->m_gll_x.get_csi();
        const auto&     csi_y = this->m_gll_y.get_csi();
        WriteDOFView<2> x{dmda, solution, VecType::Local};

        for (const auto j : this->m_dm_info.get_global_elem_range(1))
            for (const auto i : this->m_dm_info.get_global_elem_range(0))
            {
                auto x_idxer = Indexer<PetscReal>{x.as_span(i, j, ndofs), x_nds, y_nds};
                for (const auto k : std::views::iota(0uz, x_nds))
                    for (const auto l : std::views::iota(0uz, y_nds))
                    {
                        const auto x_pos = h_x * (i + 0.5 * (1.0 + csi_x[k]));
                        const auto y_pos = h_y * (j + 0.5 * (1.0 + csi_y[l]));
                        const auto state = m_bc.initial_cond(x_pos, y_pos);

                        if (this->m_opts.get_cond_sys())
                        {
                            // Convert conservative initial state to primitive variable workspace
                            // storage
                            const auto [t_rho, u_vel, v_vel, t_ie] = cons_to_prim_elem(
                                state.rho, state.rhoE, state.rhoU[0], state.rhoU[1]);
                            x_idxer(0, k, l) = t_rho;
                            x_idxer(1, k, l) = u_vel;
                            x_idxer(2, k, l) = v_vel;
                            x_idxer(3, k, l) = t_ie;
                        }
                        else
                        {
                            x_idxer(0, k, l) = state.rho;
                            x_idxer(1, k, l) = state.rhoU[0];
                            x_idxer(2, k, l) = state.rhoU[1];
                            x_idxer(3, k, l) = state.rhoE;
                        }
                    }
            }
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    template <class Traits>
    std::array<typename Traits::T, Traits::n_dims + 2>
    PhysicsEvaluator<Traits, 2>::compute_error(const T time, const Vec solution,
                                               const DM dmda) const
    {
        std::array<PetscReal, 4> errs{};
        std::array<PetscReal, 4> glob_errs{};
        const auto               x_nds = this->m_gll_x.get_n_nodes();
        const auto               y_nds = this->m_gll_y.get_n_nodes();
        const auto               ndofs = this->get_n_dof_per_element();
        const auto&              wei_x = this->m_gll_x.get_ws();
        const auto&              wei_y = this->m_gll_y.get_ws();
        WriteDOFView<2>          x{dmda, solution, VecType::Local};

        for (const auto j : this->m_dm_info.get_global_elem_range(1))
            for (const auto i : this->m_dm_info.get_global_elem_range(0))
            {
                auto x_idxer = Indexer<PetscReal>{x.as_span(i, j, ndofs), x_nds, y_nds};
                for (const auto k : std::views::iota(0uz, x_nds))
                    for (const auto l : std::views::iota(0uz, y_nds))
                    {
                        const auto [x_pos, y_pos] = get_x_y_pos(i, j, k, l);
                        const auto metrics        = get_metrics(i, j, k, l);
                        const auto state          = m_bc.exact_sol(time, x_pos, y_pos);

                        PetscReal rho, rhoU, rhoV, rhoE;
                        if (this->m_opts.get_cond_sys())
                        {
                            // Unpack primitives from workspace storage and convert back to
                            // conservative variables
                            auto cons_tuple = prim_to_cons_elem(x_idxer(0, k, l), x_idxer(3, k, l),
                                                                x_idxer(1, k, l), x_idxer(2, k, l));
                            rho             = std::get<0>(cons_tuple);
                            rhoU            = std::get<1>(cons_tuple);
                            rhoV            = std::get<2>(cons_tuple);
                            rhoE            = std::get<3>(cons_tuple);
                        }
                        else
                        {
                            rho  = x_idxer(0, k, l);
                            rhoU = x_idxer(1, k, l);
                            rhoV = x_idxer(2, k, l);
                            rhoE = x_idxer(3, k, l);
                        }

                        const auto dRho  = rho - state.rho;
                        const auto dRhoU = rhoU - state.rhoU[0];
                        const auto dRhoV = rhoV - state.rhoU[1];
                        const auto dRhoE = rhoE - state.rhoE;

                        const auto constant_value = wei_x[k] * wei_y[l] * metrics.detJ;

                        errs[0] += static_cast<PetscReal>(constant_value * dRho * dRho);
                        errs[1] += static_cast<PetscReal>(constant_value * dRhoU * dRhoU);
                        errs[2] += static_cast<PetscReal>(constant_value * dRhoV * dRhoV);
                        errs[3] += static_cast<PetscReal>(constant_value * dRhoE * dRhoE);
                    }
            }
        MPI_Reduce(errs.data(), glob_errs.data(), errs.size(), MPIU_REAL, MPIU_SUM, 0,
                   MPI_COMM_WORLD);
        std::ranges::transform(glob_errs, glob_errs.begin(), [](auto v) { return std::sqrt(v); });
        return glob_errs;
    }

    template <class Traits>
    template <real_or_complex_floating_point U>
    void PhysicsEvaluator<Traits, 2>::compute_volume_term(const std::span<const U> u_span,
                                                          std::span<U>             res_span,
                                                          std::integral auto       i,
                                                          std::integral auto       j) const
    {
        constexpr auto n_eqs = this->get_n_eqs();
        const auto     x_nds = this->m_gll_x.get_n_nodes();
        const auto     y_nds = this->m_gll_y.get_n_nodes();
        const auto     ndofs = this->get_n_dof_per_element();
        const auto&    D_x   = this->m_cached_data_x.get_diff_mat();
        const auto&    D_y   = this->m_cached_data_y.get_diff_mat();
        std::ranges::fill(res_span, U{0.0});

        const Indexer<const U> u_C(u_span, x_nds, y_nds);
        Indexer<U>             res_idx(res_span, x_nds, y_nds);

        std::vector<U> F_buffer(ndofs, U{0.0});
        std::vector<U> G_buffer(ndofs, U{0.0});
        Indexer<U>     F(F_buffer, x_nds, y_nds);
        Indexer<U>     G(G_buffer, x_nds, y_nds);

        for (std::size_t l = 0; l < y_nds; ++l)
            for (std::size_t k = 0; k < x_nds; ++k)
            {
                ComprFD::NodeState<U, 2> state{.rho  = u_C(0, k, l),
                                               .rhoU = {u_C(1, k, l), u_C(2, k, l)},
                                               .rhoE = u_C(3, k, l)};

                // const auto [flux_x, flux_y] = ComprFD::euler_fluxes_2d(state, this->m_gas_mod);
                const auto [flux_x, flux_y, pres] =
                    ComprFD::euler_fluxes_2d_w_p(state, this->m_gas_mod);
                for (std::size_t eq = 0; eq < n_eqs; ++eq)
                {
                    F(eq, k, l) = flux_x[eq];
                    G(eq, k, l) = flux_y[eq];
                }

                // Special treatment for the radial coordinates
                if constexpr (Traits::coord_sys == CoordinateSystem::Radial)
                {
                    const auto [x_pos, y_pos] = get_x_y_pos(i, j, k, l);
                    const auto radius         = 0.5 * y_pos;
                    const auto inv_r          = (radius > 1e-8) ? 1.0 / radius : 0.0;
                    for (auto eq{0uz}; eq < get_n_eqs(); ++eq)
                    {
                        res_idx(eq, k, l) += G(eq, k, l) * inv_r;
                    }

                    constexpr auto rad_dim = 2uz;
                    res_idx(rad_dim, k, l) -= pres * inv_r;
                }
            }

        // (Curvilinear Chain-Rule Interior Fluxes)
        for (std::size_t l = 0; l < y_nds; ++l)
            for (std::size_t k = 0; k < x_nds; ++k)
            {

                const auto metrics = this->get_metrics(i, j, k, l);
                for (std::size_t eq = 0; eq < n_eqs; ++eq)
                {
                    U dF_dxi{0.0};
                    U dG_dxi{0.0};
                    for (std::size_t m = 0; m < x_nds; ++m)
                    {
                        dG_dxi += D_x(k, m) * G(eq, m, l);
                        dF_dxi += D_x(k, m) * F(eq, m, l);
                    }

                    U dG_deta{0.0};
                    U dF_deta{0.0};
                    for (std::size_t m = 0; m < y_nds; ++m)
                    {
                        dG_deta += D_y(l, m) * G(eq, k, m);
                        dF_deta += D_y(l, m) * F(eq, k, m);
                    }

                    // PURE CONTRAVARIANT PROJECTION
                    // The metrics (xi_x, eta_x) natively contain the volume scaling (2.0/h)
                    // and the spatial curvature transformation. No outer factors needed!
                    const U phys_deriv_x = metrics.xi_x * dF_dxi + metrics.eta_x * dF_deta;
                    const U phys_deriv_y = metrics.xi_y * dG_dxi + metrics.eta_y * dG_deta;

                    res_idx(eq, k, l) += phys_deriv_x + phys_deriv_y;
                }
            }

        return;
    }

#define OLD_VERSION false
#if OLD_VERSION
    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 2>::residual_kernel(const RV& x_view, WV& f_view,
                                                                T time) const
    {
        PetscFunctionBeginUser;

        const auto x_nds = this->m_gll_x.get_n_nodes();
        const auto y_nds = this->m_gll_y.get_n_nodes();
        const auto ndofs = this->get_n_dof_per_element();

        const auto& D_x  = this->m_cached_data_x.get_diff_mat();
        const auto& D_y  = this->m_cached_data_y.get_diff_mat();
        const auto& dL_x = this->m_cached_data_x.get_dL();
        const auto& dR_x = this->m_cached_data_x.get_dR();
        const auto& dL_y = this->m_cached_data_y.get_dL();
        const auto& dR_y = this->m_cached_data_y.get_dR();

        const auto            inv_jac_x = this->compute_inv_jacobian(0uz);
        const auto            inv_jac_y = this->compute_inv_jacobian(1uz);
        constexpr std::size_t n_eqs     = 4;

        // Local MPI boundaries
        const auto& i_start = this->m_dm_info.start[0];
        const auto& i_max   = this->m_dm_info.max[0];
        const auto& j_start = this->m_dm_info.start[1];
        const auto& j_max   = this->m_dm_info.max[1];

        const auto& global_max_i = this->m_opts.get_n_elems(0uz);
        const auto& global_max_j = this->m_opts.get_n_elems(1uz);

        // ====================================================================
        // 1. VOLUME INTEGRATION (Interior Fluxes and Derivatives)
        // ====================================================================
        for (const auto j : std::views::iota(j_start, j_max))
        {
            for (const auto i : std::views::iota(i_start, i_max))
            {
                const auto u_span   = x_view.as_span(i, j, ndofs);
                auto       res_span = f_view.as_span(i, j, ndofs);
                std::ranges::fill(res_span, U{0.0});

                Indexer<const U> u_C(u_span, x_nds, y_nds);
                Indexer<U>       res_idx(res_span, x_nds, y_nds);

                std::vector<U> F_buffer(ndofs, U{0.0});
                std::vector<U> G_buffer(ndofs, U{0.0});
                Indexer<U>     F(F_buffer, x_nds, y_nds);
                Indexer<U>     G(G_buffer, x_nds, y_nds);

                for (std::size_t l = 0; l < y_nds; ++l)
                {
                    for (std::size_t k = 0; k < x_nds; ++k)
                    {
                        ComprFD::NodeState<U, 2> state{.rho  = u_C(0, k, l),
                                                       .rhoU = {u_C(1, k, l), u_C(2, k, l)},
                                                       .rhoE = u_C(3, k, l)};
                        const auto [flux_x, flux_y] =
                            ComprFD::euler_fluxes_2d(state, this->m_gas_mod);
                        for (std::size_t eq = 0; eq < n_eqs; ++eq)
                        {
                            F(eq, k, l) = flux_x[eq];
                            G(eq, k, l) = flux_y[eq];
                        }
                    }
                }

                for (std::size_t l = 0; l < y_nds; ++l)
                {
                    for (std::size_t k = 0; k < x_nds; ++k)
                    {
                        for (std::size_t eq = 0; eq < n_eqs; ++eq)
                        {
                            U deriv_x{0.0};
                            for (std::size_t m = 0; m < x_nds; ++m)
                                deriv_x += D_x(k, m) * F(eq, m, l);
                            U deriv_y{0.0};
                            for (std::size_t m = 0; m < y_nds; ++m)
                                deriv_y += D_y(l, m) * G(eq, k, m);

                            res_idx(eq, k, l) += (inv_jac_x * deriv_x) + (inv_jac_y * deriv_y);
                        }
                    }
                }
            }
        }

        // ====================================================================
        // 2. X-INTERFACE COUPLING
        // ====================================================================
        auto eval_x_edge =
            [&]<bool UpdateL, bool UpdateR, bool GlobalLeftBC = false, bool GlobalRightBC = false>(
                std::integral auto I, std::integral auto j)
        {
            Indexer<const U> u_L(x_view.as_span(I - 1, j, ndofs), x_nds, y_nds);
            Indexer<const U> u_R(x_view.as_span(I, j, ndofs), x_nds, y_nds);

            for (std::size_t l = 0; l < y_nds; ++l)
            {
                std::array<U, 4> UL, UR;

                // Load internal data assuming standard exchange
                if constexpr (!GlobalRightBC)
                    UR = {u_R(0, 0, l), u_R(1, 0, l), u_R(2, 0, l), u_R(3, 0, l)};
                if constexpr (!GlobalLeftBC)
                    UL = {u_L(0, x_nds - 1, l), u_L(1, x_nds - 1, l), u_L(2, x_nds - 1, l),
                          u_L(3, x_nds - 1, l)};

                // ── OVERRIDE EXTERIOR GHOSTS WITH PHYSICAL BOUNDARY CONDITIONS ──
                if constexpr (GlobalLeftBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UR[0], .rhoU = {UR[1], UR[2]}, .rhoE = UR[3]};
                    const auto [x_pos, y_pos] = this->get_x_y_pos(I, j, 0uz, l);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Left, time, x_pos,
                                                           y_pos, interior);
                    // numPDE::println("Working eval_bc LEFT");
                    UL = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                if constexpr (GlobalRightBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UL[0], .rhoU = {UL[1], UL[2]}, .rhoE = UL[3]};
                    const auto [x_pos, y_pos] =
                        this->get_x_y_pos(I, j, this->m_gll_x.get_poly_deg(), l);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Right, time, x_pos,
                                                           y_pos, interior);
                    // numPDE::println("Working eval_bc RIGHT");
                    UR = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                ComprFD::NodeState<U, 2> s_L{.rho = UL[0], .rhoU = {UL[1], UL[2]}, .rhoE = UL[3]};
                ComprFD::NodeState<U, 2> s_R{.rho = UR[0], .rhoU = {UR[1], UR[2]}, .rhoE = UR[3]};

                const auto [flux_x_L, _1] = ComprFD::euler_fluxes_2d(s_L, this->m_gas_mod);
                const auto [flux_x_R, _2] = ComprFD::euler_fluxes_2d(s_R, this->m_gas_mod);

                const auto F_star =
                    this->m_riemann_solver.compute_flux(UL, UR, this->m_gas_mod, Axis::X);

                for (std::size_t eq = 0; eq < n_eqs; ++eq)
                {
                    if constexpr (UpdateL)
                    {
                        Indexer<U> res_L(f_view.as_span(I - 1, j, ndofs), x_nds, y_nds);
                        for (std::size_t k = 0; k < x_nds; ++k)
                            res_L(eq, k, l) += inv_jac_x * (F_star[eq] - flux_x_L[eq]) * dR_x(k);
                    }
                    if constexpr (UpdateR)
                    {
                        Indexer<U> res_R(f_view.as_span(I, j, ndofs), x_nds, y_nds);
                        for (std::size_t k = 0; k < x_nds; ++k)
                            res_R(eq, k, l) += inv_jac_x * (F_star[eq] - flux_x_R[eq]) * dL_x(k);
                    }
                }
            }
        };

        {
            // Left Face (I = i_start)
            const bool is_physical_left = (i_start == 0) && !this->is_periodic_x;
            // Right Face (I = i_max)
            const bool is_physical_right =
                (static_cast<std::size_t>(i_max) == global_max_i) && !this->is_periodic_x;
            for (const auto j : std::views::iota(j_start, j_max))
            {
                if (is_physical_left)
                    eval_x_edge.template operator()<false, true, true, false>(i_start, j);
                else
                    eval_x_edge.template operator()<false, true, false, false>(i_start, j);

                // Internal Interfaces
                for (const auto I : std::views::iota(i_start + 1, i_max))
                    eval_x_edge.template operator()<true, true, false, false>(I, j);

                if (is_physical_right)
                    eval_x_edge.template operator()<true, false, false, true>(i_max, j);
                else
                    eval_x_edge.template operator()<true, false, false, false>(i_max, j);
            }
        }

        // ====================================================================
        // 3. Y-INTERFACE COUPLING
        // ====================================================================
        auto eval_y_edge =
            [&]<bool UpdateB, bool UpdateT, bool GlobalBottomBC = false, bool GlobalTopBC = false>(
                std::integral auto i, std::integral auto J)
        {
            Indexer<const U> u_B(x_view.as_span(i, J - 1, ndofs), x_nds, y_nds);
            Indexer<const U> u_T(x_view.as_span(i, J, ndofs), x_nds, y_nds);

            for (std::size_t k = 0; k < x_nds; ++k)
            {
                std::array<U, 4> UB, UT;

                if constexpr (!GlobalTopBC)
                    UT = {u_T(0, k, 0), u_T(1, k, 0), u_T(2, k, 0), u_T(3, k, 0)};
                if constexpr (!GlobalBottomBC)
                    UB = {u_B(0, k, y_nds - 1), u_B(1, k, y_nds - 1), u_B(2, k, y_nds - 1),
                          u_B(3, k, y_nds - 1)};

                // ── OVERRIDE EXTERIOR GHOSTS WITH PHYSICAL BOUNDARY CONDITIONS ──
                if constexpr (GlobalBottomBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UT[0], .rhoU = {UT[1], UT[2]}, .rhoE = UT[3]};
                    const auto [x_pos, y_pos] = this->get_x_y_pos(i, J, k, 0uz);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Bottom, time, x_pos,
                                                           y_pos, interior);
                    // numPDE::println("Working eval_bc BOTTOM");
                    UB = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                if constexpr (GlobalTopBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UB[0], .rhoU = {UB[1], UB[2]}, .rhoE = UB[3]};
                    const auto [x_pos, y_pos] =
                        this->get_x_y_pos(i, J, k, this->m_gll_y.get_poly_deg());
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Top, time, x_pos,
                                                           y_pos, interior);
                    // numPDE::println("Working eval_bc");
                    UT = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                ComprFD::NodeState<U, 2> s_B{.rho = UB[0], .rhoU = {UB[1], UB[2]}, .rhoE = UB[3]};
                ComprFD::NodeState<U, 2> s_T{.rho = UT[0], .rhoU = {UT[1], UT[2]}, .rhoE = UT[3]};

                const auto [_1, flux_y_B] = ComprFD::euler_fluxes_2d(s_B, this->m_gas_mod);
                const auto [_2, flux_y_T] = ComprFD::euler_fluxes_2d(s_T, this->m_gas_mod);

                const auto G_star =
                    this->m_riemann_solver.compute_flux(UB, UT, this->m_gas_mod, Axis::Y);

                for (std::size_t eq = 0; eq < n_eqs; ++eq)
                {
                    if constexpr (UpdateB)
                    {
                        Indexer<U> res_B(f_view.as_span(i, J - 1, ndofs), x_nds, y_nds);
                        for (std::size_t l = 0; l < y_nds; ++l)
                            res_B(eq, k, l) += inv_jac_y * (G_star[eq] - flux_y_B[eq]) * dR_y(l);
                    }
                    if constexpr (UpdateT)
                    {
                        Indexer<U> res_T(f_view.as_span(i, J, ndofs), x_nds, y_nds);
                        for (std::size_t l = 0; l < y_nds; ++l)
                            res_T(eq, k, l) += inv_jac_y * (G_star[eq] - flux_y_T[eq]) * dL_y(l);
                    }
                }
            }
        };

        {

            // Bottom Face (J = j_start)
            const bool is_physical_bot = (j_start == 0) && !this->is_periodic_y;
            // Top Face (J = j_max)
            const bool is_physical_top =
                (static_cast<std::size_t>(j_max) == global_max_j) && !this->is_periodic_y;
            for (const auto i : std::views::iota(i_start, i_max))
            {
                if (is_physical_bot)
                    eval_y_edge.template operator()<false, true, true, false>(i, j_start);
                else
                    eval_y_edge.template operator()<false, true, false, false>(i, j_start);

                // Internal Interfaces
                for (const auto J : std::views::iota(1 + j_start, j_max))
                    eval_y_edge.template operator()<true, true, false, false>(i, J);

                if (is_physical_top)
                    eval_y_edge.template operator()<true, false, false, true>(i, j_max);
                else
                    eval_y_edge.template operator()<true, false, false, false>(i, j_max);
            }
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }

#else
    template <class Traits>
    template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
    PetscErrorCode PhysicsEvaluator<Traits, 2>::residual_kernel(const RV& x_view, WV& f_view,
                                                                T time) const
    {
        PetscFunctionBeginUser;

        const auto x_nds = this->m_gll_x.get_n_nodes();
        const auto y_nds = this->m_gll_y.get_n_nodes();
        const auto ndofs = this->get_n_dof_per_element();

        const auto& dL_x = this->m_cached_data_x.get_dL();
        const auto& dR_x = this->m_cached_data_x.get_dR();
        const auto& dL_y = this->m_cached_data_y.get_dL();
        const auto& dR_y = this->m_cached_data_y.get_dR();

        constexpr std::size_t n_eqs = 4;

        // Local partition bounding constraints
        const auto& i_start = this->m_dm_info.start[0];
        const auto& i_max   = this->m_dm_info.max[0];
        const auto& j_start = this->m_dm_info.start[1];
        const auto& j_max   = this->m_dm_info.max[1];

        const auto& global_max_i = this->m_opts.get_n_elems(0uz);
        const auto& global_max_j = this->m_opts.get_n_elems(1uz);

        // Normalized mapping increments and GLL node distributions

        // ====================================================================
        // VOLUME INTEGRATION
        // ====================================================================
        for (const auto j : std::views::iota(j_start, j_max))
        {
            for (const auto i : std::views::iota(i_start, i_max))
            {
                const auto u_span   = x_view.as_span(i, j, ndofs);
                auto       res_span = f_view.as_span(i, j, ndofs);
                this->compute_volume_term<U>(u_span, res_span, i, j);
            }
        }

        // ====================================================================
        // X-INTERFACE COUPLING (Vertical Edges)
        // ====================================================================
        auto eval_x_edge =
            [&]<bool UpdateL, bool UpdateR, bool GlobalLeftBC = false, bool GlobalRightBC = false>(
                std::integral auto I, std::integral auto j)
        {
            Indexer<const U> u_L(x_view.as_span(I - 1, j, ndofs), x_nds, y_nds);
            Indexer<const U> u_R(x_view.as_span(I, j, ndofs), x_nds, y_nds);

            // Interface X coordinate is fixed cleanly on the element boundary
            const T comp_x = get_comp_coord_x(I, 0uz);

            for (std::size_t l = 0; l < y_nds; ++l)
            {
                // Trace precise interface location along the Y distribution
                const T comp_y = get_comp_coord_y(j, l);

                const auto face_geom = this->m_geometry.get_x_face_metrics(comp_x, comp_y);
                const auto metrics   = this->get_metrics(comp_x, comp_y);

                const T nx = face_geom.nx;
                const T ny = face_geom.ny;

                // Contravariant gradient magnitude (||∇ξ||)
                const T scale_x = metrics.norm_grad_xi();

                std::array<U, 4> UL, UR;

                if constexpr (!GlobalRightBC)
                    UR = {u_R(0, 0, l), u_R(1, 0, l), u_R(2, 0, l), u_R(3, 0, l)};
                if constexpr (!GlobalLeftBC)
                    UL = {u_L(0, x_nds - 1, l), u_L(1, x_nds - 1, l), u_L(2, x_nds - 1, l),
                          u_L(3, x_nds - 1, l)};

                if constexpr (GlobalLeftBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UR[0], .rhoU = {UR[1], UR[2]}, .rhoE = UR[3]};
                    const auto [x_pos, y_pos] =
                        this->m_geometry.get_physical_coords(comp_x, comp_y);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Left, time, x_pos,
                                                           y_pos, interior);
                    UL = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                if constexpr (GlobalRightBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UL[0], .rhoU = {UL[1], UL[2]}, .rhoE = UL[3]};
                    const auto [x_pos, y_pos] =
                        this->m_geometry.get_physical_coords(comp_x, comp_y);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Right, time, x_pos,
                                                           y_pos, interior);
                    UR = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                ComprFD::NodeState<U, 2> s_L{.rho = UL[0], .rhoU = {UL[1], UL[2]}, .rhoE = UL[3]};
                ComprFD::NodeState<U, 2> s_R{.rho = UR[0], .rhoU = {UR[1], UR[2]}, .rhoE = UR[3]};

                const auto [flux_x_L, flux_y_L] = ComprFD::euler_fluxes_2d(s_L, this->m_gas_mod);
                const auto [flux_x_R, flux_y_R] = ComprFD::euler_fluxes_2d(s_R, this->m_gas_mod);

                const auto F_star =
                    this->m_riemann_solver.compute_flux(UL, UR, this->m_gas_mod, nx, ny);

                for (std::size_t eq = 0; eq < n_eqs; ++eq)
                {
                    if constexpr (UpdateL)
                    {
                        Indexer<U> res_L(f_view.as_span(I - 1, j, ndofs), x_nds, y_nds);
                        const U    normal_flux_L = flux_x_L[eq] * nx + flux_y_L[eq] * ny;
                        for (std::size_t k = 0; k < x_nds; ++k)
                            res_L(eq, k, l) += scale_x * (F_star[eq] - normal_flux_L) * dR_x(k);
                    }
                    if constexpr (UpdateR)
                    {
                        Indexer<U> res_R(f_view.as_span(I, j, ndofs), x_nds, y_nds);
                        const U    normal_flux_R = flux_x_R[eq] * nx + flux_y_R[eq] * ny;
                        for (std::size_t k = 0; k < x_nds; ++k)
                            res_R(eq, k, l) += scale_x * (F_star[eq] - normal_flux_R) * dL_x(k);
                    }
                }
            }
        };

        {
            const bool is_physical_left = (i_start == 0) && !this->is_periodic_x;
            const bool is_physical_right =
                (static_cast<std::size_t>(i_max) == global_max_i) && !this->is_periodic_x;

            for (const auto j : std::views::iota(j_start, j_max))
            {
                if (is_physical_left)
                    eval_x_edge.template operator()<false, true, true, false>(i_start, j);
                else
                    eval_x_edge.template operator()<false, true, false, false>(i_start, j);

                for (const auto I : std::views::iota(i_start + 1, i_max))
                    eval_x_edge.template operator()<true, true, false, false>(I, j);

                if (is_physical_right)
                    eval_x_edge.template operator()<true, false, false, true>(i_max, j);
                else
                    eval_x_edge.template operator()<true, false, false, false>(i_max, j);
            }
        }

        // ====================================================================
        // 3. Y-INTERFACE COUPLING (Horizontal Tilting Edges)
        // ====================================================================
        auto eval_y_edge =
            [&]<bool UpdateB, bool UpdateT, bool GlobalBottomBC = false, bool GlobalTopBC = false>(
                std::integral auto i, std::integral auto J)
        {
            Indexer<const U> u_B(x_view.as_span(i, J - 1, ndofs), x_nds, y_nds);
            Indexer<const U> u_T(x_view.as_span(i, J, ndofs), x_nds, y_nds);

            // Interface Y coordinate is fixed cleanly on the element boundary
            const T comp_y = get_comp_coord_y(J, 0uz);

            for (std::size_t k = 0; k < x_nds; ++k)
            {
                // Trace precise interface location along the X distribution
                const T comp_x = get_comp_coord_x(i, k);

                const auto face_geom = this->m_geometry.get_y_face_metrics(comp_x, comp_y);
                const auto metrics   = this->get_metrics(comp_x, comp_y);

                const T nx = face_geom.nx;
                const T ny = face_geom.ny;

                // Contravariant gradient magnitude (||∇η||)
                const T scale_y = metrics.norm_grad_eta();

                std::array<U, 4> UB, UT;
                if constexpr (!GlobalTopBC)
                    UT = {u_T(0, k, 0), u_T(1, k, 0), u_T(2, k, 0), u_T(3, k, 0)};
                if constexpr (!GlobalBottomBC)
                    UB = {u_B(0, k, y_nds - 1), u_B(1, k, y_nds - 1), u_B(2, k, y_nds - 1),
                          u_B(3, k, y_nds - 1)};

                if constexpr (GlobalBottomBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UT[0], .rhoU = {UT[1], UT[2]}, .rhoE = UT[3]};
                    const auto [x_pos, y_pos] =
                        this->m_geometry.get_physical_coords(comp_x, comp_y);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Bottom, time, x_pos,
                                                           y_pos, interior);
                    UB = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                if constexpr (GlobalTopBC)
                {
                    ComprFD::NodeState<U, 2> interior{
                        .rho = UB[0], .rhoU = {UB[1], UB[2]}, .rhoE = UB[3]};
                    const auto [x_pos, y_pos] =
                        this->m_geometry.get_physical_coords(comp_x, comp_y);
                    auto exterior = this->m_bc.evaluate_bc(numPDE::Direction::Top, time, x_pos,
                                                           y_pos, interior);
                    UT = {exterior.rho, exterior.rhoU[0], exterior.rhoU[1], exterior.rhoE};
                }

                ComprFD::NodeState<U, 2> s_B{.rho = UB[0], .rhoU = {UB[1], UB[2]}, .rhoE = UB[3]};
                ComprFD::NodeState<U, 2> s_T{.rho = UT[0], .rhoU = {UT[1], UT[2]}, .rhoE = UT[3]};

                const auto [flux_x_B, flux_y_B] = ComprFD::euler_fluxes_2d(s_B, this->m_gas_mod);
                const auto [flux_x_T, flux_y_T] = ComprFD::euler_fluxes_2d(s_T, this->m_gas_mod);

                const auto G_star =
                    this->m_riemann_solver.compute_flux(UB, UT, this->m_gas_mod, nx, ny);

                for (std::size_t eq = 0; eq < n_eqs; ++eq)
                {
                    if constexpr (UpdateB)
                    {
                        Indexer<U> res_B(f_view.as_span(i, J - 1, ndofs), x_nds, y_nds);
                        const U    normal_flux_B = flux_x_B[eq] * nx + flux_y_B[eq] * ny;
                        for (std::size_t l = 0; l < y_nds; ++l)
                            res_B(eq, k, l) += scale_y * (G_star[eq] - normal_flux_B) * dR_y(l);
                    }
                    if constexpr (UpdateT)
                    {
                        Indexer<U> res_T(f_view.as_span(i, J, ndofs), x_nds, y_nds);
                        const U    normal_flux_T = flux_x_T[eq] * nx + flux_y_T[eq] * ny;
                        for (std::size_t l = 0; l < y_nds; ++l)
                            res_T(eq, k, l) += scale_y * (G_star[eq] - normal_flux_T) * dL_y(l);
                    }
                }
            }
        };

        {
            const bool is_physical_bot = (j_start == 0) && !this->is_periodic_y;
            const bool is_physical_top =
                (static_cast<std::size_t>(j_max) == global_max_j) && !this->is_periodic_y;

            for (const auto i : std::views::iota(i_start, i_max))
            {
                if (is_physical_bot)
                    eval_y_edge.template operator()<false, true, true, false>(i, j_start);
                else
                    eval_y_edge.template operator()<false, true, false, false>(i, j_start);

                for (const auto J : std::views::iota(1 + j_start, j_max))
                    eval_y_edge.template operator()<true, true, false, false>(i, J);

                if (is_physical_top)
                    eval_y_edge.template operator()<true, false, false, true>(i, j_max);
                else
                    eval_y_edge.template operator()<true, false, false, false>(i, j_max);
            }
        }

        PetscFunctionReturn(PETSC_SUCCESS);
    }
#endif
} // namespace numPDE
