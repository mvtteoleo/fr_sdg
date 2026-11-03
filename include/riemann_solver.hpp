#pragma once
#include "gas_models.hpp"
#include "macro_and_libc.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <utility>

namespace numPDE::ComprFD
{
    enum class RiemannScheme
    {
        Rusanov,
        HLLC
    };

    // Primitive vars and thermodynamic state from conservatives
    template <real_or_complex_floating_point U>
    auto make_state(const NodeState<U, 1>& ns) -> ComprFD::ThermodynamicState<U>
    {
        const U rho    = wrap_positive<U>(ns.rho);
        const U tot_en = wrap_positive<U>(ns.rhoE);
        const U u      = ns.rhoU[0] / rho;
        return {.density = rho, .energy = (tot_en / rho - U(0.5) * u * u)};
    }

    template <real_or_complex_floating_point U>
    auto make_state(const NodeState<U, 2>& ns) -> ComprFD::ThermodynamicState<U>
    {
        const U rho    = wrap_positive<U>(ns.rho);
        const U tot_en = wrap_positive<U>(ns.rhoE);
        const U u      = ns.rhoU[0] / rho;
        const U v      = ns.rhoU[1] / rho;
        return {.density = rho, .energy = (tot_en / rho - U(0.5) * (u * u + v * v))};
    }

    template <real_or_complex_floating_point U, GasModelConc<PetscReal> GasModel,
              std::size_t n_dims = 1>
    auto euler_flux(const NodeState<U, 1>& ns, const GasModel& gas_mod) -> std::array<U, n_dims + 2>
    {
        const U p = gas_mod.template get_pressure<U>(make_state(ns));
        const U u = ns.rhoU[0] / wrap_positive<U>(ns.rho);
        return {ns.rhoU[0], ns.rhoU[0] * u + p, (ns.rhoE + p) * u};
    }

    template <real_or_complex_floating_point U, typename GasModel>
    auto euler_fluxes_2d_w_p(const NodeState<U, 2>& ns, const GasModel& gas_mod)
    {
        const U rho     = wrap_positive<U>(ns.rho);
        const U inv_rho = U(1.0) / rho;
        const U u       = ns.rhoU[0] * inv_rho;
        const U v       = ns.rhoU[1] * inv_rho;

        const U p = gas_mod.template get_pressure<U>(make_state(ns));

        const U rhoE_plus_p = ns.rhoE + p;
        const U rhou_v      = ns.rhoU[0] * v;

        std::array<U, 4> F{ns.rhoU[0], ns.rhoU[0] * u + p, rhou_v, u * rhoE_plus_p};

        std::array<U, 4> G{ns.rhoU[1], rhou_v, ns.rhoU[1] * v + p, v * rhoE_plus_p};

        return std::make_tuple(F, G, p);
    }
    /**
     * @brief Computes both the X-flux (F) and Y-flux (G) simultaneously for a 2D node.
     */
    template <real_or_complex_floating_point U, typename GasModel>
    auto euler_fluxes_2d(const NodeState<U, 2>& ns, const GasModel& gas_mod)
        -> std::pair<std::array<U, 4>, std::array<U, 4>>
    {
        const auto [F, G, pres] = euler_fluxes_2d_w_p(ns, gas_mod);

        return std::make_pair(F, G);
    }

    // Primary templates for Riemann Solvers
    template <real_or_complex_floating_point T, RiemannScheme Scheme, GasModelConc<T> GasModel>
    struct RiemannSolver;

    // UPDATED: Primary declaration using geometric normal projections
    template <real_or_complex_floating_point T, RiemannScheme Scheme, typename GasModel>
    struct RiemannSolver2D;

    template <real_or_complex_floating_point T, GasModelConc<T> GasModel>
    struct RiemannSolver<T, RiemannScheme::Rusanov, GasModel>
    {
        static constexpr int n_eqs = 3;

        template <real_or_complex_floating_point U>
        static std::array<U, n_eqs> compute_flux(const std::array<U, n_eqs>& UL,
                                                 const std::array<U, n_eqs>& UR,
                                                 const GasModel&             gas_mod)
        {
            NodeState<U, 1> s_l{.rho = UL[0], .rhoU = {UL[1]}, .rhoE = UL[2]};
            NodeState<U, 1> s_r{.rho = UR[0], .rhoU = {UR[1]}, .rhoE = UR[2]};
            const auto      stateL = make_state(s_l);
            const auto      stateR = make_state(s_r);

            const U sL   = numPDE::smooth_abs(UL[1] / UL[0]) + gas_mod.get_sound_speed(stateL);
            const U sR   = numPDE::smooth_abs(UR[1] / UR[0]) + gas_mod.get_sound_speed(stateR);
            const U smax = numPDE::smooth_max(sL, sR);

            const auto FL = euler_flux(s_l, gas_mod);
            const auto FR = euler_flux(s_r, gas_mod);

            return {U(0.5) * (FL[0] + FR[0]) - U(0.5) * smax * (UR[0] - UL[0]),
                    U(0.5) * (FL[1] + FR[1]) - U(0.5) * smax * (UR[1] - UL[1]),
                    U(0.5) * (FL[2] + FR[2]) - U(0.5) * smax * (UR[2] - UL[2])};
        }
    };

    template <real_or_complex_floating_point T, GasModelConc<T> GasModel>
    struct RiemannSolver<T, RiemannScheme::HLLC, GasModel>
    {
        static constexpr int n_eqs = 3;

        template <real_or_complex_floating_point U>
        static std::array<U, n_eqs> compute_flux(const std::array<U, n_eqs>& UL,
                                                 const std::array<U, n_eqs>& UR,
                                                 const GasModel&             gas_mod)
        {
            NodeState<U, 1> s_l{.rho = UL[0], .rhoU = {UL[1]}, .rhoE = UL[2]};
            NodeState<U, 1> s_r{.rho = UR[0], .rhoU = {UR[1]}, .rhoE = UR[2]};

            const auto stateL = make_state(s_l);
            const auto stateR = make_state(s_r);

            const U uL = UL[1] / UL[0];
            const U uR = UR[1] / UR[0];
            const U aL = gas_mod.get_sound_speed(stateL);
            const U aR = gas_mod.get_sound_speed(stateR);

            const auto FL = euler_flux(s_l, gas_mod);
            const auto FR = euler_flux(s_r, gas_mod);

            const U pL = FL[1] - uL * UL[1];
            const U pR = FR[1] - uR * UR[1];

            const U sL = numPDE::smooth_min(uL - aL, uR - aR);
            const U sR = numPDE::smooth_max(uL + aL, uR + aR);

            const U rhoL_diff = UL[0] * (sL - uL);
            const U rhoR_diff = UR[0] * (sR - uR);
            const U sM = (pR - pL + uL * rhoL_diff - uR * rhoR_diff) / (rhoL_diff - rhoR_diff);

            auto get_real = [](const U& val)
            {
                if constexpr (numPDE::is_complex_floating_point<std::remove_const_t<U>>::value)
                    return val.real() + val.imag();
                else
                    return val;
            };

            const auto real_sL = get_real(sL);
            const auto real_sM = get_real(sM);
            const auto real_sR = get_real(sR);

            if (real_sL >= 0.0)
            {
                return FL;
            }
            else if (real_sM >= 0.0)
            {
                const U p_star = pL + UL[0] * (sL - uL) * (sM - uL);
                return {(sM * (sL * UL[0] - FL[0])) / (sL - sM),
                        (sM * (sL * UL[1] - FL[1]) + sL * p_star) / (sL - sM),
                        (sM * (sL * UL[2] - FL[2]) + sL * p_star * sM) / (sL - sM)};
            }
            else if (real_sR >= 0.0)
            {
                const U p_star = pR + UR[0] * (sR - uR) * (sM - uR);
                return {(sM * (sR * UR[0] - FR[0])) / (sR - sM),
                        (sM * (sR * UR[1] - FR[1]) + sR * p_star) / (sR - sM),
                        (sM * (sR * UR[2] - FR[2]) + sR * p_star * sM) / (sR - sM)};
            }
            else
            {
                return FR;
            }
        }
    };

    // ====================================================================
    // REFACTORED 2D REGIMES: Fully Curvilinear and Geometric Normal Bound
    // ====================================================================
    template <real_or_complex_floating_point T, GasModelConc<T> GasModel>
    struct RiemannSolver2D<T, RiemannScheme::Rusanov, GasModel>
    {
        static constexpr std::size_t n_eqs = 4;

        template <real_or_complex_floating_point U>
        static std::array<U, n_eqs>
        compute_flux(const std::array<U, n_eqs>& UL, const std::array<U, n_eqs>& UR,
                     const GasModel& gas_mod, std::floating_point auto nx,
                     std::floating_point auto ny)
        {
            const U inv_rhoL = U(1.0) / wrap_positive<U>(UL[0]);
            const U inv_rhoR = U(1.0) / wrap_positive<U>(UR[0]);

            const U uL = UL[1] * inv_rhoL;
            const U vL = UL[2] * inv_rhoL;
            const U uR = UR[1] * inv_rhoR;
            const U vR = UR[2] * inv_rhoR;

            // 1. Rotate Cartesian velocities into Face Normal/Tangential Components
            const U unL = uL * nx + vL * ny;
            const U utL = -uL * ny + vL * nx;
            const U unR = uR * nx + vR * ny;
            const U utR = -uR * ny + vR * nx;

            ComprFD::NodeState<U, 2> state_L{.rho = UL[0], .rhoU = {UL[1], UL[2]}, .rhoE = UL[3]};
            ComprFD::NodeState<U, 2> state_R{.rho = UR[0], .rhoU = {UR[1], UR[2]}, .rhoE = UR[3]};
            const auto               thermoL = make_state(state_L);
            const auto               thermoR = make_state(state_R);

            const U cL = gas_mod.get_sound_speed(thermoL);
            const U cR = gas_mod.get_sound_speed(thermoR);

            const U smax = smooth_max(smooth_abs(unL) + cL, smooth_abs(unR) + cR);

            const U pL = gas_mod.template get_pressure<U>(thermoL);
            const U pR = gas_mod.template get_pressure<U>(thermoR);

            // 2. Compute 1D normal structural fluxes in the face reference frame
            std::array<U, 4> FL_norm = {UL[0] * unL, UL[0] * unL * unL + pL, UL[0] * unL * utL,
                                        (UL[3] + pL) * unL};
            std::array<U, 4> FR_norm = {UR[0] * unR, UR[0] * unR * unR + pR, UR[0] * unR * utR,
                                        (UR[3] + pR) * unR};

            // 3. Central Rusanov sampling with components mapping
            U flux_rho_star = 0.5 * (FL_norm[0] + FR_norm[0]) - 0.5 * smax * (UR[0] - UL[0]);
            U flux_un_star =
                0.5 * (FL_norm[1] + FR_norm[1]) - 0.5 * smax * (UR[0] * unR - UL[0] * unL);
            U flux_ut_star =
                0.5 * (FL_norm[2] + FR_norm[2]) - 0.5 * smax * (UR[0] * utR - UL[0] * utL);
            U flux_E_star = 0.5 * (FL_norm[3] + FR_norm[3]) - 0.5 * smax * (UR[3] - UL[3]);

            // 4. Inverse Rotate Face Fluxes back into Global Cartesian Residuals
            return {flux_rho_star,
                    flux_un_star * nx - flux_ut_star * ny, // Global X-Momentum flux
                    flux_un_star * ny + flux_ut_star * nx, // Global Y-Momentum flux
                    flux_E_star};
        }
    };

    template <real_or_complex_floating_point T, GasModelConc<T> GasModel>
    struct RiemannSolver2D<T, RiemannScheme::HLLC, GasModel>
    {
        static constexpr int n_eqs = 4;

        template <real_or_complex_floating_point U>
        static std::array<U, n_eqs>
        compute_flux(const std::array<U, n_eqs>& UL, const std::array<U, n_eqs>& UR,
                     const GasModel& gas_mod, std::floating_point auto nx,
                     std::floating_point auto ny)
        {
            const U inv_rhoL = U(1.0) / wrap_positive<U>(UL[0]);
            const U inv_rhoR = U(1.0) / wrap_positive<U>(UR[0]);

            const U uL = UL[1] * inv_rhoL;
            const U vL = UL[2] * inv_rhoL;
            const U uR = UR[1] * inv_rhoR;
            const U vR = UR[2] * inv_rhoR;

            // 1. Rotate velocities to Face Frame
            const U unL = uL * nx + vL * ny;
            const U utL = -uL * ny + vL * nx;
            const U unR = uR * nx + vR * ny;
            const U utR = -uR * ny + vR * nx;

            NodeState<U, 2> s_l{.rho = UL[0], .rhoU = {UL[1], UL[2]}, .rhoE = UL[3]};
            NodeState<U, 2> s_r{.rho = UR[0], .rhoU = {UR[1], UR[2]}, .rhoE = UR[3]};

            const auto stateL = make_state(s_l);
            const auto stateR = make_state(s_r);

            const U aL = gas_mod.get_sound_speed(stateL);
            const U aR = gas_mod.get_sound_speed(stateR);
            const U pL = gas_mod.template get_pressure<U>(stateL);
            const U pR = gas_mod.template get_pressure<U>(stateR);

            // 2. Generate 1D Local Normal Flux Tensors
            std::array<U, 4> FL_norm = {UL[0] * unL, UL[0] * unL * unL + pL, UL[0] * unL * utL,
                                        (UL[3] + pL) * unL};
            std::array<U, 4> FR_norm = {UR[0] * unR, UR[0] * unR * unR + pR, UR[0] * unR * utR,
                                        (UR[3] + pR) * unR};

            // 3. Complex Step Tracking Safe Bounds
            const U sL = numPDE::smooth_min(unL - aL, unR - aR);
            const U sR = numPDE::smooth_max(unL + aL, unR + aR);

            const U rhoL_diff = UL[0] * (sL - unL);
            const U rhoR_diff = UR[0] * (sR - unR);
            const U sM = (pR - pL + unL * rhoL_diff - unR * rhoR_diff) / (rhoL_diff - rhoR_diff);

            auto get_real = [](const U& val)
            {
                if constexpr (numPDE::is_complex_floating_point<std::remove_const_t<U>>::value)
                    return val.real() + val.imag();
                else
                    return val;
            };

            const auto real_sL = get_real(sL);
            const auto real_sM = get_real(sM);
            const auto real_sR = get_real(sR);

            std::array<U, 4> flux_norm_star;

            // 4. Perform localized 1D HLLC Sampling
            if (real_sL >= 0.0)
            {
                flux_norm_star = FL_norm;
            }
            else if (real_sM >= 0.0)
            {
                const U p_star = pL + UL[0] * (sL - unL) * (sM - unL);

                flux_norm_star[0] = (sM * (sL * UL[0] - FL_norm[0])) / (sL - sM);
                flux_norm_star[1] =
                    (sM * (sL * (UL[0] * unL) - FL_norm[1]) + sL * p_star) / (sL - sM);
                flux_norm_star[2] = utL * flux_norm_star[0]; // Tangential Momentum Advection
                flux_norm_star[3] = (sM * (sL * UL[3] - FL_norm[3]) + sL * p_star * sM) / (sL - sM);
            }
            else if (real_sR >= 0.0)
            {
                const U p_star = pR + UR[0] * (sR - unR) * (sM - unR);

                flux_norm_star[0] = (sM * (sR * UR[0] - FR_norm[0])) / (sR - sM);
                flux_norm_star[1] =
                    (sM * (sR * (UR[0] * unR) - FR_norm[1]) + sR * p_star) / (sR - sM);
                flux_norm_star[2] = utR * flux_norm_star[0]; // Tangential Momentum Advection
                flux_norm_star[3] = (sM * (sR * UR[3] - FR_norm[3]) + sR * p_star * sM) / (sR - sM);
            }
            else
            {
                flux_norm_star = FR_norm;
            }

            // 5. Inverse Rotate Face Fluxes back to Cartesian World
            return {flux_norm_star[0],
                    flux_norm_star[1] * nx - flux_norm_star[2] * ny, // Global X-Momentum flux
                    flux_norm_star[1] * ny + flux_norm_star[2] * nx, // Global Y-Momentum flux
                    flux_norm_star[3]};
        }
    };
} // namespace numPDE::ComprFD
