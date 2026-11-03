#pragma once

#include "gas_models.hpp"
#include "riemann_solver.hpp"
namespace numPDE
{

    template <class FluxType, typename T>
    concept FluxConc = requires(FluxType flux, T scal) {
        { flux.n_equations() } -> std::convertible_to<std::size_t>;
        { flux.get_phys_flux(state) } -> std::convertible_to<T>;
        { flux.get_num_flux(scal, scal, scal) } -> std::convertible_to<T>;
    };

    template <std::floating_point T, std::size_t n_dims, ComprFD::GasModelConc<T> GasModel,
              typename RiemannSolver>
    class EulerEquationsAgnostic
    {
        const GasModel      m_gas_mod;
        const RiemannSolver m_riemann;

      public:
        consteval auto n_equations() const { return n_dims + 2; };

        template <std::ranges::random_access_range R>
        auto get_phys_flux(R state)
        {
            using U = std::ranges::range_value_t<R>;
            return ComprFD::euler_fluxes_2d(s_L, this->m_gas_mod);
        };
    };

} // namespace numPDE
