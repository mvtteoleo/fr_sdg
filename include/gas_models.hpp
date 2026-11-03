#pragma once
#include "macro_and_libc.hpp"
#include "parser.hpp"

namespace numPDE::ComprFD
{

    /*
     * Density in dimensional form [ kg * m^-3]
     * Specific internal energy [ m^2 sec^-2 ]
     */
    template <real_or_complex_floating_point T>
    struct ThermodynamicState
    {
        T density{};
        T energy{};
    };

    /*
        template <real_or_complex_floating_point U>
        struct NodeState1D
        {
            U rho, rhoU, rhoE;
        };
    */

    template <real_or_complex_floating_point U, std::size_t Dims = 1>
    struct NodeState
    {
        U                   rho;
        std::array<U, Dims> rhoU;
        U                   rhoE;

        // Zero-cost abstraction to get scalar velocity squared
        constexpr U get_vel_sq() const
        {
            U v_sq{0.0};
            for (const auto& mom : rhoU)
            {
                v_sq += (mom * mom);
            }
            return v_sq / (rho * rho);
        }

        // Helper to get specific internal energy cleanly
        constexpr U get_internal_energy() const { return (rhoE / rho) - U(0.5) * get_vel_sq(); }
    };

    template <class GasModelType, typename T>
    concept GasModelConc = requires(GasModelType g_law, ThermodynamicState<T> state, T scal) {
        { g_law.get_pressure(state) } -> std::convertible_to<T>;
        { g_law.get_sound_speed(state) } -> std::convertible_to<T>;
        { g_law.get_rhoE(scal, scal, scal) } -> std::convertible_to<T>;
    };

    template <real_or_complex_floating_point T>
    class IdealGas
    {
      private:
        T gamma{1.4};
        T R_spec{287.05};

      public:
        IdealGas() = default;
        explicit IdealGas(T g, T R) : gamma{g}, R_spec(R) {}

        // Template on U so that I can then leverage the Complex Step derivative
        // WARNING a phisical solution MUST ensure 'density > 0'
        template <real_or_complex_floating_point U>
        U get_pressure(const ThermodynamicState<U>& ts) const
        {
            return (gamma - U{1.0}) * ts.density * ts.energy;
        }

        template <real_or_complex_floating_point U>
        U get_sound_speed(const ThermodynamicState<U>& ts) const
        {
            return std::sqrt(gamma * get_pressure(ts) / ts.density);
        }

        T get_rhoE(T rho, T vel, T pres) const
        {
            const T internal_energy = pres / (gamma - 1.0);
            const T kinetic_energy  = 0.5 * rho * vel * vel;
            return internal_energy + kinetic_energy;
        };
    };

    template <real_or_complex_floating_point T>
    class VanDerWaals
    {
      private:
        T gamma{1.4};
        T R_spec{287.05};
        T a{0.0}; // Attraction parameter [Pa * m^6 / kg^2]
        T b{0.0}; // Co-volume / Repulsion parameter [m^3 / kg]

      public:
        VanDerWaals() = default;

        explicit VanDerWaals(T g, T R, T attr_a, T rep_b) : gamma{g}, R_spec(R), a{attr_a}, b{rep_b}
        {
        }

        template <real_or_complex_floating_point U>
        U get_pressure(const ThermodynamicState<U>& ts) const
        {
            U rho           = ts.density;
            U covolume_term = U{1.0} - U{b} * rho;

            // T_corr represents the thermal contribution (cv * T = e + a * rho)
            U cvT = ts.energy + U{a} * rho;

            U p_ideal      = ((U{gamma} - U{1.0}) * rho * cvT) / covolume_term;
            U p_attraction = U{a} * rho * rho;

            return p_ideal - p_attraction;
        }

        template <real_or_complex_floating_point U>
        U get_sound_speed(const ThermodynamicState<U>& ts) const
        {

            const U rho           = ts.density;
            const U covolume_term = U{1.0} - U{b} * rho;
            const U denom_sq      = covolume_term * covolume_term;

            const U thermal_energy = ts.energy + U{a} * rho;

            // Term 1: Ideal thermal speed component
            const U term1 = (U{gamma} * (U{gamma} - U{1.0}) * thermal_energy) / denom_sq;

            // Term 2: Van der Waals specific internal energy correction mapping coupling
            const U term2 = ((U{gamma} - U{1.0}) * U{a} * rho * (U{2.0} - U{b} * rho)) / denom_sq;

            // Term 3: Direct attraction modification
            const U term3 = U{2.0} * U{a} * rho;

            const U c_squared = term1 + term2 - term3;

            return std::sqrt(c_squared);
        }

        T get_rhoE(T rho, T vel, T pres) const
        {
            T covolume_term = T{1.0} - T{b} * rho;
            T a_rho_sq      = T{a} * rho * rho;

            // 1. Calculate volumetric internal energy (rho * e)
            T rho_e = ((pres + a_rho_sq) * covolume_term) / (T{gamma} - T{1.0}) - a_rho_sq;

            // 2. Calculate kinetic energy
            T kinetic_energy = T{0.5} * rho * vel * vel;

            // 3. Return the fully formed conservative state vector
            return rho_e + kinetic_energy;
        };
    };

    // Since this class respects the GasModelConc I can simply use it to switch
    // betweeen gas models as well as allow the user to create his gas model to
    // feed to the Solver class
    template <real_or_complex_floating_point T>
    class GasModels
    {
      private:
        const ConstitutiveLaw m_const_law;

        // Explicitly const variant containing only the active law
        const std::variant<IdealGas<T>, VanDerWaals<T>> m_gas;

      public:
        explicit GasModels(const ConfigOptions<T>& opts)
            : m_const_law{[&]() -> ConstitutiveLaw
                          {
                              const auto g_opts = opts.get_gas_model();
                              return g_opts.c_law;
                          }()},
              m_gas{[&]() -> std::variant<IdealGas<T>, VanDerWaals<T>>
                    {
                        const GasModelConstants gas_opts = opts.get_gas_model();
                        switch (m_const_law)
                        {
                            case ConstitutiveLaw::IdealGas:
                                return IdealGas<T>{gas_opts.gamma, gas_opts.R_spec};
                            case ConstitutiveLaw::VanDerWaals:
                                return VanDerWaals<T>{gas_opts.gamma, gas_opts.R_spec,
                                                      gas_opts.VdW_a, gas_opts.VdW_b};
                            default:
                                throw std::runtime_error("Invalid model");
                        }
                    }()}
        {
        }

        template <real_or_complex_floating_point U>
        U get_pressure(const ThermodynamicState<U>& ts) const
        {
            // std::visit maps perfectly to the std::variant correct type
            return std::visit([&](auto&& gas) -> U { return gas.get_pressure(ts); }, m_gas);
        }
        template <real_or_complex_floating_point U>
        U get_sound_speed(const ThermodynamicState<U>& ts) const
        {
            return std::visit([&](auto&& gas) -> U { return gas.get_sound_speed(ts); }, m_gas);
        }

        T get_rhoE(T rho, T vel, T pres) const
        {
            return std::visit([&](auto&& gas) -> T { return gas.get_rhoE(rho, vel, pres); }, m_gas);
        }
    };

} // namespace numPDE::ComprFD
