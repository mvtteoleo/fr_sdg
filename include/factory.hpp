#pragma once
// Factory with some precooked different problems

#include "boundary_conditions.hpp"
#include "gas_models.hpp"
#include "macro_and_libc.hpp"
#include "parser.hpp"
#include "problem_def.hpp"
#include <cmath>
#include <print>
namespace numPDE
{

    enum class ProblemType
    {
        DoNothing // Sanity check to ensure the solver was working as intended
        ,
        AcousticWave // Non-linear entropy wave (for convergence)
        ,
        LinearizedMOC // Linearized acoustic pulse (for BC and more physical checks)
        ,
        SodShockTube,
        CircShockRadial
    };

    // Forward declaration
    template <ProblemType prob>
    struct ProblemFactory;

    // ==========================================
    // Template specializations
    // ==========================================
    template <>
    struct ProblemFactory<ProblemType::DoNothing>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const AcousticWaveData& /*data*/)
        {
            using AirTrait = ProblemTraits<PetscReal, 1, ComprFD::IdealGas<PetscReal>,
                                           ComprFD::RiemannScheme::HLLC>;
            BoundaryConditions<PetscReal, 1> bcs{};

            bcs.set_right_bc(BCType::Periodic);
            bcs.set_left_bc(BCType::Periodic);

            const auto state_one =
                ComprFD::NodeState<PetscReal, 1>{.rho = 1.0, .rhoU = {2.0}, .rhoE = 3.0};

            bcs.initial_cond = [=](const PetscReal) -> ComprFD::NodeState<PetscReal, 1>
            { return state_one; };
            bcs.exact_sol = [=](const PetscReal,
                                const PetscReal) -> ComprFD::NodeState<PetscReal, 1>
            { return state_one; };

            return std::make_unique<Problem<AirTrait>>(conf, bcs, AirTrait::GasModel{});
        }
    };

    template <>
    struct ProblemFactory<ProblemType::AcousticWave>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const AcousticWaveData& data)
        {
            using AirTrait = ProblemTraits<PetscReal, 1, ComprFD::IdealGas<PetscReal>,
                                           ComprFD::RiemannScheme::HLLC>;
            BoundaryConditions<PetscReal, 1> bcs{};

            const PetscReal gamma  = 1.4;
            const PetscReal length = conf.get_box_length(0);
            const PetscReal x_c    = length / 2.0;

            auto exact_entropy_wave = [=](const PetscReal time,
                                          const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            {
                PetscReal shifted_pos = pos - data.u_0 * time;
                // Safe periodic wrapping
                PetscReal wrapped_pos = std::fmod(std::fmod(shifted_pos, length) + length, length);

                const PetscReal rho =
                    data.rho_0 +
                    data.epsilon * std::exp(-data.alpha * std::pow((wrapped_pos - x_c), 2));
                const PetscReal vel  = data.u_0;
                const PetscReal pres = data.p_0;

                const PetscReal internal_energy = pres / (gamma - 1.0);
                const PetscReal kinetic_energy  = 0.5 * rho * vel * vel;

                return ComprFD::NodeState<PetscReal, 1>{
                    .rho = rho, .rhoU = {rho * vel}, .rhoE = internal_energy + kinetic_energy};
            };
            bcs.set_left_bc(BCType::Dirichlet,
                            [=](PetscReal t, const ComprFD::NodeState<PetscReal, 1>& /* interior */)
                                -> ComprFD::NodeState<PetscReal, 1>
                            {
                                // Evaluate the exact wave at the left boundary (x = 0)
                                return exact_entropy_wave(t, 0.0);
                            });

            bcs.set_right_bc(
                BCType::Dirichlet,
                [=](PetscReal t, const ComprFD::NodeState<PetscReal, 1>& /* interior */)
                    -> ComprFD::NodeState<PetscReal, 1>
                {
                    // Evaluate the exact wave at the right boundary (x = length)
                    return exact_entropy_wave(t, length);
                });
            bcs.initial_cond = [=](const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return exact_entropy_wave(0.0, pos); };
            bcs.exact_sol = [=](const PetscReal curr_time,
                                const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return exact_entropy_wave(curr_time, pos); };

            return std::make_unique<Problem<AirTrait>>(conf, bcs, AirTrait::GasModel{});
        }
    };

    template <>
    struct ProblemFactory<ProblemType::LinearizedMOC>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const AcousticWaveData& data)
        {
            using AirTrait = ProblemTraits<PetscReal, 1, ComprFD::IdealGas<PetscReal>,
                                           ComprFD::RiemannScheme::HLLC>;
            BoundaryConditions<PetscReal, 1> bcs{};
            AirTrait::GasModel               gas_model{};

            // MOC with Wall Boundaries
            bcs.set_right_bc(BCType::Wall);
            bcs.set_left_bc(BCType::Wall);

            const PetscReal gamma  = 1.4;
            const PetscReal length = conf.get_box_length(0);
            const PetscReal x_c    = length / 2.0;

            const auto c_0          = std::sqrt(gamma * data.p_0 / data.rho_0);
            const auto lambda_plus  = data.u_0 + c_0;
            const auto lambda_minus = data.u_0 - c_0;

            auto perturbation = [=](const PetscReal pos) -> PetscReal
            {
                // Method of Images for Wall reflections
                PetscReal p_main = data.epsilon * std::exp(-data.alpha * std::pow((pos - x_c), 2));
                PetscReal p_left = data.epsilon * std::exp(-data.alpha * std::pow((pos + x_c), 2));
                PetscReal p_right =
                    data.epsilon *
                    std::exp(-data.alpha * std::pow((pos - (2.0 * length - x_c)), 2));
                return p_main + p_left + p_right;
            };

            auto MOC_solution = [=](const PetscReal time,
                                    const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            {
                const PetscReal rrw = perturbation(pos - lambda_plus * time);
                const PetscReal lrw = perturbation(pos - lambda_minus * time);

                const PetscReal rho  = data.rho_0 + 0.5 * (rrw + lrw);
                const PetscReal vel  = data.u_0 + c_0 / (2.0 * data.rho_0) * (rrw - lrw);
                const PetscReal pres = data.p_0 * std::pow(rho / data.rho_0, gamma);

                return ComprFD::NodeState<PetscReal, 1>{.rho  = rho,
                                                        .rhoU = {rho * vel},
                                                        .rhoE = pres / (gamma - 1.0) +
                                                                0.5 * rho * vel * vel};
            };

            bcs.initial_cond = [=](const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return MOC_solution(0.0, pos); };
            bcs.exact_sol = [=](const PetscReal curr_time,
                                const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return MOC_solution(curr_time, pos); };

            return std::make_unique<Problem<AirTrait>>(conf, bcs, gas_model);
        }
    };

    template <>
    struct ProblemFactory<ProblemType::SodShockTube>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const SodProblemValues& data)
        {
            using GasType             = ComprFD::GasModels<PetscReal>;
            constexpr auto RiemanType = ComprFD::RiemannScheme::HLLC;
            using AirTrait            = ProblemTraits<PetscReal, 1, GasType, RiemanType>;
            BoundaryConditions<PetscReal, 1> bcs{};
            GasType                          gas_model{conf};

            // Sod Shock Tube usually uses Outflow/Transmissive bounds.
            // Wall boundaries are fine ONLY IF the final simulation time
            // is short enough that the waves don't reach the edges.
            bcs.set_right_bc(BCType::Outflow);
            bcs.set_left_bc(BCType::Outflow);

            const PetscReal length = conf.get_box_length(0);
            const PetscReal x_c    = length / 2.0; // Diaphragm position

            // Discontinuous Initial Conditions
            auto sod_initial = [=](const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            {
                PetscReal rho, vel, pres;
                if (pos < x_c)
                {
                    rho  = data.rho_L;
                    vel  = data.vel_L;
                    pres = data.pres_L;
                }
                else
                {
                    rho  = data.rho_R;
                    vel  = data.vel_R;
                    pres = data.pres_R;
                }

                return ComprFD::NodeState<PetscReal, 1>{
                    .rho = rho, .rhoU = {rho * vel}, .rhoE = gas_model.get_rhoE(rho, vel, pres)};
            };

            bcs.initial_cond = [=](const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return sod_initial(pos); };

            // WARNING: For L2 error computations, this exact solution is WRONG for t > 0.
            // Writing an exact Riemann solver requires an iterative root-finding method
            // for the star-region pressure. For now, we return the initial state just
            // to satisfy the callback signature.
            bcs.exact_sol = [=](const PetscReal /*curr_time*/,
                                const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return sod_initial(pos); };

            return std::make_unique<Problem<AirTrait>>(conf, bcs, gas_model);
        }
    };
    template <>
    struct ProblemFactory<ProblemType::CircShockRadial>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const SodProblemValues& data)
        {
            using GasType             = ComprFD::GasModels<PetscReal>;
            constexpr auto RiemanType = ComprFD::RiemannScheme::HLLC;
            using AirTrait =
                ProblemTraits<PetscReal, 1, GasType, RiemanType, CoordinateSystem::Radial>;
            BoundaryConditions<PetscReal, 1> bcs{};
            AirTrait::GasModel               gas_model{conf};

            // Sod Shock Tube usually uses Outflow/Transmissive bounds.
            // Wall boundaries are fine ONLY IF the final simulation time
            // is short enough that the waves don't reach the edges.
            bcs.set_right_bc(BCType::Outflow);
            bcs.set_left_bc(BCType::Wall);

            const PetscReal length = conf.get_box_length(0);
            const PetscReal x_c    = data.radius; // Diaphragm position

            // Discontinuous Initial Conditions
            auto sod_initial = [=](const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            {
                PetscReal rho, vel, pres;
                if (pos < x_c)
                {
                    rho  = data.rho_L;
                    vel  = data.vel_L;
                    pres = data.pres_L;
                }
                else
                {
                    rho  = data.rho_R;
                    vel  = data.vel_R;
                    pres = data.pres_R;
                }

                return ComprFD::NodeState<PetscReal, 1>{
                    .rho = rho, .rhoU = {rho * vel}, .rhoE = gas_model.get_rhoE(rho, vel, pres)};
            };

            bcs.initial_cond = [=](const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return sod_initial(pos); };

            // WARNING: For L2 error computations, this exact solution is WRONG for t > 0.
            // Writing an exact Riemann solver requires an iterative root-finding method
            // for the star-region pressure. For now, we return the initial state just
            // to satisfy the callback signature.
            bcs.exact_sol = [=](const PetscReal /*curr_time*/,
                                const PetscReal pos) -> ComprFD::NodeState<PetscReal, 1>
            { return sod_initial(pos); };

            return std::make_unique<Problem<AirTrait>>(conf, bcs, gas_model);
        }
    };

    // New Problem Types for the 2D Factory
    enum class ProblemType2D
    {
        IsentropicVortex,
        DensityWave,
        IsentropicVortexDirichlet,
        CircularExplosion
    };

    // Data structures to hold initialization parameters
    struct VortexData
    {
        PetscReal u_0{0.0};  // Background X-velocity
        PetscReal v_0{5.0};  // Background Y-velocity
        PetscReal beta{5.0}; // Vortex strength/intensity
        PetscReal r_v{1.0};
        PetscReal l{10.0};
        PetscReal t_final{2.0};
    };

    // Primary Template for 2D Problems
    template <ProblemType2D prob>
    struct ProblemFactory2D;

    template <real_or_complex_floating_point T>
    auto exact_vortex_solution(const T time, const T x, const T y, const VortexData& vd)
        -> ComprFD::NodeState<T, 2>
    {
        const auto gamma{1.4};
        const auto pi{std::numbers::pi_v<PetscReal>};

        // 1. DYNAMIC CENTER TRACKING (Accounts for Periodic Domain Wrapping)
        // Adjust the center position based on the advection velocity and time
        T x_c = vd.l / 2.0 + vd.u_0 * time;
        T y_c = vd.l / 2.0 + vd.v_0 * time;

        // Apply periodic mapping over the domain length [0, vd.l]
        // This ensures tracking works flawlessly past t=2.0
        x_c = x_c - vd.l * std::floor(x_c / vd.l);
        y_c = y_c - vd.l * std::floor(y_c / vd.l);

        // 2. RELATIVE SPATIAL DISTANCE
        const auto dx = x - x_c;
        const auto dy = y - y_c;
        const auto r2 = (dx * dx) + (dy * dy);

        // 3. EXPONENTIAL DECAY FACTOR WITH ENFORCED CORE SCALING
        // f = 1 - (r / r_v)^2
        const auto f     = 1.0 - (r2 / (vd.r_v * vd.r_v));
        const auto exp_f = std::exp(f);

        // 4. ALIASED ACCURACY-PRESERVING DENSITY
        const auto rho = std::pow(1.0 - (vd.beta * vd.beta * (gamma - 1.0) * (exp_f * exp_f)) /
                                            (16.0 * gamma * pi * pi),
                                  1.0 / (gamma - 1.0));

        // 5. CORRECT VELOCITY FORMULATION (Fixed parenthesis and added exp_f multiplier)
        const auto u = vd.u_0 - (vd.beta * exp_f * dy) / (2.0 * pi * vd.r_v);
        const auto v = vd.v_0 + (vd.beta * exp_f * dx) / (2.0 * pi * vd.r_v);

        // 6. TOTAL CONSERVATIVE ENERGY MAPPING
        const auto p = std::pow(rho, gamma);
        const auto E = (p / (gamma - 1.0)) + 0.5 * rho * (u * u + v * v);

        return ComprFD::NodeState<T, 2>{.rho = rho, .rhoU = {rho * u, rho * v}, .rhoE = E};
    }

    // ====================================================================
    // 1. ISENTROPIC VORTEX PROPAGATION (Smooth, exact analytical solution)
    // ====================================================================
    template <>
    struct ProblemFactory2D<ProblemType2D::IsentropicVortex>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const VortexData& data)
        {
            // Define 2D problem traits using a 2D HLLC solver
            using AirTrait2D = ProblemTraits<PetscReal, 2, ComprFD::IdealGas<PetscReal>,
                                             ComprFD::RiemannScheme::HLLC>;

            BoundaryConditions<PetscReal, 2> bcs{};

            if (std::abs(conf.get_box_length(0) - conf.get_box_length(0uz)) >= 1e-12)
            {
                numPDE::println("Lx must meet the problem data to have a consistent Isoentropic "
                                "vortex problem!!");
                std::terminate();
            }
            if (std::abs(conf.get_box_length(1) - conf.get_box_length(1uz)) >= 1e-12)

            {
                numPDE::println("Lx must meet the problem data to have a consistent Isoentropic "
                                "vortex problem!!");
                std::terminate();
            }
            // Vortex tests are conventionally run on a fully periodic domain
            bcs.set_left_bc(BCType::Periodic);
            bcs.set_right_bc(BCType::Periodic);
            bcs.set_bottom_bc(BCType::Periodic);
            bcs.set_top_bc(BCType::Periodic);

            // Register initial conditions and analytical tracking
            bcs.initial_cond = [=](const PetscReal x, const PetscReal y)
            { return exact_vortex_solution<PetscReal>(0.0, x, y, data); };
            bcs.exact_sol = [=](const PetscReal t, const PetscReal x, const PetscReal y)
            { return exact_vortex_solution<PetscReal>(t, x, y, data); };

            return std::make_unique<Problem<AirTrait2D>>(conf, bcs, AirTrait2D::GasModel{});
        }
    };

    // ====================================================================
    // 2. RADIAL SOD EXPLOSION (Discontinuous multi-D shock test)
    // ====================================================================
    template <>
    struct ProblemFactory2D<ProblemType2D::CircularExplosion>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const SodProblemValues& data)
        {
            using GasType    = ComprFD::GasModels<PetscReal>;
            using AirTrait2D = ProblemTraits<PetscReal, 2, GasType,
                                             // ComprFD::RiemannScheme::HLLC>;
                                             ComprFD::RiemannScheme::Rusanov>;
            AirTrait2D::GasModel             gas_model{conf};
            BoundaryConditions<PetscReal, 2> bcs{};

            // Set non-periodic transmissive boundaries at domain boundaries
            bcs.set_left_bc(BCType::Outflow);
            bcs.set_right_bc(BCType::Outflow);
            bcs.set_bottom_bc(BCType::Outflow);
            bcs.set_top_bc(BCType::Outflow);

            const PetscReal gamma = 1.4;
            const PetscReal xc    = conf.get_box_length(0) / 2.0;
            const PetscReal yc    = conf.get_box_length(1) / 2.0;

            // Discontinuous radial profile
            auto explosion_initial = [=](const PetscReal x,
                                         const PetscReal y) -> ComprFD::NodeState<PetscReal, 2>
            {
                PetscReal dx = x - xc;
                PetscReal dy = y - yc;
                PetscReal r  = std::sqrt(dx * dx + dy * dy);

                PetscReal rho, pres;
                if (r < data.radius)
                {
                    rho  = data.rho_L;
                    pres = data.pres_L;
                }
                else
                {
                    rho  = data.rho_R;
                    pres = data.pres_R;
                }

                if (std::abs(data.vel_L) >= 1e-12 or std::abs(data.vel_R) >= 1e-12)
                    numPDE::println("Circular explosion assumes fluid at rest!! The velocities "
                                    "imposed are ignored");

                // Initially stagnant fluid field
                PetscReal u   = 0.0;
                PetscReal v   = 0.0;
                auto      vel = u + v;

                return ComprFD::NodeState<PetscReal, 2>{.rho  = rho,
                                                        .rhoU = {rho * u, rho * v},
                                                        .rhoE = gas_model.get_rhoE(rho, vel, pres)};
            };

            bcs.initial_cond = [=](const PetscReal x, const PetscReal y)
            { return explosion_initial(x, y); };

            // Return initial state for exact_sol signature to handle error reporting safely
            bcs.exact_sol = [=](const PetscReal, const PetscReal x, const PetscReal y)
            { return explosion_initial(x, y); };

            return std::make_unique<Problem<AirTrait2D>>(conf, bcs, gas_model);
        }
    };
    // ====================================================================
    // 3. ISENTROPIC VORTEX PROPAGATION WITH TIME-DEPENDENT DIRICHLET BCs
    // ====================================================================
    template <>
    struct ProblemFactory2D<ProblemType2D::IsentropicVortexDirichlet>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, const VortexData& data)
        {
            using AirTrait2D = ProblemTraits<PetscReal, 2, ComprFD::IdealGas<PetscReal>,
                                             ComprFD::RiemannScheme::HLLC>;

            BoundaryConditions<PetscReal, 2> bcs{};

            const PetscReal Lx = conf.get_box_length(0);
            const PetscReal Ly = conf.get_box_length(1);

            if (std::abs(Lx - data.l) >= 1e-12)
            {
                numPDE::println("Lx must meet the problem data to have a consistent Isoentropic "
                                "vortex problem!!");
                std::terminate();
            }
            if (std::abs(Ly - data.l) >= 1e-12)

            {
                numPDE::println("Lx must meet the problem data to have a consistent Isoentropic "
                                "vortex problem!!");
                std::terminate();
            }
            // Define the exact moving vortex profile
            auto exact_vortex = [=](const PetscReal time, const PetscReal x,
                                    const PetscReal y) -> ComprFD::NodeState<PetscReal, 2>
            { return exact_vortex_solution<PetscReal>(time, x, y, data); };

            // ── Fixed Dirichlet Boundary Registrations ──
            // Removed BCType parameter and matched explicit std::function signature

            // Left Face (x = 0.0, y varies)
            bcs.set_left_bc([=](PetscReal t, PetscReal /*x*/, PetscReal y,
                                const numPDE::ComprFD::NodeState<PetscReal, 2>& /*s_int*/)
                            { return exact_vortex(t, 0.0, y); });

            // Right Face (x = Lx, y varies)
            bcs.set_right_bc([=](PetscReal t, PetscReal /*x*/, PetscReal y,
                                 const numPDE::ComprFD::NodeState<PetscReal, 2>& /*s_int*/)
                             { return exact_vortex(t, Lx, y); });

            bcs.set_bottom_bc(BCType::Periodic);
            bcs.set_top_bc(BCType::Periodic);

            //
            //  // Bottom Face (x varies, y = 0.0)
            //  bcs.set_bottom_bc([=](PetscReal t, PetscReal x, PetscReal /*y*/,
            //                        const numPDE::ComprFD::NodeState<PetscReal, 2>& /*s_int*/)
            //                    { return exact_vortex(t, x, 0.0); });
            //
            //  // Top Face (x varies, y = Ly)
            //  bcs.set_top_bc([=](PetscReal t, PetscReal x, PetscReal /*y*/,
            //                     const numPDE::ComprFD::NodeState<PetscReal, 2>& /*s_int*/)
            //                 { return exact_vortex(t, x, Ly); });

            bcs.initial_cond = [=](const PetscReal x, const PetscReal y)
            { return exact_vortex(0.0, x, y); };
            bcs.exact_sol = [=](const PetscReal t, const PetscReal x, const PetscReal y)
            { return exact_vortex(t, x, y); };

            return std::make_unique<Problem<AirTrait2D>>(conf, bcs, AirTrait2D::GasModel{});
        }
    };

    // ====================================================================
    // 4. 2D SMOOTH DENSITY WAVE ADVECTION WITH DIRICHLET BCs
    // ====================================================================
    template <>
    struct ProblemFactory2D<ProblemType2D::DensityWave>
    {
        static auto create(const ConfigOptions<PetscReal>& conf, VortexData)
        {
            using AirTrait2D = ProblemTraits<PetscReal, 2, ComprFD::IdealGas<PetscReal>,
                                             ComprFD::RiemannScheme::HLLC>;

            BoundaryConditions<PetscReal, 2> bcs{};

            const PetscReal gamma = 1.4;
            const PetscReal Lx    = conf.get_box_length(0);
            const PetscReal Ly    = conf.get_box_length(1);

            // Constant uniform background fields (No pressure gradients!)
            const PetscReal u_0 = 1.0;
            const PetscReal v_0 = 0.5;
            const PetscReal p_0 = 1.0;

            // Define the exact advecting density profile
            auto exact_solution = [=](const PetscReal time, const PetscReal x,
                                      const PetscReal y) -> ComprFD::NodeState<PetscReal, 2>
            {
                // Displace coordinates linearly by advection background velocities
                PetscReal x_shifted = x - u_0 * time;
                PetscReal y_shifted = y - v_0 * time;

                // Handle safe periodic mapping over analytical box constraints
                PetscReal x_wrapped = std::fmod(std::fmod(x_shifted, Lx) + Lx, Lx);
                PetscReal y_wrapped = std::fmod(std::fmod(y_shifted, Ly) + Ly, Ly);

                // Smooth, continuous sine wave density perturbation
                PetscReal rho = 1.0 + 0.2 * std::sin(2.0 * M_PI * x_wrapped / Lx) *
                                          std::sin(2.0 * M_PI * y_wrapped / Ly);
                PetscReal u = u_0;
                PetscReal v = v_0;

                PetscReal internal_energy = p_0 / (gamma - 1.0);
                PetscReal kinetic_energy  = 0.5 * rho * (u * u + v * v);

                return ComprFD::NodeState<PetscReal, 2>{.rho  = rho,
                                                        .rhoU = {rho * u, rho * v},
                                                        .rhoE = internal_energy + kinetic_energy};
            };

            // ── Register Dirichlet Boundary Constraints (Single-Argument Overloads) ──

            bcs.set_left_bc(BCType::Periodic);
            bcs.set_right_bc(BCType::Periodic);
            bcs.set_bottom_bc(BCType::Periodic);
            bcs.set_top_bc(BCType::Periodic);
            bcs.initial_cond = [=](const PetscReal x, const PetscReal y)
            { return exact_solution(0.0, x, y); };
            bcs.exact_sol = [=](const PetscReal t, const PetscReal x, const PetscReal y)
            { return exact_solution(t, x, y); };

            return std::make_unique<Problem<AirTrait2D>>(conf, bcs, AirTrait2D::GasModel{});
        }
    };

} // namespace numPDE
