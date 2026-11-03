#pragma once
#include "cached_computations.hpp"
#include "macro_and_libc.hpp"
#include <print>

namespace numPDE
{
    enum class JacobianType
    {
        FD = 0,
        ComplexFD,
    };

    enum class FilterTypes
    {
        None = 0,
        Spectral,
        WENO,
        ArtificialViscosity
    };

    enum class ConstitutiveLaw
    {
        IdealGas,
        VanDerWaals
    };

    enum class PrecondJacMat
    {
        None = 0,
        Coloring,
        p_MG,
        FD
    };

    // Needed array of strings to handle the PETSc options
    // Indices MUST match with the ones of the enum
    static const char* JacobianTypes[] = {"fd", "complexfd", "JacobianType", "JAC_", NULL};

    static const char* PrecondJacMats[] = {"none",          "coloring", "p_mg", "fd",
                                           "PrecondJacMat", "PRE_",     NULL};

    static const char* ConstitutiveLaws[] = {"ideal", "van_der_waals", "ConstitutiveLaw",
                                             "GAS_MOD_", NULL};
    struct SpectFiltVals
    {
        std::size_t n_cut_off{2};
        std::size_t filter_strength{32};
        PetscReal   alpha{36};
        bool        apply_filter{false};
    };

    struct WenoConstants
    {
        PetscReal gamma_elem{0.998};
        bool      apply_limiter{false};
    };

    struct ShockSensorConstants
    {
        PetscReal theta{1.0};
        PetscReal threshold{0.8};
    };

    struct GasModelConstants
    {

        ConstitutiveLaw c_law{ConstitutiveLaw::IdealGas};
        PetscReal       gamma{1.4};
        PetscReal       R_spec{287.05};  // Air
        PetscReal       VdW_a{161.94};   // [Pa * m^6 / kg^2]
        PetscReal       VdW_b{0.001265}; // [m^3 / kg]
    };

    template <std::floating_point T>
    class ConfigOptions
    {
      private:
        static constexpr std::size_t    n_dims = 3;
        ShockSensorConstants            m_shock_sens{};
        GasModelConstants               m_gas_csts{};
        SpectFiltVals                   m_spec_filt{};
        bool                            post_ts{true};
        bool                            condition_system{false};
        T                               Radau_corr{};
        std::array<std::size_t, n_dims> n_elems{};
        std::array<std::size_t, n_dims> poly_deg{};
        std::array<T, n_dims>           box_length{};
        JacobianType                    j_type{JacobianType::FD};
        PrecondJacMat                   pre_j_type{PrecondJacMat::None};
        WenoConstants                   weno_csts{};

      public:
        /**** SETTERS *****/
        void set_shock_sens(const ShockSensorConstants& v) { this->m_shock_sens = v; };
        void set_gas_model(const GasModelConstants& g_mod) { this->m_gas_csts = g_mod; };
        void set_weno_params(const WenoConstants v) { this->weno_csts = v; };
        void set_do_post_ts(const bool v) { this->post_ts = v; };
        void set_jac_type(const JacobianType jac) { this->j_type = jac; };
        void set_pre_type(const PrecondJacMat jac) { this->pre_j_type = jac; };
        void set_rad_corr(const T v) { this->Radau_corr = v; };
        void set_filt_vals(SpectFiltVals v) { m_spec_filt = v; };
        template <std::integral I>
        void set_n_elems(const I v)
        {
            std::fill(n_elems.begin(), n_elems.end(), v);
        };
        template <std::integral I>
        void set_poly_deg(const I v)
        {
            std::ranges::fill(poly_deg, v);
        };
        void set_box_length(const T v) { std::ranges::fill(box_length, v); };

        template <std::integral I>
        void set_n_elems(const I v, const I i)
        {
            this->n_elems[i] = v;
        };
        template <std::integral I>
        void set_poly_deg(const I v, const I i)
        {
            this->poly_deg[i] = v;
        };
        template <std::integral I>
        void set_box_length(const T v, const I i)
        {
            this->box_length[i] = v;
        };

        void set_cond_sys(const bool v) { this->condition_system = v; };
        void set_apply_filter(const bool v) { this->m_spec_filt.apply_filter = v; };

        /**** GETTERS *****/
        auto get_do_post_ts() const { return this->post_ts; };
        auto get_jac_type() const { return this->j_type; };
        auto get_pre_type() const { return this->pre_j_type; };
        auto get_rad_corr() const { return this->Radau_corr; };
        auto get_cond_sys() const { return this->condition_system; };

        auto get_apply_filter() const { return this->m_spec_filt.apply_filter; };
        auto get_filt_vals() const { return m_spec_filt; };

        auto get_weno_params() const { return this->weno_csts; };
        auto get_weno_gamma() const { return this->weno_csts.gamma_elem; };
        auto get_apply_weno() const { return this->weno_csts.apply_limiter; };

        auto get_gas_model() const { return this->m_gas_csts; };

        auto get_shock_sens() const { return this->m_shock_sens; };

        template <std::integral I>
        const auto& get_box_length(const I i = 0) const
        {
            return this->box_length[i];
        };
        template <std::integral I>
        const auto& get_n_elems(const I i = 0) const
        {
            return this->n_elems[i];
        };
        template <std::integral I>
        const auto& get_poly_deg(const I i = 0) const
        {
            return this->poly_deg[i];
        };

        void print() const
        {
            std::println("{:=^50}", " GLL configuration ");
            std::println("Corr factor: {}", this->Radau_corr);
            std::println("N elems    : {}", this->n_elems[0]);
            std::println("Poly degree: {}", this->poly_deg[0]);
            std::println("Dom Length : {}", this->box_length[0]);
            std::println("Jac routine: {}", JacobianTypes[static_cast<int>(this->j_type)]);
            std::println("Precond    : {}", PrecondJacMats[static_cast<int>(this->pre_j_type)]);
            std::println("Change var : {}", this->condition_system);
            std::println("{:=^50}", " FILTER configuration ");
            std::println("Is active  : {}", this->m_spec_filt.apply_filter);
            std::println("Strength   : {}", this->m_spec_filt.filter_strength);
            std::println("Cut-off    : {}", this->m_spec_filt.n_cut_off);
            std::println("Alpha      : {}", this->m_spec_filt.alpha);
            std::println("{:=^50}", " WENO configuration ");
            std::println("Is active  : {}", this->weno_csts.apply_limiter);
            std::println("Gamma      : {}", this->weno_csts.gamma_elem);
            std::println("{:=^50}", " SHOCK SENSOR configuration ");
            std::println("Theta      : {}", this->m_shock_sens.theta);
            std::println("Epsilon    : {}", this->m_shock_sens.threshold);
            std::println("{:=^50}", "");
        };
    };

    struct AcousticWaveData
    {
        PetscReal epsilon = 0.01;
        PetscReal alpha   = 10.0;
        PetscReal u_0     = 1;
        PetscReal rho_0   = 10;
        PetscReal p_0     = 10;

        void print() const
        {
            std::println("{:=^50}", " AcousticWave configuration ");
            std::println("epsilon : {}", epsilon);
            std::println("alpha   : {}", alpha);
            std::println("u_0     : {}", u_0);
            std::println("rho_0   : {}", rho_0);
            std::println("p_0     : {}", p_0);
            return;
        }
    };

    struct SodProblemValues
    {
        PetscReal radius = 0;
        PetscReal rho_R, pres_R, vel_R;
        PetscReal rho_L, pres_L, vel_L;
    };

    class Parser
    {
      private:
        ConfigOptions<PetscReal> opts{};

      public:
        Parser(int* argc, char** argv[])
        {
            static constexpr const char* help_message =
                "Work in progress\n.Possible usage: mpirun ./build/test/euler_1d -n_elems 11 "
                "-poly_deg 12  -box_l 10"
                "-radau_corr 1.30  -jac_type complexfd -pre_type p_mg\n"
                "Some other helpful options fropm PETSc are:\n"
                "-ts_monitor_error -ts_dt 0.01 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type none "
                "-snes_converged_reason   -snes_max_funcs unlimited -snes_max_it 1000000   "
                "-snes_view";
            PetscInitialize(argc, argv, /*Filename*/ nullptr, help_message);
            parse_options(opts, true);
        };
        Parser(Parser&&)                 = delete;
        Parser(const Parser&)            = delete;
        Parser& operator=(Parser&&)      = delete;
        Parser& operator=(const Parser&) = delete;
        ~Parser() { PetscFinalize(); };

        const auto&      get_options() { return this->opts; };
        AcousticWaveData parse_acoust_wave_data(bool print = false) const;
        SodProblemValues parse_sod_shock_vals(bool print = false) const;

      private:
        PetscErrorCode parse_options(ConfigOptions<PetscReal>& opts, bool print = false);
    };

    PetscErrorCode Parser::parse_options(ConfigOptions<PetscReal>& opts, const bool print_opts)
    {
        PetscFunctionBeginUser;

        PetscInt  n_elems{10};
        PetscInt  poly_deg{3};
        PetscReal radau_corr{0.0};
        PetscReal box_l{1.0};
        // Setup the Jacobian options
        JacobianType  jac_default = JacobianType::FD;    // Default
        PrecondJacMat pre_default = PrecondJacMat::None; // Default
        PetscBool     set_jac{false}, set_pre{false};
        PetscBool     post_ts{true};
        PetscBool     condition_system{false};

        // Filter options
        SpectFiltVals spec_vals{};

        WenoConstants weno_vals{};

        GasModelConstants g_mod_csts{};

        ShockSensorConstants shock_csts{};
        PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Solver options", NULL);
        // Shock sensor options
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-shock_sens_activation_value",
                                      &shock_csts.threshold, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-shock_sens_theta", &shock_csts.theta, NULL));
        // Gas models constants
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-gas_mod_R", &g_mod_csts.R_spec, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-gas_mod_gamma", &g_mod_csts.gamma, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-gas_mod_VdW_a", &g_mod_csts.VdW_a, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-gas_mod_VdW_b", &g_mod_csts.VdW_b, NULL));
        PetscBool set_g_mod{false};
        PetscOptionsEnum(
            "-gas_mod_law",                                  // Command line flag
            "Choose a constitutive gas law",                 // Description
            NULL,                                            // Source file (optional)
            ConstitutiveLaws,                                // The lookup table
            static_cast<PetscEnum>(g_mod_csts.c_law),        // Current/Default value
            reinterpret_cast<PetscEnum*>(&g_mod_csts.c_law), // Where to store the result
            &set_g_mod);                                     // Was it actually changed?

        PetscCall(PetscOptionsGetBool(NULL, NULL, "-apply_weno", &weno_vals.apply_limiter, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-weno_gamma", &weno_vals.gamma_elem, NULL));

        PetscInt __fil_str = {(PetscInt) spec_vals.filter_strength};
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-filt_strength", &__fil_str, NULL));

        spec_vals.filter_strength = static_cast<std::size_t>(__fil_str);

        PetscInt __n_cut_off = {(PetscInt) spec_vals.n_cut_off};
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-filt_cut_off_deg", &__n_cut_off, NULL));
        spec_vals.n_cut_off = static_cast<std::size_t>(__n_cut_off);

        PetscCall(PetscOptionsGetBool(NULL, NULL, "-apply_filter", &spec_vals.apply_filter, NULL));
        PetscCall(
            PetscOptionsGetBool(NULL, NULL, "-solve_positive_preserving", &condition_system, NULL));
        PetscCall(PetscOptionsGetBool(NULL, NULL, "-do_post_ts", &post_ts, NULL));
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_elems", &n_elems, NULL));
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-poly_deg", &poly_deg, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-radau_corr", &radau_corr, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-box_l", &box_l, NULL));
        PetscOptionsEnum("-jac_type",                                     // Command line flag
                         "Choose an evaluation routine for the jacobian", // Description
                         NULL,                                            // Source file (optional)
                         JacobianTypes,                                   // The lookup table
                         static_cast<PetscEnum>(jac_default),             // Current/Default value
                         reinterpret_cast<PetscEnum*>(&jac_default), // Where to store the result
                         &set_jac);                                  // Was it actually changed?

        PetscOptionsEnum("-pre_type", // Command line flag
                         "Choose an evaluation routine for the preconditioning", // Description
                         NULL,                                       // Source file (optional)
                         PrecondJacMats,                             // The lookup table
                         static_cast<PetscEnum>(pre_default),        // Current/Default value
                         reinterpret_cast<PetscEnum*>(&pre_default), // Where to store the result
                         &set_pre);                                  // Was it actually changed?
        PetscOptionsEnd();

        // Enforce the options
        opts.set_shock_sens(shock_csts);
        opts.set_filt_vals(spec_vals);
        opts.set_weno_params(weno_vals);
        opts.set_rad_corr(radau_corr);
        opts.set_n_elems(n_elems);
        opts.set_poly_deg(poly_deg);
        opts.set_jac_type(jac_default);
        opts.set_pre_type(pre_default);
        opts.set_box_length(box_l);
        opts.set_do_post_ts(post_ts);
        opts.set_cond_sys(condition_system);
        opts.set_gas_model(g_mod_csts);

        if (is_rank_0() and print_opts) opts.print();

        PetscFunctionReturn(PETSC_SUCCESS);
    }
    SodProblemValues Parser::parse_sod_shock_vals(const bool print_data) const
    {
        SodProblemValues data{};

        PetscOptionsGetReal(NULL, NULL, "-rho_L", &data.rho_L, NULL);
        PetscOptionsGetReal(NULL, NULL, "-vel_L", &data.vel_L, NULL);
        PetscOptionsGetReal(NULL, NULL, "-pres_L", &data.pres_L, NULL);
        PetscOptionsGetReal(NULL, NULL, "-rho_R", &data.rho_R, NULL);
        PetscOptionsGetReal(NULL, NULL, "-vel_R", &data.vel_R, NULL);
        PetscOptionsGetReal(NULL, NULL, "-pres_R", &data.pres_R, NULL);
        PetscOptionsGetReal(NULL, NULL, "-radius", &data.radius, NULL);
        if (is_rank_0() and print_data)
        {
            std::println("Right state:");
            std::println("\trho : {:} [kg/m^3], vel : {:} [m/s], pressure {:}[Pa || N/m^2]",
                         data.rho_R, data.vel_R, data.pres_R);
            std::println("Left state:");
            std::println("\trho : {:} [kg/m^3], vel : {:} [m/s], pressure {:}[Pa || N/m^2]",
                         data.rho_L, data.vel_L, data.pres_L);
        }
        return data;
    };
    AcousticWaveData Parser::parse_acoust_wave_data(const bool print_data) const
    {
        AcousticWaveData data{};
        PetscOptionsGetReal(NULL, NULL, "-epsilon", &data.epsilon, NULL);
        PetscOptionsGetReal(NULL, NULL, "-alpha", &data.alpha, NULL);
        PetscOptionsGetReal(NULL, NULL, "-u_0", &data.u_0, NULL);
        PetscOptionsGetReal(NULL, NULL, "-rho_0", &data.rho_0, NULL);
        PetscOptionsGetReal(NULL, NULL, "-p_0", &data.p_0, NULL);
        if (is_rank_0() and print_data) data.print();
        return data;
    }
} // namespace numPDE
