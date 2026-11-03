#include "../include/my_lib.hpp"
#include <boost/math/tools/roots.hpp>
#include <cmath>
#include <memory>
#include <tuple>
#include <utility>

using namespace numPDE;
using Real = PetscReal;

// ====================================================================
// CURVED WEDGE REGIME POLICY
// ====================================================================
template <std::floating_point T>
struct GeometryCurvedWedge2D
{
    T m_L_x{0.0}, m_L_y{0.0};
    T m_N_x{0.0}, m_N_y{0.0};

    GeometryCurvedWedge2D(const ConfigOptions<T>& opts)
    {
        m_L_x = static_cast<T>(opts.get_box_length(0uz));
        m_L_y = static_cast<T>(opts.get_box_length(1uz));
        m_N_x = static_cast<T>(opts.get_n_elems(0uz));
        m_N_y = static_cast<T>(opts.get_n_elems(1uz));
    }

    // Simple parabolic wedge starting at the middle of the domain
    // C1 Continuous Wedge: Flat -> Parabolic Fillet -> Straight Line
    std::pair<T, T> get_wedge_height_and_slope(T x) const
    {
        // --- WEDGE PARAMETERS ---
        const T x_start     = 0.3 * m_L_x; // Where the curve begins
        const T x_straight  = 0.5 * m_L_x; // Where the pure straight line begins
        const T wedge_slope = 0.2;         // Final slope of the wedge (~11.3 degrees)

        // Mathematically compute the parabola coefficient 'A' so that the
        // slope exactly matches 'wedge_slope' when x reaches 'x_straight'.
        // h'(x) = 2*A*(x - x_start)  =>  A = wedge_slope / (2 * dx)
        const T A = wedge_slope / (2.0 * (x_straight - x_start));

        if (x <= x_start)
        {
            // Zone 1: Flat incoming wall
            return {0.0, 0.0};
        }
        else if (x <= x_straight)
        {
            // Zone 2: Parabolic transition (fillet)
            const T dx    = x - x_start;
            T       h_x   = A * dx * dx;
            T       dh_dx = 2.0 * A * dx;
            return {-h_x, -dh_dx};
        }
        else
        {
            // Zone 3: Straight line wedge
            // Base height at the end of the parabolic transition
            const T dx_trans = x_straight - x_start;
            const T h_base   = A * dx_trans * dx_trans;

            // Linear projection
            const T dx    = x - x_straight;
            T       h_x   = h_base + wedge_slope * dx;
            T       dh_dx = wedge_slope; // Constant slope
            return {-h_x, -dh_dx};
        }
    }

    auto get_physical_coords(T csi, T eta) const
    {
        const T x_phys      = m_L_x * csi;
        const auto [h_x, _] = get_wedge_height_and_slope(x_phys);

        // Transfinite Interpolation bounds mapping
        // Bottom wall is h_x, Top wall is flat at m_L_y
        const T y_phys = h_x * (1.0 - eta) + m_L_y * eta;

        return std::make_tuple(x_phys, y_phys);
    }

    GeomJacobian2D<T> get_metrics(T csi, T eta) const
    {
        const T x_phys          = m_L_x * csi;
        const auto [h_x, dh_dx] = get_wedge_height_and_slope(x_phys);

        // Exact analytical derivatives of get_physical_coords
        const T dx_dcsi = m_L_x;
        const T dx_deta = 0.0;
        const T dy_dcsi = dh_dx * m_L_x * (1.0 - eta);
        const T dy_deta = m_L_y - h_x;

        return GeomJacobian2D<T>{
            .dx_dcsi = dx_dcsi, .dy_dcsi = dy_dcsi, .dx_deta = dx_deta, .dy_deta = dy_deta};
    }

    FaceMetrics2D<T> get_x_face_metrics(T /*csi*/, T /*eta*/) const
    {
        // Vertical faces: The tangent is (dx_deta, dy_deta).
        // Normal points strictly in the +/- X direction.
        return FaceMetrics2D<T>{.nx = 1.0, .ny = 0.0};
    }

    FaceMetrics2D<T> get_y_face_metrics(T csi, T eta) const
    {
        const T x_phys        = m_L_x * csi;
        const auto [_, dh_dx] = get_wedge_height_and_slope(x_phys);

        // The tangent vector to the constant-eta lines is (dx_dcsi, dy_dcsi)
        // dx_dcsi = m_L_x
        // dy_dcsi = dh_dx * m_L_x * (1.0 - eta)
        // An orthogonal vector (-dy, dx) gives the unnormalized normal direction.

        T unscaled_nx = -dh_dx * m_L_x * (1.0 - eta);
        T unscaled_ny = m_L_x;

        T inv_norm = 1.0 / std::sqrt(unscaled_nx * unscaled_nx + unscaled_ny * unscaled_ny);
        return FaceMetrics2D<T>{.nx = unscaled_nx * inv_norm, .ny = unscaled_ny * inv_norm};
    }
};

// ====================================================================
// MAIN APPLICATION
// ====================================================================
int main(int argc, char* argv[])
{
    // Initialize PETSc environmental frameworks and process incoming runtime choices
    Parser      parser(&argc, &argv);
    const auto& conf_p = parser.get_options();

    auto conf{conf_p};

    // Standard square domain for testing the wedge
    const auto L_x = 1.0;
    const auto L_y = 1.0;

    conf.set_box_length(L_x, 0);
    conf.set_box_length(L_y, 1);

    // ====================================================================
    // 2. COMPILE-TIME PROBLEM TRAITS SELECTION
    // ====================================================================
    // Hook up the new curved wedge geometry policy
    using WedgeGeometry = GeometryCurvedWedge2D<Real>;

    using WedgeTraits =
        ProblemTraits<Real, 2, ComprFD::IdealGas<Real>, ComprFD::RiemannScheme::Rusanov,
                      CoordinateSystem::Cartesian, WedgeGeometry>;

    // ====================================================================
    // 3. PHYSICAL BOUNDARY CONDITIONS LAYOUT
    // ====================================================================
    ComprFD::IdealGas<Real> gas_model{};
    constexpr Real          gamma = 1.4;

    const Real rho_in  = 1.0;
    const Real u_in    = 2.0; // Supersonic entry (Mach > 1) to generate a nice shock off the wedge
    const Real v_in    = 0.0;
    const Real pres_in = 1.0;

    const auto left_state =
        ComprFD::NodeState<Real, 2>{.rho  = rho_in,
                                    .rhoU = {rho_in * u_in, rho_in * v_in},
                                    .rhoE = gas_model.get_rhoE(rho_in, u_in, pres_in)};
    const auto zero_state =
        ComprFD::NodeState<Real, 2>{.rho  = rho_in,
                                    .rhoU = {rho_in * v_in, rho_in * v_in},
                                    .rhoE = gas_model.get_rhoE(rho_in, v_in, pres_in)};
    /*
    // ── Supersonic Inflow (Real SI Units) ──
    const Real rho_in  = 1.225;       // Standard sea-level density (kg/m^3)
    const Real pres_in = 101325.0;    // Standard atmospheric pressure (Pa = Kg m / s^2)

    // Explicitly define your desired Mach number
    const Real Mach_in = 2.5;

    // Calculate the physical speed of sound (~340 m/s) and velocity (~850 m/s)
    const Real c_sound = std::sqrt(gamma * pres_in / rho_in);
    const Real u_in    = Mach_in * c_sound;
    const Real v_in    = 0.0;

    // The energy calculation stays exactly the same, but the numbers will be massive!
    const Real internal_energy_in = pres_in / (gamma - 1.0);
    const Real kinetic_energy_in  = 0.5 * rho_in * (u_in * u_in + v_in * v_in);

    const auto left_state =
        ComprFD::NodeState<Real, 2>{.rho  = rho_in,
                                    .rhoU = {rho_in * u_in, rho_in * v_in},
                                    .rhoE = internal_energy_in + kinetic_energy_in};
                                    */

    BoundaryConditions<Real, 2> bcs{};

    // Bottom is our curved wedge wall. Top is a solid flat wall.
    // 1. Instantiate the geometry to query normals
    WedgeGeometry geom(conf);

    // 2. Custom Curvilinear Slip-Wall
    bcs.set_bottom_bc(
        [=](Real /*t*/, Real x, Real /*y*/, const auto& interior)
        {
            auto exterior = interior;

            Real csi          = x / L_x;
            auto face_metrics = geom.get_y_face_metrics(csi, 0.0);
            Real nx           = face_metrics.nx;
            Real ny           = face_metrics.ny;

            Real u_in = interior.rhoU[0] / interior.rho;
            Real v_in = interior.rhoU[1] / interior.rho;

            Real vn_in = u_in * nx + v_in * ny;

            Real u_out = u_in - 2.0 * vn_in * nx;
            Real v_out = v_in - 2.0 * vn_in * ny;

            exterior.rhoU[0] = exterior.rho * u_out;
            exterior.rhoU[1] = exterior.rho * v_out;

            return exterior;
        });

    bcs.set_top_bc([=](Real /*t*/, Real /*x*/, Real /*y*/, const auto& /*interior*/)
                   { return left_state; });

    // Supersonic inflow Dirichlet
    bcs.set_left_bc([=](Real /*t*/, Real /*x*/, Real /*y*/, const auto& /*interior*/)
                    { return left_state; });

    // Supersonic outflow (zero-gradient extrapolation)
    bcs.set_right_bc(BCType::Outflow);

    // Uniform initial condition matching the inflow
    bcs.initial_cond = [=](Real /*x*/, Real /*y*/) -> ComprFD::NodeState<Real, 2>
    { return left_state; };

    // ====================================================================
    // EXACT ANALYTICAL SOLUTION (METHOD OF CHARACTERISTICS)
    // ====================================================================
    // 1. Analytical Stagnation & Freestream Constants
    const Real c_in = std::sqrt(gamma * pres_in / rho_in);
    const Real M_in = u_in / c_in;
    
    // Lambda for Prandtl-Meyer Angle (returns radians)
    auto prandtl_meyer = [](Real M) {
        constexpr Real g = 1.4;
        return std::sqrt((g+1.0)/(g-1.0)) * std::atan(std::sqrt((g-1.0)/(g+1.0)*(M*M-1.0))) 
               - std::atan(std::sqrt(M*M-1.0));
    };
    
    const Real nu_in = prandtl_meyer(M_in);
    const Real P0 = pres_in * std::pow(1.0 + 0.5*(gamma-1.0)*M_in*M_in, gamma/(gamma-1.0));
    const Real rho0 = rho_in * std::pow(1.0 + 0.5*(gamma-1.0)*M_in*M_in, 1.0/(gamma-1.0));

    // Lambda to invert PM function via Boost Bisection
    auto inv_prandtl_meyer = [prandtl_meyer](Real nu_target) {
        auto func = [nu_target, prandtl_meyer](Real M) { return prandtl_meyer(M) - nu_target; };
        auto tol = [](Real l, Real r) { return std::abs(r - l) <= 1e-8; };
        boost::uintmax_t max_iter = 100;
        // Mach number must be >= 1.0. Bounding upper limit at Mach 20.
        auto r = boost::math::tools::bisect(func, static_cast<Real>(1.0001), static_cast<Real>(20.0), tol, max_iter);
        return (r.first + r.second) / 2.0;
    };

    bcs.exact_sol = [=](Real /*t*/, Real x, Real y) -> ComprFD::NodeState<Real, 2> {
        const Real x_start = 0.3 * L_x;
        const Real x_straight = 0.5 * L_x;
        const Real mu_in = std::asin(1.0 / M_in);
        
        // Characteristic equation of the very first Mach wave
        const Real y_first_wave = (x - x_start) * std::tan(mu_in);

        // Region 1: Freestream (Upstream of expansion fan)
        if (x <= x_start || y >= y_first_wave) {
            return left_state;
        }

        // Region 2 & 3: Find the origin wall coordinate (x_w) of the characteristic line passing through (x,y)
        auto characteristic_eq = [&](Real x_w) {
            auto [minus_h, minus_dh_dx] = geom.get_wedge_height_and_slope(x_w);
            Real y_w = minus_h;
            Real theta_w = std::atan(minus_dh_dx);         // Flow turning angle (negative)
            Real M_w = inv_prandtl_meyer(nu_in - theta_w); // nu increases as flow expands
            Real mu_w = std::asin(1.0 / M_w);
            
            Real alpha = theta_w + mu_w; // Angle of the left-running characteristic
            return y_w + (x - x_w) * std::tan(alpha) - y;
        };

        Real x_root = x_start;
        Real f_last = characteristic_eq(x_straight);

        if (f_last > 0.0) {
            // Region 3: Downstream of the fan. The flow is uniform and parallel to the final wedge angle.
            x_root = x_straight;
        } else {
            // Region 2: Inside the Expansion Fan. 
            // Solve implicitly for the origin wall point x_w using Boost Bisection.
            auto tol = [](Real l, Real r) { return std::abs(r - l) <= 1e-6; };
            boost::uintmax_t max_iter = 100;
            auto r = boost::math::tools::bisect(characteristic_eq, x_start, x_straight, tol, max_iter);
            x_root = (r.first + r.second) / 2.0;
        }

        // Compute local exact primitive variables based on the wave origin state
        auto [minus_h, minus_dh_dx] = geom.get_wedge_height_and_slope(x_root);
        Real theta_exact = std::atan(minus_dh_dx);
        Real M_exact = inv_prandtl_meyer(nu_in - theta_exact);
        
        Real T_ratio = 1.0 / (1.0 + 0.5 * (gamma - 1.0) * M_exact * M_exact);
        Real P_exact = P0 * std::pow(T_ratio, gamma / (gamma - 1.0));
        Real rho_exact = rho0 * std::pow(T_ratio, 1.0 / (gamma - 1.0));
        
        Real c_exact = std::sqrt(gamma * P_exact / rho_exact);
        Real V_exact = M_exact * c_exact;
        
        Real u_exact = V_exact * std::cos(theta_exact);
        Real v_exact = V_exact * std::sin(theta_exact);
        Real rhoE_exact = P_exact / (gamma - 1.0) + 0.5 * rho_exact * (u_exact * u_exact + v_exact * v_exact);

        return ComprFD::NodeState<Real, 2>{
            .rho = rho_exact,
            .rhoU = {rho_exact * u_exact, rho_exact * v_exact},
            .rhoE = rhoE_exact
        };
    };

    // ====================================================================
    // 5. SOLVER DISPATCH SEQUENCE
    // ====================================================================
    numPDE::println("Assembling Curved Wedge Mesh Topology and Injected Policies...");

    // Instantiate problem structure bypassing factory wrappers entirely
    auto prob = std::make_unique<Problem<WedgeTraits>>(conf, bcs);

    numPDE::println("Executing Wedge Flow Solver Loop...");
    prob->solve();
    auto errs = prob->compute_error();
    numPDE::println("Error logs");
    numPDE::println("Rho : {:3e}", errs[0]);
    numPDE::println("RhoU : {:3e}", errs[1]);
    numPDE::println("RhoV : {:3e}", errs[2]);
    numPDE::println("RhoE : {:3e}", errs[3]);

    // Finalize performance tracking and clear memory allocations
    numPDE::println("Simulation Run Complete.");
    return 0;
}
