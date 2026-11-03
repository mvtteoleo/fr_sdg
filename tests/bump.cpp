#include "../include/my_lib.hpp"
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
    // Zone boundaries (fractions of domain length), in order:
    //   [0, x_start]                : flat incoming wall
    //   (x_start, x_c_straight]     : parabolic compression blend
    //   (x_c_straight, x_e_start]   : straight compression ramp -> oblique shock,
    //                                 exact theta-beta-M reference valid here
    //   (x_e_start, x_e_end]        : parabolic expansion blend -> Prandtl-Meyer fan
    //   (x_e_end, L_x]              : straight expansion ramp -> second exact
    //                                 (Prandtl-Meyer) reference valid here
    const T x_start    = 0.15 * m_L_x;
    const T x_c_straight = 0.30 * m_L_x; // end of compression parabola
    const T x_e_start  = 0.55 * m_L_x;   // end of straight shock ramp / start of fan blend
    const T x_e_end    = 0.70 * m_L_x;   // end of expansion parabola

    const T compress_slope = 0.2;  // ~11.3 deg, unchanged oblique-shock deflection
    const T expand_slope   = -0.2; // ~-11.3 deg, symmetric expansion deflection
                                     // (keep |expand_slope| below the Prandtl-Meyer
                                     // max turning angle for the chosen M_inf, or the
                                     // flow will try to form a vacuum rather than a
                                     // clean fan)

    // --- Zone 2: parabolic compression blend ---
    // h'(x) = 2*A_c*(x - x_start)  =>  A_c = compress_slope / (2*dx)
    const T A_c = compress_slope / (2.0 * (x_c_straight - x_start));

    // Height and slope at the end of the compression parabola (needed by
    // every later zone to stay C1-continuous).
    const T dx_c   = x_c_straight - x_start;
    const T h_c_end = A_c * dx_c * dx_c; // = compress_slope * dx_c / 2

    if (x <= x_start)
    {
        // Zone 1: Flat incoming wall
        return {0.0, 0.0};
    }
    else if (x <= x_c_straight)
    {
        // Zone 2: Parabolic compression transition (fillet)
        const T dx     = x - x_start;
        const T h_x    = A_c * dx * dx;
        const T dh_dx  = 2.0 * A_c * dx;
        return {-h_x, -dh_dx};
    }
    else if (x <= x_e_start)
    {
        // Zone 3: Straight compression ramp -- the oblique shock forms and
        // stabilizes here. theta-beta-M gives the exact post-shock state
        // analytically anywhere strictly inside this zone.
        const T dx    = x - x_c_straight;
        const T h_x   = h_c_end + compress_slope * dx;
        const T dh_dx = compress_slope; // constant slope
        return {-h_x, -dh_dx};
    }

    // Height/slope at the start of the expansion blend -- continuity anchor
    // for both Zone 4 and Zone 5.
    const T dx_run    = x_e_start - x_c_straight;
    const T h_e_start = h_c_end + compress_slope * dx_run;

    // --- Zone 4: parabolic expansion blend ---
    // Mirrors the Zone-2 construction, but starts from a nonzero slope
    // (compress_slope) instead of 0, and blends down to expand_slope.
    // h'(x) = compress_slope + 2*A_e*(x - x_e_start)
    //   =>  A_e = (expand_slope - compress_slope) / (2*dx_e)
    const T dx_e = x_e_end - x_e_start;
    const T A_e  = (expand_slope - compress_slope) / (2.0 * dx_e);

    if (x <= x_e_end)
    {
        // Zone 4: Parabolic expansion transition -- this is where the
        // centered Prandtl-Meyer fan originates. M(x) along this zone
        // follows the simple-wave relation nu(M_inf) - |theta(x)| = nu(M(x)),
        // with theta(x) = atan(dh_dx) the local wall angle below.
        const T dx    = x - x_e_start;
        const T h_x   = h_e_start + compress_slope * dx + A_e * dx * dx;
        const T dh_dx = compress_slope + 2.0 * A_e * dx;
        return {-h_x, -dh_dx};
    }
    else
    {
        // Zone 5: Straight expansion ramp -- uniform post-fan state,
        // exact via Prandtl-Meyer theory (second closed-form reference).
        const T h_e_end = h_e_start + compress_slope * dx_e + A_e * dx_e * dx_e;
        const T dx       = x - x_e_end;
        const T h_x      = h_e_end + expand_slope * dx;
        const T dh_dx    = expand_slope; // constant slope
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

    bcs.exact_sol = [=](Real /*t*/, Real /*x*/, Real /*y*/) -> ComprFD::NodeState<Real, 2>
    { return left_state; };

    // ====================================================================
    // 5. SOLVER DISPATCH SEQUENCE
    // ====================================================================
    numPDE::println("Assembling Curved Wedge Mesh Topology and Injected Policies...");

    // Instantiate problem structure bypassing factory wrappers entirely
    auto prob = std::make_unique<Problem<WedgeTraits>>(conf, bcs);

    numPDE::println("Executing Wedge Flow Solver Loop...");
    prob->solve();

    // Finalize performance tracking and clear memory allocations
    numPDE::println("Simulation Run Complete.");
    return 0;
}
