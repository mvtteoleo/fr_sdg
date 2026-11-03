#pragma once
#include "macro_and_libc.hpp"
#include "parser.hpp"
#include <cmath>
#include <concepts>
#include <tuple>

namespace numPDE
{
    // ====================================================================
    // GEOMETRIC METRICS CONTAINERS
    // ====================================================================
    // Raw forward Jacobian of the PURE PHYSICAL SHAPE MAP:  (csi, eta) -> (x, y),
    // csi, eta ∈ [0,1] being the computational domain coordinates.
    // No element-count / reference-element scaling is folded in here — that
    // composition happens one level up, in PhysicsEvaluator, which is the only
    // place that knows about GLL reference coordinates r,s ∈ [-1,1] and the
    // number of elements per axis.
    template <std::floating_point T>
    struct GeomJacobian2D
    {
        T dx_dcsi{1.0}, dy_dcsi{0.0};
        T dx_deta{0.0}, dy_deta{1.0};

        [[nodiscard]] constexpr T detJ() const { return dx_dcsi * dy_deta - dx_deta * dy_dcsi; }
    };

    // Composite metrics at a GLL node: the CONTRAVARIANT metric tensor of the
    // full map (r, s) -> (x, y), i.e. after chaining the physical shape map
    // through the per-element affine reference-element map. Built by
    // PhysicsEvaluator, not by GeometryModel.
    template <std::floating_point T>
    struct VolumeMetrics2D
    {
        T xi_x{1.0}, xi_y{0.0};
        T eta_x{0.0}, eta_y{1.0};
        T detJ{1.0}; // The full physical volume determinant |J| (Replaces inv_jac)

        // ── Helper methods for Interface Scaling (Contravariant Gradient Magnitudes) ──
        [[nodiscard]] constexpr T norm_grad_xi() const
        {
            return std::sqrt(xi_x * xi_x + xi_y * xi_y);
        }

        [[nodiscard]] constexpr T norm_grad_eta() const
        {
            return std::sqrt(eta_x * eta_x + eta_y * eta_y);
        }
    };

    template <std::floating_point T>
    struct FaceMetrics2D
    {
        T nx{1.0}; // Unit normal component X pointing toward increasing index direction
        T ny{0.0}; // Unit normal component Y pointing toward increasing index direction
    };

    // ====================================================================
    // GEOMETRY MODEL CONCEPT (Pure Mapping Abstraction)
    // ====================================================================
    template <typename GeomMod, typename T>
    concept GeometryModelConc = requires(const GeomMod geom, T csi, T eta) {
        { geom.get_physical_coords(csi, eta) } -> std::convertible_to<std::tuple<T, T>>;
        { geom.get_metrics(csi, eta) } -> std::same_as<GeomJacobian2D<T>>;
        { geom.get_x_face_metrics(csi, eta) } -> std::same_as<FaceMetrics2D<T>>;
        { geom.get_y_face_metrics(csi, eta) } -> std::same_as<FaceMetrics2D<T>>;
    };

    // ====================================================================
    // CARTESIAN REGIME POLICY
    // ====================================================================
    template <std::floating_point T>
    struct GeometryCartesian2D
    {
        T m_L_x{0.0}, m_L_y{0.0};
        T m_N_x{0.0}, m_N_y{0.0};

        GeometryCartesian2D(const ConfigOptions<T>& opts)
        {
            m_L_x = static_cast<T>(opts.get_box_length(0uz));
            m_L_y = static_cast<T>(opts.get_box_length(1uz));
            m_N_x = static_cast<T>(opts.get_n_elems(0uz));
            m_N_y = static_cast<T>(opts.get_n_elems(1uz));
        }

        auto get_physical_coords(T csi, T eta) const
        {
            return std::make_tuple(m_L_x * csi, m_L_y * eta);
        }

        GeomJacobian2D<T> get_metrics(T /*csi*/, T /*eta*/) const
        {
            // x = L_x * csi, y = L_y * eta  →  constant, diagonal shape Jacobian.
            // No 2*N_x scaling here anymore — PhysicsEvaluator folds that in.
            return GeomJacobian2D<T>{
                .dx_dcsi = m_L_x, .dy_dcsi = 0.0, .dx_deta = 0.0, .dy_deta = m_L_y};
        }

        FaceMetrics2D<T> get_x_face_metrics(T, T) const
        {
            return FaceMetrics2D<T>{.nx = 1.0, .ny = 0.0};
        }
        FaceMetrics2D<T> get_y_face_metrics(T, T) const
        {
            return FaceMetrics2D<T>{.nx = 0.0, .ny = 1.0};
        }
    };

    // ====================================================================
    // DE LAVAL NOZZLE REGIME POLICY
    // ====================================================================
    struct StraightLineProfile
    {
        // Default channel: Constant half-height of 0.5, zero slope.
        constexpr auto operator()(auto) const { return std::make_pair(0.5, 0.0); }
    };

    /*
     * Class that acts as metric tensor mapping from computational domain
     * csi, eta ∈ [0, 1] to the physical curved domain.
     * * X1 = X2 = L_x * csi
     * Y1 = 0.0
     * Y2 = h(x)
     * x(csi, eta) = X1 * (1 - eta) + X2 * eta
     * y(csi, eta) = Y1 * (1 - eta) + Y2 * eta
     */
    template <std::floating_point T, typename ProfileFunc = StraightLineProfile>
    struct GeometryDeLavalNozzle2D
    {
        T           m_L_x{0.0}, m_L_y{0.0};
        T           m_N_x{0.0}, m_N_y{0.0};
        ProfileFunc m_profile;

        GeometryDeLavalNozzle2D(const ConfigOptions<T>& opts, ProfileFunc profile)
            : m_profile(profile)
        {
            m_L_x = static_cast<T>(opts.get_box_length(0uz));
            m_L_y = static_cast<T>(opts.get_box_length(1uz));
            m_N_x = static_cast<T>(opts.get_n_elems(0uz));
            m_N_y = static_cast<T>(opts.get_n_elems(1uz));
        }

        GeometryDeLavalNozzle2D(const ConfigOptions<T>& opts)
            : GeometryDeLavalNozzle2D(opts, ProfileFunc{})
        {
        }

        auto get_physical_coords(T csi, T eta) const
        {
            const T x_phys      = m_L_x * csi;
            const auto [h_x, _] = m_profile(x_phys);

            // Transfinite Interpolation bounds mapping
            const T Y1     = -h_x; // 0.5 * (m_L_y - h_x);
            const T Y2     = h_x;  // 0.5 * (m_L_y + h_x);
            const T y_phys = Y1 * (1.0 - eta) + Y2 * eta;

            return std::make_tuple(x_phys, y_phys);
        }

        GeomJacobian2D<T> get_metrics(T csi, T eta) const
        {
            const T x_phys          = m_L_x * csi;
            const auto [h_x, dh_dx] = m_profile(x_phys);

            // Forward Jacobian of the PHYSICAL SHAPE MAP ONLY: (csi, eta) -> (x, y).
            // csi, eta ∈ [0,1] (computational domain). No reference-element /
            // element-count scaling here — PhysicsEvaluator composes that in.
            const T dx_dcsi = m_L_x;
            const T dy_dcsi = dh_dx * m_L_x * eta;
            const T dy_deta = h_x;
            // dx_deta = 0: x only depends on csi in this profile.

            return GeomJacobian2D<T>{
                .dx_dcsi = dx_dcsi, .dy_dcsi = dy_dcsi, .dx_deta = 0.0, .dy_deta = dy_deta};
        }

        FaceMetrics2D<T> get_x_face_metrics(T /*csi*/, T /*eta*/) const
        {
            return FaceMetrics2D<T>{.nx = 1.0, .ny = 0.0};
        }

        FaceMetrics2D<T> get_y_face_metrics(T csi, T eta) const
        {
            const T x_phys        = m_L_x * csi;
            const auto [_, dh_dx] = m_profile(x_phys);

            // Unscaled normal tilting
            T unscaled_nx = -dh_dx * eta;
            T unscaled_ny = 1.0;

            T inv_norm = 1.0 / std::sqrt(unscaled_nx * unscaled_nx + unscaled_ny * unscaled_ny);
            return FaceMetrics2D<T>{.nx = unscaled_nx * inv_norm, .ny = unscaled_ny * inv_norm};
        }
    };
} // namespace numPDE
