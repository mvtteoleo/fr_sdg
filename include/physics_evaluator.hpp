
#pragma once
// ============================================================
//  PhysicsEvaluator<Traits>
//
//  Owns everything that is pure numerics:
//    GLL nodes/weights, differentiation matrix, Radau corrections,
//    spectral filter, gas model, Riemann solver, BCs.
//
//  Knows NOTHING about PETSc's TS/SNES/Mat machinery.
//  PETSc types that appear here (Vec, DM) are only parameters
//  passed in from outside — never stored as members.
//
//  This means:
//    • residual_kernel, local_steady_res, filter_sol are all
//      independently unit-testable without PetscInitialize.
//    • "current time" is a plain parameter, not fetched from TS.
// ============================================================

#include "boundary_conditions.hpp"
#include "cached_computations.hpp"
#include "filter.hpp"
#include "gas_models.hpp"
#include "geometry_models.hpp"
#include "indexer.hpp"
#include "macro_and_libc.hpp"
#include "parser.hpp"
#include "reference_element.hpp"
#include "riemann_solver.hpp"

#include "vec_dof_view.hpp"
#include <exception>
#include <print>

namespace numPDE
{
    template <std::floating_point ScalarT = double, // Scalar type
              std::size_t         Dims    = 1,      // Dimension of the problem
              // Gas model and Riemann solver
              ComprFD::GasModelConc<ScalarT> Gas          = ComprFD::IdealGas<ScalarT>,
              ComprFD::RiemannScheme         RiemSolvType = ComprFD::RiemannScheme::Rusanov,
              CoordinateSystem               CSYS         = CoordinateSystem::Cartesian,
              GeometryModelConc<ScalarT>     Geometry     = GeometryCartesian2D<ScalarT>>
    struct ProblemTraits
    {
        using T                                           = ScalarT;
        static constexpr std::size_t n_dims               = Dims;
        using GasModel                                    = Gas;
        using GeomModel                                   = Geometry;
        static constexpr ComprFD::RiemannScheme RiemType  = RiemSolvType;
        static constexpr CoordinateSystem       coord_sys = CSYS;
    };

    template <class Traits, std::size_t n_dims = Traits::n_dims>
    class PhysicsEvaluator
    {
      public:
        PhysicsEvaluator() { std::println("Sono il {:}D", Traits::n_dims); };
    };

    template <class Traits>
    class PhysicsEvaluator<Traits, 1>
    {
        template <real_or_complex_floating_point T>
        struct NeighView
        {
            ComprFD::NodeState<T, 1> prev, next;
        };

      public:
        using T                      = typename Traits::T;
        using GasModel               = typename Traits::GasModel;
        using RiemannSolverT         = ComprFD::RiemannSolver<T, Traits::RiemType, GasModel>;
        static constexpr auto n_dims = Traits::n_dims;

        // ── Owned numerical state ──────────────────────────────────────────────
        ConfigOptions<T>           m_opts;
        GLL<T>                     m_gll;
        CachedElementData<T>       m_cached_data;
        const GasModel             m_gas_mod;
        const RiemannSolverT       m_riemann_solver;
        BoundaryConditions<T, 1>   m_bc;
        StabilizationTechniques<T> m_stab;
        bool                       is_periodic{false};

        // Set by PetscSolver after DMDA construction — just integer ranges,
        // no ownership of a PETSc object.
        DMInfos<1> m_dm_info;

        // ── Construction ──────────────────────────────────────────────────────
        explicit PhysicsEvaluator(const ConfigOptions<T>& opts, const BoundaryConditions<T, 1>& bcs,
                                  const GasModel& gmod = GasModel{})
            : m_opts{opts}, m_gll{opts.get_poly_deg(0)}, m_cached_data{m_gll, opts.get_rad_corr()},
              m_gas_mod{gmod}, m_riemann_solver{}, m_bc{bcs}, m_stab{m_gll, m_opts, m_cached_data},
              is_periodic{bcs.get_type(Direction::Left) == BCType::Periodic &&
                          bcs.get_type(Direction::Right) == BCType::Periodic}
        {
        }

        // ── Core DG kernels ───────────────────────────────────────────────────
        //  All work on abstract Read/WriteView pairs and operate on T or
        //  CplxType transparently — the complex-step Jacobian reuses these
        //  kernels directly without any separate specialisation.

        // Outer kernel: iterates over elements, dispatches boundary vs interior.
        // 'time' is an explicit parameter so PETSc's TS is never touched here.
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode residual_kernel(const RV& x_view, WV& f_view, T time);

        // Change-of-variable kernels (primitive ↔ conservative)
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode prim_to_cons_kernel(const RV& prim, WV& cons);

        // Apply weno limiter
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode limit_solution(const RV& prim, WV& cons);

        // Spectral filter applied equation-by-equation on each element
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode filter_sol(const RV& sol, WV& fil);

        // ── Utility ───────────────────────────────────────────────────────────

        // Fill 'solution' with the IC.  Vec and DM are passed in so this
        // method has no knowledge of where the vector lives.
        PetscErrorCode impose_ic(Vec solution, DM dmda);

        // Compute L2 error against exact_sol at 'time'.
        // The current solution is passed in as a plain Vec.
        std::array<typename Traits::T, Traits::n_dims + 2>
        compute_error(const T time, const Vec solution, const DM dmda) const;

        T compute_dx(auto i) const { return (m_opts.get_box_length(i) / m_opts.get_n_elems(i)); };
        T compute_inv_jacobian(auto i) const { return 2.0 / compute_dx(i); }
        T compute_inv_jacobian() const { return compute_inv_jacobian(0); }

        auto get_n_dof_per_element() const
        {
            return static_cast<PetscInt>((n_dims + 2) * m_gll.get_n_nodes());
        }

      private:
        // Inner kernel: single element, reads from elem_u, writes to res.
        template <real_or_complex_floating_point U>
        PetscErrorCode local_steady_res(std::span<const U> elem_u, const NeighView<U>& nv,
                                        std::span<U> res);

        auto get_r(std::integral auto elem, std::integral auto node) const
        {
            const auto& csi = this->m_gll.get_csi();

            const auto h = this->compute_dx(0uz);

            return (elem + 0.5 * (csi[node] + 1.0)) * h;
        };
    };

    template <class Traits>
    struct PhysicsEvaluator<Traits, 2>
    {
        using T                      = typename Traits::T;
        using GasModel               = typename Traits::GasModel;
        using GeomModel              = typename Traits::GeomModel;
        using RiemannSolverT         = ComprFD::RiemannSolver2D<T, Traits::RiemType, GasModel>;
        static constexpr auto n_dims = Traits::n_dims;

        // ── Owned numerical state ──────────────────────────────────────────────
        ConfigOptions<T>                      m_opts;
        const GLL<T>                          m_gll_x, m_gll_y;
        const CachedElementData<T>            m_cached_data_x, m_cached_data_y;
        const GasModel                        m_gas_mod;
        const RiemannSolverT                  m_riemann_solver;
        const GeomModel                       m_geometry;
        BoundaryConditions<T, Traits::n_dims> m_bc;
        struct Stab2D
        {
            StabilizationTechniques<T> x, y;
            bool filters_active() const { return x.filters_active() || y.filters_active(); }
        } m_stab;
        bool is_periodic_x{false};
        bool is_periodic_y{false};

        // Set by PetscSolver after DMDA construction — just integer ranges,
        // no ownership of a PETSc object.
        DMInfos<2> m_dm_info;

        // ── Construction ──────────────────────────────────────────────────────
        explicit PhysicsEvaluator(const ConfigOptions<T>&              opts,
                                  const BoundaryConditions<T, n_dims>& bcs,
                                  const GasModel&                      gmod = GasModel{})
            : m_opts{opts}, m_gll_x{opts.get_poly_deg(0)}, m_gll_y{opts.get_poly_deg(1)},
              m_cached_data_x{m_gll_x, opts.get_rad_corr()},
              m_cached_data_y{m_gll_y, opts.get_rad_corr()}, m_gas_mod{gmod}, m_riemann_solver{},
              m_geometry{opts}, m_bc{bcs}, m_stab{.x{m_gll_x, m_opts, m_cached_data_x, 0},
                                                  .y{m_gll_y, m_opts, m_cached_data_y, 1}},
              is_periodic_x{bcs.get_type(Direction::Left) == BCType::Periodic &&
                            bcs.get_type(Direction::Right) == BCType::Periodic},
              is_periodic_y{bcs.get_type(Direction::Top) == BCType::Periodic &&
                            bcs.get_type(Direction::Bottom) == BCType::Periodic}
        {
        }
        consteval std::size_t get_n_eqs() const { return 4; };
        auto                  get_n_dof_per_element() const
        {
            return static_cast<PetscInt>((n_dims + 2) * m_gll_x.get_n_nodes() *
                                         m_gll_y.get_n_nodes());
        }

        // Outer kernel: iterates over elements, dispatches boundary vs interior.
        // 'time' is an explicit parameter so PETSc's TS is never touched here.
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode residual_kernel(const RV& x_view, WV& f_view, T time) const;

        // Change-of-variable kernels (primitive ↔ conservative)
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode prim_to_cons_kernel(const RV& prim, WV& cons);

        // Apply weno limiter
        template <real_or_complex_floating_point U, ReadViewConc<U> RV, WriteViewConc<U> WV>
        PetscErrorCode limit_solution(const RV& prim, WV& cons);

        // ── Utility ───────────────────────────────────────────────────────────

        // Fill 'solution' with the IC.  Vec and DM are passed in so this
        // method has no knowledge of where the vector lives.
        PetscErrorCode impose_ic(Vec solution, DM dmda) const;

        // Compute L2 error against exact_sol at 'time'.
        // The current solution is passed in as a plain Vec.
        std::array<typename Traits::T, Traits::n_dims + 2> compute_error(T time, Vec solution,
                                                                         DM dmda) const;

      private:
        // ── Computational Coordinates in [0, 1]² ──
        T get_comp_coord_x(std::integral auto i, std::integral auto k) const
        {
            const auto& csi_x = m_gll_x.get_csi();
            const T     h_x   = 1.0 / static_cast<T>(this->m_opts.get_n_elems(0uz));
            return h_x * (static_cast<T>(i) + 0.5 * (1.0 + csi_x[k]));
        }
        T get_comp_coord_y(std::integral auto j, std::integral auto l) const
        {
            const auto& csi_y = m_gll_y.get_csi();
            const T     h_y   = 1.0 / static_cast<T>(this->m_opts.get_n_elems(1uz));
            return h_y * (static_cast<T>(j) + 0.5 * (1.0 + csi_y[l]));
        }

        auto get_computational_coords(std::integral auto i, std::integral auto j,
                                      std::integral auto k, std::integral auto l) const
        {

            const T comp_x = get_comp_coord_x(i, k);
            const T comp_y = get_comp_coord_y(j, l);

            return std::make_tuple(comp_x, comp_y);
        }

        // ── Physical Coordinates from Geometry ──
        auto get_x_y_pos(std::integral auto i, std::integral auto j, std::integral auto k,
                         std::integral auto l) const
        {
            const auto [comp_x, comp_y] = get_computational_coords(i, j, k, l);
            return this->m_geometry.get_physical_coords(comp_x, comp_y);
        }

        // ── Per-element affine scaling: GLL reference coords r,s ∈ [-1,1]  ──
        // ── to computational domain coords csi,eta ∈ [0,1].                ──
        // csi(r) = h_x * (i + 0.5*(1+r))  ⇒  dcsi/dr = 0.5 * h_x = 0.5 / N_x
        T get_dcsi_dr() const { return 0.5 / static_cast<T>(this->m_opts.get_n_elems(0uz)); }
        T get_deta_ds() const { return 0.5 / static_cast<T>(this->m_opts.get_n_elems(1uz)); }

        /* Metrics at a given computational-domain location (Used in Volume
         * Integration and at interfaces).
         *
         * This is where the two mappings are composed:
         * GeometryModel supplies d(x,y)/d(csi,eta) ,
         * hysicsEvaluator supplies d(csi,eta)/d(r,s)
         *
         * The chain rule gives the contravariant metric tensor of the full map r,s -> x,y.
         * */

        VolumeMetrics2D<T> get_metrics(T comp_x, T comp_y) const
        {
            const auto geom_jac = this->m_geometry.get_metrics(comp_x, comp_y);

            const T dcsi_dr = this->get_dcsi_dr();
            const T deta_ds = this->get_deta_ds();

            // Composite forward Jacobian d(x,y)/d(r,s). Tensor-product elements
            // are axis-aligned in the r,s <-> csi,eta map (r only maps to csi,
            // s only maps to eta), so the cross terms vanish here exactly.
            const T dx_dr = geom_jac.dx_dcsi * dcsi_dr;
            const T dy_dr = geom_jac.dy_dcsi * dcsi_dr;
            const T dx_ds = geom_jac.dx_deta * deta_ds;
            const T dy_ds = geom_jac.dy_deta * deta_ds;

            const T detJ = dx_dr * dy_ds - dx_ds * dy_dr;

            // Exact inverse (contravariant) metric tensor of the full map.
            return VolumeMetrics2D<T>{.xi_x  = dy_ds / detJ,
                                      .xi_y  = -dx_ds / detJ,
                                      .eta_x = -dy_dr / detJ,
                                      .eta_y = dx_dr / detJ,
                                      .detJ  = detJ};
        }

        // ── Metrics from Geometry (Used in Volume Integration) ──
        VolumeMetrics2D<T> get_metrics(std::integral auto i, std::integral auto j,
                                       std::integral auto k, std::integral auto l) const
        {
            const auto [comp_x, comp_y] = get_computational_coords(i, j, k, l);
            return this->get_metrics(comp_x, comp_y);
        }

        template <real_or_complex_floating_point U>
        void compute_volume_term(const std::span<const U> u_span, std::span<U> res_span,
                                 std::integral auto i, std::integral auto j) const;
    };
} // namespace numPDE
#include "physics_impl_1D.hpp"
#include "physics_impl_2D.hpp"
