#pragma once
#include "cached_computations.hpp"
#include "macro_and_libc.hpp"
#include "parser.hpp"
#include "reference_element.hpp"
#include <exception>
#include <numeric>
#include <vector>

#define with_ranges true
#if !with_ranges

namespace numPDE
{

    template <std::floating_point T>
    class SpectralFilter
    {
      private:
        SpectFiltVals filt_vals{};
        EigMat<T>     m_Filter{};

      public:
        SpectralFilter(const GLL<T>& gll, const SpectFiltVals f_vals,
                       const std::integral auto p_val)
            : filt_vals{f_vals}, m_Filter{create_filter_mat(gll)}
        {
            if (filt_vals.n_cut_off >= p_val)
                std::println("Cut off higher than polynomial degree set!! {} vs {}!!",
                             filt_vals.n_cut_off, p_val);
        };

        explicit SpectralFilter(const GLL<T>& gll, const ConfigOptions<T>& opts)
            : filt_vals{opts.get_filt_vals()}, m_Filter{create_filter_mat(gll)}
        {
            if (filt_vals.n_cut_off >= opts.get_poly_deg(0))
                std::println("Cut off higher than polynomial degree set!! {} vs {}!!",
                             filt_vals.n_cut_off, opts.get_poly_deg(0));
        };

        const auto& get_filt_mat() const { return m_Filter; };

        template <real_or_complex_floating_point U>
        void apply_filter(const std::span<const U> sp_in, std::span<U> sp_fil) const
        {
            // Safety check to ensure the spans match the filter matrix dimension
            assert((int) sp_in.size() == (int) m_Filter.rows() && "Input span size mismatch!");
            assert((int) sp_fil.size() == (int) m_Filter.rows() && "Output span size mismatch!");

            // Map the contiguous C++ spans to Eigen Vectors (Zero-cost abstraction)
            Eigen::Map<const Eigen::Vector<U, Eigen::Dynamic>> vec_in(sp_in.data(), sp_in.size());
            Eigen::Map<Eigen::Vector<U, Eigen::Dynamic>> vec_out(sp_fil.data(), sp_fil.size());

            // Perform the filter matrix multiplication
            vec_out = this->m_Filter * vec_in;
            return;
        };

      private:
        EigMat<T> create_filter_mat(const GLL<T>& gll);
    };

    template <std::floating_point T>
    EigMat<T> SpectralFilter<T>::create_filter_mat(const GLL<T>& gll)
    {
        const std::size_t n               = gll.get_n_nodes();
        EigMat<T>         Vandermonde_mat = EigMat<T>::Zero(n, n);
        EigMat<T>         sigma_mat       = EigMat<T>::Zero(n, n);

        auto sigma = [&](auto i) -> T
        {
            if (i < this->filt_vals.n_cut_off)
                return T{1.0};
            else
            {
                const T ratio = static_cast<T>(i - this->filt_vals.n_cut_off) /
                                static_cast<T>((n - 1) - this->filt_vals.n_cut_off);
                return T{std::exp(-this->filt_vals.alpha *
                                  std::pow(ratio, this->filt_vals.filter_strength))};
            }
        };

        const auto  rows = std::views::iota(0uz, n);
        const auto  cols = rows;
        const auto& csi  = gll.get_csi();

        for (const auto i : cols)
        {
            sigma_mat(i, i) = sigma(i);

            for (const auto j : rows)
                Vandermonde_mat(i, j) = boost::math::legendre_p(j, csi[i]);
        }

        return EigMat<T>{Vandermonde_mat * sigma_mat * Vandermonde_mat.inverse()};
    };

    template <std::floating_point T>
    struct WENOLimiter
    {
      public:
        using RangeTypeGammas = std::array<T, 3>;

      private:
        // WENO linear weights as defined in Zhong & Shu (Eq 3.11)
        static constexpr T epsilon = 1.0e-6;

        EigVec<T>             m_quad_weights; // GLL integration weights
        EigMat<T>             m_D;            // First derivative matrix
        const RangeTypeGammas gammas;

      public:
        explicit WENOLimiter(const GLL<T>& gll, const CachedElementData<T>& cached, const T gamma)
            : m_quad_weights{EigVec<T>::Zero(gll.get_n_nodes())}, m_D{cached.get_diff_mat()},
              gammas{{(1 - gamma) * 0.5, gamma, (1 - gamma) * 0.5}}
        {
            if (gamma > 1.0 or gamma < 0)
            {
                std::println(stderr,
                             "Gamma value makes no sense, it has to be below 1 and bigger than 0!");
                std::terminate();
            }
            const auto& wheights = gll.get_ws();
            std::copy_n(wheights.begin(), wheights.size(), m_quad_weights.begin());
        }

        const auto get_gammas() const { return gammas; };

        // Acts element-wise for a single scalar field (e.g., density)
        // Computation of omegas is delegated to the Shock sensor.
        template <real_or_complex_floating_point U>
        void apply_weno_scalar(std::span<const U> p_prev, std::span<const U> p_curr,
                               std::span<const U> p_next, std::span<U> p_out,
                               const RangeTypeGammas& norm_omegas) const
        {
            assert(norm_omegas.size() == gammas.size());
            const std::size_t n = p_curr.size();

            // Extract averages for clean readability in the reconstruction loop
            const U avg_prev = compute_cell_average(p_prev);
            const U avg_curr = compute_cell_average(p_curr);
            const U avg_next = compute_cell_average(p_next);

            // Shift polynomials and reconstruct [Eq 3.6, 3.7]
            for (std::size_t i = 0; i < n; ++i)
            {
                const U tilde_prev = p_prev[i] - avg_prev + avg_curr;
                const U tilde_next = p_next[i] - avg_next + avg_curr;

                p_out[i] = norm_omegas[0] * tilde_prev + norm_omegas[1] * p_curr[i] +
                           norm_omegas[2] * tilde_next;
            }
            return;
        }

        template <real_or_complex_floating_point U>
        void apply_weno_scalar(std::span<const U> p_prev, std::span<const U> p_curr,
                               std::span<const U> p_next, std::span<U> p_out) const
        {

            // 1. & 2. Compute cell averages and smoothness indicators (beta)
            const std::array<std::pair<U, U>, 3> smoothness = {
                compute_smoothness(p_prev), compute_smoothness(p_curr), compute_smoothness(p_next)};

            // 3. Compute unnormalized Non-linear Weights (w_bar) [Eq 3.8, 3.9]
            RangeTypeGammas omegas;
            std::ranges::transform(smoothness, this->gammas, omegas.begin(),
                                   [](const auto& beta, const T gamma)
                                   { return gamma / ((epsilon + beta) * (epsilon + beta)); });

            const U w_sum = std::reduce(omegas.begin(), omegas.end(), U{0.0});

            for (auto& omega : omegas)
                omega /= w_sum;

            apply_weno_scalar(p_prev, p_curr, p_next, p_out, omegas);
            return;
        }

        template <real_or_complex_floating_point U>
        U compute_cell_average(std::span<const U> poly) const
        {
            U avg{0.0};
            for (std::size_t i = 0; i < poly.size(); ++i)
                avg += m_quad_weights[i] * poly[i];
            return 0.5 * avg; // 0.5 factor maps from standard element [-1, 1]
        }

        template <real_or_complex_floating_point U>
        U compute_smoothness(std::span<const U> poly) const
        {
            // Map the input span to an Eigen vector view
            Eigen::Map<const Eigen::Vector<U, Eigen::Dynamic>> p_vec(poly.data(), poly.size());

            U beta{0.0};

            // Allocate a working buffer for successive derivative calculations
            Eigen::Vector<U, Eigen::Dynamic> current_derivative = p_vec;

            // The maximum derivative order is the polynomial degree: k = poly.size() - 1
            const std::size_t k = poly.size() - 1;

            // The scaling factor in front of compute_integral_sq is 2^(2s).
            // For s = 1, 2^(2*1) = 4.0
            U factor = 4.0;

            for (std::size_t s = 1; s <= k; ++s)
            {
                // Differentiate one more time: d^s(p) / d\xi^s
                current_derivative = m_D * current_derivative;

                // Accumulate the reference integral scaled by the paper's weight
                beta += factor * compute_integral_sq(current_derivative);

                // Update factor for the next derivative order: 2^(2*(s+1)) = 4 * 2^(2s)
                factor *= 4.0;
            }

            return beta;
        }

        template <real_or_complex_floating_point U>
        U compute_integral_sq(const Eigen::Vector<U, Eigen::Dynamic>& vec) const
        {
            const U integral = std::transform_reduce(
                vec.begin(), vec.end(),                 
                m_quad_weights.begin(),                 
                U{0.0},                                 
                std::plus<>(),                          // Reduction operation 
                [](const U& v, const auto& w) {         // Transformation operation
                    return w * (v * v); 
                } 
            );

            return 0.5 * integral;
        }
    };

    template <std::floating_point T, std::ranges::range RR>
    class ShockSensor
    {
      private:
        static constexpr T epsilon = 1.0e-6;
        const T            m_theta;
        const T            m_threshold;

        // Ideal weights (d_l)
        const RR m_ideal_vals;

        // Precomputed normalization denominator
        T m_one_over_norm_denominator;

      public:
        // Defaults based on the paper: theta = 1.0 pairs well with threshold = 0.8
        explicit ShockSensor(RR gammas, T theta = 1.0, T threshold = 0.8)
            : m_theta{theta}, m_threshold{threshold}, m_ideal_vals{gammas}
        {
            // Precompute denominator (eq 21 art: On shock sensors for hybrid compact/WENO schemes)
            const T d_min = std::ranges::min(m_ideal_vals);
            this->m_one_over_norm_denominator =
                1.0 / (std::pow(std::abs(1.0 / d_min - 1.0), m_theta) + gammas.size() - 1);
        };

        // Returns true if a shock is detected. Populates omega weights to prevent recomputation.
        // Input the betas from WENO (Smoothness indicators)
        std::pair<bool, RR> detect_shock(const RR& betas) const
        {

            // Compute unnormalized nonlinear weights (w_bar)
            RR corrected_gammas = this->m_ideal_vals;

            for (const auto i : std::views::iota(0uz, m_ideal_vals.size()))
                corrected_gammas[i] /= std::pow(epsilon + betas[i], 2);

            // Normalize to get omega

            const T w_sum = std::reduce(corrected_gammas.begin(), corrected_gammas.end());

            for (auto& gamma : corrected_gammas)
                gamma /= w_sum;

            // Compute the numerator of the sensor (deviation from ideal weights)
            const T num = std::transform_reduce(corrected_gammas.begin(), corrected_gammas.end(),
                                                m_ideal_vals.begin(), T{0}, std::plus<>(),
                                                [&](auto w_l, auto d_l)
                                                {
                                                    T v_ = std::abs(w_l / d_l - T{1.0});
                                                    return std::pow(v_, m_theta);
                                                });

            // Final sensor value r_j
            const T r_j = num * m_one_over_norm_denominator;

            // Trigger WENO if threshold is exceeded
            return std::make_pair(static_cast<bool>(std::abs(r_j) >= m_threshold),
                                  corrected_gammas);
        }
    };
    /*
     *   Class that handles the stabilization techniques.
     *   The classes are empty and contain simply the methods and the constants.
     *
     *   The stabilization is applied AFTER the step.
     *   The only exception is the *Artificial viscosity* that
     *   actually modifies the equation so needs to be
     *   called during the residual kernel routine and (possibly) store 2 std::vectors if
     *   the Jacobian is evaluated using the complex step.
     */

    template <std::floating_point T>
    class StabilizationTechniques
    {
        const bool                                               m_apply_spec{false};
        const SpectralFilter<T>                                  m_spect;
        const bool                                               m_apply_weno{false};
        WENOLimiter<T>                                           m_weno;
        ShockSensor<T, typename WENOLimiter<T>::RangeTypeGammas> m_sensor;

      public:
        explicit StabilizationTechniques(const GLL<T>& gll, const ConfigOptions<T>& opts,
                                         const CachedElementData<T>& cached_data)
            : m_apply_spec{opts.get_apply_filter()}, m_spect{gll, opts},
              m_apply_weno{opts.get_apply_weno()}, m_weno{gll, cached_data, opts.get_weno_gamma()},
              m_sensor{m_weno.get_gammas(), opts.get_shock_sens().theta,
                       opts.get_shock_sens().threshold}
        {
        }
        explicit StabilizationTechniques(const GLL<T>& gll, const ConfigOptions<T>& opts,
                                         const CachedElementData<T>& cached_data,
                                         std::integral auto          axis)
            : m_apply_spec{opts.get_apply_filter()},
              m_spect{gll, opts.get_filt_vals(), opts.get_poly_deg(axis)},
              m_apply_weno{opts.get_apply_weno()}, m_weno{gll, cached_data, opts.get_weno_gamma()},
              m_sensor{m_weno.get_gammas(), opts.get_shock_sens().theta,
                       opts.get_shock_sens().threshold}
        {
        }

        bool filters_active() const { return static_cast<bool>(m_apply_weno or m_apply_spec); };
        /**
         * @brief Unified element-wise stabilization pipeline.
         * Exposes a single clean interface to the Physics Evaluator.
         */
        template <real_or_complex_floating_point U>
        void apply_element_stabilization(std::span<const U> sp_prev, std::span<const U> sp_curr,
                                         std::span<const U> sp_next, std::span<U> sp_out) const
        {
            if (!m_apply_spec && !m_apply_weno)
            {
                std::ranges::copy(sp_curr, sp_out.begin());
                return;
            }

            std::span<const U> sp_working = sp_curr;

            // 1. WENO with Shock Sensor (Only runs if active)
            if (m_apply_weno)
            {
                // Calculate smoothness indicators ONLY
                typename WENOLimiter<T>::RangeTypeGammas betas = {
                    m_weno.compute_smoothness(sp_prev),
                    m_weno.compute_smoothness(sp_working), // Use filtered data if filter ran
                    m_weno.compute_smoothness(sp_next)};

                // Sensor makes the decision and calculates omegas
                const auto [is_shock, omegas] = m_sensor.detect_shock(betas);

                // Fast path: Apply WENO without recalculating omegas
                if (is_shock)
                    m_weno.apply_weno_scalar(sp_prev, sp_working, sp_next, sp_out, omegas);
                // Sensor says smooth: bypass WENO
                else
                    std::ranges::copy(sp_working, sp_out.begin());

                sp_working = sp_out;
            }

            if (m_apply_spec)
            {
                std::vector<U> m_scratch(sp_curr.size());
                std::span<U>   sp_scratch(m_scratch.data(), scratch.size());

                m_spect.template apply_filter<U>(sp_working, sp_scratch);

                std::ranges::copy(sp_scratch, sp_out.begin());
            }
        };
        // add_av_to_res()
        // vompu
    };

} // namespace numPDE
#else
namespace numPDE
{

    template <std::floating_point T>
    class SpectralFilter
    {
      private:
        SpectFiltVals filt_vals{};
        EigMat<T>     m_Filter{};

      public:
        SpectralFilter(const GLL<T>& gll, const SpectFiltVals f_vals,
                       const std::integral auto p_val)
            : filt_vals{f_vals}, m_Filter{create_filter_mat(gll)}
        {
            if (filt_vals.n_cut_off >= p_val)
                std::println("Cut off higher than polynomial degree set!! {} vs {}!!",
                             filt_vals.n_cut_off, p_val);
        }

        explicit SpectralFilter(const GLL<T>& gll, const ConfigOptions<T>& opts)
            : filt_vals{opts.get_filt_vals()}, m_Filter{create_filter_mat(gll)}
        {
            if (filt_vals.n_cut_off >= opts.get_poly_deg(0))
                std::println("Cut off higher than polynomial degree set!! {} vs {}!!",
                             filt_vals.n_cut_off, opts.get_poly_deg(0));
        }

        const auto& get_filt_mat() const { return m_Filter; };

        // ── Generalized Ranges Interface for Spectral Filtering ──
        template <std::ranges::random_access_range RIn, std::ranges::random_access_range ROut>
        void apply_filter(RIn&& range_in, ROut&& range_out) const
        {
            using U = std::ranges::range_value_t<ROut>;
            assert(std::ranges::size(range_in) == static_cast<std::size_t>(m_Filter.cols()) &&
                   "Input range size mismatch!");
            assert(std::ranges::size(range_out) == static_cast<std::size_t>(m_Filter.rows()) &&
                   "Output range size mismatch!");

            const std::size_t rows = m_Filter.rows();
            const std::size_t cols = m_Filter.cols();

            // Zero Eigen mapping. We use an inner product view pipeline over the columns.
            for (std::size_t r = 0; r < rows; ++r)
            {
                range_out[r] = std::transform_reduce(
                    std::views::iota(0uz, cols).begin(), std::views::iota(0uz, cols).end(), U{0.0},
                    std::plus<>(),
                    [&](std::size_t c) { return static_cast<U>(m_Filter(r, c)) * range_in[c]; });
            }
        }

      private:
        EigMat<T> create_filter_mat(const GLL<T>& gll);
    };

    template <std::floating_point T>
    EigMat<T> SpectralFilter<T>::create_filter_mat(const GLL<T>& gll)
    {
        const std::size_t n               = gll.get_n_nodes();
        EigMat<T>         Vandermonde_mat = EigMat<T>::Zero(n, n);
        EigMat<T>         sigma_mat       = EigMat<T>::Zero(n, n);

        auto sigma = [&](auto i) -> T
        {
            if (i < this->filt_vals.n_cut_off)
                return T{1.0};
            else
            {
                const T ratio = static_cast<T>(i - this->filt_vals.n_cut_off) /
                                static_cast<T>((n - 1) - this->filt_vals.n_cut_off);
                return T{std::exp(-this->filt_vals.alpha *
                                  std::pow(ratio, this->filt_vals.filter_strength))};
            }
        };

        const auto  rows = std::views::iota(0uz, n);
        const auto  cols = rows;
        const auto& csi  = gll.get_csi();

        for (const auto i : cols)
        {
            sigma_mat(i, i) = sigma(i);

            for (const auto j : rows)
                Vandermonde_mat(i, j) = boost::math::legendre_p(j, csi[i]);
        }

        return EigMat<T>{Vandermonde_mat * sigma_mat * Vandermonde_mat.inverse()};
    }

    template <std::floating_point T, std::size_t MaxNum = 9>
    struct WENOLimiter
    {
      public:
        using RangeTypeGammas = std::array<T, 3>;

      private:
        static constexpr T epsilon = 1.0e-6;

        EigVec<T>             m_quad_weights;
        EigMat<T>             m_D;
        const RangeTypeGammas gammas;

      public:
        explicit WENOLimiter(const GLL<T>& gll, const CachedElementData<T>& cached, const T gamma)
            : m_quad_weights{EigVec<T>::Zero(gll.get_n_nodes())}, m_D{cached.get_diff_mat()},
              gammas{{(1 - gamma) * 0.5, gamma, (1 - gamma) * 0.5}}
        {
            if (gamma > 1.0 || gamma < 0)
            {
                std::println(stderr,
                             "Gamma value makes no sense, it has to be below 1 and bigger than 0!");
                std::terminate();
            }
            const auto& wheights = gll.get_ws();
            std::copy_n(wheights.begin(), wheights.size(), m_quad_weights.begin());
        }

        const auto get_gammas() const { return gammas; };

        // ── C++23 Zip-View Enabled Vectorized WENO Reconstruction ──
        template <std::ranges::random_access_range RPrev, std::ranges::random_access_range RCurr,
                  std::ranges::random_access_range RNext, std::ranges::random_access_range ROut>
        void apply_weno_scalar(RPrev&& p_prev, RCurr&& p_curr, RNext&& p_next, ROut&& p_out,
                               const RangeTypeGammas& norm_omegas) const
        {
            assert(norm_omegas.size() == gammas.size());
            using U = std::ranges::range_value_t<RCurr>;

            const U avg_prev = compute_cell_average(p_prev);
            const U avg_curr = compute_cell_average(p_curr);
            const U avg_next = compute_cell_average(p_next);

            // Zip views eliminate manual loop counters and index grouping entirely
            auto zipped = std::views::zip(p_prev, p_curr, p_next, p_out);

            std::ranges::for_each(zipped,
                                  [&](auto&& tuple_pack)
                                  {
                                      auto& [prev_val, curr_val, next_val, out_val] = tuple_pack;

                                      const U tilde_prev = prev_val - avg_prev + avg_curr;
                                      const U tilde_next = next_val - avg_next + avg_curr;

                                      out_val = norm_omegas[0] * tilde_prev +
                                                norm_omegas[1] * curr_val +
                                                norm_omegas[2] * tilde_next;
                                  });
        }

        template <std::ranges::random_access_range RPrev, std::ranges::random_access_range RCurr,
                  std::ranges::random_access_range RNext, std::ranges::random_access_range ROut>
        void apply_weno_scalar(RPrev&& p_prev, RCurr&& p_curr, RNext&& p_next, ROut&& p_out) const
        {
            using U = std::ranges::range_value_t<RCurr>;

            const std::array<std::pair<U, U>, 3> smoothness = {
                compute_smoothness(p_prev), compute_smoothness(p_curr), compute_smoothness(p_next)};

            RangeTypeGammas omegas;
            std::ranges::transform(smoothness, this->gammas, omegas.begin(),
                                   [](const auto& beta, const T gamma)
                                   { return gamma / ((epsilon + beta) * (epsilon + beta)); });

            const U w_sum = std::reduce(omegas.begin(), omegas.end(), U{0.0});

            for (auto& omega : omegas)
                omega /= w_sum;

            apply_weno_scalar(p_prev, p_curr, p_next, p_out, omegas);
        }

        // ── Zip-Based Pointwise Quadrature Reductions ──
        template <std::ranges::random_access_range R>
        auto compute_cell_average(R&& poly) const
        {
            using U     = std::ranges::range_value_t<R>;
            auto zipped = std::views::zip(m_quad_weights, poly);

            const U integral =
                std::transform_reduce(zipped.begin(), zipped.end(), U{0.0}, std::plus<>(),
                                      [](const auto& tuple_pair)
                                      {
                                          const auto& [w, p] = tuple_pair;
                                          return static_cast<U>(w) * p;
                                      });
            return U{0.5} * integral;
        }

        // ── Functional Derivative-Free Matrix Smoothness Pipeline ──
        template <std::ranges::random_access_range R>
        auto compute_smoothness(R&& poly) const
        {
            using U = std::ranges::range_value_t<R>;
            std::array<T, MaxNum> m_curr_deriv, m_next_deriv;
            const std::size_t     n = std::ranges::size(poly);

            std::ranges::copy(poly, m_curr_deriv.begin());

            U                 beta{0.0};
            const std::size_t k      = n - 1;
            U                 factor = 4.0;

            for (std::size_t s = 1; s <= k; ++s)
            {

                // Linear multiplication via functional zip rows
                for (std::size_t r = 0; r < n; ++r)
                {
                    m_next_deriv[r] = std::transform_reduce(
                        std::views::iota(0uz, n).begin(), std::views::iota(0uz, n).end(), U{0.0},
                        std::plus<>(),
                        [&](std::size_t c) { return static_cast<U>(m_D(r, c)) * m_curr_deriv[c]; });
                }
                std::swap(m_curr_deriv, m_next_deriv);

                beta += factor * compute_integral_sq(m_curr_deriv);
                factor *= 4.0;
            }
            return beta;
        }

        template <std::ranges::random_access_range R>
        auto compute_integral_sq(R&& vec) const
        {
            using U     = std::ranges::range_value_t<R>;
            auto zipped = std::views::zip(m_quad_weights, vec);

            const U integral =
                std::transform_reduce(zipped.begin(), zipped.end(), U{0.0}, std::plus<>(),
                                      [](const auto& tuple_pair)
                                      {
                                          const auto& [w, v] = tuple_pair;
                                          return static_cast<U>(w) * v * v;
                                      });
            return U{0.5} * integral;
        }
    };

    template <std::floating_point T, std::ranges::range RR>
    class ShockSensor
    {
      private:
        static constexpr T epsilon = 1.0e-6;
        const T            m_theta;
        const T            m_threshold;
        const RR           m_ideal_vals;
        T                  m_one_over_norm_denominator;

      public:
        explicit ShockSensor(RR gammas, T theta = 1.0, T threshold = 0.8)
            : m_theta{theta}, m_threshold{threshold}, m_ideal_vals{gammas}
        {
            const T d_min = std::ranges::min(m_ideal_vals);
            this->m_one_over_norm_denominator =
                1.0 / (std::pow(std::abs(1.0 / d_min - 1.0), m_theta) + gammas.size() - 1);
        }

        std::pair<bool, RR> detect_shock(const RR& betas) const
        {
            RR corrected_gammas = this->m_ideal_vals;

            for (const auto i : std::views::iota(0uz, m_ideal_vals.size()))
                corrected_gammas[i] /= std::pow(epsilon + betas[i], 2);

            const T w_sum = std::reduce(corrected_gammas.begin(), corrected_gammas.end());

            for (auto& gamma : corrected_gammas)
                gamma /= w_sum;

            const T num = std::transform_reduce(corrected_gammas.begin(), corrected_gammas.end(),
                                                m_ideal_vals.begin(), T{0}, std::plus<>(),
                                                [&](auto w_l, auto d_l)
                                                {
                                                    T v_ = std::abs(w_l / d_l - T{1.0});
                                                    return std::pow(v_, m_theta);
                                                });

            const T r_j = num * m_one_over_norm_denominator;

            return std::make_pair(static_cast<bool>(std::abs(r_j) >= m_threshold),
                                  corrected_gammas);
        }
    };

    template <std::floating_point T>
    class StabilizationTechniques
    {
        const bool                                               m_apply_spec{false};
        SpectralFilter<T>                                        m_spect;
        const bool                                               m_apply_weno{false};
        WENOLimiter<T, GLL<T>::MaxNumElem>                       m_weno;
        ShockSensor<T, typename WENOLimiter<T>::RangeTypeGammas> m_sensor;

      public:
        explicit StabilizationTechniques(const GLL<T>& gll, const ConfigOptions<T>& opts,
                                         const CachedElementData<T>& cached_data)
            : m_apply_spec{opts.get_apply_filter()}, m_spect{gll, opts},
              m_apply_weno{opts.get_apply_weno()}, m_weno{gll, cached_data, opts.get_weno_gamma()},
              m_sensor{m_weno.get_gammas(), opts.get_shock_sens().theta,
                       opts.get_shock_sens().threshold}
        {
        }

        explicit StabilizationTechniques(const GLL<T>& gll, const ConfigOptions<T>& opts,
                                         const CachedElementData<T>& cached_data,
                                         std::integral auto          axis)
            : m_apply_spec{opts.get_apply_filter()},
              m_spect{gll, opts.get_filt_vals(), opts.get_poly_deg(axis)},
              m_apply_weno{opts.get_apply_weno()}, m_weno{gll, cached_data, opts.get_weno_gamma()},
              m_sensor{m_weno.get_gammas(), opts.get_shock_sens().theta,
                       opts.get_shock_sens().threshold}
        {
        }

        bool filters_active() const { return static_cast<bool>(m_apply_weno || m_apply_spec); };

        // ── Global Pipeline Acceptant of Any Random Access Range Configuration ──
        template <std::ranges::random_access_range RPrev, std::ranges::random_access_range RCurr,
                  std::ranges::random_access_range RNext, std::ranges::random_access_range ROut>
        void apply_element_stabilization(RPrev&& sp_prev, RCurr&& sp_curr, RNext&& sp_next,
                                         ROut&& sp_out) const
        {
            if (!m_apply_spec && !m_apply_weno)
            {
                std::ranges::copy(sp_curr, sp_out.begin());
                return;
            }

            if (m_apply_weno)
            {
                typename WENOLimiter<T>::RangeTypeGammas betas = {
                    m_weno.compute_smoothness(sp_prev), m_weno.compute_smoothness(sp_curr),
                    m_weno.compute_smoothness(sp_next)};

                const auto [is_shock, omegas] = m_sensor.detect_shock(betas);

                if (is_shock)
                {
                    m_weno.apply_weno_scalar(sp_prev, sp_curr, sp_next, sp_out, omegas);
                    if (m_apply_spec)
                    {
                        std::array<T, GLL<T>::MaxNumElem> m_scratch;
                        std::ranges::copy(sp_curr, m_scratch.begin());
                        m_spect.apply_filter(std::span(m_scratch.begin(), sp_out.size()), sp_out);
                    }
                    return;
                }
            }

            // Smooth flow fallback sequence
            if (m_apply_spec)
            {
                m_spect.apply_filter(sp_curr, sp_out);
            }
            else
            {
                std::ranges::copy(sp_curr, sp_out.begin());
            }
        }
    };

} // namespace numPDE
#endif
