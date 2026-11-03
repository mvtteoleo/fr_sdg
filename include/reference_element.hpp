#pragma once
#include "macro_and_libc.hpp"
#include <exception>
#include <numbers>
#include <print>

namespace numPDE
{

    template <std::floating_point T = PetscReal>
    class GLL
    {
      private:
        /* WARNING!! */
        /*
         It may be more benefitial to fix the maximum order of the polynomial and
         stack allocate the data
        */
        std::size_t    m_poly_deg;
        std::vector<T> m_weights;
        std::vector<T> m_csi;

      public:
        static constexpr std::size_t MaxNumElem = 9;
        explicit GLL(std::size_t p_deg);
        // Getters (No setters, just constructor!!)
        auto        get_n_nodes() const { return m_poly_deg + 1; };
        auto        get_poly_deg() const { return m_poly_deg; };
        const auto& get_csi() const { return m_csi; };
        const auto& get_ws() const { return m_weights; };

        void print() const;

      private:
        void initialize_csi_ws(std::size_t p_deg);
        void compute_csi(const EigMat<T>& J);
        void compute_ws();
        auto compute_J_mat(std::size_t n, T alpha, T beta);

        // A sanity check, at poly_deg=270 is still ok
        void check_ws()
        {
            // Asserts are ignored when using -O3 !!
            assert(std::abs(std::reduce(m_weights.begin(), m_weights.end(), T{}) - 2) <= 1e-14);
        };
    };

    /* METHODS DEFINITIONS */

    template <std::floating_point T>
    GLL<T>::GLL(std::size_t p_deg)
        : m_poly_deg(p_deg), m_weights(p_deg + 1, T{}), m_csi(p_deg + 1, T{})
    {
        assert(p_deg >= 1 && "Degree of the polynomial space needs to be at least 1");
        if (MaxNumElem < m_poly_deg + 1)
        {
            std::println("Polynomial degree too high, modify GLL::MaxNumElem and recompile");
            std::terminate();
        }

        // TODO
        // This has to be done just on rank to avoid overheads
        this->initialize_csi_ws(p_deg);
        this->check_ws();
    }

    template <std::floating_point T>
    void GLL<T>::initialize_csi_ws(std::size_t p_deg)
    {
        if (p_deg >= 2)
        {
            const auto J = this->compute_J_mat(p_deg - 1, 1.0, 1.0);
            compute_csi(J);
            compute_ws();
        }
        if (p_deg == 1)
        {
            this->m_csi[0] = T{-1};
            this->m_csi[1] = T{+1};

            this->m_weights[0] = T{1};
            this->m_weights[1] = T{1};
        }
        if (p_deg == 0)
        {
            this->m_csi[0]     = T{0};
            this->m_weights[0] = T{2};
        }
    }

    template <std::floating_point T>
    void GLL<T>::compute_csi(const EigMat<T>& J)
    {
        m_csi[0]                = -1.0;
        m_csi[m_csi.size() - 1] = 1.0;

        Eigen::SelfAdjointEigenSolver<EigMat<T>> solver(J, Eigen::EigenvaluesOnly);

        auto vals = solver.eigenvalues(); // already sorted ascending

        std::copy_n(vals.begin(), vals.size(), std::addressof(m_csi[1]));
    }

    template <std::floating_point T>
    void GLL<T>::compute_ws()
    {
        const double denominator_const = m_poly_deg * (m_poly_deg + 1);

        std::ranges::transform(m_csi, m_weights.begin(),
                               [deg = m_poly_deg, &denominator_const](auto csi_val)
                               {
                                   double p_n = std::legendre(deg, csi_val);
                                   return 2.0 / (denominator_const * p_n * p_n);
                               });
    }

    // Returns the roots of P_n^(alpha,beta) — equivalent to roots_jacobi(n, alpha, beta)
    template <std::floating_point T>
    auto GLL<T>::compute_J_mat(std::size_t n, T alpha, T beta)
    {
        // Diagonal entries of the Jacobi matrix
        EigVec<T> diag(n), subdiag(n - 1);

        for (std::size_t k = 0; k < n; ++k)
        {
            T ab    = alpha + beta;
            T a2b2  = beta * beta - alpha * alpha;
            T denom = (2.0 * k + ab) * (2.0 * k + ab + 2.0);
            diag(k) = (denom == 0.0) ? 0.0 : a2b2 / denom;
        }

        for (std::size_t k = 1; k < n; ++k)
        {
            T ab           = alpha + beta;
            T num          = 4.0 * k * (k + alpha) * (k + beta) * (k + ab);
            T d            = (2.0 * k + ab) * (2.0 * k + ab);
            T den          = d * (2.0 * k + ab - 1.0) * (2.0 * k + ab + 1.0);
            subdiag(k - 1) = std::sqrt(num / den);
        }

        // Symmetric tridiagonal matrix
        EigMat<T> J    = EigMat<T>::Zero(n, n);
        J.diagonal()   = diag;
        J.diagonal(-1) = subdiag;
        J.diagonal(1)  = subdiag;
        return J;
    }
    template <std::floating_point T>
    void GLL<T>::print() const
    {
        std::cout << "\n==========================\n";
        std::cout << "\nOrder : " << this->get_poly_deg();

        std::cout << "\nCsi    : ";

        std::copy(this->get_csi().begin(), this->get_csi().end(),
                  std::ostream_iterator<T>(std::cout, " "));

        std::cout << "\nWeights: ";
        std::copy(this->get_ws().begin(), this->get_ws().end(),
                  std::ostream_iterator<T>(std::cout, " "));

        std::cout << "\n==========================\n";
        return;
    }
}; // namespace numPDE
