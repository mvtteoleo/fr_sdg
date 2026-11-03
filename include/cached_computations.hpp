#pragma once

#include "../include/reference_element.hpp"

namespace numPDE
{

    template <std::floating_point T>
    class CachedElementData
    {
      private:
        EigMat<T> m_D{};

        // Derivative of both Radau's polys + correction term
        EigVec<T> m_dR_dx{};
        EigVec<T> m_dL_dx{};

      public:
        explicit CachedElementData(const GLL<T>& gll, const T c_Rad = T{});
        const auto& get_dR() const { return m_dR_dx; };
        const auto& get_dL() const { return m_dL_dx; };
        const auto& get_diff_mat() const { return m_D; };

      private:
        EigMat<T> create_diff_matrix(const std::vector<T>& csi);
        void      build_Radaus_poly_deriv(const GLL<T>& gll, const T c_Rad);
    };
    template <std::floating_point T>
    CachedElementData<T>::CachedElementData(const GLL<T>& gll, const T c_Rad)
        : m_D{create_diff_matrix(gll.get_csi())}
    {
        build_Radaus_poly_deriv(gll, c_Rad);
    }

    // ==================== Class method definitions ======================================== //

    template <std::floating_point T>
    // Implementation based on the paper ```A New Class of High-Order Energy
    // Stable Flux Reconstruction Schemes P.E. Vincent · P. Castonguay · A.
    // Jameson```
    void CachedElementData<T>::build_Radaus_poly_deriv(const GLL<T>& gll, const T c_Rad)
    {
        const auto& csi = gll.get_csi();
        const int   n   = gll.get_n_nodes();
        const int   p   = gll.get_poly_deg();

        m_dR_dx.resize(n);
        m_dL_dx.resize(n);

        const T sign = (p % 2 == 0) ? T(1) : T(-1); // (-1)^k

#if true
        // Helper lambda to calculate factorials for the a_k term
        auto factorial = [](int num)
        {
            T fact = T(1);
            for (int i = 2; i <= num; ++i)
                fact *= T(i);
            return fact;
        };

        // Calculate (a_k * k!) = (2k)! / (2^k * k!) based on Formula 3.25
        const T p_fact = factorial(p);
        const T ak     = factorial(2 * p) / (std::pow(T(2), p) * p_fact * p_fact);

        const T ak_fact_k = ak * p_fact;
        // Calculate eta_k based on Formula 3.45
        const T eta_k = c_Rad * T(2 * p + 1) * ak_fact_k * ak_fact_k / T(2);

        std::ranges::for_each(std::views::iota(0, n),
                              [&](int i)
                              {
                                  const auto& xi = csi[i];

                                  const T dL  = boost::math::legendre_p_prime(p, xi);
                                  const T dLp = boost::math::legendre_p_prime(p - 1, xi);
                                  const T dLn = boost::math::legendre_p_prime(p + 1, xi);

                                  const auto term_ = (eta_k * dLp + dLn) / (1 + eta_k);
                                  // Final derivatives of g_R and g_L

                                  m_dL_dx[i] = sign * T(0.5) * (dL - term_);
                                  m_dR_dx[i] = T(0.5) * (dL + term_);
                              });
#else
        std::ranges::for_each(std::views::iota(0, n),
                              [&](int i)
                              {
                                  const auto& xi = csi[i];

                                  const T Lp   = boost::math::legendre_p(p, xi);
                                  const T dLp  = boost::math::legendre_p_prime(p, xi);
                                  const T dLp1 = boost::math::legendre_p_prime(p + 1, xi);

                                  // VCJH Correction term derivatives
                                  const T corr_R = c_Rad * (T(0.5) * Lp + (T(1) + xi) / T(2) * dLp);
                                  const T corr_L =
                                      c_Rad * (-T(0.5) * Lp + (T(1) - xi) / T(2) * dLp);

                                  m_dR_dx[i] = T(0.5) * (dLp + dLp1 + corr_R);
                                  m_dL_dx[i] = sign * T(0.5) * (dLp - dLp1 + corr_L);
                              });

#endif
    }

    template <std::floating_point T>
    EigMat<T> CachedElementData<T>::create_diff_matrix(const std::vector<T>& csi)
    {
        const int n = csi.size();
        EigMat<T> D = EigMat<T>::Zero(n, n);

        // Pass 1: a[i] = ∏_{k≠i} (csi[i] - csi[k])
        EigVec<T> a = EigVec<T>::Ones(n);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                if (i != k) a(i) *= csi[i] - csi[k];

        // Pass 2: populate D
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (i != j)
                {
                    D(i, j) = a(i) / (a(j) * (csi[i] - csi[j]));
                }
                else
                {
                    T s = T{};
                    for (int k = 0; k < n; ++k)
                        if (k != i) s += T(1) / (csi[i] - csi[k]);
                    D(i, i) = s;
                }
            }
        }
        return D;
    }
}; // namespace numPDE
