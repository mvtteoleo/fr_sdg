#pragma once

/******* EIGEN *******/
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

/******* PETSc *******/
#include "petscvec.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscerror.h>
#include <petscsnes.h>
#include <petscts.h>

/******* BOOST *******/
#include <boost/math/special_functions/legendre.hpp>

/******* C++ STD *******/
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <print>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace numPDE
{
    // Some aliases to avoid fake templates
    template <std::floating_point T>
    using EigMat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

    template <std::floating_point T>
    using EigVec = Eigen::Vector<T, Eigen::Dynamic>;

    template <std::floating_point T>
    using EigMap = Eigen::Map<EigVec<T>>;

    using CplxType = std::complex<PetscReal>;

    template <typename T>
    struct is_complex_floating_point : public std::false_type
    {
    };

    template <typename T>
    struct is_complex_floating_point<std::complex<T>>
        : public std::bool_constant<std::is_floating_point_v<T>>
    {
        using value_type = T;
    };

    template <typename T>
    concept real_or_complex_floating_point =
        std::floating_point<T> || is_complex_floating_point<std::remove_const_t<T>>::value;

    bool is_rank_0()
    {
        int rank{};
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        return static_cast<bool>(!rank);
    };

    // CSD-Safe Absolute Value
    template <real_or_complex_floating_point T>
    T constexpr smooth_abs(const T& val)
    {
        constexpr T eps{1e-7};
        return T{std::sqrt(val * val + eps * eps)} - eps;

        //  if constexpr (is_complex_floating_point<std::remove_const_t<T>>::value)
        //      return (val.real() >= 0.0) ? val : -val;
        //  else
        //      return std::abs(val);
    }

    // CSD-Safe Max
    template <real_or_complex_floating_point T>
    T constexpr smooth_max(const T& a, const T& b)
    {
        return 0.5 * (a + b + smooth_abs(a - b));
        //  if constexpr (is_complex_floating_point<std::remove_const_t<T>>::value)
        //      return (a.real() > b.real()) ? a : b;
        //  else
        //      return std::max(a, b);
    }

    // CSD-Safe Min
    template <real_or_complex_floating_point T>
    T constexpr smooth_min(const T& a, const T& b)
    {
        return 0.5 * (a + b - smooth_abs(a - b));
        //  if constexpr (is_complex_floating_point<std::remove_const_t<T>>::value)
        //      return (a.real() < b.real()) ? a : b;
        //  else
        //      return std::min(a, b);
    }

    // Simple function to track simply where /0 may occur
    // TODO Check the correctness of the complex case implementation
    template <real_or_complex_floating_point T>
    auto wrap_positive(T rho)
    {
        // Normal floating point case
        if constexpr (std::is_floating_point_v<T>)
        {
            return std::max(std::numeric_limits<T>::min(), rho);
        }
        // Complex case
        if constexpr (is_complex_floating_point<std::remove_const_t<T>>::value)
        {
            // This case is a bit tricky.
            // In the case I have a rho = rho_r + i*rho.
            // If im too close to 0 (ie floor) I want the
            // solver to understand that this direction is WRONG.
            const auto floor = std::numeric_limits<typename T::value_type>::min();
            if (std::real(rho) < floor)
                return T(floor, 0.0);
            else
                return rho;
        }
    }

    // Helper wrapper that preserves compile-time format string checking
    template <typename... Args>
    void println(std::format_string<Args...> fmt, Args&&... args)
    {
        if (numPDE::is_rank_0())
        {
            std::println(fmt, std::forward<Args>(args)...);
        }
    };
    enum class Axis
    {
        X = 0,
        Y = 1
    };
    enum class CoordinateSystem
    {
        Cartesian,
        Radial,
        Curvilinear
    };
} // namespace numPDE
