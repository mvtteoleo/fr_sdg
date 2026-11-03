#pragma once

#include "macro_and_libc.hpp"

namespace numPDE
{

    enum class VecType
    {
        Global,
        Local
    };

    template <std::size_t n_dims>
    struct PetscPtrDOF;

    template <>
    struct PetscPtrDOF<0>
    {
        using PtrType = PetscReal*;
    };

    // Nice recursive relation
    template <std::size_t n_dims>
    struct PetscPtrDOF
    {
        using PtrType = typename PetscPtrDOF<n_dims - 1>::PtrType*;
    };

    template <bool IsConst, std::size_t n_dims = 1>
    class PetscDOFView
    {
      private:
        DM  m_da{nullptr};
        Vec m_vec_orig{nullptr};
        Vec m_vec_work{nullptr};

        // Automatically resolve to const or non-const pointer
        using PtrType = std::conditional_t<IsConst, const typename PetscPtrDOF<n_dims>::PtrType,
                                           typename PetscPtrDOF<n_dims>::PtrType>;
        PtrType m_vec_ptr{nullptr};

        bool m_needs_scatter{false};

      public:
        PetscDOFView()                                = delete;
        PetscDOFView(const PetscDOFView&)             = delete;
        PetscDOFView& operator=(const PetscDOFView&)  = delete;
        PetscDOFView& operator=(PetscDOFView&& other) = delete;
        PetscDOFView(PetscDOFView&& other)            = delete;

        // Main Constructor
        explicit PetscDOFView(DM da, Vec v, VecType v_type)
            : m_da(da), m_vec_orig(v), m_needs_scatter(v_type == VecType::Local)
        {
            if constexpr (IsConst)
            {
                if (m_needs_scatter)
                {
                    // Global Read: Needs to scatter to a temporary local vector for ghost cells
                    PetscCallVoid(DMGetLocalVector(m_da, &m_vec_work));
                    PetscCallVoid(
                        DMGlobalToLocalBegin(m_da, m_vec_orig, INSERT_VALUES, m_vec_work));
                    PetscCallVoid(DMGlobalToLocalEnd(m_da, m_vec_orig, INSERT_VALUES, m_vec_work));
                    PetscCallVoid(DMDAVecGetArrayDOFRead(m_da, m_vec_work, (void*) &m_vec_ptr));
                }
                else
                {
                    // Local Read: Already ghosted, just get the array
                    m_vec_work = m_vec_orig;
                    PetscCallVoid(DMDAVecGetArrayDOFRead(m_da, m_vec_work, (void*) &m_vec_ptr));
                }
            }
            else if constexpr (!IsConst)
            {
                // Write directly into the owned global elements
                m_vec_work = m_vec_orig;
                PetscCallVoid(DMDAVecGetArrayDOF(m_da, m_vec_work, (void*) &m_vec_ptr));
            }
        }

        ~PetscDOFView()
        {
            if (m_vec_ptr)
            {
                if constexpr (IsConst)
                {
                    PetscCallVoid(DMDAVecRestoreArrayDOFRead(m_da, m_vec_work, (void*) &m_vec_ptr));
                    if (m_needs_scatter)
                    {
                        // Return the temporary local vector to the DM pool
                        PetscCallVoid(DMRestoreLocalVector(m_da, &m_vec_work));
                    }
                }
                else
                {
                    PetscCallVoid(DMDAVecRestoreArrayDOF(m_da, m_vec_work, (void*) &m_vec_ptr));
                }
            }
        }

        template <typename T = PetscReal>
            requires(n_dims == 1)
        std::span<std::conditional_t<IsConst, const T, T>> as_span(std::integral auto elem,
                                                                   std::integral auto n_dofs) const
        {
            return {m_vec_ptr[elem], static_cast<std::size_t>(n_dofs)};
        }

        template <typename T = PetscReal>
            requires(n_dims == 2)
        std::span<std::conditional_t<IsConst, const T, T>> as_span(std::integral auto elem_x,
                                                                   std::integral auto elem_y,
                                                                   std::integral auto n_dofs) const
        {
            return {m_vec_ptr[elem_y][elem_x], static_cast<std::size_t>(n_dofs)};
        }
    };

    template <bool IsConst, std::size_t n_dims = 1>
    struct ComplexDOFView
    {
        using T = std::conditional_t<IsConst, const CplxType, CplxType>;
        std::span<T>           m_data;
        const DMInfos<n_dims>& m_info;

        ComplexDOFView(std::vector<CplxType>& vec, const DMInfos<n_dims>& info)
            : m_data{vec}, m_info{info} {};

        // Mimics the exact same API as PetscDOFView!

        std::span<T> as_span(PetscInt elem, PetscInt ndofs) const
        {
            static_assert(n_dims == 1);
            return m_data.subspan((elem - m_info.ghost_start[0]) * ndofs, ndofs);
        }

        std::span<T> as_span(PetscInt elem_x, PetscInt elem_y, PetscInt ndofs) const
        {
            static_assert(n_dims == 2);
            // Flatten the 2D ghosted Cartesian grid into a 1D offset
            const PetscInt local_x = elem_x - m_info.ghost_start[0];
            const PetscInt local_y = elem_y - m_info.ghost_start[1];

            // Stride by the total width of the ghosted domain
            const PetscInt row_width = m_info.ghost_size[0];
            const PetscInt flat_idx  = (local_y * row_width) + local_x;

            return m_data.subspan(flat_idx * ndofs, ndofs);
        }
    };

    // Some nice aliases
    template <std::size_t n_dims = 1>
    using ReadCplxView = ComplexDOFView<true, n_dims>;
    template <std::size_t n_dims = 1>
    using WriteCplxView = ComplexDOFView<false, n_dims>;

    template <std::size_t n_dims = 1>
    using ReadDOFView = PetscDOFView<true, n_dims>;
    template <std::size_t n_dims = 1>
    using WriteDOFView = PetscDOFView<false, n_dims>;

    // ====================================================================
    // 1. COMPILE-TIME DIMENSION DETECTOR
    // ====================================================================
    template <typename T>
    constexpr std::size_t detect_dims = 1; // Primary template default fallback

    template <bool IsConst, std::size_t n_dims>
    constexpr std::size_t detect_dims<PetscDOFView<IsConst, n_dims>> = n_dims;

    template <bool IsConst, std::size_t n_dims>
    constexpr std::size_t detect_dims<ComplexDOFView<IsConst, n_dims>> = n_dims;

    // ====================================================================
    // 2. N-DIMENSIONAL VIEW CONCEPTS
    // ====================================================================
    template <class View, typename SpT>
    concept ViewConc = (detect_dims<std::remove_cvref_t<View>> == 1 &&
                        requires(View view, PetscInt elem, PetscInt n_dofs) {
                            { view.as_span(elem, n_dofs) } -> std::convertible_to<std::span<SpT>>;
                        }) ||
                       (detect_dims<std::remove_cvref_t<View>> == 2 &&
                        requires(View view, PetscInt elem_x, PetscInt elem_y, PetscInt n_dofs) {
                            {
                                view.as_span(elem_x, elem_y, n_dofs)
                            } -> std::convertible_to<std::span<SpT>>;
                        });

    template <class View, typename SpT>
    concept WriteViewConc = ViewConc<View, SpT>;

    template <class View, typename SpT>
    concept ReadViewConc = ViewConc<View, const SpT>;

} // namespace numPDE
