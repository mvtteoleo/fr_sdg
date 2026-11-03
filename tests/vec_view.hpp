#pragma once
#include "../include/macro_and_libc.hpp"

namespace numPDE
{
    template <std::floating_point T>
    struct VecSpan
    {
      private:
        Vec          m_vec;
        std::span<T> m_span;

        static constexpr bool is_const = std::is_const_v<T>;
        using RawT                     = std::remove_const_t<T>;

      public:
        VecSpan()                          = delete;
        VecSpan(const VecSpan&)            = delete;
        VecSpan& operator=(const VecSpan&) = delete;

        explicit VecSpan(Vec v) : m_vec(v)
        {
            PetscInt size = 0;
            PetscCallVoid(VecGetLocalSize(m_vec, &size));

            if constexpr (is_const)
            {
                const RawT* ptr = nullptr;
                PetscCallVoid(VecGetArrayRead(m_vec, &ptr));
                m_span = {ptr, static_cast<std::size_t>(size)};
            }
            else
            {
                RawT* ptr = nullptr;
                PetscCallVoid(VecGetArray(m_vec, &ptr));
                m_span = {ptr, static_cast<std::size_t>(size)};
            }
        }

        ~VecSpan()
        {
            if constexpr (is_const)
            {
                const RawT* ptr = m_span.data();
                PetscCallVoid(VecRestoreArrayRead(m_vec, &ptr));
            }
            else
            {
                RawT* ptr = m_span.data();
                PetscCallVoid(VecRestoreArray(m_vec, &ptr));
            }
        }

        // Expose the span directly
        std::span<T> span() { return m_span; }
        T&           operator[](std::size_t i) { return m_span[i]; }
        std::size_t  size() const { return m_span.size(); }
    };

    using VecReadSpan  = VecSpan<const PetscReal>;
    using VecWriteSpan = VecSpan<PetscReal>;
} // namespace numPDE
