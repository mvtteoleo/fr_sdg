#pragma once
#include "macro_and_libc.hpp"

namespace numPDE
{
    template <std::size_t Dims = 1>
    struct DMInfos
    {
        std::array<PetscInt, Dims> start{};
        std::array<PetscInt, Dims> size{};
        std::array<PetscInt, Dims> max{};
        std::array<PetscInt, Dims> ghost_start{};
        std::array<PetscInt, Dims> ghost_size{};
        std::array<PetscInt, Dims> ghost_max{};

        DMInfos() = default;

        explicit DMInfos(DM da)
        {
            if constexpr (Dims == 1)
            {
                PetscCallVoid(DMDAGetCorners(da, &start[0], NULL, NULL, &size[0], NULL, NULL));
                PetscCallVoid(DMDAGetGhostCorners(da, &ghost_start[0], NULL, NULL, &ghost_size[0],
                                                  NULL, NULL));
                max[0]       = start[0] + size[0];
                ghost_max[0] = ghost_start[0] + ghost_size[0];
            }
            else if constexpr (Dims == 2)
            {
                // Note: PETSc 2D corners are typically (x, y)
                PetscCallVoid(
                    DMDAGetCorners(da, &start[0], &start[1], NULL, &size[0], &size[1], NULL));
                PetscCallVoid(DMDAGetGhostCorners(da, &ghost_start[0], &ghost_start[1], NULL,
                                                  &ghost_size[0], &ghost_size[1], NULL));
                max[1]       = start[1] + size[1];
                ghost_max[1] = ghost_start[1] + ghost_size[1];
                max[0]       = start[0] + size[0];
                ghost_max[0] = ghost_start[0] + ghost_size[0];
            }
        }
        auto get_global_elem_range(std::size_t i = 0uz) const
        {
            assert(i < Dims);
            return std::views::iota(this->start[i], this->max[i]);
        };
        auto get_ghosted_elem_range(std::size_t i = 0uz) const
        {
            assert(i < Dims);
            return std::views::iota(this->ghost_start[i], this->ghost_max[i]);
        };
    };

} // namespace numPDE
