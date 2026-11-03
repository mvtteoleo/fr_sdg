
#include "../include/my_lib.hpp"
#include <fstream>

using namespace numPDE;
using Real = PetscReal;

int main(int argc, char* argv[])
{
    Parser      parser(&argc, &argv);
    const auto& config = parser.get_options();
    const auto  data   = parser.parse_acoust_wave_data(true);
    // TODO make the configs, run the convergence tests (dt, n_elems, P)
    /* Initialize problem */
    auto prob = ProblemFactory<ProblemType::AcousticWave>::create(config, data);
    /* Solve */
    prob->solve();

    /* Post proc */
    auto err = prob->compute_error();

    PetscPrintf(PETSC_COMM_WORLD, "Rho : %1.15e\n", err[0]);
    PetscPrintf(PETSC_COMM_WORLD, "RhoU: %1.15e\n", err[1]);
    PetscPrintf(PETSC_COMM_WORLD, "RhoE: %1.15e\n", err[2]);

    return 0;
}
