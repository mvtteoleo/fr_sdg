#include "../include/factory.hpp"

using namespace numPDE;
using Real = PetscReal;
int main(int argc, char* argv[])
{
    Parser      parser(&argc, &argv);
    const auto& config = parser.get_options();
    const auto  data   = parser.parse_sod_shock_vals(true);

    /* Initialize problem */
    auto prob = ProblemFactory2D<ProblemType2D::CircularExplosion>::create(config, data);
    /* Solve */
    prob->solve();
    auto errs = prob->compute_error();

    numPDE::println("Error logs");
    numPDE::println("Rho : {:3e}", errs[0]);
    numPDE::println("RhoU : {:3e}", errs[1]);
    numPDE::println("RhoV : {:3e}", errs[2]);
    numPDE::println("RhoE : {:3e}", errs[3]);

    /* Post proc */
    return 0;
}
