
#include "../include/factory.hpp"

using namespace numPDE;
using Real = PetscReal;
int main(int argc, char* argv[])
{
    Parser      parser(&argc, &argv);
    const auto& config = parser.get_options();
    // const auto  data   = parser.parse_acoust_wave_data(false);
    // const auto data = parser.parse_sod_shock_vals(true);
    VortexData data{};
    /*
    data.beta = 0.5;
    data.u_0  = 2.0;
    data.v_0  = 2.0;
    */
    /* Initialize problem */
    auto prob = ProblemFactory2D<ProblemType2D::DensityWave>::create(config, data);
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
