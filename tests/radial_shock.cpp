#include "../include/my_lib.hpp"
using namespace numPDE;

int main(int argc, char* argv[])
{
    Parser      parser(&argc, &argv);
    const auto& config = parser.get_options();
    // const auto  data   = parser.parse_acoust_wave_data(false);
    const auto data = parser.parse_sod_shock_vals(true);
    /* Initialize problem */
    auto prob = ProblemFactory<ProblemType::CircShockRadial>::create(config, data);
    /* Solve */
    prob->solve();

    /* Post proc */

    return 0;

    return 0;
}
