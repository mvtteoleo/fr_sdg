#include "../include/my_lib.hpp"
#include <random>

using Real = PetscReal;
using namespace numPDE;

int main(int argc, char* argv[])
{
    Parser            parser(&argc, &argv);
    const auto&       opts     = parser.get_options();
    const auto        poly_deg = opts.get_poly_deg(0);
    numPDE::GLL<Real> gll{poly_deg};

    numPDE::SpectralFilter<Real> filter{gll, opts};
    const auto&                  filt_mat = filter.get_filt_mat();

    std::println("The filter matrix is: ");
    std::cout << filt_mat << std::endl;

    // Compute eigenvalues
    const auto eigs = filt_mat.eigenvalues();

    std::println("The eigenvaluse of the filter matrix are: ");
    std::cout << eigs << std::endl;

    EigVec<Real> v_in  = EigVec<Real>::Zero(poly_deg + 1);
    EigVec<Real> v_out = v_in;

    std::mt19937                         g;
    std::uniform_real_distribution<Real> u;
    std::generate(v_in.begin(), v_in.end(), [&]() { return u(g); });

    v_out = filt_mat * v_in;

    std::println("The unfiltered values are: ");
    std::cout << v_in << std::endl;

    std::println("The filtered values are: ");
    std::cout << v_out << std::endl;

    return 0;
}
