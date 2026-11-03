#include "../include/cached_computations.hpp"
#include "../include/macro_and_libc.hpp"
#include "../include/reference_element.hpp"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <string>

void verify_sbp_property(std::size_t poly_degree)
{
    // 1. Initialize your classes (c_Rad = 0.0 recovers pure Nodal DG)
    numPDE::GLL<double>               gll(poly_degree);
    numPDE::CachedElementData<double> cache(gll, 0.0);

    const auto& ws      = gll.get_ws();
    const auto& D       = cache.get_diff_mat();
    int         n_nodes = gll.get_n_nodes();

    // 2. Build the Mass Matrix M (Diagonal matrix of GLL weights)
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> M(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
    {
        M.diagonal()[i] = ws[i];
    }

    // 3. Compute Q = M * D
    Eigen::MatrixXd Q = M * D;

    // 4. Compute SBP = Q + Q^T
    Eigen::MatrixXd SBP = Q + Q.transpose();

    // 5. Build the expected Boundary Matrix B: diag(-1, 0, ..., 0, 1)
    Eigen::MatrixXd B           = Eigen::MatrixXd::Zero(n_nodes, n_nodes);
    B(0, 0)                     = -1.0;
    B(n_nodes - 1, n_nodes - 1) = 1.0;

    // 6. Check the difference
    double error_norm = (SBP - B).norm();

    std::cout << "======================================\n";
    std::cout << " SBP Verification for Polynomial Order: " << poly_degree << "\n";
    std::cout << "======================================\n";
    std::cout << "Error Norm || (MD + D^T M) - B || = " << std::scientific << error_norm << "\n\n";

    // Optional: Print the matrices for low polynomial orders to see it visually
    if (poly_degree <= 3)
    {
        std::cout << "Calculated SBP Matrix (MD + D^T M):\n"
                  << std::fixed << std::setprecision(2) << SBP << "\n";
    }
}

int main(int argc, char* argv[])
{

    int max_p = 4;
    if (argc > 1) max_p = std::stoul(argv[1]);

    for (auto n : std::views::iota(2, max_p + 1))
        verify_sbp_property(n);
    return 0;
}
