#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    // Gas and Geometry Parameters
    const double gamma    = 1.4;
    const double nu       = 2.0; // 2D Cylindrical
    std::size_t  N_points = (argc > 1) ? std::stoul(argv[1]) : 5000;

    // Velocity Boundaries (Shock front to just outside the core)
    double V0    = 4.0 / ((nu + 2.0) * (gamma + 1.0));
    double Vc    = 2.0 / ((nu + 2.0) * gamma);
    double V_end = Vc + 1e-4; // Stop slightly short of center to prevent Z -> infinity

    // Dimensionless Sound Speed Squared (z)
    auto z_func = [gamma, nu](double V)
    {
        double num = (gamma - 1.0) * V * V * (V - 2.0 / (nu + 2.0));
        double den = 2.0 * (2.0 / ((nu + 2.0) * gamma) - V);
        return num / den;
    };

    // The inverted ODE: d(ln Csi) / dV
    auto dlnlambda_dV = [gamma, nu, &z_func](double V)
    {
        double z    = z_func(V);
        double term = V - 2.0 / (nu + 2.0);
        double num  = z - term * term;
        double den  = V * (V - 1.0) * term + nu * (2.0 / ((nu + 2.0) * gamma) - V) * z;
        return num / den;
    };

    // Sedov Density Constant
    double C = (8.0 * gamma * (gamma - 1.0)) / (std::pow(nu + 2.0, 2) * std::pow(gamma + 1.0, 2)) *
               std::pow((gamma - 1.0) / (gamma + 1.0), gamma - 2.0) *
               (4.0 / ((nu + 2.0) * (gamma + 1.0)) - 2.0 / (nu + 2.0));

    // Baseline Normalizers
    double z0    = z_func(V0);
    double term0 = V0 - 2.0 / (nu + 2.0);
    double R0    = std::pow((z0 * term0) / C, 1.0 / (gamma - 2.0));

    // Arrays to hold your specific variables
    std::vector<double> Csi_sol, Z_sol, Omega_sol, V_sol;

    // 1. Inject the Shock Front initial conditions
    double Omega0 = (gamma + 1.0) / (gamma - 1.0); // Exact shock jump for Density
    double Z0     = (z0 * Omega0) / gamma;         // Exact shock jump for Pressure ratio

    Csi_sol.push_back(1.0);
    Z_sol.push_back(Z0);
    Omega_sol.push_back(Omega0);
    V_sol.push_back(V0);

    // 2. Integration Loop
    double dV               = (V0 - V_end) / N_points;
    double current_integral = 0.0;
    double prev_integrand   = dlnlambda_dV(V0);

    for (int i = 1; i <= N_points; ++i)
    {
        double V = V0 - i * dV;

        // Trapezoidal Integration for Space (Csi)
        double integrand = dlnlambda_dV(V);
        current_integral += 0.5 * (prev_integrand + integrand) * dV;
        double csi = std::exp(-current_integral);

        // Algebra for Z and Omega
        double z    = z_func(V);
        double term = V - 2.0 / (nu + 2.0);

        double R_val = std::pow((z * term * std::pow(csi, nu + 2.0)) / C, 1.0 / (gamma - 2.0));
        double D     = R_val / R0; // Normalized density (0 to 1)

        // Map to your specific variables
        double Omega = D * Omega0;
        double Z_val = (z * Omega) / gamma;

        // Store
        Csi_sol.push_back(csi);
        Z_sol.push_back(Z_val);
        Omega_sol.push_back(Omega);
        V_sol.push_back(V);

        prev_integrand = integrand;
    }

    // 3. Write to CSV
    std::string   filename = "sedov_variables.csv";
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << "Csi,Z,Omega,V\n";
        file << std::fixed << std::setprecision(8);
        for (size_t i = 0; i < Csi_sol.size(); ++i)
        {
            file << Csi_sol[i] << "," << Z_sol[i] << "," << Omega_sol[i] << "," << V_sol[i] << "\n";
        }
        file.close();
        std::cout << "Successfully wrote exact solution to " << filename << "\n";
    }
    else
    {
        std::cerr << "Failed to open file for writing.\n";
    }

    return 0;
}
