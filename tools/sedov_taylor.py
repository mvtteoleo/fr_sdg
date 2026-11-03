#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp, simpson
import matplotlib.pyplot as plt

from post_proc import RiemannIC


def solve_adimensional_2d(gamma, eps, q=2):

    def A_operator(xi, Y):
        # State vector Y matches the order [Omega, V, Z]
        Omega = Y[0]
        V = Y[1]
        Z = Y[2]

        eta = V - 0.5
        Delta = eta**2 - gamma * Z / Omega

        # Explicitly calculate the forcing vector components (b)
        f0 = -2 * Omega * V
        f1 = -V * (V - 1.0) - 2 * Z / Omega
        f2 = -2 * (V - 1.0)

        # Explicitly perform the row-by-column multiplication (A^-1 * b)
        factor = 1.0 / (xi * Delta)

        # R0 computes V'
        R1 = factor * ((-gamma * Z / Omega**2) * f0 + (eta) * f1 + (-Z / Omega) * f2)
        # R1 computes Omega'
        R0 = factor * ((eta) * f0 + (-Omega) * f1 + (Z / eta) * f2)
        # R2 computes Z'
        R2 = factor * (
            (gamma * eta * Z / Omega) * f0 + (-gamma * Z) * f1 + (eta * Z) * f2
        )

        # MUST return in the order of Y: [Omega', V', Z']
        return [R0, R1, R2]

    # Rankine-Hugoniot boundary conditions at the shock front (xi = 1)
    D_1 = (gamma + 1) / (gamma - 1)
    V_1 = 1 / (gamma + 1)
    P_1 = 1 / (2 * (gamma + 1))

    # Integrating backwards from 1.0 down to eps
    xi_eval = np.linspace(1.0, eps, 10000)

    sol = solve_ivp(
        A_operator,
        [1.0, eps],
        [D_1, V_1, P_1],
        t_eval=xi_eval,
        method="BDF",  # BDF is perfect here due to stiffness near the origin
    )

    # Reverse arrays so xi goes from 0 to 1 for Simpson's rule and plotting
    xi = sol.t[::-1]
    D, V, P = sol.y
    D = D[::-1]
    V = V[::-1]
    P = P[::-1]

    # Filter out unphysical negative values that might occur right at the singularity
    condition = (D > 0) & (V > 0) & (P > 0)
    xi = xi[condition]
    D = D[condition]
    V = V[condition]
    P = P[condition]

    # Energy integral J with the correct xi**3 weighting for 2D cylindrical
    integrand = (0.5 * D * V**2 + P / (gamma - 1)) * (xi**3)
    J = simpson(integrand, x=xi)

    # Calculate beta parameter
    beta = (2 * np.pi * J) ** (-0.25)

    return beta, xi, D, V, P


def to_dimensional(data, time, E0, rho_0):

    beta, xi, D, V, P = data
    R_final = beta * (E0 / rho_0) ** 0.25 * time**0.5

    r = R_final * xi
    vel = r * V / time
    density = rho_0 * D

    pressure = rho_0 * r**2 * P / time**2

    return r, density, vel, pressure


if __name__ == "__main__":

    # Ratio of specific heats (e.g., Monoatomic gas)
    gamma = 5 / 3.0

    # Small number to avoid division by zero at the exact center
    eps = 1.0e-03

    data = solve_adimensional_2d(gamma, eps)
    beta, xi, D, V, P = data

    time = 1.25
    p_0 = 1000000
    radius = 1.0
    e0 = 2 * np.pi * radius**2 * p_0 / (gamma - 1.0)
    print(e0)
    r, density, vel, pres = to_dimensional(data, time, e0, radius)

    print(f"Integration successful. Calculated Beta = {beta:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(r, density, label=r"Density")
    plt.plot(r, pres, label=r"Pressure")
    plt.plot(r, vel, label=r"velocity")
    plt.xlabel(r"Radial distance from the initial shock", fontsize=12)
    plt.ylabel("Physical Properties", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plotting the normalized results
    plt.figure(figsize=(8, 6))
    plt.title(
        rf"Cylindrical Sedov-Taylor Blast Wave"
        "\n"
        rf"$\beta = {beta:.3f}$, $\gamma = {gamma:.2f}$"
    )

    plt.plot(xi, D / D[-1], label=r"Density ($D/D_1$)", color="blue", linewidth=2)
    plt.plot(xi, P / P[-1], label=r"Pressure ($P/P_1$)", color="red", linewidth=2)
    plt.plot(xi, V / V[-1], label=r"Velocity ($V/V_1$)", color="green", linewidth=2)

    plt.xlabel(r"Similarity Variable $\xi = r/R(t)$", fontsize=12)
    plt.ylabel("Normalized Properties", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.2)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
