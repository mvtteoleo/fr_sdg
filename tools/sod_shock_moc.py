import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def exact_riemann_solver(
    rho_L, u_L, p_L, rho_R, u_R, p_R, t, x_arr, x_disc=0.0, gamma=1.4
):
    """
    Computes the exact solution to the 1D Euler Riemann problem.
    """
    # Calculate initial sound speeds
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    p_max = 10 * max(p_R, p_L)

    # 1. ROOT FINDING FOR STAR REGION PRESSURE (p_star)
    # The function f(p) relates pressure across a shock or rarefaction
    def f_K(p, rho_K, u_K, p_K, a_K):
        if p > p_K:  # Shock wave
            A = 2.0 / ((gamma + 1.0) * rho_K)
            B = (gamma - 1.0) / (gamma + 1.0) * p_K
            return (p - p_K) * np.sqrt(A / (p + B))
        else:  # Rarefaction wave
            return (
                2.0
                * a_K
                / (gamma - 1.0)
                * ((p / p_K) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
            )

    # The matching condition across the contact discontinuity
    def f_total(p):
        return f_K(p, rho_L, u_L, p_L, a_L) + f_K(p, rho_R, u_R, p_R, a_R) + (u_R - u_L)

    # Solve for p_star using a robust root finder
    res = root_scalar(f_total, bracket=[1e-12, p_max], method="brentq")
    p_star = res.root

    # 2. CALCULATE STAR REGION VELOCITY (u_star)
    u_star = 0.5 * (
        u_L
        + u_R
        + f_K(p_star, rho_R, u_R, p_R, a_R)
        - f_K(p_star, rho_L, u_L, p_L, a_L)
    )

    # Calculate densities in the star region
    if p_star > p_L:  # Left Shock
        rho_star_L = (
            rho_L
            * (p_star / p_L + (gamma - 1) / (gamma + 1))
            / ((gamma - 1) / (gamma + 1) * p_star / p_L + 1.0)
        )
    else:  # Left Rarefaction
        rho_star_L = rho_L * (p_star / p_L) ** (1.0 / gamma)

    if p_star > p_R:  # Right Shock
        rho_star_R = (
            rho_R
            * (p_star / p_R + (gamma - 1) / (gamma + 1))
            / ((gamma - 1) / (gamma + 1) * p_star / p_R + 1.0)
        )
    else:  # Right Rarefaction
        rho_star_R = rho_R * (p_star / p_R) ** (1.0 / gamma)

    a_star_L = np.sqrt(gamma * p_star / rho_star_L)
    a_star_R = np.sqrt(gamma * p_star / rho_star_R)

    # 3. SAMPLE THE SOLUTION ALONG CHARACTERISTICS (S = x/t)
    rho = np.zeros_like(x_arr)
    u = np.zeros_like(x_arr)
    p = np.zeros_like(x_arr)

    for i, x in enumerate(x_arr):
        S = (x - x_disc) / t if t > 0 else 0.0

        if S < u_star:  # To the left of the contact discontinuity
            if p_star > p_L:  # Left Shock
                S_L = u_L - a_L * np.sqrt(
                    (gamma + 1) / (2 * gamma) * p_star / p_L + (gamma - 1) / (2 * gamma)
                )
                if S < S_L:
                    rho[i], u[i], p[i] = rho_L, u_L, p_L
                else:
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
            else:  # Left Rarefaction
                S_HL = u_L - a_L
                S_TL = u_star - a_star_L
                if S < S_HL:
                    rho[i], u[i], p[i] = rho_L, u_L, p_L
                elif S > S_TL:
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
                else:  # Inside the fan
                    u_fan = 2.0 / (gamma + 1.0) * (a_L + (gamma - 1.0) / 2.0 * u_L + S)
                    a_fan = u_fan - S
                    p_fan = p_L * (a_fan / a_L) ** (2.0 * gamma / (gamma - 1.0))
                    rho_fan = rho_L * (a_fan / a_L) ** (2.0 / (gamma - 1.0))
                    rho[i], u[i], p[i] = rho_fan, u_fan, p_fan

        else:  # To the right of the contact discontinuity
            if p_star > p_R:  # Right Shock
                S_R = u_R + a_R * np.sqrt(
                    (gamma + 1) / (2 * gamma) * p_star / p_R + (gamma - 1) / (2 * gamma)
                )
                if S > S_R:
                    rho[i], u[i], p[i] = rho_R, u_R, p_R
                else:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
            else:  # Right Rarefaction
                S_HR = u_R + a_R
                S_TR = u_star + a_star_R
                if S > S_HR:
                    rho[i], u[i], p[i] = rho_R, u_R, p_R
                elif S < S_TR:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
                else:  # Inside the fan
                    u_fan = 2.0 / (gamma + 1.0) * (-a_R + (gamma - 1.0) / 2.0 * u_R + S)
                    a_fan = S - u_fan
                    p_fan = p_R * (a_fan / a_R) ** (2.0 * gamma / (gamma - 1.0))
                    rho_fan = rho_R * (a_fan / a_R) ** (2.0 / (gamma - 1.0))
                    rho[i], u[i], p[i] = rho_fan, u_fan, p_fan

    return rho, u, p


# ==========================================
# TEST CASE: Sod Shock Tube
# ==========================================
if __name__ == "__main__":
    # Domain
    x = np.linspace(0, 10, 500)
    t = 0.3  # Evaluate at time t

    # Standard Sod Shock Tube Left and Right States
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1

    # Calculate exact solution
    rho_exact, u_exact, p_exact = exact_riemann_solver(
        rho_L,
        u_L,
        p_L,
        rho_R,
        u_R,
        p_R,
        t,
        x,
        x0=5.0,  # Shock starts in the middle of the domain (x=5)
    )

    # --- Overlay your C++ Data Here ---
    # cpp_data = np.loadtxt("your_cpp_output.csv", delimiter=",")
    # x_cpp = cpp_data[:, 0]
    # rho_cpp = cpp_data[:, 1]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, rho_exact, "k-", linewidth=2, label="Rho Exact Analytical Solution")
    plt.plot(
        x,
        u_exact * rho_exact,
        "r-",
        linewidth=2,
        label="RhoU Exact Analytical Solution",
    )
    # plt.plot(x_cpp, rho_cpp, 'r--', label='C++ Spectral Solver')

    plt.title(f"Sod Shock Tube Exact Solution at t={t}")
    plt.xlabel("Position (x)")
    plt.ylabel("Density (rho)")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.show()
