import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# ====================================================================
# ANALYTICAL FORMULATION (MOC / ISENTROPIC COMPRESSION)
# ====================================================================
def prandtl_meyer(M, gamma=1.4):
    """Returns Prandtl-Meyer angle in radians"""
    res = np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(
        np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))
    )
    res -= np.arctan(np.sqrt(M**2 - 1))
    return res


def inverse_pm(nu_target, gamma=1.4, M_guess=1.5):
    """Numerically inverts the PM function to find the Mach number"""
    func = lambda M: prandtl_meyer(M, gamma) - nu_target
    return fsolve(func, M_guess)[0]


# 1. Freestream Conditions


def get_wedge_sol():
    M_inf = 1.6903
    rho_inf = 1.0
    nu_inf = prandtl_meyer(M_inf)

    # 2. Domain Generation
    x_wall = np.linspace(0, 1.0, 500)
    mach_analytical = np.zeros_like(x_wall)
    rho_analytical = np.zeros_like(x_wall)

    # 3. Calculate Analytical States
    for i, x in enumerate(x_wall):
        # Match the C++ Geometry derivative
        if x <= 0.3:
            dh_dx = 0.0  # Flat[cite: 2]
        elif x <= 0.5:
            dh_dx = x - 0.3  # Parabolic transition[cite: 2]
        else:
            dh_dx = 0.2  # Straight wedge[cite: 2]

        theta = np.arctan(dh_dx)

        # Isentropic compression decreases the PM angle
        nu_local = nu_inf - theta

        # Invert to find local Mach
        M_local = inverse_pm(nu_local)
        mach_analytical[i] = M_local

        # Isentropic Density Relation
        T_ratio = (1.0 + 0.2 * M_inf**2) / (1.0 + 0.2 * M_local**2)
        rho_analytical[i] = rho_inf * (T_ratio) ** 2.5

    return {"mesh": x_wall, "rho": rho_analytical, "mach": mach_analytical}


if __name__ == "__main__":
    out = get_wedge_sol()
    # ====================================================================
    # PLOTTING
    # ====================================================================
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: Density tracking
    color = "tab:blue"
    ax1.set_xlabel("Axial Coordinate ($x$)")
    ax1.set_ylabel("Analytical Density ($\\rho$)", color=color)
    ax1.plot(out["mesh"], out["rho"], color=color, lw=2.5, label="Density")
    ax1.tick_params(axis="y", labelcolor=color)

    # Right axis: Mach number tracking
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Analytical Mach Number ($M$)", color=color)
    ax2.plot(out["mesh"], out["mach"], color=color, lw=2.5, label="Mach Number")

    # Geometry Markers
    ax2.axvline(
        x=0.3, color="gray", linestyle=":", alpha=0.7, label="Compression Start (x=0.3)"
    )
    ax2.axvline(
        x=0.5, color="gray", linestyle="--", alpha=0.7, label="Straight Wedge (x=0.5)"
    )

    ax2.tick_params(axis="y", labelcolor=color)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left")

    plt.title("Exact Analytical Solution: Isentropic Compression Over Wedge")
    fig.tight_layout()
    plt.show()
