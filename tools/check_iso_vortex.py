#!/usr/bin/env python3
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from plot_func import plot_flow_fields
from plot import csi_ws


def get_sim_cmd(
    test_running,
    N_ELEMS,
    POLY_DEG,
    DT,
    MAX_TIME,
    FILENAME,
    BOX_L,
    CHANGE_VAR,
    RADIUS=4,
    implicit=True,
    MPI_PROCS=None,
    WRITE_SOL=True,
):

    executable = {
        "vortex": "./build/test/test_2d",
        "acoustic": "./build/test/acoustic2d",
        "circ_explosion": "./build/test/sod_shock_2d",
    }

    EXEC_CMD = []
    if test_running in executable.keys():
        if MPI_PROCS is not None:
            EXEC_CMD = ["mpirun", "-n", str(MPI_PROCS), executable[test_running]]  # ] #
        else:
            EXEC_CMD = ["mpirun", executable[test_running]]  # ] #

    # Initialize the list with an empty array
    BASE_ARGS = []

    # Conditionally append circ_explosion flags
    if test_running == "circ_explosion":
        BASE_ARGS.extend(
            ["-options_file", "conf_files/sod_shock.opts", "-radius", str(RADIUS)]
        )

    # Append standard flags
    BASE_ARGS.extend(
        [
            "-box_l",
            str(BOX_L),
            "-jac_type",
            "complexfd",
            # "-ts_adapt_type", "none",
        ]
    )

    if implicit == True:
        BASE_ARGS.extend(
            [
                "-ts_type",
                "bdf",
                "-ts_bdf_order",
                "2",
            ]
        )
    if implicit == False:
        BASE_ARGS.extend(
            [
                "-ts_type",
                "ssp",
                "-ts_ssp_type",
                "rks3",
            ]
        )

    # Conditionally append non-circ_explosion flags
    if test_running != "circ_explosion":
        BASE_ARGS.extend(
            [
                "-apply_filter",
                "false",
                "-filt_strength",
                "32",
                "-filt_cut_off_deg",
                "0",
                "-apply_weno",
                "false",
                "-weno_gamma",
                "0.998",
            ]
        )

    # Append the remaining standard flags
    BASE_ARGS.extend(
        [
            "-do_post_ts",
            "true",
            "-n_elems",
            str(N_ELEMS),
            "-poly_deg",
            str(POLY_DEG),
            "-ts_dt",
            str(DT),
            "-ts_max_time",
            str(MAX_TIME),
            "-solve_positive_preserving",
            "true" if CHANGE_VAR else "false",
            f"-ts_monitor_solution \t binary:{FILENAME}" if WRITE_SOL else "",
        ]
    )
    return EXEC_CMD, BASE_ARGS


# =============================================================================
# 2. ANALYTICAL SOLUTION WORKSPACE (Matches C++ engine perfectly)
# =============================================================================
def similarity_sedov(
    X,
    Y,
    t,
    Lx=10.0,
    Ly=10.0,
    beta=5.0,
    rho_0=1.0,
    p_0=1.0,
    u_0=1.0,
    v_0=0.5,
    r_v=1.0,
    gamma=1.4,
):
    # beta, xi, D, V, P = solve_adimensional(gamma, 1e-3, 2)
    # Move them back to physical domain
    return 0.0 * X, 0.0 * X, 0.0 * X, 0.0 * X


def exact_acoustic(
    X,
    Y,
    t,
    Lx=10.0,
    Ly=10.0,
    beta=5.0,
    rho_0=1.0,
    p_0=1.0,
    u_0=1.0,
    v_0=0.5,
    r_v=1.0,
    gamma=1.4,
):
    """Computes exact analytical fields wrapped periodically in a box."""
    X_shifted = X - u_0 * t
    Y_shifted = Y - v_0 * t

    X_wrapped = np.mod(np.mod(X_shifted, Lx) + Lx, Lx)
    Y_wrapped = np.mod(np.mod(Y_shifted, Ly) + Ly, Ly)

    rho = rho_0 * (
        0.2 * np.sin(2 * np.pi * X_wrapped / Lx) * np.sin(2 * np.pi * Y_wrapped / Ly)
    )
    u = u_0 * rho
    v = v_0 * rho
    rhoE = p_0 / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return rho, u, v, rhoE


def exact_isentropic_vortex(
    X,
    Y,
    t,
    Lx=10.0,
    Ly=10.0,
    beta=5.0,
    rho_0=1.0,
    p_0=1.0,
    u_0=0.0,
    v_0=5.0,
    r_v=1.0,
    gamma=1.4,
):
    """Computes exact analytical fields wrapped periodically in a box."""
    X_shifted = X - u_0 * t
    Y_shifted = Y - v_0 * t

    X_wrapped = np.mod(np.mod(X_shifted, Lx) + Lx, Lx)
    Y_wrapped = np.mod(np.mod(Y_shifted, Ly) + Ly, Ly)

    xc, yc = Lx / 2.0, Ly / 2.0
    rx = X_wrapped - xc
    ry = Y_wrapped - yc
    r_sq = rx**2 + ry**2

    # ── FIXED: Integrated core radius tracking variable r_v ──
    f = 1.0 - (r_sq / (r_v * r_v))
    exp_f = np.exp(f)

    factor = (gamma - 1.0) * beta**2 / (16.0 * gamma * np.pi**2)
    T = 1.0 - factor * (exp_f * exp_f)

    rho = rho_0 * (T ** (1.0 / (gamma - 1.0)))
    pres = p_0 * (T ** (gamma / (gamma - 1.0)))

    u = u_0 - (beta * exp_f * ry) / (2.0 * np.pi * r_v)
    v = v_0 + (beta * exp_f * rx) / (2.0 * np.pi * r_v)

    rhoE = pres / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return rho, u, v, rhoE


# =============================================================================
# 4. PETSc STREAMING BINARY PARSER
# =============================================================================
def parse_last_streaming_step(filename, expected_size):
    with open(filename, "rb") as f:
        last_valid_vector = None
        while True:
            header = np.fromfile(f, dtype=">i4", count=2)
            if len(header) < 2:
                break

            cookie, size = header[0], header[1]
            if cookie != 1211214:
                raise ValueError("Corrupted PETSc binary matrix layout cookie.")

            data = np.fromfile(f, dtype=">f8", count=size)
            if len(data) == size:
                last_valid_vector = data

        if last_valid_vector is None:
            raise ValueError(
                f"File '{filename}' contains no decipherable data streams."
            )
        return last_valid_vector


def extract_sol(FILENAME, N_ELEMS, POLY_DEG, BOX_L, CHANGE_VAR):
    x_nodes = POLY_DEG + 1
    y_nodes = POLY_DEG + 1
    n_eqs = 4
    expected_total = N_ELEMS * N_ELEMS * x_nodes * y_nodes * n_eqs
    flat_vector = parse_last_streaming_step(FILENAME, expected_total)

    grid_data = flat_vector.reshape((N_ELEMS, N_ELEMS, y_nodes, x_nodes, n_eqs))

    # ── FIXED: Coordinate transformations scaled to match total BOX_L ──
    h_x, h_y = BOX_L / N_ELEMS, BOX_L / N_ELEMS
    gll_x, _ = csi_ws(POLY_DEG)
    gll_y, _ = csi_ws(POLY_DEG)

    X_glob = np.zeros((N_ELEMS * y_nodes, N_ELEMS * x_nodes))
    Y_glob = np.zeros((N_ELEMS * y_nodes, N_ELEMS * x_nodes))
    rho_num = np.zeros_like(X_glob)
    rhou_num = np.zeros_like(X_glob)
    rhov_num = np.zeros_like(X_glob)
    rhoE_num = np.zeros_like(X_glob)

    for j in range(N_ELEMS):
        for i in range(N_ELEMS):
            for l in range(y_nodes):
                for k in range(x_nodes):
                    g_y = j * y_nodes + l
                    g_x = i * x_nodes + k

                    X_glob[g_y, g_x] = h_x * (i + 0.5 * (1.0 + gll_x[k]))
                    Y_glob[g_y, g_x] = h_y * (j + 0.5 * (1.0 + gll_y[l]))

                    w = grid_data[j, i, l, k, :]

                    if CHANGE_VAR:
                        r = np.exp(w[0])
                        u = w[1]
                        v = w[2]
                        e_int = np.exp(w[3])

                        rho_num[g_y, g_x] = r
                        rhou_num[g_y, g_x] = r * u
                        rhov_num[g_y, g_x] = r * v
                        rhoE_num[g_y, g_x] = (r * e_int) + 0.5 * r * (u**2 + v**2)
                    else:
                        rho_num[g_y, g_x] = w[0]
                        rhou_num[g_y, g_x] = w[1]
                        rhov_num[g_y, g_x] = w[2]
                        rhoE_num[g_y, g_x] = w[3]
    return X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num


# =============================================================================
# 5. EXECUTION & PIPELINE EXECUTION
# =============================================================================
def main():
    # =============================================================================
    # 1. RUNTIME SIMULATION PARAMETERS
    # =============================================================================
    N_ELEMS = 20  # Number of elements in X and Y directions
    POLY_DEG = 1  # Polynomial degree
    DT = 0.1  # Time step size
    MAX_TIME = 2.0  # 2.0  # Final simulation target time
    FILENAME = "build/solution.dat"
    CHANGE_VAR = False  # True if running with -solve_positive_preserving true
    BOX_L = 10.0  # Matches -box_l argument
    if os.path.exists(FILENAME):
        os.remove(FILENAME)

    test_running = "vortex"  # "acoustic" #
    EXEC_CMD, BASE_ARGS = get_sim_cmd(
        test_running,
        N_ELEMS,
        POLY_DEG,
        DT,
        MAX_TIME,
        FILENAME,
        BOX_L,
        CHANGE_VAR,
        RADIUS=0.0,
        implicit=False,
    )
    cmd = EXEC_CMD + BASE_ARGS
    print("=" * 60)
    print(f"LAUNCHING SOLVER VIA: {' '.join(cmd)}")
    print("=" * 60)
    subprocess.run(cmd, check=True)

    print("\nParsing streaming output file...")

    X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
        FILENAME, N_ELEMS, POLY_DEG, BOX_L, CHANGE_VAR
    )

    def exact_sol(
        X,
        Y,
        t,
        Lx=10.0,
        Ly=10.0,
        beta=5.0,
        rho_0=1.0,
        p_0=1.0,
        u_0=0.0,
        v_0=5.0,
        r_v=1.0,
        gamma=1.4,
    ):
        if test_running == "vortex":
            return exact_isentropic_vortex(X, Y, t, Lx=BOX_L, Ly=BOX_L)
        if test_running == "acoustic":
            return exact_acoustic(X, Y, t, Lx=BOX_L, Ly=BOX_L)
        if test_running == "circ_explosion":
            return similarity_sedov(X, Y, t)
        else:
            return 0.0, 0.0, 0.0, 0.0

    rho_ex, u_ex, v_ex, rhoE_ex = exact_sol(
        X_glob, Y_glob, MAX_TIME, Lx=BOX_L, Ly=BOX_L
    )
    rhou_ex = rho_ex * u_ex
    rhov_ex = rho_ex * v_ex

    scalars = {
        "rho": rho_ex,
        "rhou": rhou_ex,
        "rhov": rhov_ex,
        "rhoE": rhoE_ex,
    }
    plot_flow_fields(
        X_glob,
        Y_glob,
        scalars,
        BOX_L,
        save_path="vortex_test.png",
        show=True,
    )
    exit()

    # ─────────────────────────────────────────────────────────────────────────
    # PLOTS
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    c1 = axes[0].contourf(X_glob, Y_glob, rho_num, levels=50, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title(f"Numerical Density ($\\rho$)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    c2 = axes[1].contourf(X_glob, Y_glob, rho_ex, levels=50, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title(f"Exact Density ($\\rho$)")
    axes[1].set_xlabel("X")

    c3 = axes[2].contourf(
        X_glob, Y_glob, np.abs(rho_num - rho_ex), levels=50, cmap="inferno"
    )
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title("Absolute Density Error")
    axes[2].set_xlabel("X")

    plt.suptitle(
        f"Isentropic Vortex Field Comparisons at Time t={MAX_TIME}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Centerline cut slice tracking
    mid_y_idx = (N_ELEMS * POLY_DEG + 1) // 2
    actual_y = Y_glob[mid_y_idx, 0]
    x_line = X_glob[mid_y_idx, :]

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    fig2.suptitle(
        f"Centerline Slices across Vortex Core Axis ($Y = {actual_y:.3f}$)",
        fontsize=14,
        fontweight="bold",
    )

    axes2[0, 0].plot(x_line, rho_ex[mid_y_idx, :], "k-", lw=2, label="Exact Analytical")
    axes2[0, 0].plot(
        x_line, rho_num[mid_y_idx, :], "r--", lw=1.5, label="Numerical SEM/DG"
    )
    axes2[0, 0].set_title("Density ($\\rho$) Cut")
    axes2[0, 0].grid(True, ls="--", alpha=0.5)
    axes2[0, 0].legend()

    axes2[0, 1].plot(
        x_line, rhou_ex[mid_y_idx, :], "k-", lw=2, label="Exact Analytical"
    )
    axes2[0, 1].plot(
        x_line, rhou_num[mid_y_idx, :], "r--", lw=1.5, label="Numerical SEM/DG"
    )
    axes2[0, 1].set_title("X-Momentum ($\\rho u$) Cut")
    axes2[0, 1].grid(True, ls="--", alpha=0.5)
    axes2[0, 1].legend()

    axes2[1, 0].plot(
        x_line, rhov_ex[mid_y_idx, :], "k-", lw=2, label="Exact Analytical"
    )
    axes2[1, 0].plot(
        x_line, rhov_num[mid_y_idx, :], "r--", lw=1.5, label="Numerical SEM/DG"
    )
    axes2[1, 0].set_title("Y-Momentum ($\\rho v$) Cut")
    axes2[1, 0].grid(True, ls="--", alpha=0.5)
    axes2[1, 0].legend()

    axes2[1, 1].plot(
        x_line, rhoE_ex[mid_y_idx, :], "k-", lw=2, label="Exact Analytical"
    )
    axes2[1, 1].plot(
        x_line, rhoE_num[mid_y_idx, :], "r--", lw=1.5, label="Numerical SEM/DG"
    )
    axes2[1, 1].set_title("Total Energy ($\\rho E$) Cut")
    axes2[1, 1].grid(True, ls="--", alpha=0.5)
    axes2[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
