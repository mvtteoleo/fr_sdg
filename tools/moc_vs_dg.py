import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sod_shock_moc import exact_riemann_solver
from plot import read_all_petsc_vectors, get_mesh_on_gll, read_all_petsc_vectors

import sys
import os
import subprocess
import time


# ---------------------------------------------------------------------------
# Solver registry
# Each entry:  key -> (label, extra_args, timestep_type, output_filename)
#   timestep_type: "impl" | "expl"  (selects dt_impl or dt_expl at call time)
# ---------------------------------------------------------------------------
SOLVER_REGISTRY = {
    "fd": ("FD-BDF", ["-jac_type", "fd"], "impl", "fd_bdf"),
    "csd": ("CSD-BDF", ["-jac_type", "complexfd"], "impl", "csd_bdf"),
    "rk": ("RK-2a", ["-ts_type", "rk", "-ts_rk_type", "2a"], "expl", "rk_run"),
    "fd_pp": (
        "FD-BDF-PP",
        ["-jac_type", "fd", "-solve_positive_preserving", "true"],
        "impl",
        "fd_bdf_pp",
    ),
    "csd_pp": (
        "CSD-BDF-PP",
        ["-jac_type", "complexfd", "-solve_positive_preserving", "true"],
        "impl",
        "csd_bdf_pp",
    ),
}

# Keys that store results in log-space and need conversion back to conservatives
PP_KEYS = {"fd_pp", "csd_pp"}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def run_sim_get_hyst(
    l_0, n_elems, poly_deg, dt, t_max, out_file_name, extra_args, run_sim=True
):
    """Runs the C++ solver and returns (rho, rhoU, rhoE) history matrices."""
    sanitized_file_name = f"build/{out_file_name}.bin"
    nds = poly_deg + 1

    EXEC_CMD = [
        "mpirun",
        # "-n", "3",
        "./build/test/euler_1d",
    ]
    BASE_ARGS = ["-options_file", "conf_files/sod_shock.opts"]
    cmd = (
        EXEC_CMD
        + BASE_ARGS
        + extra_args
        + [
            "-box_l",
            str(l_0),
            "-n_elems",
            str(n_elems),
            "-poly_deg",
            str(poly_deg),
            "-ts_dt",
            str(dt),
            "-ts_max_time",
            str(t_max),
            "-ts_monitor_solution",
            f"binary:{sanitized_file_name}",
            "-rho_L",
            "2.0",
            "-vel_L",
            "0.0",
            "-pres_L",
            "15",
            "-rho_R",
            "1.0",
            "-vel_R",
            "0.0",
            "-pres_R",
            "2.0",
        ]
    )

    if run_sim:
        print(
            f"Running {out_file_name} with : N={n_elems:<3} | P={poly_deg} | dt={dt:<7}"
        )
        str_time = time.perf_counter()
        log_file = "build/log.log"
        with open(log_file, "w") as outfile:
            subprocess.run(cmd, capture_output=False, text=True, stdout=outfile)

        fin_time = time.perf_counter()
        delta_t = fin_time - str_time
        print(f"Time elapsed for {out_file_name} : {delta_t}")

    all_steps = read_all_petsc_vectors(sanitized_file_name)
    num_steps = len(all_steps)
    num_points = n_elems * nds
    print(f"Successfully extracted {num_steps} time steps from {sanitized_file_name}")

    rho_history = np.zeros((num_steps, num_points))
    rhoU_history = np.zeros((num_steps, num_points))
    rhoE_history = np.zeros((num_steps, num_points))

    for i, step_data in enumerate(all_steps):
        data_3d = step_data.reshape((n_elems, 3, nds))
        rho_history[i, :] = data_3d[:, 0, :].flatten()
        rhoU_history[i, :] = data_3d[:, 1, :].flatten()
        rhoE_history[i, :] = data_3d[:, 2, :].flatten()

    return rho_history, rhoU_history, rhoE_history


def pp_to_cons(hyst):
    """Convert positive-preserving (log-space) history to conservative variables."""
    log_rho, u, log_e = hyst
    rho = np.exp(log_rho)
    int_e = np.exp(log_e)
    return rho, rho * u, rho * (int_e + 0.5 * u * u)


def get_moc_sol(l_0, n_elems, poly_deg, t_max):
    mesh_x = get_mesh_on_gll(n_elems, poly_deg, l_0)
    rho_L = 2.0
    u_L = 0.0
    p_L = 15.0
    rho_R = 1.0
    u_R = 0.0
    p_R = 2.0
    """
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    """
    rho_e, u_e, p_e = exact_riemann_solver(
        rho_L,
        u_L,
        p_L,
        rho_R,
        u_R,
        p_R,
        t_max,
        mesh_x,
        x_disc=l_0 / 2.0,
    )
    c_L = np.sqrt(1.4 * p_L / rho_L)
    c_R = np.sqrt(1.4 * p_R / rho_R)
    c_max = max(c_L, c_R)
    return mesh_x, c_max, rho_e, rho_e * u_e, (p_e / 0.4 + 0.5 * rho_e * u_e**2)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_check_plot(
    l_0,
    n_elems,
    poly_deg,
    t_max,
    dt_impl,
    dt_expl,
    # --- What to run (set of solver keys, or "all") ---
    run_keys="all",
    # --- What to plot (set of solver keys + optionally "moc", or "all") ---
    plot_keys="all",
    # --- Pairs to overlay as difference curves: list of (key_a, key_b) tuples ---
    diff_pairs=None,
    # --- Whether to actually execute the solver binaries ---
    run_sim=True,
):
    """
    Parameters
    ----------
    run_keys   : "all"  OR  a set/list of keys from SOLVER_REGISTRY, e.g. {"fd", "csd_pp"}
    plot_keys  : "all"  OR  a set/list of keys (plus "moc"), e.g. {"moc", "rk", "csd_pp"}
    diff_pairs : list of 2-tuples of solver keys whose difference should be plotted,
                 e.g. [("csd_pp", "fd_pp")].  Pass None or [] to skip.
    run_sim    : if False, only load existing binary files (skip running the solver).
    """

    if run_keys == "all":
        run_keys = set(SOLVER_REGISTRY.keys())
    else:
        run_keys = set(run_keys)

    if plot_keys == "all":
        plot_keys = {"moc"} | set(SOLVER_REGISTRY.keys())
    else:
        plot_keys = set(plot_keys)

    if diff_pairs is None:
        diff_pairs = []

    l_0 = float(l_0)

    # Keys we need loaded = plot_keys (minus "moc") + all keys in diff_pairs
    needed_keys = (plot_keys - {"moc"}) | {k for pair in diff_pairs for k in pair}
    keys_to_run = run_keys & needed_keys  # only run what will actually be used

    # ---- MOC reference ----
    x_coords, c_max, rho_moc, rhoU_moc, rhoE_moc = get_moc_sol(
        l_0, n_elems, poly_deg, t_max
    )
    dx = l_0 / n_elems
    CFL_impl = dt_impl * c_max / dx
    CFL_expl = dt_expl * c_max / dx
    print(f"{CFL_expl=:.4f},  {CFL_impl=:.4f}")

    # ---- Run / load solvers ----
    results = {}  # key -> (rho_history, rhoU_history, rhoE_history) in conservatives
    for key in needed_keys:
        label, extra_args, ts_type, fname = SOLVER_REGISTRY[key]
        dt = dt_impl if ts_type == "impl" else dt_expl
        hyst = run_sim_get_hyst(
            l_0,
            n_elems,
            poly_deg,
            dt,
            t_max,
            str(fname + "__" + str(poly_deg) + "__" + str(n_elems)),
            extra_args,
            run_sim=(run_sim and key in keys_to_run),
        )
        results[key] = pp_to_cons(hyst) if key in PP_KEYS else hyst

    # ---- Plot ----
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    var_labels = [
        r"Density ($\rho$)",
        r"Momentum ($\rho U$)",
        r"Total Energy ($\rho E$)",
    ]

    def _plot_ith(key_or_label, arrays, ith):
        lbl = (
            key_or_label
            if isinstance(key_or_label, str)
            else SOLVER_REGISTRY[key_or_label][0]
        )
        for ax, arr in zip(axs, arrays):
            ax.plot(x_coords, arr[ith, :], label=lbl)

    def _plot_last(key_or_label, arrays):
        _plot_ith(key_or_label, arrays, -1)

    # MOC reference
    if "moc" in plot_keys:
        moc_arrays = [
            np.array([rho_moc]),
            np.array([rhoU_moc]),
            np.array([rhoE_moc]),
        ]
        _plot_last("MOC", moc_arrays)

    # Solver curves
    for key in SOLVER_REGISTRY:
        if key in plot_keys and key in results:
            _plot_last(key, results[key])  # uses registry label

    # ---- Formatting ----
    fig.suptitle(
        f"1D Euler: Sod shock tube, solution at {t_max:.3} s. (#El: {n_elems}, P : {poly_deg})",
        fontsize=16,
    )

    """
    x_elems = [i * l_0 / n_elems for i in range(n_elems)]
    axs[0].vlines(
        x_elems,
        ymin=min(rho_moc),
        ymax=max(rho_moc),
        colors="gray",
        linestyles="dashed",
        alpha=0.5,
    )
    axs[1].vlines(
        x_elems,
        ymin=min(rhoU_moc),
        ymax=max(rhoU_moc),
        colors="gray",
        linestyles="dashed",
        alpha=0.5,
    )
    axs[2].vlines(
        x_elems,
        ymin=min(rhoE_moc),
        ymax=max(rhoE_moc),
        colors="gray",
        linestyles="dashed",
        alpha=0.5,
    )
    """

    for ax, ylabel in zip(axs, var_labels):
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper right")
    axs[2].set_xlabel("Position (x)")

    plt.tight_layout()
    plt.show()

    # ==========================================
    # PLOT ANIMATION
    # ==========================================
    if False and results:
        # 1. Get total number of frames from the first loaded solver
        first_key = list(results.keys())[0]
        num_frames = results[first_key][0].shape[0]

        fig_anim, axs_anim = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig_anim.suptitle("1D Euler Animation: Sod Shock Tube", fontsize=16)

        # 2. Setup Axes
        for ax, ylabel in zip(axs_anim, var_labels):
            ax.set_xlim(0, l_0)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.7)
        axs_anim[2].set_xlabel("Position (x)")

        # Set approximate fixed Y-limits for the standard Sod shock tube
        axs_anim[0].set_ylim(-0.05, 1.15)  # Density
        axs_anim[1].set_ylim(-0.15, 0.45)  # Momentum
        axs_anim[2].set_ylim(0.00, 2.75)  # Total Energy

        # 3. Draw MOC as a static reference background (if requested)
        if "moc" in plot_keys:
            axs_anim[0].plot(x_coords, rho_moc, "k--", label="MOC (Static)", alpha=0.5)
            axs_anim[1].plot(x_coords, rhoU_moc, "k--", label="MOC (Static)", alpha=0.5)
            axs_anim[2].plot(x_coords, rhoE_moc, "k--", label="MOC (Static)", alpha=0.5)

        # 4. Initialize animation lines for each solver
        anim_lines = {}
        for key in SOLVER_REGISTRY:
            if key in plot_keys and key in results:
                lbl = SOLVER_REGISTRY[key][0]
                (line_rho,) = axs_anim[0].plot([], [], lw=2, label=lbl)
                (line_rhoU,) = axs_anim[1].plot([], [], lw=2, label=lbl)
                (line_rhoE,) = axs_anim[2].plot([], [], lw=2, label=lbl)
                anim_lines[key] = (line_rho, line_rhoU, line_rhoE)

        for ax in axs_anim:
            ax.legend(loc="upper right")
        plt.tight_layout()

        # 5. Animation functions
        def init_anim():
            for l_rho, l_rhoU, l_rhoE in anim_lines.values():
                l_rho.set_data([], [])
                l_rhoU.set_data([], [])
                l_rhoE.set_data([], [])
            # Return flattened list of line objects for blitting
            return [l for tup in anim_lines.values() for l in tup]

        def update_anim(i):
            i -= 1
            for key, (l_rho, l_rhoU, l_rhoE) in anim_lines.items():
                # results[key] is a tuple of: (rho_history, rhoU_history, rhoE_history)
                # index 0, 1, and 2 respectively, then slice the current time step [i, :]
                l_rho.set_data(x_coords, results[key][0][i, :])
                l_rhoU.set_data(x_coords, results[key][1][i, :])
                l_rhoE.set_data(x_coords, results[key][2][i, :])
            return [l for tup in anim_lines.values() for l in tup]

        anim = FuncAnimation(
            fig_anim,
            update_anim,
            init_func=init_anim,
            frames=num_frames,
            interval=20,
            blit=True,
        )
        plt.show()

    if diff_pairs != []:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        def _plot_diff(key_a, key_b):
            ra, rua, rea = results[key_a]
            rb, rub, reb = results[key_b]
            lbl_a = SOLVER_REGISTRY[key_a][0]
            lbl_b = SOLVER_REGISTRY[key_b][0]
            lbl = f"Δ({lbl_a} − {lbl_b})"
            for ax, a, b in zip(axs, [ra, rua, rea], [rb, rub, reb]):
                ax.plot(x_coords, a[-1, :] - b[-1, :], linestyle="-", label=lbl)

        # Difference curves
        for key_a, key_b in diff_pairs:
            if key_a in results and key_b in results:
                _plot_diff(key_a, key_b)
            else:
                print(
                    f"[warn] diff pair ({key_a}, {key_b}) skipped — one or both keys not loaded."
                )

        fig.suptitle(
            f"1D Euler: Sod shock tube, solution at {t_max} s difference", fontsize=16
        )
        for ax, ylabel in zip(axs, var_labels):
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="upper right")
        axs[2].set_xlabel("Position (x)")

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Entry point — edit these lines to control what happens
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    n_elems, poly_deg, t_max = 20, 5, 0.5
    n_elems, poly_deg, t_max = 500, 3, 0.5
    n_elems, poly_deg, t_max = 400, 4, 0.5
    n_elems, poly_deg, t_max = 100, 19, 0.5
    """
    n_elems, poly_deg, t_max = 500, 3, 0.5

    l_0 = 10
    dt_impl = t_max / 200
    dt_expl = t_max / 5000

    run_check_plot(
        l_0,
        n_elems,
        poly_deg,
        t_max,
        dt_impl,
        dt_expl,
        # Keys to actually execute (subset of SOLVER_REGISTRY keys, or "all")
        run_keys={"csd"},  # "csd_pp", , "rk"},  #
        # Keys to draw on the plot (add "moc" for the exact solution)
        plot_keys={"csd", "moc"},  # , "moc", "rk" "fd_pp",
        # Pairs whose point-wise difference should be overlaid as dashed curves
        diff_pairs=None,  # [("csd_pp", "moc")],#
        # Set False to skip running the binary and only load existing .bin files
        run_sim=True,  # False,  #
    )
