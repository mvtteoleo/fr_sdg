import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import data
from scipy.special import binom
from sod_shock_moc import exact_riemann_solver
from plot import csi_ws, read_all_petsc_vectors, get_mesh_on_gll

import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Initial condition (Riemann left / right states)
# ---------------------------------------------------------------------------


@dataclass
class RiemannIC:
    """Left and right primitive states for a Riemann problem."""

    rho_L: float = 1.0
    vel_L: float = 0.0
    pres_L: float = 1.0
    rho_R: float = 0.125
    vel_R: float = 0.0
    pres_R: float = 0.1
    gamma: float = 1.4

    def as_cli_args(self) -> list[str]:
        """Return the flags expected by the C++ solver."""
        return [
            "-rho_L",
            str(self.rho_L),
            "-vel_L",
            str(self.vel_L),
            "-pres_L",
            str(self.pres_L),
            "-rho_R",
            str(self.rho_R),
            "-vel_R",
            str(self.vel_R),
            "-pres_R",
            str(self.pres_R),
        ]

    def c_max(self) -> float:
        c_L = np.sqrt(self.gamma * self.pres_L / self.rho_L) + self.vel_L
        c_R = np.sqrt(self.gamma * self.pres_R / self.rho_R) + self.vel_R
        return max(c_L, c_R)


# Sod shock tube default — use this or override at the call site
SOD_IC = RiemannIC()


# ---------------------------------------------------------------------------
# Solver registry
# Each entry:  key -> (label, extra_args, timestep_type, output_stem)
#   timestep_type: "impl" | "expl"  (selects dt_impl or dt_expl at call time)
# ---------------------------------------------------------------------------
SOLVER_REGISTRY = {
    "fd": ("FD-BDF", ["-jac_type", "fd"], "impl", "fd_bdf"),
    "csd": ("CSD-BDF", ["-jac_type", "complexfd"], "impl", "csd_bdf"),
    "rk": ("RK-2a", ["-ts_type", "rk", "-ts_rk_type", "2a"], "expl", "rk_run"),
    "ssp3": (
        "SSP-3",
        ["-ts_type", "ssp", "-ts_ssp_type", "rks3", "ts_ssp_num_stages", "5"],
        "expl",
        "ssp_run",
    ),
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

# Keys whose output is in log-space (need conversion back to conservatives)
PP_KEYS = {"fd_pp", "csd_pp"}


# ---------------------------------------------------------------------------
# Output-directory helpers
# ---------------------------------------------------------------------------


def _make_run_dir(base_dir: Path, solver_key: str) -> Path:
    """
    Create  <base_dir>/<solver_key>/  and return the path.
    Safe to call even if it already exists.
    """
    run_dir = base_dir / solver_key
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_simulation_input(run_dir: Path, meta: dict) -> None:
    """Dump all run parameters as a human-readable JSON file."""
    input_file = run_dir / "simulation.input"
    with open(input_file, "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def load_1d_vec(bin_file: str, inp: dict):

    n_elems = inp["n_elems"]
    nds = inp["poly_deg"] + 1
    all_steps = read_all_petsc_vectors(str(bin_file))
    num_steps = len(all_steps)
    num_points = n_elems * nds

    rho_history = np.zeros((num_steps, num_points))
    rhoU_history = np.zeros((num_steps, num_points))
    rhoE_history = np.zeros((num_steps, num_points))

    for i, step_data in enumerate(all_steps):
        data_3d = step_data.reshape((n_elems, 3, nds))
        rho_history[i, :] = data_3d[:, 0, :].flatten()
        rhoU_history[i, :] = data_3d[:, 1, :].flatten()
        rhoE_history[i, :] = data_3d[:, 2, :].flatten()

    out = dict()
    out["data"] = [rho_history, rhoU_history, rhoE_history]
    out["inp"] = inp
    return out


def run_sim_get_hyst(
    l_0,
    n_elems,
    poly_deg,
    dt,
    t_max,
    run_dir: Path,
    solver_key: str,
    extra_args: list,
    ic: RiemannIC,
    run_sim: bool = True,
):
    """
    Runs the C++ solver (optionally) and returns (rho, rhoU, rhoE) history matrices.

    All output files (binary + log) land in *run_dir*.
    A simulation.input file with every parameter is written there too.
    """
    bin_file = run_dir / "solution.bin"
    log_file = run_dir / "run.log"
    nds = poly_deg + 1

    EXEC_CMD = ["mpirun", "./build/test/euler_1d"]
    BASE_ARGS = ["-options_file", "conf_files/basic_options.opts"]
    #   BASE_ARGS = ["-options_file", "conf_files/sod_shock.opts"]
    cmd = (
        EXEC_CMD
        + BASE_ARGS
        + extra_args
        + ic.as_cli_args()
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
            f"binary:{bin_file}",
            "-ts_monitor_solution_interval",
            "-1",
        ]
    )

    # (Over)write the metadata so it reflects the latest run
    meta = {
        "solver_key": solver_key,
        "solver_label": SOLVER_REGISTRY[solver_key][0],
        "l_0": l_0,
        "n_elems": n_elems,
        "poly_deg": poly_deg,
        "dt": dt,
        "t_max": t_max,
        "ic": asdict(ic),
        "extra_args": extra_args,
        "command": " ".join(str(c) for c in cmd),
    }
    if run_sim:
        _write_simulation_input(run_dir, meta)

        print(
            f"  [{solver_key}] N={n_elems:<4} P={poly_deg}  dt={dt:<10.4g}  → {run_dir}"
        )
        elapsed = 0
        with open(log_file, "w") as lf:
            t0 = time.perf_counter()
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
            elapsed = time.perf_counter() - t0
            print(f"  [{solver_key}] done in {elapsed:.2f} s")

    sim_data = load_1d_vec(str(bin_file), meta)
    rho_history, rhoU_history, rhoE_history = sim_data["data"]

    return rho_history, rhoU_history, rhoE_history


def pp_to_cons(hyst):
    """Convert positive-preserving (log-space) history to conservative variables."""
    log_rho, u, log_e = hyst
    rho = np.exp(log_rho)
    int_e = np.exp(log_e)
    return rho, rho * u, rho * (int_e + 0.5 * u * u)


def get_moc_sol(l_0, n_elems, poly_deg, t_max, ic: RiemannIC):
    mesh_x = get_mesh_on_gll(n_elems, poly_deg, l_0)
    rho_e, u_e, p_e = exact_riemann_solver(
        ic.rho_L,
        ic.vel_L,
        ic.pres_L,
        ic.rho_R,
        ic.vel_R,
        ic.pres_R,
        t_max,
        mesh_x,
        x_disc=l_0 / 2.0,
        gamma=ic.gamma,
    )
    return mesh_x, rho_e, rho_e * u_e, (p_e / (ic.gamma - 1.0) + 0.5 * rho_e * u_e**2)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
@dataclass
class OutputData:
    rho_err: float
    rhoU_err: float
    rhoE_err: float

    sim_data: tuple[np.ndarray]


def run_check_plot(
    l_0,
    n_elems,
    poly_deg,
    t_max,
    dt_impl,
    dt_expl,
    # --- Initial condition ---
    ic: RiemannIC = SOD_IC,
    # --- Output root (each solver gets its own sub-directory here) ---
    output_dir: str | Path = "build/runs",
    # --- What to run (set of solver keys, or "all") ---
    run_keys="all",
    # --- What to plot (set of solver keys + optionally "moc", or "all") ---
    plot_keys="all",
    # --- Pairs to plot as separate difference figure: list of (key_a, key_b) ---
    diff_pairs=None,
    err_keys="all",
    # --- Whether to actually execute the solver binaries ---
    run_sim: bool = True,
):
    """
    Parameters
    ----------
    ic         : RiemannIC dataclass with the left/right primitive states.
    output_dir : root directory; each solver gets its own sub-folder with
                 solution.bin, run.log and simulation.input inside.
    run_keys   : "all"  OR a set/list of SOLVER_REGISTRY keys.
    plot_keys  : "all"  OR a set/list of keys (add "moc" for the exact solution).
    diff_pairs : list of (key_a, key_b) tuples — each pair gets a separate
                 difference figure.  None / [] to skip.
    run_sim    : False → skip the binary, only reload existing .bin files.
    """

    print(f"{ic=}")

    l_0 = float(l_0)
    output_dir = Path(output_dir)

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

    needed_keys = (plot_keys - {"moc"}) | {
        k for pair in diff_pairs for k in pair if k != "moc"
    }
    keys_to_run = run_keys & needed_keys

    # ---- CFL info ----
    dx = l_0 / (n_elems * poly_deg**2)
    c_max = ic.c_max()
    print(f"{c_max = }")
    CFL_impl = dt_impl * c_max / dx
    CFL_expl = dt_expl * c_max / dx
    print(f"CFL_expl={CFL_expl:.4f},  CFL_impl={CFL_impl:.4f}")

    # ---- MOC reference ----
    x_coords, rho_moc, rhoU_moc, rhoE_moc = get_moc_sol(
        l_0, n_elems, poly_deg, t_max, ic
    )

    # ---- Run / load solvers ----
    results: dict[str, tuple] = {}
    for key in needed_keys:
        label, extra_args, ts_type, _ = SOLVER_REGISTRY[key]
        dt = dt_impl if ts_type == "impl" else dt_expl
        run_dir = _make_run_dir(output_dir, key)

        hyst = run_sim_get_hyst(
            l_0,
            n_elems,
            poly_deg,
            dt,
            t_max,
            run_dir,
            key,
            extra_args,
            ic,
            run_sim=(run_sim and key in keys_to_run),
        )
        results[key] = pp_to_cons(hyst) if key in PP_KEYS else hyst

    # Compute errors
    def compute_l2_diff(vec1, vec2):
        vec1 = vec1.reshape((n_elems, poly_deg + 1))
        vec2 = vec2.reshape((n_elems, poly_deg + 1))
        _, ws = csi_ws(poly_deg)
        err = np.sum((vec1 - vec2) ** 2 * ws)

        err *= l_0 / n_elems
        return np.sqrt(err)

    output: dict[str, OutputData] = {}
    for key in err_keys:
        if key not in results:
            print(f"Warning: Skipping '{key}' because it is missing from the results.")
            continue
        if len(results[key][0]) == 0:
            print(f"Error {key}: Solver failed to produce output (0 steps loaded).")
            continue

        err_rho = compute_l2_diff(results[key][0], rho_moc)
        err_rhoU = compute_l2_diff(results[key][1], rhoU_moc)
        err_rhoE = compute_l2_diff(results[key][2], rhoE_moc)

        print(f"Error {key}:")
        print(f"\trho  :{err_rho}")
        print(f"\trhoU :{err_rhoU}")
        print(f"\trhoE :{err_rhoE}")
        output[key] = OutputData(
            rho_err=err_rho, rhoU_err=err_rhoU, rhoE_err=err_rhoE, sim_data=results[key]
        )

    # ---- Shared plot helpers ----
    var_labels = [
        r"Density ($\rho$)",
        r"Momentum ($\rho U$)",
        r"Total Energy ($\rho E$)",
    ]

    def _fmt_axes(fig, axs, title):
        fig.suptitle(title, fontsize=16)
        for ax, ylabel in zip(axs, var_labels):
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc="upper right")
        axs[2].set_xlabel("Position (x)")
        plt.tight_layout()

    def _plot_last(axs, key_or_label, arrays):
        lbl = (
            key_or_label
            if isinstance(key_or_label, str)
            else SOLVER_REGISTRY[key_or_label][0]
        )
        for ax, arr in zip(axs, arrays):
            if len(arr) == 0:
                print(f"Skipping plot for {lbl} because data is empty.")
                return

            ax.plot(x_coords, arr[-1, :], label=lbl)

    # ---- Figure 1: solution curves ----
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    if "moc" in plot_keys:
        _plot_last(
            axs1,
            "MOC",
            [np.array([rho_moc]), np.array([rhoU_moc]), np.array([rhoE_moc])],
        )

    for key in SOLVER_REGISTRY:
        if key in plot_keys and key in results:
            _plot_last(axs1, key, results[key])

    _fmt_axes(
        fig1,
        axs1,
        f"1D Euler: solution at t={t_max:.3g} s  |  #El={n_elems}, P={poly_deg}"
        f"\n IC: ρ=({ic.rho_L}|{ic.rho_R})  u=({ic.vel_L}|{ic.vel_R})  p=({ic.pres_L}|{ic.pres_R})",
    )
    plt.show()

    # ---- Figure 2+: one difference figure per pair ----
    for key_a, key_b in diff_pairs:
        if key_a not in results or key_b not in results:
            print(
                f"[warn] diff pair ({key_a}, {key_b}) skipped — one or both keys not loaded."
            )
            continue

        fig_d, axs_d = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        lbl_a = SOLVER_REGISTRY[key_a][0]
        lbl_b = SOLVER_REGISTRY[key_b][0]
        lbl = f"Δ({lbl_a} − {lbl_b})"

        for ax, a_arr, b_arr in zip(
            axs_d,
            results[key_a],
            results[key_b],
        ):
            ax.plot(x_coords, a_arr[-1, :] - b_arr[-1, :], label=lbl)

        _fmt_axes(
            fig_d,
            axs_d,
            f"1D Euler: difference  {lbl_a} − {lbl_b}  at t={t_max:.3g} s"
            f"\n #El={n_elems}, P={poly_deg}",
        )
        plt.show()

    # ==========================================
    # PLOT ANIMATION  (set animate=True below to enable)
    # ==========================================
    animate = False
    if animate and results:
        first_key = next(iter(results))
        num_frames = results[first_key][0].shape[0]

        fig_a, axs_a = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig_a.suptitle("1D Euler Animation: Sod Shock Tube", fontsize=16)

        for ax, ylabel in zip(axs_a, var_labels):
            ax.set_xlim(0, l_0)
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.7)
        axs_a[2].set_xlabel("Position (x)")
        axs_a[0].set_ylim(-0.05, 1.15)
        axs_a[1].set_ylim(-0.15, 0.45)
        axs_a[2].set_ylim(0.00, 2.75)

        if "moc" in plot_keys:
            axs_a[0].plot(x_coords, rho_moc, "k--", label="MOC", alpha=0.5)
            axs_a[1].plot(x_coords, rhoU_moc, "k--", label="MOC", alpha=0.5)
            axs_a[2].plot(x_coords, rhoE_moc, "k--", label="MOC", alpha=0.5)

        anim_lines = {}
        for key in SOLVER_REGISTRY:
            if key in plot_keys and key in results:
                lbl = SOLVER_REGISTRY[key][0]
                anim_lines[key] = (
                    axs_a[0].plot([], [], lw=2, label=lbl)[0],
                    axs_a[1].plot([], [], lw=2, label=lbl)[0],
                    axs_a[2].plot([], [], lw=2, label=lbl)[0],
                )
        for ax in axs_a:
            ax.legend(loc="upper right")
        plt.tight_layout()

        def _init():
            for lines in anim_lines.values():
                for ln in lines:
                    ln.set_data([], [])
            return [ln for tup in anim_lines.values() for ln in tup]

        def _update(i):
            for key, (l0, l1, l2) in anim_lines.items():
                l0.set_data(x_coords, results[key][0][i, :])
                l1.set_data(x_coords, results[key][1][i, :])
                l2.set_data(x_coords, results[key][2][i, :])
            return [ln for tup in anim_lines.values() for ln in tup]

        FuncAnimation(
            fig_a, _update, init_func=_init, frames=num_frames, interval=20, blit=True
        )
        plt.show()

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n_elems, poly_deg, t_max = 1000, 1, 0.1
    l_0 = 10.0
    dt_impl = t_max / 10
    dt_expl = t_max / 1000

    for n_elems in [100]:  # [200, 400, 800]:
        for poly_deg in [1]:  # range(1, 6):
            ris = run_check_plot(
                l_0,
                n_elems,
                poly_deg,
                t_max,
                dt_impl,
                dt_expl,
                output_dir=Path(
                    str("build/runs" + "__" + str(n_elems) + "__" + str(poly_deg))
                ),
                # --- Initial condition ---
                ic=RiemannIC(
                    rho_L=1.0,
                    vel_L=0.0,
                    pres_L=1,
                    rho_R=0.125,
                    vel_R=0.0,
                    pres_R=0.1,
                    gamma=1.4,
                ),
                # Keys to actually execute (subset of SOLVER_REGISTRY keys, or "all")
                run_keys={"fd"},  # , "ssp3", "csd_pp"},  # , "fd_pp""csd_pp",
                # Keys to draw on the solution plot (add "moc" for the exact solution)
                plot_keys={"fd"},  # , "ssp3", "moc", "csd_pp"},
                # Each tuple produces a separate difference figure
                diff_pairs=None,
                # Checks error and plots the delta with respect to MOC
                err_keys={"fd"},  # , "ssp3", "csd_pp"},
                # False → skip the binary, only reload existing .bin files
                run_sim=True,
            )
