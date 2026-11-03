import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import data
from sod_shock_moc import exact_riemann_solver
from plot import csi_ws, read_all_petsc_vectors, get_mesh_on_gll

import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


# ---------------------------------------------------------------------------
# Initial condition (Riemann left / right states)
# ---------------------------------------------------------------------------


@dataclass
class ProblemData:
    l_0: float = 10.0
    dt_impl: float = 0.0
    dt_expl: float = 0.0
    t_max: float = 1.0

    poly_deg: int = 1
    n_elems: int = 1


@dataclass
class OutputData:
    rho_err: float
    rhoU_err: float
    rhoE_err: float

    sim_data: tuple[np.ndarray]
    time_to_sol: float
    state: str


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
    EXEC_CMD=["mpirun", "./build/test/euler_1d"],
):
    """
    Runs the C++ solver (optionally) and returns (rho, rhoU, rhoE) history matrices.

    All output files (binary + log) land in *run_dir*.
    A simulation.input file with every parameter is written there too.
    """
    bin_file = run_dir / "solution.bin"
    log_file = run_dir / "run.log"
    nds = poly_deg + 1

    BASE_ARGS = []  # ["-options_file", "conf_files/basic_options.opts"]
    #   BASE_ARGS = ["-options_file", "conf_files/sod_shock.opts"]
    cmd = (
        EXEC_CMD
        + BASE_ARGS
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
        + extra_args
    )

    # (Over)write the metadata so it reflects the latest run
    elapsed = -1.0
    if run_sim:
        print(
            f"  [{solver_key}] N={n_elems:<4} P={poly_deg}  dt={dt:<10.4g}  → {run_dir}"
        )
        with open(log_file, "w") as lf:
            t0 = time.perf_counter()
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
            elapsed = time.perf_counter() - t0
            print(f"  [{solver_key}] done in {elapsed:.2f} s")
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
            "time_needed": elapsed,
        }
        _write_simulation_input(run_dir, meta)

    all_steps = read_all_petsc_vectors(str(bin_file))
    num_steps = len(all_steps)
    num_points = n_elems * nds
    print(f"  [{solver_key}] loaded {num_steps} steps from {bin_file}")

    rho_history = np.zeros((num_steps, num_points))
    rhoU_history = np.zeros((num_steps, num_points))
    rhoE_history = np.zeros((num_steps, num_points))

    for i, step_data in enumerate(all_steps):
        data_3d = step_data.reshape((n_elems, 3, nds))
        rho_history[i, :] = data_3d[:, 0, :].flatten()
        rhoU_history[i, :] = data_3d[:, 1, :].flatten()
        rhoE_history[i, :] = data_3d[:, 2, :].flatten()

    return rho_history, rhoU_history, rhoE_history, elapsed


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
def run_extract_ris(
    data: ProblemData,
    # --- Initial condition ---
    ic: RiemannIC = SOD_IC,
    # --- Output root (each solver gets its own sub-directory here) ---
    output_dir: str | Path = "build/runs",
    # --- What to run (set of solver keys, or "all") ---
    run_keys="all",
    # --- Pairs to plot as separate difference figure: list of (key_a, key_b) ---
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

    l_0 = data.l_0
    n_elems = data.n_elems
    poly_deg = data.poly_deg

    t_max = data.t_max
    dt_impl = data.dt_impl
    dt_expl = data.dt_expl
    l_0 = float(l_0)
    output_dir = Path(output_dir)

    if run_keys == "all":
        run_keys = set(SOLVER_REGISTRY.keys())
    else:
        run_keys = set(run_keys)
    keys_to_run = run_keys

    # ---- CFL info ----
    dx = l_0 / (n_elems * poly_deg**2)
    c_max = ic.c_max()
    CFL_impl = dt_impl * c_max / dx
    CFL_expl = dt_expl * c_max / dx

    # ---- MOC reference ----
    x_coords, rho_moc, rhoU_moc, rhoE_moc = get_moc_sol(
        l_0, n_elems, poly_deg, t_max, ic
    )

    # ---- Run / load solvers ----
    results: dict[str, tuple] = {}
    times: dict[str, float] = {}
    ris: dict[str, str] = {}
    for key in keys_to_run:
        label, extra_args, ts_type, _ = SOLVER_REGISTRY[key]
        dt = dt_impl if ts_type == "impl" else dt_expl
        run_dir = _make_run_dir(output_dir, key)

        _r, _rU, _rE, time = run_sim_get_hyst(
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
        hyst = (_r, _rU, _rE)
        results[key] = pp_to_cons(hyst) if key in PP_KEYS else hyst
        times[key] = time

    # Check that the last time  is logged
    for key in keys_to_run:
        ris[key] = "success"
    """
        with open(Path(str(f"{out_dir}" + f"/{key}" + "/run.log")), "r") as f:
            final_t = 0.0
            lines_iter = iter(f)
            for line in lines_iter:
                match = re.search(rf'Time\s:\s*([0-9.]+)+', line)
                if match:
                    final_t  = max(final_t, float(match.group(1)))

            if (abs(final_t - t_max) < 1e-9):
                ris[key] = "success"
                print(f"[{key}] SUCCEDED to  reach final  time {final_t}")
            else:
                ris[key] = "failed"
                print(f"[{key}] FAILED to  reach final  time {final_t}")
    """

    if run_sim == False:
        for key in keys_to_run:
            with open(
                Path(str(f"{out_dir}" + f"/{key}" + "/simulation.input")), "r"
            ) as f:
                lines_iter = iter(f)
                for line in lines_iter:
                    match = re.search(rf'"time_needed":\s*([0-9.]+)+', line)
                    if match:
                        time_s = float(match.group(1))
                        times[key] = time_s

    # Compute errors
    def compute_l2_diff(vec1_, vec2):
        vec1 = vec1_[-1, :]
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

        if ris[key] == "success":
            err_rho = compute_l2_diff(results[key][0], rho_moc)
            err_rhoU = compute_l2_diff(results[key][1], rhoU_moc)
            err_rhoE = compute_l2_diff(results[key][2], rhoE_moc)

            print(f"Error {key}:")
            print(f"\trho  :{err_rho}")
            print(f"\trhoU :{err_rhoU}")
            print(f"\trhoE :{err_rhoE}")
            output[key] = OutputData(
                rho_err=err_rho,
                rhoU_err=err_rhoU,
                rhoE_err=err_rhoE,
                sim_data=results[key],
                time_to_sol=times[key],
                state=ris[key],
            )

    output["moc"] = OutputData(
        rho_err=0.0,
        rhoU_err=0.0,
        rhoE_err=0.0,
        sim_data=(rho_moc, rhoU_moc, rhoE_moc),
        time_to_sol=0.0,
        state="success",
    )

    return output, x_coords


def plot_results(results: dict, x_coords: np.ndarray, ic: RiemannIC, data: ProblemData):
    # ---- Shared plot helpers ----
    var_labels = [
        r"Density ($\rho$)",
        r"Momentum ($\rho U$)",
        r"Total Energy ($\rho E$)",
    ]
    t_max = data.t_max
    n_elems = data.n_elems
    poly_deg = data.poly_deg
    # ---- Figure: 3 rows (Variables), 2 columns (Solution vs Error) ----
    # Make the figure a bit wider to comfortably fit two columns
    fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    # Correct slicing: [all_rows, column_index]
    left_col = axs[:, 0]
    right_col = axs[:, 1]

    def _plot_last(axs_col, key_or_label, data_val):
        lbl = (
            key_or_label
            if isinstance(key_or_label, str)
            else SOLVER_REGISTRY[key_or_label][0]
        )

        # Unpack the arrays
        arrays = data_val.sim_data if hasattr(data_val, "sim_data") else data_val

        for ax, arr in zip(axs_col, arrays):
            if len(arr) == 0:
                print(f"Skipping plot for {lbl} because data is empty.")
                return

            # --- THE FIX: Dimension checking ---
            arr_np = np.array(arr)

            if arr_np.ndim > 1:
                # It's a 2D solver array: grab the last time step
                plot_data = arr_np[-1, :]
            else:
                # It's already a 1D MOC array
                plot_data = arr_np

            # Flatten to guarantee shape is (N,) not (1, N)
            plot_data = plot_data.flatten()

            ax.plot(x_coords, plot_data, label=lbl)

    def _plot_err(axs_col, key_or_label, data_val):
        lbl = (
            key_or_label
            if isinstance(key_or_label, str)
            else SOLVER_REGISTRY[key_or_label][0]
        )

        moc_data = results.get("moc")
        if moc_data is None:
            return

        # Unpack the arrays
        arrays = data_val.sim_data if hasattr(data_val, "sim_data") else data_val
        moc_arrays = moc_data.sim_data if hasattr(moc_data, "sim_data") else moc_data

        for i, (ax, arr) in enumerate(zip(axs_col, arrays)):
            if len(arr) == 0 or len(moc_arrays[i]) == 0:
                print(f"Skipping error plot for {lbl} because data is empty.")
                return

            error_arr = arr[:] - moc_arrays[i]

            # Now error_arr is a perfect 1D array matching x_coords
            ax.plot(x_coords, error_arr[-1, :], label=lbl)

    # ---- Plotting Loop ----
    for key, values in results.items():
        # Always plot the raw solution in the left column
        _plot_last(left_col, key, values)

        # Plot the error in the right column (skip MOC vs MOC)
        if key != "moc":
            _plot_err(right_col, key, values)

    # ---- Formatting ----
    fig.suptitle(
        f"1D Euler: solution at t={t_max:.3g} s  |  #El={n_elems}, P={poly_deg}\n"
        f"IC: ρ=({ic.rho_L}|{ic.rho_R})  u=({ic.vel_L}|{ic.vel_R})  p=({ic.pres_L}|{ic.pres_R})",
        fontsize=16,
    )

    # Column Titles
    left_col[0].set_title("Solution", fontsize=14, fontweight="bold")
    right_col[0].set_title("Error (Solver - MOC)", fontsize=14, fontweight="bold")

    # Apply grids, labels, and legends
    for i in range(3):
        # Set y-labels
        left_col[i].set_ylabel(var_labels[i])
        left_col[0].set_ylim([0.0, 1.15])
        left_col[1].set_ylim([-0.1, 0.55])
        left_col[2].set_ylim([0.0, 2.65])
        right_col[i].set_ylabel(f"Δ {var_labels[i]}")
        right_col[i].set_ylim([-0.50, 0.50])

        for col in (0, 1):
            axs[i, col].grid(True, linestyle="--", alpha=0.7)
            # Use 'best' to let matplotlib avoid covering the curves
            axs[i, col].legend(loc="best")

    # Set x-labels only on the bottom row
    left_col[-1].set_xlabel("Position (x)")
    right_col[-1].set_xlabel("Position (x)")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/result.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        f"../recaps/01StateOf1D/imgs/{n_elems}__{poly_deg}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


from typing import List
from matplotlib.lines import Line2D
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunMetrics:
    solver: str
    N: int
    P: int
    exec_time_s: float
    err_rho: float
    failed: bool


def plot_pareto(all_runs_dict: dict):
    # 1. Convert your all_runs dictionary into the RunMetrics list
    runs: List[RunMetrics] = []
    for (n_elems, poly_deg), (ris, x_cords) in all_runs_dict.items():
        for solver_key, out_data in ris.items():
            if solver_key == "moc":
                continue

            t = out_data.time_to_sol
            e = out_data.rho_err
            # Mark as failed if time is 0 or error is NaN/Inf
            is_failed = t <= 0 or np.isnan(e) or np.isinf(e)

            runs.append(
                RunMetrics(
                    solver=solver_key,
                    N=n_elems,
                    P=poly_deg,
                    exec_time_s=t,
                    err_rho=e,
                    failed=is_failed,
                )
            )

    # 2. Filter valid runs
    valid_runs = [r for r in runs if not r.failed]
    if not valid_runs:
        print("No valid data to plot!")
        return

    plt.figure(figsize=(12, 8))

    # 3. Plot the scatter points (Grouped by solver and N)
    # --- CHANGED: Colors now map to N_elements instead of P ---
    colors = {20: "blue", 200: "green", 2000: "red"}
    markers = {"csd_pp": "o", "ssp3": "s", "ssp": "^"}

    for r in valid_runs:
        c = colors.get(r.N, "black")  # Extract color based on N
        m = markers.get(r.solver, "x")

        # Plot individual points (zorder=3 ensures they sit on top of the lines)
        plt.scatter(
            r.exec_time_s,
            r.err_rho,
            marker=m,
            color=c,
            s=100,
            alpha=0.9,
            edgecolors="k",
            zorder=3,
        )

        # Add a tiny text label to each point
        plt.annotate(
            f"{r.solver}\nN={r.N}, P={r.P}",
            (r.exec_time_s, r.err_rho),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            zorder=4,
        )

    # 4. Connect points with the same N and P
    runs_by_config = defaultdict(list)
    for r in valid_runs:
        runs_by_config[(r.N, r.P)].append(r)

    for (N, P), group in runs_by_config.items():
        if len(group) > 1:
            group_sorted = sorted(group, key=lambda x: x.exec_time_s)
            x_vals = [r.exec_time_s for r in group_sorted]
            y_vals = [r.err_rho for r in group_sorted]
            plt.plot(
                x_vals,
                y_vals,
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                # zorder=1,
            )

    # 7. Formatting
    plt.title(
        "Performance vs Accuracy (Color mapped to Grid Size N)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Execution Time (seconds) - Lower is better", fontsize=12)
    plt.ylabel(r"$L_2$ Density Error - Lower is better", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # 8. Create custom legend handles
    legend_elements = [
        Line2D(
            [0], [0], color="gray", lw=1.5, linestyle=":", label="Same Config (N, P)"
        ),
        # Solvers (Markers)
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=10,
            label="Explicit (ssp3)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=10,
            label="Implicit (csd_pp)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=10,
            label="Finite Diff (fd)",
        ),
        # --- CHANGED: Grid Sizes (Colors) ---
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="k",
            markersize=10,
            label="N=20",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markeredgecolor="k",
            markersize=10,
            label="N=200",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markeredgecolor="k",
            markersize=10,
            label="N=2000",
        ),
    ]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("true_pareto_front.png", dpi=300, bbox_inches="tight")
    plt.savefig(
        f"../recaps/01StateOf1D/imgs/convergences.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_max = 1.00
    l_0 = 10.0
    dt_impl = t_max / 10
    dt_expl = t_max / 1000
    ric = RiemannIC(
        rho_L=1.0,
        vel_L=0.0,
        pres_L=1,
        rho_R=0.125,
        vel_R=0.0,
        pres_R=0.1,
        gamma=1.4,
    )
    all_runs = {}
    for n_elems in [20, 200, 2000]:  #
        for poly_deg in [3, 5]:  #
            dx_eff = l_0 / (n_elems * poly_deg**2)
            out_dir = Path(
                str("build/runs" + "__" + str(n_elems) + "__" + str(poly_deg))
            )
            data = ProblemData(
                dt_impl=0.5 / ric.c_max() * dx_eff,
                dt_expl=0.01 / ric.c_max() * dx_eff,
                n_elems=n_elems,
                poly_deg=poly_deg,
                l_0=l_0,
                t_max=t_max,
            )
            ris, x_cords = run_extract_ris(
                data=data,
                output_dir=out_dir,
                # --- Initial condition ---
                ic=ric,
                # Keys to actually execute (subset of SOLVER_REGISTRY keys, or "all")
                run_keys={"csd_pp", "ssp3", "csd", "fd", "fd_pp"},  #
                err_keys={"csd_pp", "ssp3", "csd", "fd", "fd_pp"},  #
                # Checks error and plots the delta with respect to MOC
                # False → skip the binary, only reload existing .bin files
                run_sim=0,
            )
            plot_all_sims = 0
            if plot_all_sims:
                plot_results(ris, x_cords, ric, data)
            all_runs[(n_elems, poly_deg)] = (ris, x_cords)
    plot_pareto(all_runs)
