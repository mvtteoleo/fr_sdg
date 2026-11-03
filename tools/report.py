import re
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
from collections import defaultdict


# ==========================================
# 1. Data Structure
# ==========================================
@dataclass
class RunMetrics:
    solver: str
    N: int
    P: int
    dt: float
    exec_time_s: Optional[float] = None
    err_rho: Optional[float] = None
    err_rhoU: Optional[float] = None
    err_rhoE: Optional[float] = None

    # PETSc Diagnostics
    mean_snes_its: Optional[float] = None
    mean_ksp_its: Optional[float] = None
    max_snes_its: Optional[float] = None
    max_ksp_its: Optional[float] = None

    @property
    def failed(self) -> bool:
        return self.err_rho is None


# ==========================================
# 2. Parsing Logic
# ==========================================
def parse_terminal_output(filepath: str) -> List[RunMetrics]:
    runs_dict = {}  # Use a dict to track runs by (Solver, N, P)

    current_N = None
    current_P = None
    current_dt = None
    current_error_solver = None

    # Updated regexes to match your exact output
    cfg_pattern = re.compile(r"\[(\w+)\]\s+N=(\d+)\s+P=(\d+)\s+dt=([0-9.]+)")
    time_pattern = re.compile(r"\[(\w+)\]\s+done in ([0-9.]+) s")
    err_start_pattern = re.compile(r"Error (\w+):")
    rho_pattern = re.compile(r"\s*rho\s+:([0-9.]+)")

    with open(filepath, "r") as f:
        for line in f:
            # 1. Catch the configuration headers (even if only one solver prints it)
            cfg_match = cfg_pattern.search(line)
            if cfg_match:
                solver = cfg_match.group(1)
                current_N = int(cfg_match.group(2))
                current_P = int(cfg_match.group(3))
                current_dt = float(cfg_match.group(4))

                # Register the run
                runs_dict[(solver, current_N, current_P)] = RunMetrics(
                    solver=solver, N=current_N, P=current_P, dt=current_dt
                )

            # 2. Catch execution times
            time_match = time_pattern.search(line)
            if time_match and current_N is not None:
                solver = time_match.group(1)
                if (solver, current_N, current_P) in runs_dict:
                    runs_dict[(solver, current_N, current_P)].exec_time_s = float(
                        time_match.group(2)
                    )

            # 3. Catch which solver's error block we are entering
            err_start_match = err_start_pattern.search(line)
            if err_start_match:
                current_error_solver = err_start_match.group(1)

                # If we loaded data from a .bin but didn't parse a config header for it,
                # create a dummy entry so it doesn't crash
                key = (current_error_solver, current_N, current_P)
                if key not in runs_dict and current_N is not None:
                    runs_dict[key] = RunMetrics(
                        solver=current_error_solver,
                        N=current_N,
                        P=current_P,
                        dt=current_dt,
                    )

            # 4. Catch the actual errors and assign to the CORRECT solver
            rho_match = rho_pattern.search(line)
            if rho_match and current_error_solver and current_N is not None:
                key = (current_error_solver, current_N, current_P)
                val = float(rho_match.group(1))
                # Ignore NaN values so they stay marked as FAILED
                import math

                if not math.isnan(val):
                    runs_dict[key].err_rho = val

    return list(runs_dict.values())


def parse_petsc_log(filepath: str, metrics: RunMetrics):
    snes_its = []
    ksp_its = []

    snes_pattern = re.compile(r"^\s*(\d+)\s+SNES Function norm")
    ksp_pattern = re.compile(r"^\s*(\d+)\s+KSP Residual norm")

    try:
        with open(filepath, "r") as f:
            for line in f:
                s_match = snes_pattern.search(line)
                if s_match:
                    snes_its.append(int(s_match.group(1)))

                k_match = ksp_pattern.search(line)
                if k_match:
                    ksp_its.append(int(k_match.group(1)))

        if snes_its:
            metrics.mean_snes_its = sum(snes_its) / len(snes_its)
            metrics.max_snes_its = max(snes_its)
        if ksp_its:
            metrics.mean_ksp_its = sum(ksp_its) / len(ksp_its)
            metrics.max_ksp_its = max(ksp_its)

    except FileNotFoundError:
        pass


# ==========================================
# 3. Plotting Logic
# ==========================================
def plot_pareto_front(runs: List[RunMetrics]):
    valid_runs = [r for r in runs if not r.failed]

    plt.figure(figsize=(12, 8))

    # 1. Plot the scatter points (grouped by solver and P)
    colors = {1: "blue", 2: "orange", 3: "green", 4: "red", 5: "purple"}
    markers = {"csd_pp": "o", "ssp3": "s", "fd": "^"}

    for r in valid_runs:
        c = colors.get(r.P, "black")
        m = markers.get(r.solver, "x")
        # Plot individual points without connecting them all
        plt.scatter(r.exec_time_s, r.err_rho, marker=m, color=c, s=100, alpha=0.7)
        # Add a tiny text label to each point
        plt.annotate(
            f"{r.solver}\nN={r.N}, P={r.P}",
            (r.exec_time_s, r.err_rho),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
        )

    # 2. Calculate the True Pareto Envelope
    # Sort all runs primarily by Execution Time (fastest first)
    sorted_runs = sorted(valid_runs, key=lambda x: x.exec_time_s)

    pareto_front_x = []
    pareto_front_y = []
    current_best_error = float("inf")

    for r in sorted_runs:
        # If this run is faster (due to sorting) AND has a better error than anything we've seen
        if r.err_rho < current_best_error:
            pareto_front_x.append(r.exec_time_s)
            pareto_front_y.append(r.err_rho)
            current_best_error = r.err_rho

    # 3. Draw the Pareto envelope line
    plt.plot(
        pareto_front_x,
        pareto_front_y,
        color="black",
        linewidth=2,
        linestyle="--",
        label="True Pareto Front (Optimal Runs)",
    )

    # Formatting
    plt.title("Performance vs Accuracy (True Pareto Front)")
    plt.xlabel("Execution Time (seconds) - Lower is better")
    plt.ylabel("L2 Density Error - Lower is better")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Create custom legend handles
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="Pareto Envelope"),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Explicit (ssp3)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Implicit (csd_pp)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="P=1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="P=3",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="P=4",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="purple",
            markersize=10,
            label="P=5",
        ),
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("true_pareto_front.png", dpi=300)
    plt.show()


# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    terminal_log_file = "tools/terminal_out.txt"

    print(f"Parsing {terminal_log_file}...\n")
    try:
        metrics_list = parse_terminal_output(terminal_log_file)

        for run in metrics_list:
            if not run.failed:
                # Dynamically construct path using the parsed solver name
                petsc_log_path = f"build/runs__{run.N}__{run.P}/{run.solver}/run.log"
                parse_petsc_log(petsc_log_path, run)

        # Updated Table Header to include Solver
        header = f"| {'Solver':<8} | {'N':<5} | {'P':<3} | {'Time (s)':<9} | {'Err (rho)':<10} | {'Mean SNES':<9} | {'Mean KSP':<9} | {'Max SNES':<8} | {'Max KSP':<8} |"
        print(header)
        print(f"| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |")

        for r in metrics_list:
            if r.failed:
                print(
                    f"| {r.solver:<8} | {r.N:<5} | {r.P:<3} | {'FAILED':<9} | {'N/A':<10} | {'N/A':<9} | {'N/A':<9} | {'N/A':<8} | {'N/A':<8} |"
                )
            else:
                s_mean = f"{r.mean_snes_its:.1f}" if r.mean_snes_its else "N/A"
                k_mean = f"{r.mean_ksp_its:.1f}" if r.mean_ksp_its else "N/A"
                s_max = f"{r.max_snes_its:.0f}" if r.max_snes_its else "N/A"
                k_max = f"{r.max_ksp_its:.0f}" if r.max_ksp_its else "N/A"

                print(
                    f"| {r.solver:<8} | {r.N:<5} | {r.P:<3} | {r.exec_time_s:<9.2f} | {r.err_rho:<10.5f} | {s_mean:<9} | {k_mean:<9} | {s_max:<8} | {k_max:<8} |"
                )

    except FileNotFoundError:
        print(f"Could not find {terminal_log_file}.")
