#!/usr/bin/env python3
import matplotlib.ticker as ticker
import json
import argparse
from pathlib import Path
import subprocess
import math
import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ====================================================================
# CONFIGURATION AND HELPER FUNCTIONS (Preserved from original)
# ====================================================================
@dataclass
class ConvTest:
    p_h_test: int = 2
    dt_h_test: float = 0.0001
    t_final_h_test: float = 2.0
    n_range_h_test: np.ndarray = field(default_factory=lambda: np.array([10, 20, 40]))

    n_p_test: int = 20
    dt_p_test: float = 0.001
    t_final_p_test: float = 2.0
    p_range_p_test: np.ndarray = field(
        default_factory=lambda: np.array([2 * x + 1 for x in range(4)])
    )


PROFILES = {
    "acoustic": {
        "exec": ["mpirun", "./build/test/acoustic2d"],
        "args": [
            "-snes_mf",
            "-ts_type",
            "bdf",
            "-ts_bdf_order",
            "2",  # Unified flag naming
            # "-ts_type", "ssp", "-ts_ssp_type", "rks3",
            "-ts_adapt_type",
            "none",
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
            "-do_post_ts",
            "true",
        ],
        "sim_data": ConvTest(dt_h_test=1.0e-6, t_final_h_test=1.0e-5),
    },
    "vortex": {
        "exec": ["mpirun", "-n", "5", "./build/test/test_2d"],
        "args": [
            "-box_l",
            "10.0",
            "-jac_type",
            "complexfd",
            "-snes_mf",
            "-ts_type",
            "bdf",
            "-ts_bdf_order",
            "2",  # Unified flag naming
            # "-ts_type", "ssp", "-ts_ssp_type", "rks3",
            "-ts_adapt_type",
            "none",
            "-apply_filter",
            "false",
            "-filt_strength",
            "32",
            "-filt_cut_off_deg",
            "2",
            "-apply_weno",
            "false",
            "-weno_gamma",
            "0.998",
            "-do_post_ts",
            "true",
            "-solve_positive_preserving",
            "false",
        ],
        "sim_data": {
            "max_time": 2.0,
        },
    },
}

EXEC_CMD = []
BASE_ARGS = []
PROBLEM_NAME = ""


def run_sim_get_errs(n_elems, poly_deg, dt, max_time=2.0):
    """Runs the 2D solver overriding specific elements, polynomial degree, and dt parameters."""
    cmd = (
        EXEC_CMD
        + BASE_ARGS
        + [
            "-n_elems",
            str(n_elems),
            "-poly_deg",
            str(poly_deg),
            "-ts_dt",
            str(dt),
            "-ts_max_time",
            str(max_time),
        ]
    )
    print(f"Running: N={n_elems:<2}x{n_elems:<2} | P={poly_deg} | dt={dt:<7} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    m_rho = re.search(r"Rho\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
    m_rhou = re.search(r"RhoU\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
    m_rhov = re.search(r"RhoV\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
    m_rhoe = re.search(r"RhoE\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)

    if m_rho and m_rhou and m_rhov and m_rhoe:
        rho_val, rhou_val = float(m_rho.group(1)), float(m_rhou.group(1))
        rhov_val, rhoe_val = float(m_rhov.group(1)), float(m_rhoe.group(1))

        print(
            f"  -> Extracted Errs: Rho={rho_val:.4e} | RhoU={rhou_val:.4e} | RhoV={rhov_val:.4e} | RhoE={rhoe_val:.4e}"
        )
        return rho_val, rhou_val, rhov_val, rhoe_val
    return None, None, None, None


def get_convergence_order(x_array, y_array, is_spatial):
    """Calculates the numerical order of convergence using linear regression on log-log data."""
    slope, _ = np.polyfit(np.log(x_array), np.log(y_array), 1)
    return abs(slope) if is_spatial else slope


# ====================================================================
# NEW FUNCTIONS: Data Generation and Visualization
# ====================================================================


def run_convergence(configs, problem, output_file="tools/conv_ris.json"):
    """
    Runs simulations for provided configurations, calculates convergence orders,
    and dumps the data to a JSON file.
    configs should be a list of tuples: (p_order, dt, [N1, N2, N3...])
    """
    print("\n" + "=" * 50)
    print(f" RUNNING CONVERGENCE TESTS [{PROBLEM_NAME.upper()}]")
    print("=" * 50)

    results = []
    max_time = problem["max_time"]

    for p_order, dt, N_array in configs:
        errs_rho, errs_rhou, errs_rhov, errs_rhoe = [], [], [], []
        runs = []

        for N in N_array:
            rho, rhou, rhov, rhoe = run_sim_get_errs(N, p_order, dt, max_time=max_time)
            if rho is None:
                continue

            errs_rho.append(rho)
            errs_rhou.append(rhou)
            errs_rhov.append(rhov)
            errs_rhoe.append(rhoe)

            runs.append(
                {
                    "grid_size": int(N),
                    "errors": {"rho": rho, "rho_u": rhou, "rho_v": rhov, "rho_e": rhoe},
                }
            )

        # Calculate convergence orders
        order_rho = get_convergence_order(N_array, errs_rho, is_spatial=True)
        order_rhou = get_convergence_order(N_array, errs_rhou, is_spatial=True)
        order_rhov = get_convergence_order(N_array, errs_rhov, is_spatial=True)
        order_rhoe = get_convergence_order(N_array, errs_rhoe, is_spatial=True)

        last_run = {
            "p_order": int(p_order),
            "dt": float(dt),
            "runs": runs,
            "convergence_orders": {
                "rho": round(float(order_rho), 4),
                "rho_u": round(float(order_rhou), 4),
                "rho_v": round(float(order_rhov), 4),
                "rho_e": round(float(order_rhoe), 4),
            },
        }
        results.append(last_run)
        print(last_run)

    # Create directory if it doesn't exist and save JSON
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n[+] Results successfully written to {output_file}")
    return results


def show_ris(json_file="tools/conv_ris.json"):
    """
    Reads the JSON file containing convergence data, prints a LaTeX table,
    and generates the corresponding Matplotlib plots.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}. Run run_convergence first.")
        return

    # 1. Print LaTeX Table
    print("\n" + "=" * 50)
    print(" LATEX CONVERGENCE TABLE")
    print("=" * 50 + "\n")

    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\begin{tabular}{cc|cccc}")
    print("\\hline")
    print(
        "$P$ & $\\Delta t$ & $\\mathcal{O}(\\rho)$ & $\\mathcal{O}(\\rho u)$ & $\\mathcal{O}(\\rho v)$ & $\\mathcal{O}(\\rho e)$ \\\\"
    )
    print("\\hline")

    for run in data:
        p = run["p_order"]
        dt = run["dt"]
        orders = run["convergence_orders"]
        print(
            f"{p} & {dt} & {orders['rho']:.4f} & {orders['rho_u']:.4f} & {orders['rho_v']:.4f} & {orders['rho_e']:.4f} \\\\"
        )

    print("\\hline")
    print("\\end{tabular}")
    print(
        "\\caption{Spatial convergence orders for different polynomial degrees ($P$).}"
    )
    print("\\label{tab:convergence_orders}")
    print("\\end{table}\n")

    # 2. Generate Plots
    # Generate conv_total
    plt.rcParams.update({"font.size": 14})

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    for run in data:
        p = run["p_order"]
        N_array = np.array([r["grid_size"] for r in run["runs"]])
        errs_rho = np.array([r["errors"]["rho"] for r in run["runs"]])
        errs_rhou = np.array([r["errors"]["rho_u"] for r in run["runs"]])
        errs_rhov = np.array([r["errors"]["rho_v"] for r in run["runs"]])
        errs_rhoe = np.array([r["errors"]["rho_e"] for r in run["runs"]])

        # Plot data - we capture the first line object (l_rho) to extract its color
        # so we can use the EXACT same color for the other subplots and theoretical lines
        (l_rho,) = ax[0, 0].loglog(N_array, errs_rho, marker="o", label=f"P={p}")
        c = l_rho.get_color()

        ax[1, 0].loglog(N_array, errs_rhou, marker="o", color=c)
        ax[0, 1].loglog(N_array, errs_rhov, marker="o", color=c)
        ax[1, 1].loglog(N_array, errs_rhoe, marker="o", color=c)

        # Theoretical Reference Lines (Dashed, matching color, slightly transparent)
        ref_rho = errs_rho[0] * (N_array[0] / N_array) ** (p + 1)
        c = "black"
        ax[0, 0].loglog(N_array, ref_rho, linestyle="--", color=c)

        ref_rhou = errs_rhou[0] * (N_array[0] / N_array) ** (p + 1)
        ax[1, 0].loglog(N_array, ref_rhou, linestyle="--", color=c)

        ref_rhov = errs_rhov[0] * (N_array[0] / N_array) ** (p + 1)
        ax[0, 1].loglog(N_array, ref_rhov, linestyle="--", color=c)

        ref_rhoe = errs_rhoe[0] * (N_array[0] / N_array) ** (p + 1)
        ax[1, 1].loglog(N_array, ref_rhoe, linestyle="--", color=c)

    # Format individual subplots
    ax[0, 0].set_title(r"$\rho$ error")
    ax[1, 0].set_title(r"$\rho u$ error")
    ax[0, 1].set_title(r"$\rho v$ error")
    ax[1, 1].set_title(r"$\rho e$ error")

    ticks = [10, 20, 40]
    for axis in ax.flat:

        axis.grid(True, which="both", ls="--", alpha=0.5)
        axis.set_xticks(ticks)

        axis.xaxis.set_major_formatter(ticker.ScalarFormatter())
        axis.ticklabel_format(style="plain", axis="x")
        axis.xaxis.set_minor_formatter(ticker.NullFormatter())

    # Apply global figure formatting (OUTSIDE the loop)
    fig.suptitle(f"2D Spatial Convergence", fontsize=18)
    fig.supxlabel("Grid Resolution Dimensions ($N \\times N$)")
    fig.supylabel("Continuous integrated $L_2$ Error")

    # Create a unified legend
    from matplotlib.lines import Line2D

    handles, labels = ax[0, 0].get_legend_handles_labels()
    # Add a dummy line to the legend to explain what the dashed lines mean
    handles.append(
        Line2D([0], [0], color="gray", linestyle="--", label="$\\mathcal{O}(h^{P+1})$")
    )

    # Place the legend outside the subplots
    fig.legend(handles=handles)

    plt.tight_layout()  # Prevents labels from overlapping

    # Save and show
    img_path = Path("./imgs/conv/")
    img_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        img_path / "conv_total.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def analyze_p_convergence(json_file="tools/conv_ris.json"):
    """
    Analyzes p-convergence across all N, fits the relation E(N,p) = A * (p+1)^B * N^C,
    and generates a p-convergence plot.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}.")
        return

    img_path = Path("./imgs/conv/")
    print("\n" + "=" * 50)
    print(" P-CONVERGENCE & MULTIVARIABLE FIT")
    print("=" * 50 + "\n")

    # Dictionaries to aggregate data for the fit
    fit_data = {"rho": [], "rho_u": [], "rho_v": [], "rho_e": []}

    for run in data:
        p = run["p_order"]
        for r in run["runs"]:
            N = r["grid_size"]
            errs = r["errors"]

            # Filter out failed runs
            if errs["rho"] is not None and errs["rho"] > 0:
                fit_data["rho"].append((p, N, errs["rho"]))
                fit_data["rho_u"].append((p, N, errs["rho_u"]))
                fit_data["rho_v"].append((p, N, errs["rho_v"]))
                fit_data["rho_e"].append((p, N, errs["rho_e"]))

    # Perform least-squares fit for E(N,p) = A * h^(B*p) assuming h = 1/N
    print("Fitted relations for E(N,p) = A * h^(B*p):")
    print("-" * 50)

    latex = ""
    latex += "\\centering\n"
    latex += "\\begin{tabular}{ c | c }\n"
    latex += "  N & o(p) \\\\ \n"
    for var, tuples in fit_data.items():
        if not tuples:
            continue

        p_vals = np.array([t[0] for t in tuples])
        N_vals = np.array([t[1] for t in tuples])
        E_vals = np.array([t[2] for t in tuples])

        # Calculate h (assuming domain length L=1, or absorbed into A)
        h_vals = 1.0 / N_vals

        # Design matrix X: [1, p * ln(h)]
        X = np.column_stack((np.ones_like(p_vals), p_vals * np.log(h_vals)))
        Y = np.log(E_vals)  # ln(E)

        # Solve X * coeffs = Y
        coeffs, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

        lnA, B = coeffs
        A = np.exp(lnA)

        print(f"  {var:<6}: A = {A:.4e}, B = {B:.4f}  --> Order ~ {B:.4f}p")
        latex += f"  {var} & {B:.4f} \\\\\n"

    latex += "    \\end{tabular}\n"

    with open(img_path / "conv_ord_p.tex", "w") as f:
        f.write(latex)
    # --- Generate P-Convergence Plot (Error vs P grouped by N) ---
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # Identify unique grid sizes (N) to create separate lines
    unique_Ns = sorted(list(set([t[1] for t in fit_data["rho"]])))

    axes_map = {
        "rho": ax[0, 0],
        "rho_u": ax[1, 0],
        "rho_v": ax[0, 1],
        "rho_e": ax[1, 1],
    }
    titles = {
        "rho": r"$\rho$ error",
        "rho_u": r"$\rho u$ error",
        "rho_v": r"$\rho v$ error",
        "rho_e": r"$\rho e$ error",
    }

    for var, tuples in fit_data.items():
        axis = axes_map[var]
        axis.set_title(titles[var])

        for N in unique_Ns:
            # Extract points for this specific N, sorted by polynomial order (p)
            pts = sorted(
                [(t[0], t[2]) for t in tuples if t[1] == N], key=lambda x: x[0]
            )
            if not pts:
                continue

            p_arr = np.array([pt[0] for pt in pts])
            e_arr = np.array([pt[1] for pt in pts])

            # Semilogy is standard for P-convergence as error typically drops exponentially with P
            axis.semilogy(p_arr, e_arr, marker="s", label=f"N={N}")

        axis.grid(True, which="both", ls="--", alpha=0.5)
        axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(r"$p$ convergence", fontsize=18)
    fig.supxlabel("Polynomial Degree ($P$)")
    fig.supylabel("Continuous integrated $L_2$ Error")

    # Unified Legend placed at the bottom
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        ncol=len(unique_Ns),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    img_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_path / "conv_p_total.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    # Identify unique grid sizes (N)
    unique_Ns = sorted(list(set([t[1] for t in fit_data["rho"]])))

    print("\n" + "=" * 50)
    print(" LATEX P-CONVERGENCE TABLE (Fit: E = A * h^{Bp})")
    print("=" * 50 + "\n")

    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\begin{tabular}{c|cccc}")
    print("        \\hline")
    print("        $N$ & $B(\\rho)$ & $B(\\rho u)$ & $B(\\rho v)$ & $B(\\rho e)$ \\\\")
    print("        \\hline")

    for N in unique_Ns:
        b_vals = []
        for var in ["rho", "rho_u", "rho_v", "rho_e"]:
            # Extract points for this specific N
            tuples = [t for t in fit_data[var] if t[1] == N]

            if len(tuples) > 1:
                p_vals = np.array([t[0] for t in tuples])
                E_vals = np.array([t[2] for t in tuples])

                # Assuming domain L=1, h = 1/N
                h_val = 1.0 / N

                # Linearize: ln(E) = ln(A) + B * p * ln(h)
                # Let x = p * ln(h), y = ln(E) -> slope is B
                x = p_vals * np.log(h_val)
                y = np.log(E_vals)

                slope, intercept = np.polyfit(x, y, 1)
                b_vals.append(slope)
            else:
                b_vals.append(0.0)  # Fallback if missing data

        # Print the row for this N
        print(
            f"        {N:<3} & {b_vals[0]:.2f}      & {b_vals[1]:.2f}      & {b_vals[2]:.2f}      & {b_vals[3]:.2f}      \\\\"
        )

    print("        \\hline")
    print("    \\end{tabular}")
    print(
        "    \\caption{P-convergence fit parameter $B$ ($E = A \\cdot h^{B \\cdot p}$) for varying grid resolutions ($N$).}"
    )
    print("    \\label{tab:p_convergence_orders}")
    print("\\end{table}\n")


# ====================================================================
# MAIN EXECUTION
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Solver Convergence Tests")
    parser.add_argument("--problem", choices=["acoustic", "vortex"], default="vortex")
    parser.add_argument("--out_dir", default="./")
    parser.add_argument("--out_file", default="tools/conv_ris.json")
    parser.add_argument("--read_file", default="tools/conv_ris.json")

    args = parser.parse_args()

    PROBLEM_NAME = args.problem
    out_file = Path(Path(args.out_dir) / Path(args.out_file))
    read_file = Path(Path(args.out_dir) / Path(args.read_file))

    profile = PROFILES[PROBLEM_NAME]
    EXEC_CMD = profile["exec"]
    BASE_ARGS = profile["args"]

    # Define the configurations based on the provided JSON data structure
    """
    n_conv_list = [10, 20, 40]
    configs_to_run = [
            # (1, 0.001 , n_conv_list),
            # (2, 0.001, n_conv_list),
            # (3, 0.0001, n_conv_list),
            # (4, 1e-05, n_conv_list),
            # (5, 1e-05, n_conv_list),
            (6, 1e-06, [40]),
            ]
    """

    # 1. Run simulations and build JSON
    # run_convergence(configs=configs_to_run, problem=profile["sim_data"], output_file=out_file)

    # 2. Read JSON, output LaTeX table and Plots
    # show_ris(json_file=str(read_file))
    analyze_p_convergence(json_file=str(read_file))


"""

Running: N=10x10 | P=1 | dt=0.001   ...
  -> Extracted Errs: Rho=4.4956e-01 | RhoU=7.0477e-01 | RhoV=2.3200e+00 | RhoE=7.6439e+00
  Running: N=20x20 | P=1 | dt=0.001   ...
    -> Extracted Errs: Rho=1.2439e-01 | RhoU=2.3955e-01 | RhoV=6.5587e-01 | RhoE=2.2098e+00
    Running: N=40x40 | P=1 | dt=0.001   ...
      -> Extracted Errs: Rho=2.7773e-02 | RhoU=5.7036e-02 | RhoV=1.4778e-01 | RhoE=5.0305e-01
      {'p_order': 1, 'dt': 0.001, 'runs': [{'grid_size': 10, 'errors': {'rho': 0.4495577, 'rho_u': 0.7047672, 'rho_v': 2.320011, 'rho_e': 7.643904}}, {'grid_size': 20, 'errors': {'rho': 0.1243938, 'rho_u': 0.2395527, 'rho_v': 0.6558749, 'rho_e': 2.209818}}, {'grid_size': 40, 'errors': {'rho': 0.02777267, 'rho_u': 0.05703647, 'rho_v': 0.1477782, 'rho_e': 0.503054}}], 'convergence_orders': {'rho': 2.0084, 'rho_u': 1.8136, 'rho_v': 1.9863, 'rho_e': 1.9628}}

Running: N=10x10 | P=2 | dt=0.001   ...
  -> Extracted Errs: Rho=8.6916e-02 | RhoU=1.6708e-01 | RhoV=4.4503e-01 | RhoE=1.4229e+00
Running: N=20x20 | P=2 | dt=0.001   ...
  -> Extracted Errs: Rho=1.3639e-02 | RhoU=2.6167e-02 | RhoV=7.5674e-02 | RhoE=2.6484e-01
Running: N=40x40 | P=2 | dt=0.001   ...
  -> Extracted Errs: Rho=1.3207e-03 | RhoU=2.2529e-03 | RhoV=8.2957e-03 | RhoE=3.2194e-02
{'p_order': 2, 'dt': 0.001, 'runs': [{'grid_size': 10, 'errors': {'rho': 0.08691578, 'rho_u': 0.1670817, 'rho_v': 0.4450297, 'rho_e': 1.422878}}, {'grid_size': 20, 'errors': {'rho': 0.01363889, 'rho_u': 0.02616724, 'rho_v': 0.07567368, 'rho_e': 0.2648365}}, {'grid_size': 40, 'errors': {'rho': 0.001320676, 'rho_u': 0.002252862, 'rho_v': 0.008295702, 'rho_e': 0.03219402}}], 'convergence_orders': {'rho': 3.0201, 'rho_u': 3.1063, 'rho_v': 2.8727, 'rho_e': 2.7329}}

Running: N=10x10 | P=3 | dt=0.0001  ...
  -> Extracted Errs: Rho=2.7917e-02 | RhoU=4.6712e-02 | RhoV=1.4159e-01 | RhoE=4.5316e-01
Running: N=20x20 | P=3 | dt=0.0001  ...
  -> Extracted Errs: Rho=1.5715e-03 | RhoU=2.6615e-03 | RhoV=9.1975e-03 | RhoE=3.2538e-02
Running: N=40x40 | P=3 | dt=0.0001  ...
  -> Extracted Errs: Rho=9.1432e-05 | RhoU=1.3882e-04 | RhoV=5.4829e-04 | RhoE=2.0537e-03
{'p_order': 3, 'dt': 0.0001, 'runs': [{'grid_size': 10, 'errors': {'rho': 0.02791726, 'rho_u': 0.0467115, 'rho_v': 0.1415944, 'rho_e': 0.4531649}}, {'grid_size': 20, 'errors': {'rho': 0.001571462, 'rho_u': 0.002661455, 'rho_v': 0.009197528, 'rho_e': 0.0325376}}, {'grid_size': 40, 'errors': {'rho': 9.14316e-05, 'rho_u': 0.0001388152, 'rho_v': 0.000548286, 'rho_e': 0.002053652}}], 'convergence_orders': {'rho': 4.1271, 'rho_u': 4.1972, 'rho_v': 4.0063, 'rho_e': 3.8929}}

Running: N=10x10 | P=4 | dt=1e-05   ...
  -> Extracted Errs: Rho=6.1599e-03 | RhoU=1.0848e-02 | RhoV=3.6815e-02 | RhoE=1.3435e-01
Running: N=20x20 | P=4 | dt=1e-05   ...
  -> Extracted Errs: Rho=2.1253e-04 | RhoU=2.7270e-04 | RhoV=1.2311e-03 | RhoE=4.3590e-03
Running: N=40x40 | P=4 | dt=1e-05   ...
  -> Extracted Errs: Rho=5.6265e-06 | RhoU=8.8022e-06 | RhoV=3.3858e-05 | RhoE=1.2601e-04
{'p_order': 4, 'dt': 1e-05, 'runs': [{'grid_size': 10, 'errors': {'rho': 0.006159868, 'rho_u': 0.01084848, 'rho_v': 0.03681538, 'rho_e': 0.1343481}}, {'grid_size': 20, 'errors': {'rho': 0.0002125255, 'rho_u': 0.0002727025, 'rho_v': 0.001231111, 'rho_e': 0.004358973}}, {'grid_size': 40, 'errors': {'rho': 5.626498e-06, 'rho_u': 8.802231e-06, 'rho_v': 3.385796e-05, 'rho_e': 0.0001260117}}], 'convergence_orders': {'rho': 5.0482, 'rho_u': 5.1337, 'rho_v': 5.0433, 'rho_e': 5.0291}}

Running: N=10x10 | P=5 | dt=1e-05   ...
  -> Extracted Errs: Rho=1.7997e-03 | RhoU=2.4336e-03 | RhoV=1.1019e-02 | RhoE=3.9697e-02
Running: N=20x20 | P=5 | dt=1e-05   ...
  -> Extracted Errs: Rho=2.8166e-05 | RhoU=3.9566e-05 | RhoV=1.5088e-04 | RhoE=4.8708e-04
Running: N=40x40 | P=5 | dt=1e-05   ...
  -> Extracted Errs: Rho=4.0461e-07 | RhoU=5.7064e-07 | RhoV=2.3172e-06 | RhoE=7.9573e-06
{'p_order': 5, 'dt': 1e-05, 'runs': [{'grid_size': 10, 'errors': {'rho': 0.00179975, 'rho_u': 0.002433551, 'rho_v': 0.01101886, 'rho_e': 0.03969654}}, {'grid_size': 20, 'errors': {'rho': 2.8166e-05, 'rho_u': 3.956581e-05, 'rho_v': 0.0001508755, 'rho_e': 0.0004870826}}, {'grid_size': 40, 'errors': {'rho': 4.046126e-07, 'rho_u': 5.706376e-07, 'rho_v': 2.317203e-06, 'rho_e': 7.957273e-06}}], 'convergence_orders': {'rho': 6.0595, 'rho_u': 6.0291, 'rho_v': 6.1077, 'rho_e': 6.1422}}
Running: N=10x10 | P=6 | dt=1e-06   ...
  -> Extracted Errs: Rho=6.6626e-04 | RhoU=5.6831e-04 | RhoV=3.3681e-03 | RhoE=9.4530e-03
Running: N=20x20 | P=6 | dt=1e-06   ...
  -> Extracted Errs: Rho=4.4454e-06 | RhoU=4.5563e-06 | RhoV=2.4974e-05 | RhoE=8.1710e-05
"""
