import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines

OUT_DIR = Path("outputs/cfl_ceiling/analysis")
ERR_VAR_NAMES = ["rho", "rhou", "rhov", "rhoE"]


# --------------------------------------------------------------------------
# Loading / flattening
# --------------------------------------------------------------------------
def _flatten_dict(d, prefix=""):
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def load_campaign(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    ceiling_records = []
    run_records = []

    for p_str, p_data in data.items():
        p_val = int(p_str)
        max_cfl = p_data.get("max_cfl", -1.0)

        if max_cfl and max_cfl > 0:
            ceiling_records.append(
                {
                    "p": p_val,
                    "max_cfl": max_cfl,
                    "max_cfl_dg": max_cfl * (p_val + 1) ** 2,
                }
            )

        for run in p_data.get("runs", []):
            record = {"p": p_val}
            inp = run.get("in", {})
            record.update(_flatten_dict(inp, prefix="in"))

            out = dict(run.get("out", {}))
            err = out.pop("err", None)
            record.update(_flatten_dict(out, prefix="out"))

            if err:
                for i, val in enumerate(err):
                    name = ERR_VAR_NAMES[i] if i < len(ERR_VAR_NAMES) else str(i)
                    record[f"out_err_{name}"] = val

            if record.get("in_cfl") is not None:
                record["cfl_dg"] = record["in_cfl"] * (p_val + 1) ** 2

            run_records.append(record)

    df_ceilings = pd.DataFrame(ceiling_records).sort_values("p").reset_index(drop=True)
    df_runs = pd.DataFrame(run_records)
    return df_ceilings, df_runs


def load_campaigns(json_paths):
    if isinstance(json_paths, (str, Path)):
        json_paths = [json_paths]
    ceilings_list, runs_list = [], []
    for path in json_paths:
        df_c, df_r = load_campaign(path)
        if not df_c.empty:
            ceilings_list.append(df_c)
        if not df_r.empty:
            runs_list.append(df_r)
    df_ceilings = (
        pd.concat(ceilings_list, ignore_index=True) if ceilings_list else pd.DataFrame()
    )
    df_runs = pd.concat(runs_list, ignore_index=True) if runs_list else pd.DataFrame()
    return df_ceilings, df_runs


def numeric_columns(df, exclude=("p",)):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c not in exclude]


def _iter_success(df_runs):
    if "out_success" in df_runs:
        return df_runs[df_runs["out_success"] == True]
    return df_runs


# --------------------------------------------------------------------------
# Generic Models (Empirical Scaling Laws)
# --------------------------------------------------------------------------
def power_law(x, a, b):
    return a * (x**b)


def power_law_overhead(x, a, b, c):
    """Power law with a constant overhead baseline: y = a * x^b + c"""
    return a * (x**b) + c


def ksp_surface_model(X, A, delta, gamma):
    p, cfl = X
    return A * ((p + 1) ** delta) * (cfl**gamma)


def cfl_scaling_model(p, C, a, b):
    """CFL = C / (p+a)^b"""
    return C / ((p + a) ** b)


# --------------------------------------------------------------------------
# Analytical Plots & Tables
# --------------------------------------------------------------------------
def plot_cfl_ceiling(df_ceilings, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        df_ceilings["p"], df_ceilings["max_cfl"], color="skyblue", edgecolor="black"
    )
    axes[0].set_title("Stability Limit: Max CFL vs p")
    axes[0].set_xlabel("Polynomial Degree (p)")
    axes[0].set_ylabel("Maximum Stable CFL")
    axes[0].set_xticks(df_ceilings["p"])
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    if "max_cfl_dg" in df_ceilings:
        axes[1].bar(
            df_ceilings["p"],
            df_ceilings["max_cfl_dg"],
            color="salmon",
            edgecolor="black",
        )
        axes[1].set_title(r"Stability Limit: Max $\mathrm{CFL}_{DG}$ vs p")
        axes[1].set_xlabel("Polynomial Degree (p)")
        axes[1].set_ylabel(r"Maximum Stable $\mathrm{CFL}_{DG}$")
        axes[1].set_xticks(df_ceilings["p"])
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_dir / "cfl_ceiling_vs_p.png", dpi=300)
    plt.close(fig)


def plot_time_vs_cfl(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    if "out_time_to_solution" not in success_df or "in_cfl" not in success_df:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    unique_ps = sorted(success_df["p"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(unique_ps)}

    # 1. Plot the actual data (without adding to the automatic legend)
    for (p, is_impl), grp in success_df.groupby(["p", "in_is_impl"]):
        grp = grp.sort_values("in_cfl")

        color = color_map[p]
        marker = "o" if is_impl else "s"
        linestyle = "-" if is_impl else "--"

        ax.plot(
            grp["in_cfl"],
            grp["out_time_to_solution"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            alpha=0.85,
        )

    # 2. Build the Custom Legend
    # Part A: Color mapping for polynomial degrees
    p_handles = [
        mlines.Line2D([], [], color=color_map[p], lw=2, label=f"$p={p}$")
        for p in unique_ps
    ]

    # Part B: Shape/Line mapping for integration schemes
    scheme_handles = [
        mlines.Line2D(
            [], [], color="black", marker="o", linestyle="-", label="Implicit"
        ),
        mlines.Line2D(
            [], [], color="black", marker="s", linestyle="--", label="Explicit"
        ),
    ]

    # Combine them into a single, clean legend
    ax.legend(handles=p_handles + scheme_handles, fontsize=9, ncol=2)

    # 3. Formatting
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Computational Efficiency: Time to Solution vs CFL")
    ax.set_xlabel("CFL Number (log scale)")
    ax.set_ylabel("Time to Solution (s, log scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "time_vs_cfl.png", dpi=300)
    plt.close(fig)


def plot_error_vs_cfl(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    if "out_err_rho" not in success_df or "in_cfl" not in success_df:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    unique_ps = sorted(success_df["p"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(unique_ps)}

    for (p, is_impl), grp in success_df.groupby(["p", "in_is_impl"]):
        grp = grp.sort_values("in_cfl")
        color = color_map[p]
        marker = "o" if is_impl else "s"
        linestyle = "-" if is_impl else "--"
        ax.plot(
            grp["in_cfl"],
            grp["out_err_rho"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            alpha=0.85,
        )

    p_handles = [
        mlines.Line2D([], [], color=color_map[p], lw=2, label=f"$p={p}$")
        for p in unique_ps
    ]

    # Part B: Shape/Line mapping for integration schemes
    scheme_handles = [
        mlines.Line2D(
            [], [], color="black", marker="o", linestyle="-", label="Implicit"
        ),
        mlines.Line2D(
            [], [], color="black", marker="s", linestyle="--", label="Explicit"
        ),
    ]

    # Combine them into a single, clean legend
    ax.legend(handles=p_handles + scheme_handles, fontsize=9, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Accuracy vs Stability: Error vs CFL")
    ax.set_xlabel("CFL (log scale)")
    ax.set_ylabel(r"Density error (log scale)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "error_vs_cfl.png", dpi=300)
    plt.close(fig)


def plot_implicit_speedup(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    if success_df.empty or "out_time_to_solution" not in success_df:
        return
    speedups = []
    p_vals = sorted(success_df["p"].unique())
    for p in p_vals:
        p_data = success_df[success_df["p"] == p]
        expl_min = p_data[p_data["in_is_impl"] == False]["out_time_to_solution"].min()
        impl_min = p_data[p_data["in_is_impl"] == True]["out_time_to_solution"].min()
        if pd.notnull(expl_min) and pd.notnull(impl_min) and impl_min > 0:
            speedups.append((p, expl_min / impl_min))
        else:
            speedups.append((p, 0))
    if not any(s[1] > 0 for s in speedups):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ps, factors = zip(*speedups)
    colors = ["tab:green" if f > 1 else "tab:red" for f in factors]
    ax.bar(ps, factors, color=colors, edgecolor="black", width=0.6)
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Break-even (Factor = 1)",
    )
    ax.set_title("Optimal Speedup: Explicit Time / Implicit Time")
    ax.set_xlabel("Polynomial Degree (p)")
    ax.set_ylabel("Speedup Factor")
    ax.set_xticks(ps)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "implicit_speedup_factor.png", dpi=300)
    plt.close(fig)


def plot_global_pareto_fronts(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    if success_df.empty:
        return
    expl_df = success_df[success_df["in_is_impl"] == False]
    impl_df = success_df[success_df["in_is_impl"] == True]

    def get_pareto(df):
        df_s = df.sort_values("out_err_rho").dropna(
            subset=["out_err_rho", "out_time_to_solution"]
        )
        p_err, p_time = [], []
        m_time = float("inf")
        for _, r in df_s.iterrows():
            if r["out_time_to_solution"] < m_time:
                m_time = r["out_time_to_solution"]
                p_err.append(r["out_err_rho"])
                p_time.append(m_time)
        return p_err, p_time

    fig, ax = plt.subplots(figsize=(9, 6))
    if not expl_df.empty:
        e_err, e_time = get_pareto(expl_df)
        ax.plot(
            e_err,
            e_time,
            color="red",
            marker="s",
            linewidth=2.5,
            label="Explicit Pareto",
        )
        ax.scatter(
            expl_df["out_err_rho"],
            expl_df["out_time_to_solution"],
            color="red",
            alpha=0.15,
        )
    if not impl_df.empty:
        i_err, i_time = get_pareto(impl_df)
        ax.plot(
            i_err,
            i_time,
            color="blue",
            marker="o",
            linewidth=2.5,
            label="Implicit Pareto",
        )
        ax.scatter(
            impl_df["out_err_rho"],
            impl_df["out_time_to_solution"],
            color="blue",
            alpha=0.15,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Global Work-Precision Envelope")
    ax.set_xlabel("Density $L^2$ Error")
    ax.set_ylabel("Minimum Wall-Clock Time (s)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "global_pareto_crossover.png", dpi=300)
    plt.close(fig)


def plot_work_precision_scaling(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    needed = {"out_err_rho", "out_time_to_solution", "in_is_impl", "p"}
    if not needed.issubset(success_df.columns):
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    unique_ps = sorted(success_df["p"].unique())
    colors = {True: "blue", False: "red"}
    labels = {True: "Implicit", False: "Explicit"}

    print("\n" + "=" * 60)
    print("WORK-PRECISION EFFICIENCY INDEX (Error ∝ Time^-k)")
    print("=" * 60)

    for is_impl in [False, True]:
        scheme_df = success_df[success_df["in_is_impl"] == is_impl]
        if scheme_df.empty:
            continue

        for p in unique_ps:
            p_data = scheme_df[scheme_df["p"] == p].dropna(
                subset=["out_err_rho", "out_time_to_solution"]
            )
            if len(p_data) < 3:
                continue

            p_data = p_data.sort_values("out_time_to_solution")
            t_data = p_data["out_time_to_solution"].values
            e_data = p_data["out_err_rho"].values

            try:
                fit_params = np.polyfit(np.log(t_data), np.log(e_data), 1)
                k_opt = -fit_params[0]
                C_opt = np.exp(fit_params[1])

                print(
                    f"p={p} | {labels[is_impl]:<8} | k = {k_opt:5.2f} (Theoretical limit approx 2.0)"
                )

                c_color = colors[is_impl]
                ax.scatter(
                    t_data,
                    e_data,
                    color=c_color,
                    marker="o" if is_impl else "s",
                    alpha=0.3,
                )

                t_smooth = np.logspace(np.log10(min(t_data)), np.log10(max(t_data)), 50)
                e_fit = C_opt * (t_smooth ** (-k_opt))
                ax.plot(
                    t_smooth,
                    e_fit,
                    color=c_color,
                    linestyle="-" if is_impl else "--",
                    linewidth=2,
                    label=rf"$p={p}$ {labels[is_impl][:4]} ($E \propto T^{{{-k_opt:.2f}}}$)",
                )

            except np.linalg.LinAlgError:
                pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Work-Precision Scaling: Error vs. Wall-Clock Time")
    ax.set_xlabel("Time to Solution (seconds)")
    ax.set_ylabel(r"Density $L^2$ Error")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "work_precision_scaling.png", dpi=300)
    plt.close(fig)


def plot_cost_scaling_law(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    if success_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {True: ("blue", "b-"), False: ("red", "r--")}
    labels = {True: "Implicit", False: "Explicit"}

    print("\n" + "=" * 60)
    print("COMPUTATIONAL COST SCALING LAW (Time ∝ p^k)")
    print("=" * 60)

    has_data = False
    for is_impl in [False, True]:
        df_scheme = success_df[success_df["in_is_impl"] == is_impl]
        if df_scheme.empty:
            continue

        opt_times = (
            df_scheme[df_scheme["p"] > 0]
            .groupby("p")["out_time_to_solution"]
            .min()
            .reset_index()
        )
        if len(opt_times) < 2:
            continue

        p_data = opt_times["p"].values
        t_data = opt_times["out_time_to_solution"].values

        try:
            popt, _ = curve_fit(power_law, p_data, t_data, p0=[1.0, 3.0])
            C_opt, k_opt = popt

            print(
                f"[{labels[is_impl]}] T_opt(p) = {C_opt:.4f} * p^{k_opt:.4f}  =>  Complexity: O(p^{k_opt:.2f})"
            )

            p_smooth = np.linspace(min(p_data), max(p_data), 100)
            t_fit = power_law(p_smooth, *popt)

            ax.plot(
                p_data,
                t_data,
                marker="o",
                color=colors[is_impl][0],
                linestyle="",
                markersize=8,
                label=f"Opt. {labels[is_impl]} Times",
            )
            ax.plot(
                p_smooth,
                t_fit,
                colors[is_impl][1],
                linewidth=2,
                label=rf"Fit {labels[is_impl]}: $T = {C_opt:.2f} p^{{{k_opt:.2f}}}$",
            )
            has_data = True

        except Exception as e:
            pass

    if has_data:
        ax.set_title("Computational Cost Scaling: Implicit vs Explicit")
        ax.set_xlabel("Polynomial Degree ($p$)")
        ax.set_ylabel("Minimum Time to Solution (s)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / "cost_scaling_law.png", dpi=300)
    plt.close(fig)


def plot_cfl_scaling_law(df_runs, out_dir):
    """
    Fits the max stable CFLs to CFL = C / (p + a)^b for both explicit and implicit solvers.
    """
    success_df = _iter_success(df_runs)
    if success_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {True: ("blue", "b--"), False: ("red", "r--")}
    labels = {True: "Implicit", False: "Explicit"}

    print("\n" + "=" * 60)
    print("STABILITY LIMIT SCALING LAW (CFL ∝ 1 / (p+a)^b)")
    print("=" * 60)

    has_data = False
    for is_impl in [False, True]:
        df_scheme = success_df[success_df["in_is_impl"] == is_impl]
        if df_scheme.empty:
            continue

        # Get max CFL per p
        max_cfls = df_scheme.groupby("p")["in_cfl"].max().reset_index()
        if len(max_cfls) < 3:
            continue

        p_data = max_cfls["p"].values
        cfl_data = max_cfls["in_cfl"].values

        try:
            # Tighter bounds: C > 0, 0.001 < a < 10.0, 0.0 < b < 5.0
            # This completely prevents the optimizer from running away to infinity
            popt, _ = curve_fit(
                cfl_scaling_model,
                p_data,
                cfl_data,
                p0=[cfl_data[0], 1.0, 2.0],  # Use the first CFL as a better guess for C
                bounds=([0.0, 0.001, 0.0], [np.inf, 10.0, 5.0]),
                maxfev=10000,
            )
            C_opt, a_opt, b_opt = popt

            # If it hits the artificial bound of 5.0, note that the model isn't a perfect physical fit
            fit_warning = (
                " (Hit upper bound, nonlinear failure dominance likely)"
                if b_opt > 4.99
                else ""
            )

            print(
                f"[{labels[is_impl]}] CFL_max(p) = {C_opt:.4f} / (p + {a_opt:.4f})^{b_opt:.4f}{fit_warning}"
            )

            p_smooth = np.linspace(min(p_data), max(p_data), 100)
            cfl_fit = cfl_scaling_model(p_smooth, *popt)

            ax.plot(
                p_data,
                cfl_data,
                marker="o",
                color=colors[is_impl][0],
                linestyle="",
                markersize=8,
                label=f"Max {labels[is_impl]} CFLs",
            )
            ax.plot(
                p_smooth,
                cfl_fit,
                colors[is_impl][1],
                linewidth=2,
                label=rf"Fit {labels[is_impl]}: $\frac{{{C_opt:.2f}}}{{(p + {a_opt:.2f})^{{{b_opt:.2f}}}}}$",
            )

            if is_impl:

                def __cfl_dg__impl__(x):
                    return 1.0 / ((x + 1) ** 1)

                ax.plot(
                    p_smooth,
                    cfl_data[0] * __cfl_dg__impl__(p_smooth) / __cfl_dg__impl__(1.0),
                    color=colors[is_impl][0],
                    linewidth=1.5,
                    label=rf"$\frac{{1}}{{ (p+1) }}$ reference",
                )
            else:

                def __cfl_dg__(x):
                    return 1.0 / ((x + 1) ** 2)

                ax.plot(
                    p_smooth,
                    cfl_data[0] * __cfl_dg__(p_smooth) / __cfl_dg__(1.0),
                    colors[is_impl][0],
                    linewidth=1.5,
                    label=rf"$\frac{{1}}{{ (p+1)^2 }}$ reference",
                )
            has_data = True

        except Exception as e:
            print(f"Skipping CFL scaling fit for {labels[is_impl]}: {e}")

    if has_data:
        ax.set_title("Empirical Stability Limit Scaling: CFL vs p")
        ax.set_xlabel("Polynomial Degree ($p$)")
        ax.set_ylabel("Maximum Stable CFL")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "cfl_scaling_law.png", dpi=300)
    plt.close(fig)


def plot_unit_iteration_cost(df_runs, out_dir):
    """
    Calculates the raw computational cost (in seconds) of a single KSP iteration
    vs a single Explicit RK step, plotting how they scale with polynomial degree p.
    """
    success_df = _iter_success(df_runs).copy()
    needed = {"out_time_to_solution", "in_dt", "in_is_impl", "p"}
    if not needed.issubset(success_df.columns):
        return

    # Attempt to find the simulation end time (fallback to 1.0 if not found)
    # This just acts as a scaling constant, so the relative trends remain perfect.
    if "out_max_time" in success_df.columns:
        t_max = success_df["out_max_time"]
    elif "in_max_t" in success_df.columns:
        t_max = success_df["in_max_t"]
    else:
        t_max = 2.0

    # Calculate total steps for all runs
    success_df["N_steps"] = t_max / success_df["in_dt"]

    c_rk_list, c_ksp_list, p_rk_list, p_ksp_list = [], [], [], []

    print("\n" + "=" * 60)
    print("UNIT COST SCALING (Time per RK step vs Time per KSP iter)")
    print("=" * 60)

    unique_ps = sorted(success_df["p"].unique())

    for p in unique_ps:
        p_data = success_df[success_df["p"] == p]

        # Explicit Runs: Unit cost is Time / N_steps
        expl_data = p_data[p_data["in_is_impl"] == False]
        if not expl_data.empty:
            avg_time_per_rk = (
                expl_data["out_time_to_solution"] / (4 * expl_data["N_steps"])
            ).mean()
            c_rk_list.append(avg_time_per_rk)
            p_rk_list.append(p)

        # Implicit Runs: Unit cost is Time / (N_steps * SNES * KSP)
        impl_data = p_data[
            (p_data["in_is_impl"] == True) & (p_data["out_stats_linear_mean"] > 0)
        ]
        if not impl_data.empty:
            total_ksp = (
                impl_data["N_steps"]
                * impl_data["out_stats_nonlinear_mean"]
                * impl_data["out_stats_linear_mean"]
            )
            avg_time_per_ksp = (impl_data["out_time_to_solution"] / total_ksp).mean()
            c_ksp_list.append(avg_time_per_ksp)
            p_ksp_list.append(p)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False

    # Fit and Plot Explicit (RK Step)
    if len(p_rk_list) >= 2:
        # Added overhead model, relative error weighting (sigma), and bounds
        popt_rk, _ = curve_fit(
            power_law_overhead,
            p_rk_list,
            c_rk_list,
            p0=[1e-4, 2.0, 1e-2],
            sigma=c_rk_list,
            absolute_sigma=False,
            bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, np.inf]),
        )
        print(
            f"[Explicit] Time/RK = {popt_rk[0]:.2e} * p^{popt_rk[1]:.2f} + {popt_rk[2]:.2e}s overhead"
        )

        p_smooth = np.linspace(min(p_rk_list), max(p_rk_list), 50)
        ax.plot(p_rk_list, c_rk_list, "ro", markersize=8, label="Raw Time / RK Step")
        ax.plot(
            p_smooth,
            power_law_overhead(p_smooth, *popt_rk),
            "r--",
            linewidth=2,
            label=rf"Explicit Fit: $O(p^{{{popt_rk[1]:.2f}}}) + {popt_rk[2]:.2e}$",
        )
        has_data = True

    # Fit and Plot Implicit (KSP Iteration)
    if len(p_ksp_list) >= 2:
        popt_ksp, _ = curve_fit(
            power_law_overhead,
            p_ksp_list,
            c_ksp_list,
            p0=[1e-4, 2.0, 1e-2],
            sigma=c_ksp_list,
            absolute_sigma=False,
            bounds=([0.0, 0.0, 0.0], [np.inf, 10.0, np.inf]),
        )
        print(
            f"[Implicit] Time/KSP = {popt_ksp[0]:.2e} * p^{popt_ksp[1]:.2f} + {popt_ksp[2]:.2e}s overhead"
        )

        p_smooth = np.linspace(min(p_ksp_list), max(p_ksp_list), 50)
        ax.plot(p_ksp_list, c_ksp_list, "bo", markersize=8, label="Raw Time / KSP Iter")
        ax.plot(
            p_smooth,
            power_law_overhead(p_smooth, *popt_ksp),
            "b-",
            linewidth=2,
            label=rf"Implicit Fit: $O(p^{{{popt_ksp[1]:.2f}}}) + {popt_ksp[2]:.2e}$",
        )
        has_data = True

    if has_data:
        ax.set_title("Implementation Efficiency: Unit Cost vs Polynomial Degree")
        ax.set_xlabel("Polynomial Degree ($p$)")
        ax.set_ylabel("Wall-Clock Time per Action (seconds)")
        ax.set_yscale("log")  # Log scale helps separate the magnitudes cleanly
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "unit_iteration_cost.png", dpi=300)
    plt.close(fig)


def plot_ksp_2d_surface(df_runs, out_dir):
    impl_df = _iter_success(df_runs)
    impl_df = impl_df[
        (impl_df["in_is_impl"] == True) & (impl_df["out_stats_linear_mean"] > 0)
    ]
    if len(impl_df) < 5:
        return

    p_data = impl_df["p"].values
    cfl_data = impl_df["in_cfl"].values
    ksp_data = impl_df["out_stats_linear_mean"].values

    try:
        popt, _ = curve_fit(
            ksp_surface_model, (p_data, cfl_data), ksp_data, p0=[1.0, 1.0, 0.5]
        )
        A_opt, delta_opt, gamma_opt = popt

        print("\n" + "=" * 60)
        print("PRECONDITIONER DEGRADATION MODEL (KSP Iterations)")
        print("=" * 60)
        print(
            f"Fitted Equation: N_KSP = {A_opt:.3f} * (p+1)^{delta_opt:.3f} * CFL^{gamma_opt:.3f}"
        )
        print("=" * 60 + "\n")

        fig, ax = plt.subplots(figsize=(9, 6))
        p_grid = np.linspace(min(p_data), max(p_data), 100)
        cfl_grid = np.logspace(np.log10(min(cfl_data)), np.log10(max(cfl_data)), 100)
        P, CFL = np.meshgrid(p_grid, cfl_grid)
        Z = ksp_surface_model((P, CFL), *popt)

        contour = ax.contourf(CFL, P, Z, levels=20, cmap="viridis", alpha=0.8)
        fig.colorbar(contour, ax=ax).set_label(
            "Fitted Mean KSP Iterations", rotation=270, labelpad=15
        )
        ax.scatter(
            cfl_data, p_data, c=ksp_data, cmap="viridis", edgecolors="white", s=60
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())

        # Include the derived equation directly in the plot title
        eq_text = rf"$N_{{KSP}} = {A_opt:.2f} \cdot (p+1)^{{{delta_opt:.2f}}} \cdot CFL^{{{gamma_opt:.2f}}}$"
        ax.set_title(f"KSP Iterations Contour Map\n{eq_text}")

        ax.set_xlabel("CFL Number (Log Scale)")
        ax.set_ylabel("Polynomial Degree ($p$)")
        plt.tight_layout()
        plt.savefig(out_dir / "ksp_2d_contour_map.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Skipping KSP surface map: {e}")


# --------------------------------------------------------------------------
# Latex Exporters
# --------------------------------------------------------------------------
def export_error_saturation_matrix(df_runs, out_dir):
    impl_df = _iter_success(df_runs)
    impl_df = impl_df[impl_df["in_is_impl"] == True]
    if impl_df.empty:
        return

    pivot = impl_df.pivot_table(index="p", columns="in_cfl", values="out_err_rho")
    latex = "\\begin{table}[hbt!]\n\\centering\n\\resizebox{\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{c | " + "c" * len(pivot.columns) + "}\n"
    cfl_headers = " & ".join([f"{cfl:.2f}" for cfl in pivot.columns])
    latex += f"$p$ $\\backslash$ CFL & {cfl_headers} \\\\\n\\hline\n"

    for p, row in pivot.iterrows():
        row_str = " & ".join([f"{val:.2e}" if pd.notna(val) else "-" for val in row])
        latex += f"{p} & {row_str} \\\\\n"

    latex += "\\end{tabular}\n}\n"
    latex += "\\caption{Error saturation matrix highlighting spatial vs temporal dominance.}\n"
    latex += "\\end{table}"
    (out_dir / "error_saturation_matrix.tex").write_text(latex)


def export_optimal_configs_latex(df_runs, out_dir):
    success_df = _iter_success(df_runs)
    if success_df.empty or "out_time_to_solution" not in success_df:
        return
    unique_ps = sorted(success_df["p"].unique())

    latex = "\\begin{table}[hbt!]\n\\centering\n"
    latex += "\\begin{tabular}{c | c | c | c | c | cc | c}\n"
    latex += "$p$ & Scheme & Opt. CFL & $\\Delta t$ & $L^2(\\rho)$ Err. & KSP & SNES & Min Time (s) \\\\\n\\hline\n"

    for p in unique_ps:
        p_data = success_df[success_df["p"] == p]
        for is_impl in [False, True]:
            data_type = p_data[p_data["in_is_impl"] == is_impl]
            if not data_type.empty:
                best = data_type.loc[data_type["out_time_to_solution"].idxmin()]
                cfl = best.get("in_cfl", float("nan"))
                dt = best.get("in_dt", float("nan"))
                err = best.get("out_err_rho", float("nan"))
                time = best.get("out_time_to_solution", float("nan"))
                ksp = best.get("out_stats_linear_mean", 0.0)
                snes = best.get("out_stats_nonlinear_mean", 0.0)

                type_str = "Implicit" if is_impl else "Explicit"
                e_str = f"{err:.2e}" if pd.notnull(err) else "-"
                k_str = f"{ksp:.1f}" if pd.notnull(ksp) and is_impl else "-"
                s_str = f"{snes:.1f}" if pd.notnull(snes) and is_impl else "-"

                latex += f"{p} & {type_str} & {cfl:.2f} & {dt:.4f} & {e_str} & {k_str} & {s_str} & \\textbf{{{time:.2f}}} \\\\\n"
        if p != unique_ps[-1]:
            latex += "\\hline\n"

    latex += "\\end{tabular}\n\\caption{Pareto optimal configurations.}\n\\end{table}"
    (out_dir / "optimal_configs.tex").write_text(latex)


def export_filter_comparison_table(df_runs, out_dir):
    """
    Generates a LaTeX table comparing hardcoded 'Filter OFF' CFL ceilings
    to the dynamic 'Filter ON' CFL ceilings extracted from the current dataset.
    """
    # Hardcoded 'Filter OFF' data as provided by you
    # Format: p -> (Explicit_OFF, Implicit_OFF)
    filter_off_data = {
        1: (1.11, 1.23),
        2: (0.05, 0.05),
        3: (0.34, 1.14),
        4: (0.24, 0.33),
        5: (0, 0),
        6: (0, 0),
    }

    # Extract dynamic 'Filter ON' ceilings from the current successful runs
    success_df = _iter_success(df_runs)

    filter_on_expl = {}
    filter_on_impl = {}

    if not success_df.empty:
        for p in success_df["p"].unique():
            df_p = success_df[success_df["p"] == p]

            expl_runs = df_p[df_p["in_is_impl"] == False]
            impl_runs = df_p[df_p["in_is_impl"] == True]

            if not expl_runs.empty:
                filter_on_expl[p] = expl_runs["in_cfl"].max()
            if not impl_runs.empty:
                filter_on_impl[p] = impl_runs["in_cfl"].max()

    # Determine the range of p (combining hardcoded keys and dynamic keys)
    all_ps = sorted(
        list(
            set(
                list(filter_off_data.keys())
                + list(filter_on_expl.keys())
                + list(filter_on_impl.keys())
            )
        )
    )

    # Helper to format numbers safely
    def fmt(val):
        if isinstance(val, (int, float)):
            if val == 0:
                return "0"
            return f"{val:.2f}"
        return str(val)

    # Build the LaTeX string
    latex = ""
    # latex += "\\begin{table}[hbt!]\n"
    latex += "    \\centering\n"
    latex += "    \\begin{tabular}{c c c c c}\n"
    latex += "        \\toprule\n"
    latex += "        \\multirow{2}{*}{$p$} & \\multicolumn{2}{c}{filter OFF} & \\multicolumn{2}{c}{filter ON} \\\\\n"
    latex += "        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    latex += "                             & Explicit & Implicit & Explicit & Implicit \\\\\n"
    latex += "        \\midrule\n"

    for p in all_ps:
        off_e, off_i = filter_off_data.get(p, ("-", "-"))
        on_e = filter_on_expl.get(p, "-")
        on_i = filter_on_impl.get(p, "-")

        latex += f"        {p} & {fmt(off_e)} & {fmt(off_i)} & {fmt(on_e)} & {fmt(on_i)} \\\\\n"

    latex += "        \\bottomrule\n"
    latex += "    \\end{tabular}\n"
    # latex += "    \\caption{Comparison of maximum stable CFL with and without the smoothing filter.}\n"
    # latex += "    \\label{tab:filter_cfl_comparison}\n"
    # latex += "\\end{table}\n"

    with open(out_dir / "filter_cfl_comparison.tex", "w") as f:
        f.write(latex)

    print(f"Saved filter_cfl_comparison.tex")


def plot_temporal_convergence(df_runs, out_dir):
    """
    Plots Error vs dt on a log-log scale for both implicit and explicit solvers
    to evaluate temporal convergence orders side-by-side.
    """
    success_df = _iter_success(df_runs)

    if (
        success_df.empty
        or "in_dt" not in success_df.columns
        or "out_err_rho" not in success_df.columns
    ):
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap("tab10")

    print("\n" + "=" * 60)
    print("TEMPORAL CONVERGENCE ORDERS")
    print("=" * 60)

    # Configuration for the two subplots
    solvers = [
        {"is_impl": True, "name": "Implicit", "ax": axes[0], "ref_order": 2.0},
        {"is_impl": False, "name": "Explicit", "ax": axes[1], "ref_order": 3.0},
    ]

    for solver in solvers:
        ax = solver["ax"]
        is_impl = solver["is_impl"]
        name = solver["name"]
        ref_order = solver["ref_order"]

        df_solver = success_df[success_df["in_is_impl"] == is_impl].copy()
        if df_solver.empty:
            ax.set_title(f"{name} Solver (No Data)")
            continue

        unique_ps = sorted(df_solver["p"].unique())
        print(f"\n--- {name} Solver ---")

        # Plot actual simulation data grouped by polynomial degree
        for i, p in enumerate(unique_ps):
            grp = df_solver[df_solver["p"] == p].sort_values("in_dt")
            if len(grp) < 2:
                continue

            dt_vals = grp["in_dt"].values
            err_vals = grp["out_err_rho"].values

            # Fit a line to the log-log data to estimate local convergence order
            slope, intercept = np.polyfit(np.log(dt_vals), np.log(err_vals), 1)
            print(f"p={p} | Estimated Temporal Order: {slope:.2f}")

            ax.loglog(
                dt_vals,
                err_vals,
                marker="o",
                color=cmap(i % 10),
                linewidth=2,
                label=f"$p={p}$ (Order $\\approx {slope:.2f}$)",
            )

        # Generate a theoretical reference line
        dt_min, dt_max = df_solver["in_dt"].min(), df_solver["in_dt"].max()
        err_min = df_solver["out_err_rho"].min()

        if pd.notnull(dt_min) and pd.notnull(err_min):
            dt_ref = np.array([dt_min, dt_max])

            # Anchor the reference line slightly below the lowest error point for clarity
            c_ref = (err_min / (dt_min**ref_order)) * 0.5
            err_ref = c_ref * (dt_ref**ref_order)

            ax.loglog(
                dt_ref,
                err_ref,
                "k--",
                linewidth=2,
                label=rf"$\mathcal{{O}}(\Delta t^{int(ref_order)})$ Reference",
            )

        # Formatting
        ax.set_title(f"{name} Solver")
        ax.set_xlabel(r"Time Step ($\Delta t$)")
        if is_impl:  # Only label the y-axis on the left plot
            ax.set_ylabel(r"Density $L_2$ Error")

        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle("Temporal Convergence Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / "temporal_convergence.png", dpi=300)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_all(json_path, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_ceilings, df_runs = load_campaigns(json_path)

    run_all = False

    if not run_all:
        print("Not all analisys are being run!!")
        plot_temporal_convergence(df_runs, out_dir)
    else:
        print("All analisys are being run!!")
        if not df_ceilings.empty:
            plot_cfl_ceiling(df_ceilings, out_dir)

        if not df_runs.empty:
            # Core plots
            plot_time_vs_cfl(df_runs, out_dir)
            plot_error_vs_cfl(df_runs, out_dir)
            plot_unit_iteration_cost(df_runs, out_dir)

            # Advanced mathematical derivations
            plot_global_pareto_fronts(df_runs, out_dir)
            plot_implicit_speedup(df_runs, out_dir)
            plot_cost_scaling_law(df_runs, out_dir)
            plot_cfl_scaling_law(df_runs, out_dir)
            plot_ksp_2d_surface(df_runs, out_dir)

            # plot_work_precision_scaling(df_runs, out_dir)

            # Latex tables
            export_filter_comparison_table(df_runs, out_dir)
            export_optimal_configs_latex(df_runs, out_dir)
            export_error_saturation_matrix(df_runs, out_dir)

        print(
            f"\nDone. Processed {len(df_runs)} runs. Outputs written to: {out_dir.resolve()}"
        )
        return df_ceilings, df_runs


if __name__ == "__main__":
    json_paths = "outputs/cfl_ceiling/runs_impl_expl_smooth.json"
    outdir = OUT_DIR
    run_all(json_paths, out_dir=outdir)
## import json
## import numpy as np
## import pandas as pd
## import matplotlib.pyplot as plt
##
## def load_ceiling_data(filepath):
##     """Parses the JSON file into two DataFrames: one for the ceilings, one for the runs."""
##     with open(filepath, "r") as f:
##         data = json.load(f)
##
##     ceiling_records = []
##     run_records = []
##
##     for p_str, p_data in data.items():
##         p_val = int(p_str)
##         max_cfl = p_data.get("max_cfl", -1.0)
##
##         # Record the ceiling for this polynomial degree
##         if max_cfl > 0:
##             ceiling_records.append({"p": p_val, "max_cfl": max_cfl})
##
##         # Record every individual run attempt
##         for run in p_data.get("runs", []):
##             inp = run.get("in", {})
##             out = run.get("out", {})
##
##             err = out.get("err", [None])
##             err_rho = err[0] if err and err[0] is not None else None
##
##             stats = out.get("stats", {})
##             ksp_mean = stats.get("linear", {}).get("mean", 0)
##             snes_mean = stats.get("nonlinear", {}).get("mean", 0)
##
##             run_records.append({
##                 "p": p_val,
##                 "is_impl" : inp.get("is_impl"),
##                 "cfl": inp.get("cfl"),
##                 "dt": inp.get("dt"),
##                 "time_to_solution": out.get("time_to_solution"),
##                 "err_rho": err_rho,
##                 "ksp_mean": ksp_mean,
##                 "snes_mean": snes_mean,
##                 "success": out.get("success", False)
##             })
##
##     df_ceilings = pd.DataFrame(ceiling_records).sort_values("p")
##     df_runs = pd.DataFrame(run_records)
##
##     return df_ceilings, df_runs
##
## def plot_cfl_ceiling(df_ceilings):
##     """Plots the maximum stable CFL vs Polynomial Degree."""
##     plt.figure(figsize=(8, 5))
##     plt.bar(df_ceilings["p"], df_ceilings["max_cfl"], color="skyblue", edgecolor="black")
##
##     plt.title("Stability Limit: Maximum CFL vs Polynomial Degree")
##     plt.xlabel("Polynomial Degree (p)")
##     plt.ylabel("Maximum Stable CFL")
##     plt.xticks(df_ceilings["p"])
##     plt.grid(axis="y", linestyle="--", alpha=0.7)
##     plt.tight_layout()
##     plt.savefig("cfl_ceiling_vs_p.png", dpi=300)
##     plt.show()
##
## def plot_time_vs_cfl(df_runs):
##     """Plots the Wall-clock Time vs CFL to find the optimal running point."""
##     success_df = df_runs[df_runs["success"] == True]
##     unique_ps = sorted(success_df["p"].unique())
##
##     plt.figure(figsize=(8, 5))
##     for p in unique_ps:
##         p_data = success_df[success_df["p"] == p].sort_values("cfl")
##         plt.plot(p_data["cfl"], p_data["time_to_solution"], marker="o", label=f"p={p}")
##
##     plt.xscale("log")
##     plt.title("Computational Efficiency: Time to Solution vs CFL")
##     plt.xlabel("CFL Number (Log Scale)")
##     plt.ylabel("Time to Solution (seconds)")
##     plt.grid(True, which="both", linestyle="--", alpha=0.5)
##     plt.legend(title="Poly Deg (p)")
##     plt.tight_layout()
##     plt.savefig("time_vs_cfl.png", dpi=300)
##     plt.show()
##
## def plot_solver_effort(df_runs):
##     """Plots the Nonlinear (SNES) iterations vs CFL."""
##     success_df = df_runs[df_runs["success"] == True]
##     unique_ps = sorted(success_df["p"].unique())
##
##     plt.figure(figsize=(8, 5))
##     for p in unique_ps:
##         p_data = success_df[success_df["p"] == p].sort_values("cfl")
##         plt.plot(p_data["cfl"], p_data["snes_mean"], marker="s", linestyle="--", label=f"p={p}")
##
##     plt.xscale("log")
##     plt.title("Implicit Solver Effort: Mean SNES Iterations vs CFL")
##     plt.xlabel("CFL Number (Log Scale)")
##     plt.ylabel("Mean Nonlinear Iterations per Step")
##     plt.grid(True, which="both", linestyle="--", alpha=0.5)
##     plt.legend(title="Poly Deg (p)")
##     plt.tight_layout()
##     plt.savefig("snes_effort_vs_cfl.png", dpi=300)
##     plt.show()
##
## if __name__ == "__main__":
##     # Adjust this path to point to your actual output file
##     json_path = "outputs/cfl_ceiling/runs.json"
##
##     df_ceilings, df_runs = load_ceiling_data(json_path)
##
##     plot_cfl_ceiling(df_ceilings)
##     plot_time_vs_cfl(df_runs)
##     plot_solver_effort(df_runs)


exit()


import matplotlib.pyplot as plt

make_cfl_shock = False
make_cfl_smooth = True
if make_cfl_shock:
    ps = [1, 2]
    cfl_25 = [1.186, 0.354]
    cfl_50 = [1.179, 0.353]
    cfl_100 = [1.178, 0.348]

    cfls = [
        {"n": 25, "values": cfl_25},
        {"n": 50, "values": cfl_50},
        {"n": 100, "values": cfl_100},
    ]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    # 3. Plot each method
    for m in cfls:
        ax.plot(
            ps,
            m["values"],
            label=f"N : str({m["n"]})",
            markersize=6,
            linewidth=1.8,
            clip_on=False,
        )

    # 4. Axes & Labels
    ax.set_title(r"Maximum Stable $\mathrm{CFL}$ Ceiling (Shock Test)", pad=10)
    ax.set_xlabel(r"Polynomial Order ($p$)")
    ax.set_ylabel(r"Maximum Stable $\mathrm{CFL}$")

    # 5. Discrete ticks & bounds
    ax.set_xticks(ps)
    ax.set_xlim(0.8, 2.2)
    ax.set_ylim(bottom=0, top=2.0)

    # 6. Grid & minimal spines
    ax.grid(True, linestyle=":", alpha=0.6, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 7. Legend
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#e0e0e0",
        framealpha=0.9,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig("imgs/cfl_ceiling_shock.png")
    plt.show()


if make_cfl_smooth:

    # 1. Structured data matching table columns across p = [1, 2, 3, 4]
    p_values = [1, 2, 3, 4]

    methods = [
        {
            "label": "Filter OFF (Explicit)",
            "values": [1.11, 0.05, 0.34, 0.24],
            "color": "#1f77b4",  # Blue
            "linestyle": "--",
            "marker": "o",
        },
        {
            "label": "Filter OFF (Implicit)",
            "values": [1.23, 0.05, 1.14, 0.33],
            "color": "#d62728",  # Red
            "linestyle": "--",
            "marker": "s",
        },
        {
            "label": "Filter ON (Explicit)",
            "values": [2.06, 0.90, 0.55, 0.39],
            "color": "#1f77b4",  # Blue (consistent with Explicit)
            "linestyle": "-",
            "marker": "^",
        },
        {
            "label": "Filter ON (Implicit)",
            "values": [8.23, 1.62, 1.23, 0.98],
            "color": "#d62728",  # Red (consistent with Implicit)
            "linestyle": "-",
            "marker": "D",
        },
    ]

    # 2. Typography & figure layout
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

    # 3. Plot each method
    for m in methods:
        ax.plot(
            p_values,
            m["values"],
            label=m["label"],
            color=m["color"],
            linestyle=m["linestyle"],
            marker=m["marker"],
            markersize=6,
            linewidth=1.8,
            clip_on=False,
        )

    # 4. Axes & Labels
    ax.set_title(r"Maximum Stable $\mathrm{CFL}$ Ceiling (Smooth Test)", pad=10)
    ax.set_xlabel(r"Polynomial Order ($p$)")
    ax.set_ylabel(r"Maximum Stable $\mathrm{CFL}$")

    # 5. Discrete ticks & bounds
    ax.set_xticks(p_values)
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(bottom=0, top=9.0)

    # 6. Grid & minimal spines
    ax.grid(True, linestyle=":", alpha=0.6, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 7. Legend
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#e0e0e0",
        framealpha=0.9,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig("imgs/cfl_ceiling.png")
    plt.show()
