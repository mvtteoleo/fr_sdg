"""
compare_runs.py

Parses a "directory tree simulation summary" table (the kind your sweep
script prints out) and produces two families of comparison plots, in the
same style as the original `shocks.pdf` script:

  1) Implicit vs Explicit, matched at the same (N, P)
     -> one PDF per (N, P) pair that has both schemes present.

  2) dt / CFL sensitivity within a single scheme at fixed (N, P)
     -> one PDF per (N, P, scheme) combo that has 2+ dt values,
        showing how refining/coarsening dt changes the solution.

Plus quality metrics (L2 error, worst-point error, oscillation excess)
computed against the 1D radial reference, both annotated on the profile
plots and summarized as clean trend plots / a CSV.

========================================================================
NUMERICAL REASONING & INTERPRETATION GUIDE
========================================================================
When analyzing the generated plots and metrics, keep these principles in mind:

1. The CFL/Stability Tradeoff:
   - Explicit schemes are strictly bounded by the CFL condition. As the
     polynomial degree (P) increases, the maximum stable CFL drops drastically.
   - Implicit schemes are mathematically unconditionally stable (within reason),
     allowing much larger time steps (CFL >> 1), but this comes at the cost of
     inverting large systems and introducing temporal numerical diffusion.

2. Numerical Diffusion vs. CFL in Implicit Schemes:
   - If you look at the Implicit CFL sensitivity plots, you will likely see
     that larger time steps (higher CFL) actually result in *lower* Total
     Variation (TV) excess. This happens because the implicit time integration
     acts as a low-pass filter, smearing the solution and damping spurious
     high-frequency oscillations.

3. The "Higher Order" Paradox (TV Excess):
   - In smooth flows, higher P dramatically reduces L2 error. However, if the
     flow contains discontinuities (shocks), higher-order polynomials without
     limiters will generate spurious wiggles (Gibbs phenomenon).
   - Watch the TV excess metric: if L2 error stagnates across P=1 to P=4
     while TV excess spikes, the simulation is accurately resolving spurious
     oscillations rather than converging to the true physical shock.
========================================================================

Usage:
    python compare_runs.py                       # uses embedded SUMMARY_TEXT
    python compare_runs.py --summary sweep.txt    # parse an external file
    python compare_runs.py --out-dir imgs/compare

Requires check_radial.get_hyperref_1d and check_iso_vortex.extract_sol,
exactly like the original script.
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from check_radial import get_hyperref_1d
from check_iso_vortex import extract_sol


@dataclass
class Run:
    n: int
    p: int
    scheme: str  # "Explicit" or "Implicit"
    dt: float
    worked: bool
    time_s: float
    directory: str

    def get_cfl(self):
        c = 3.24037034920393
        dh = 1.0 / self.n
        return self.dt * c / dh

    @property
    def solution_file(self) -> str:
        return os.path.join(self.directory, "solution.bin")


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

# Matches rows like:
# 50    | 1   | Explicit | 4.32e-05   | Yes    | 19.3755    | build/.../4.320494e-05
_ROW_RE = re.compile(
    r"^\s*(?P<n>\d+)\s*\|\s*(?P<p>\d+)\s*\|\s*(?P<scheme>\w+)\s*\|\s*"
    r"(?P<dt>[\d.eE+-]+)\s*\|\s*(?P<worked>\w+)\s*\|\s*"
    r"(?P<time>[\d.]+)\s*\|\s*(?P<dir>\S+)\s*$"
)


def parse_summary(text: str) -> list[Run]:
    """Parse the pipe-delimited directory tree summary table into Runs.
    Header / title / separator lines are ignored automatically since
    they don't match the row pattern."""
    runs = []
    for line in text.splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        runs.append(
            Run(
                n=int(m["n"]),
                p=int(m["p"]),
                scheme=m["scheme"],
                dt=float(m["dt"]),
                worked=m["worked"].strip().lower() == "yes",
                time_s=float(m["time"]),
                directory=m["dir"].strip(),
            )
        )
    return runs


def parse_summary_file(path: str) -> list[Run]:
    return parse_summary(Path(path).read_text())


# ----------------------------------------------------------------------
# Grouping helpers
# ----------------------------------------------------------------------


def group_by_np(runs: list[Run]) -> dict[tuple[int, int], list[Run]]:
    groups: dict[tuple[int, int], list[Run]] = {}
    for r in runs:
        if not r.worked:
            continue
        groups.setdefault((r.n, r.p), []).append(r)
    return groups


def pick_reference_dt(runs: list[Run]) -> Run:
    """Pick the smallest-dt run in a group as the reference (most
    resolved in time -> closest to the 'true' scheme behaviour)."""
    return min(runs, key=lambda r: r.dt)


# ----------------------------------------------------------------------
# Reference + metrics
# ----------------------------------------------------------------------


def load_reference(l_half: float, l_0: float):
    """Load the 1D radial reference once. Returns (ref_x, ref_rho, ref_tv)
    where ref_tv is the reference's own total variation restricted to
    [l_half, l_0] — the "no-oscillation" baseline every run is compared
    against."""
    ris_1d = get_hyperref_1d()
    mesh_x = ris_1d["mesh"]
    rho, _, _ = ris_1d["data"]
    ref_x = mesh_x + l_half
    ref_rho = rho[-1, :]

    order = np.argsort(ref_x)
    ref_x, ref_rho = ref_x[order], ref_rho[order]

    mask = (ref_x >= l_half) & (ref_x <= l_0)
    ref_tv = float(np.sum(np.abs(np.diff(ref_rho[mask]))))

    return ref_x, ref_rho, ref_tv


def compute_metrics(x, rho, ref_x, ref_rho, ref_tv) -> dict:
    """Quality metrics for one curve against the reference:
    - l2        : RMS error (root-mean-square of rho - rho_ref). Captures overall
                  deviation, but often fails to properly penalize localized
                  oscillations around shocks.
    - linf      : Max absolute error (worst-point deviation). Usually spikes
                  exactly at the shock interface due to smearing or overshoots.
    - tv_excess : (Total variation of rho) - (Total variation of the reference).
                  TV sums |delta rho| along the curve. A physical profile has a
                  baseline TV; any excess TV represents non-physical spurious
                  wiggling (Gibbs phenomenon). This is the best indicator of
                  numerical stability/smoothness near discontinuities.
    """
    ref_interp = np.interp(x, ref_x, ref_rho)
    err = rho - ref_interp
    l2 = float(np.sqrt(np.mean(err**2)))
    linf = float(np.max(np.abs(err)))
    tv = float(np.sum(np.abs(np.diff(rho))))
    return {"l2": l2, "linf": linf, "tv": tv, "tv_excess": tv - ref_tv}


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def _load_slice(run: Run, l_0: float, l_half: float):
    X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
        run.solution_file, run.n, run.p, l_0, False
    )
    y_half_idx = np.argmin(np.abs(Y_glob[:, 0] - l_half))
    return X_glob[0, :], rho_num[y_half_idx, :]


def plot_reference(ax, ref_x, ref_rho):
    ax.plot(ref_x, ref_rho, "-", color="k", lw=1.0, label="1D radial reference")


def plot_implicit_vs_explicit(
    groups: dict[tuple[int, int], list[Run]],
    out_dir: str,
    l_0: float = 1.0,
    l_half: float = 0.5,
):
    """For every (N, P) that has BOTH an Implicit and an Explicit run,
    plot the reference-dt run from each scheme on the same axes."""
    os.makedirs(out_dir, exist_ok=True)
    ref_x, ref_rho, ref_tv = load_reference(l_half, l_0)

    for (n, p), runs in sorted(groups.items()):
        expl = [r for r in runs if r.scheme.lower().startswith("expl")]
        impl = [r for r in runs if r.scheme.lower().startswith("impl")]
        if not expl or not impl:
            continue  # need both schemes present to compare

        expl_ref = pick_reference_dt(expl)
        impl_ref = pick_reference_dt(impl)

        fig, ax = plt.subplots(1, figsize=(10, 8))
        plot_reference(ax, ref_x, ref_rho)

        for run, style, tag in (
            (expl_ref, "-o", "Explicit"),
            (impl_ref, "--s", "Implicit"),
        ):
            x, rho = _load_slice(run, l_0, l_half)
            m = compute_metrics(x, rho, ref_x, ref_rho, ref_tv)
            label = f"{tag} CFL={run.get_cfl():.3f}"  # t={run.time_s:.1f}s | " f"L2={m['l2']:.3f} TVexc={m['tv_excess']:.2f}"
            ax.plot(x, rho, style, lw=1.0, ms=4, label=label)

        ax.set_xlim(l_half, l_0)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.legend(loc="upper left", fontsize=12)
        fig.suptitle(
            rf"$\rho$ profile" + f"\n Implicit vs Explicit (N={n}, P={p})",
            fontsize=14,
            fontweight="bold",
        )

        out_path = os.path.join(out_dir, f"impl_vs_expl_n{n}_p{p}.pdf")
        plt.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_cfl_sensitivity(
    groups: dict[tuple[int, int], list[Run]],
    out_dir: str,
    l_0: float = 1.0,
    l_half: float = 0.5,
    max_curves: int = 6,
):
    """For every (N, P, scheme) combination with 2+ dt values, overlay
    all of them to show sensitivity of the solution to dt / CFL."""
    os.makedirs(out_dir, exist_ok=True)
    ref_x, ref_rho, ref_tv = load_reference(l_half, l_0)

    by_np_scheme: dict[tuple[int, int, str], list[Run]] = {}
    for runs in groups.values():
        for r in runs:
            by_np_scheme.setdefault((r.n, r.p, r.scheme), []).append(r)

    cmap = plt.get_cmap("Set1")
    metrics_log: list[dict] = []

    for (n, p, scheme), runs in sorted(by_np_scheme.items()):
        if len(runs) < 2:
            continue  # nothing to compare against

        runs_sorted = sorted(runs, key=lambda r: r.dt)[:max_curves]

        fig, ax = plt.subplots(1, figsize=(10, 8))
        plot_reference(ax, ref_x, ref_rho)

        n_curves = len(runs_sorted)
        for i, run in enumerate(runs_sorted):
            x, rho = _load_slice(run, l_0, l_half)
            m = compute_metrics(x, rho, ref_x, ref_rho, ref_tv)
            metrics_log.append(
                {
                    "n": n,
                    "p": p,
                    "scheme": scheme,
                    "dt": run.dt,
                    "cfl": run.get_cfl(),
                    **m,
                }
            )
            color = cmap(i / max(n_curves - 1, 1))
            label = (
                f"CFL={run.get_cfl():.3f} (t={run.time_s:.1f}s) | "
                # f"L2={m['l2']:.3f} TVexc={m['tv_excess']:.2f}"
            )
            ax.plot(x, rho, "-", lw=1.2, color=color, label=label)

        ax.set_xlim(l_half, l_0)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.legend(fontsize=12)
        fig.suptitle(
            rf"$\rho$ profile" + f"\n CFL sensitivity ({scheme}, N={n}, P={p})",
            fontsize=14,
            fontweight="bold",
        )

        out_path = os.path.join(out_dir, f"dt_sweep_{scheme.lower()}_n{n}_p{p}.pdf")
        plt.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"Saved: {out_path}")

    return metrics_log


def plot_metric_trends(metrics_log: list[dict], out_dir: str):
    """Turn the noisy overlapping profiles into clean quantitative trend
    plots: L2 error and TV-excess vs CFL, one figure per (N, P), one line
    per scheme. This is the 'stop squinting at spaghetti' plot."""
    os.makedirs(out_dir, exist_ok=True)

    by_np: dict[tuple[int, int], dict[str, list[dict]]] = {}
    for row in metrics_log:
        key = (row["n"], row["p"])
        by_np.setdefault(key, {}).setdefault(row["scheme"], []).append(row)

    for (n, p), by_scheme in sorted(by_np.items()):
        fig, (ax_l2, ax_tv) = plt.subplots(1, 2, figsize=(14, 6))

        for scheme, rows in by_scheme.items():
            rows_sorted = sorted(rows, key=lambda r: r["cfl"])
            cfls = [r["cfl"] for r in rows_sorted]
            l2s = [r["l2"] for r in rows_sorted]
            tvs = [r["tv_excess"] for r in rows_sorted]
            ax_l2.plot(cfls, l2s, "-o", label=scheme)
            ax_tv.plot(cfls, tvs, "-o", label=scheme)

        for ax, ylabel, title in (
            (ax_l2, "L2 error vs reference", "Accuracy"),
            (ax_tv, "TV excess (oscillation)", "Spurious oscillation"),
        ):
            ax.set_xscale("log")
            ax.set_xlabel("CFL")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=10)
            ax.grid(True, which="both", alpha=0.3)

        fig.suptitle(
            f"Quality metrics vs CFL (N={n}, P={p})",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(out_dir, f"metrics_vs_cfl_n{n}_p{p}.pdf")
        plt.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"Saved: {out_path}")


def save_metrics_csv(metrics_log: list[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metrics.csv")
    fieldnames = ["n", "p", "scheme", "dt", "cfl", "l2", "linf", "tv", "tv_excess"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_log:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# Embedded summary (from your latest sweep) — override with --summary
# ----------------------------------------------------------------------

BRIEF_CONFRONT = """
50    | 2   | Explicit | 3.09e-04   | Yes    | 6.2544     | build/circ_expl/n_50_p_2/SSP3/3.086067e-04
50    | 2   | Explicit | 1.23e-03   | Yes    | 2.5871     | build/circ_expl/n_50_p_2/SSP3/1.234427e-03
50    | 2   | Implicit | 1.23e-03   | Yes    | 87.9511    | build/circ_expl/n_50_p_2/BFD2/1.234427e-03
50    | 2   | Implicit | 1.23e-02   | Yes    | 156.2239   | build/circ_expl/n_50_p_2/BFD2/1.234427e-02
"""

# 50    | 1   | Implicit | 1.23e-02   | Yes    | 110.0195   | build/circ_expl/n_50_p_1/BFD2/1.234427e-02
# 50    | 2   | Implicit | 1.23e-02   | Yes    | 156.2239   | build/circ_expl/n_50_p_2/BFD2/1.234427e-02
# 50    | 3   | Implicit | 1.23e-02   | Yes    | 367.1649   | build/circ_expl/n_50_p_3/BFD2/1.234427e-02
SUMMARY_TEXT = """
50    | 1   | Explicit | 6.17e-04   | Yes    | 2.5181     | build/circ_expl/n_50_p_1/SSP3/6.172134e-04
50    | 1   | Explicit | 1.23e-03   | Yes    | 1.9870     | build/circ_expl/n_50_p_1/SSP3/1.234427e-03
50    | 1   | Explicit | 1.85e-03   | Yes    | 1.6476     | build/circ_expl/n_50_p_1/SSP3/1.851640e-03
50    | 1   | Explicit | 2.47e-03   | Yes    | 1.6825     | build/circ_expl/n_50_p_1/SSP3/2.468854e-03
50    | 1   | Implicit | 1.85e-03   | Yes    | 20.8995    | build/circ_expl/n_50_p_1/BFD2/1.851640e-03
50    | 1   | Implicit | 3.70e-03   | Yes    | 25.7374    | build/circ_expl/n_50_p_1/BFD2/3.703280e-03
50    | 1   | Implicit | 6.17e-03   | Yes    | 59.9910    | build/circ_expl/n_50_p_1/BFD2/6.172134e-03
50    | 1   | Implicit | 7.41e-03   | Yes    | 32.0114    | build/circ_expl/n_50_p_1/BFD2/7.406561e-03
50    | 2   | Explicit | 3.09e-04   | Yes    | 6.2544     | build/circ_expl/n_50_p_2/SSP3/3.086067e-04
50    | 2   | Explicit | 6.17e-04   | Yes    | 3.8728     | build/circ_expl/n_50_p_2/SSP3/6.172134e-04
50    | 2   | Explicit | 1.23e-03   | Yes    | 2.5871     | build/circ_expl/n_50_p_2/SSP3/1.234427e-03
50    | 2   | Implicit | 6.17e-04   | Yes    | 81.7348    | build/circ_expl/n_50_p_2/BFD2/6.172134e-04
50    | 2   | Implicit | 1.23e-03   | Yes    | 87.9511    | build/circ_expl/n_50_p_2/BFD2/1.234427e-03
50    | 2   | Implicit | 2.47e-03   | Yes    | 120.5965   | build/circ_expl/n_50_p_2/BFD2/2.468854e-03
50    | 3   | Explicit | 3.09e-04   | Yes    | 8.9988     | build/circ_expl/n_50_p_3/SSP3/3.086067e-04
50    | 3   | Explicit | 6.17e-04   | Yes    | 5.2891     | build/circ_expl/n_50_p_3/SSP3/6.172134e-04
50    | 3   | Explicit | 1.23e-03   | Yes    | 3.4158     | build/circ_expl/n_50_p_3/SSP3/1.234427e-03
50    | 3   | Implicit | 4.63e-04   | Yes    | 240.8146   | build/circ_expl/n_50_p_3/BFD2/4.629100e-04
50    | 3   | Implicit | 9.26e-04   | Yes    | 460.7774   | build/circ_expl/n_50_p_3/BFD2/9.258201e-04
50    | 3   | Implicit | 1.85e-03   | Yes    | 430.5845   | build/circ_expl/n_50_p_3/BFD2/1.851640e-03
50    | 4   | Explicit | 1.54e-04   | Yes    | 23.4139    | build/circ_expl/n_50_p_4/SSP3/1.543033e-04
50    | 4   | Explicit | 3.09e-04   | Yes    | 12.4174    | build/circ_expl/n_50_p_4/SSP3/3.086067e-04
50    | 4   | Implicit | 4.94e-04   | Yes    | 498.8946   | build/circ_expl/n_50_p_4/BFD2/4.937707e-04
50    | 4   | Implicit | 9.88e-04   | Yes    | 666.6583   | build/circ_expl/n_50_p_4/BFD2/9.875414e-04
50    | 4   | Implicit | 1.23e-02   | Yes    | 2222.4119  | build/circ_expl/n_50_p_4/BFD2/1.234427e-02
100   | 1   | Implicit | 9.26e-04   | Yes    | 128.9758   | build/circ_expl/n_100_p_1/BFD2/9.258201e-04
150   | 1   | Implicit | 6.17e-04   | Yes    | 422.2851   | build/circ_expl/n_150_p_1/BFD2/6.172134e-04
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to a text file with the directory tree summary table. "
        "If omitted, uses the SUMMARY_TEXT embedded in this file.",
    )
    parser.add_argument("--out-dir", type=str, default="imgs/compare")
    parser.add_argument(
        "--max-curves",
        type=int,
        default=6,
        help="Max number of dt curves to overlay per CFL-sensitivity plot.",
    )
    args = parser.parse_args()

    text = Path(args.summary).read_text() if args.summary else SUMMARY_TEXT
    runs = parse_summary(text)
    if not runs:
        raise SystemExit("No runs parsed — check the summary table format.")

    groups = group_by_np(runs)

    plot_implicit_vs_explicit(groups, args.out_dir)
    metrics_log = plot_cfl_sensitivity(groups, args.out_dir, max_curves=args.max_curves)
    plot_metric_trends(metrics_log, args.out_dir)
    save_metrics_csv(metrics_log, args.out_dir)

    print("\n========================================================")
    print("ANALYSIS COMPLETE")
    print("========================================================")
    print("Check the 'imgs/compare' directory. When reviewing the plots:")
    print(" 1. TV Excess: Higher values mean more non-physical wiggling (Gibbs).")
    print(" 2. Implicit CFLs: Notice how pushing Implicit CFL higher often")
    print("    dampens TV excess due to increased numerical diffusion.")
    print(" 3. L2 Stagnation: If higher P doesn't drop L2 error, it means")
    print("    the scheme is resolving spurious oscillations around a shock.")
    print("========================================================\n")


if __name__ == "__main__":
    main()
