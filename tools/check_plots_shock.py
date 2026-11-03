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

Usage:
    python compare_runs.py                       # uses embedded SUMMARY_TEXT
    python compare_runs.py --summary sweep.txt    # parse an external file
    python compare_runs.py --out-dir imgs/compare

Requires check_radial.get_hyperref_1d and check_iso_vortex.extract_sol,
exactly like the original script.
"""

import argparse
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
# Plotting
# ----------------------------------------------------------------------


def _load_slice(run: Run, l_0: float, l_half: float):
    X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
        run.solution_file, run.n, run.p, l_0, False
    )
    y_half_idx = np.argmin(np.abs(Y_glob[:, 0] - l_half))
    return X_glob[0, :], rho_num[y_half_idx, :]


def plot_reference(ax, l_half: float):
    ris_1d = get_hyperref_1d()
    mesh_x = ris_1d["mesh"]
    rho, _, _ = ris_1d["data"]
    ax.plot(
        mesh_x + l_half,
        rho[-1, :],
        "-",
        color="k",
        lw=1.0,
        label="1D radial reference",
    )


def plot_implicit_vs_explicit(
    groups: dict[tuple[int, int], list[Run]],
    out_dir: str,
    l_0: float = 1.0,
    l_half: float = 0.5,
):
    """For every (N, P) that has BOTH an Implicit and an Explicit run,
    plot the reference-dt run from each scheme on the same axes."""
    os.makedirs(out_dir, exist_ok=True)

    for (n, p), runs in sorted(groups.items()):
        expl = [r for r in runs if r.scheme.lower().startswith("expl")]
        impl = [r for r in runs if r.scheme.lower().startswith("impl")]
        if not expl or not impl:
            continue  # need both schemes present to compare

        expl_ref = pick_reference_dt(expl)
        impl_ref = pick_reference_dt(impl)

        fig, ax = plt.subplots(1, figsize=(10, 8))
        plot_reference(ax, l_half)

        for run, style, tag in (
            (expl_ref, "-o", "Explicit"),
            (impl_ref, "--s", "Implicit"),
        ):
            x, rho = _load_slice(run, l_0, l_half)
            label = f"N={n} P={p} {tag} CFL={run.get_cfl():.3f} t={run.time_s:.1f}s"
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

    by_np_scheme: dict[tuple[int, int, str], list[Run]] = {}
    for runs in groups.values():
        for r in runs:
            by_np_scheme.setdefault((r.n, r.p, r.scheme), []).append(r)

    cmap = plt.get_cmap("Set1")

    for (n, p, scheme), runs in sorted(by_np_scheme.items()):
        if len(runs) < 2:
            continue  # nothing to compare against

        runs_sorted = sorted(runs, key=lambda r: r.dt)[:max_curves]

        fig, ax = plt.subplots(1, figsize=(10, 8))
        plot_reference(ax, l_half)

        n_curves = len(runs_sorted)
        for i, run in enumerate(runs_sorted):
            x, rho = _load_slice(run, l_0, l_half)
            color = cmap(i / max(n_curves - 1, 1))
            label = f"CFL={run.get_cfl():.3f} (t={run.time_s:.1f}s)"
            ax.plot(x, rho, "-", lw=1.2, color=color, label=label)

        ax.set_xlim(l_half, l_0)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\rho$")
        ax.legend(fontsize=12)  # loc="best",
        fig.suptitle(
            rf"$\rho$ profile" + f"\n CFL sensitivity ({scheme}, N={n}, P={p})",
            fontsize=14,
            fontweight="bold",
        )

        out_path = os.path.join(out_dir, f"dt_sweep_{scheme.lower()}_n{n}_p{p}.pdf")
        plt.savefig(out_path, format="pdf")
        plt.close(fig)
        print(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# Embedded summary (from your latest sweep) — override with --summary
# ----------------------------------------------------------------------

SUMMARY_TEXT = """
N     | P   | Scheme   | dt         | Worked | Time (s)   | Directory
50    | 1   | Explicit | 4.32e-05   | Yes    | 19.3755    | build/circ_expl/n_50_p_1/SSP3/4.320494e-05
50    | 1   | Explicit | 4.32e-04   | Yes    | 3.6893     | build/circ_expl/n_50_p_1/SSP3/4.320494e-04
50    | 1   | Explicit | 1.23e-03   | Yes    | 1.7268     | build/circ_expl/n_50_p_1/SSP3/1.234427e-03
50    | 1   | Explicit | 1.85e-03   | Yes    | 1.6007     | build/circ_expl/n_50_p_1/SSP3/1.851640e-03
50    | 1   | Explicit | 2.47e-03   | Yes    | 1.5867     | build/circ_expl/n_50_p_1/SSP3/2.468854e-03
50    | 1   | Implicit | 6.17e-03   | Yes    | 18.1247    | build/circ_expl/n_50_p_1/BFD2/6.172134e-03
50    | 1   | Implicit | 6.79e-03   | Yes    | 69.6642    | build/circ_expl/n_50_p_1/BFD2/6.789347e-03
50    | 1   | Implicit | 7.41e-03   | Yes    | 25.3151    | build/circ_expl/n_50_p_1/BFD2/7.406561e-03
50    | 2   | Explicit | 4.32e-05   | Yes    | 37.0079    | build/circ_expl/n_50_p_2/SSP3/4.320494e-05
50    | 2   | Explicit | 1.23e-03   | Yes    | 2.0069     | build/circ_expl/n_50_p_2/SSP3/1.234427e-03
50    | 2   | Implicit | 2.47e-03   | Yes    | 138.8976   | build/circ_expl/n_50_p_2/BFD2/2.468854e-03
50    | 3   | Explicit | 5.99e-06   | Yes    | 440.8397   | build/circ_expl/n_50_p_3/SSP3/5.986970e-06
50    | 3   | Explicit | 1.23e-03   | Yes    | 2.4357     | build/circ_expl/n_50_p_3/SSP3/1.234427e-03
50    | 3   | Implicit | 6.17e-04   | Yes    | 263.8622   | build/circ_expl/n_50_p_3/BFD2/6.172134e-04
50    | 3   | Implicit | 1.23e-03   | Yes    | 258.6575   | build/circ_expl/n_50_p_3/BFD2/1.234427e-03
50    | 3   | Implicit | 1.85e-03   | Yes    | 328.4476   | build/circ_expl/n_50_p_3/BFD2/1.851640e-03
50    | 4   | Explicit | 3.09e-04   | Yes    | 7.7165     | build/circ_expl/n_50_p_4/SSP3/3.086067e-04
50    | 4   | Implicit | 9.26e-04   | Yes    | 649.7112   | build/circ_expl/n_50_p_4/BFD2/9.258201e-04
50    | 4   | Implicit | 9.88e-04   | Yes    | 505.0836   | build/circ_expl/n_50_p_4/BFD2/9.875414e-04
100   | 1   | Explicit | 2.16e-05   | Yes    | 140.9595   | build/circ_expl/n_100_p_1/SSP3/2.160247e-05
100   | 1   | Explicit | 2.16e-04   | Yes    | 8.4619     | build/circ_expl/n_100_p_1/SSP3/2.160247e-04
100   | 1   | Explicit | 3.09e-04   | Yes    | 6.3131     | build/circ_expl/n_100_p_1/SSP3/3.086067e-04
100   | 1   | Implicit | 3.09e-04   | Yes    | 263.6350   | build/circ_expl/n_100_p_1/BFD2/3.086067e-04
100   | 1   | Implicit | 5.55e-03   | Yes    | 278.3397   | build/circ_expl/n_100_p_1/BFD2/5.554921e-03
100   | 2   | Explicit | 3.09e-04   | Yes    | 14.4641    | build/circ_expl/n_100_p_2/SSP3/3.086067e-04
100   | 2   | Implicit | 3.09e-04   | Yes    | 756.9128   | build/circ_expl/n_100_p_2/BFD2/3.086067e-04
100   | 2   | Implicit | 5.55e-03   | Yes    | 818.4639   | build/circ_expl/n_100_p_2/BFD2/5.554921e-03
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
    plot_cfl_sensitivity(groups, args.out_dir, max_curves=args.max_curves)


if __name__ == "__main__":
    main()
