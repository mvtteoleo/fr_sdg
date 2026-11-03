"""
Comprehensive campaign runner + analyzer for the (p, CFL, c_VCJH) explicit
sweep (and, once wired up, the implicit/preconditioning/positive-preserving
studies discussed alongside it).

Usage
-----
Run the sweep and analyze in one go:
    python3 campaign.py run --outdir outputs --figdir figs

Only (re-)analyze an existing CSV (no simulation binary needed -- this is
the mode to use if you're handing the CSV to someone else, or replotting):
    python3 campaign.py analyze outputs/vortex.csv --figdir figs

Resume a previous run (skips any (p, cfl, c_key) combo already present in
the CSV instead of rerunning it):
    python3 campaign.py run --outdir outputs --figdir figs --resume

Design notes (see chat history for the reasoning):
  * `run` doubles the CFL per LEVEL, not per combo, and stops doubling only
    once every combo at that level has failed (`alive` set) -- this fixes
    the geometric-duplication bug in the original loop.
  * once a combo fails for the first time, its bracket is refined with a
    few bisection steps instead of accepting the raw 2x doubling gap.
  * `analyze` collapses the (p, cfl, c_key) cube into the six 1D views we
    actually care about: spectral convergence vs p, c-sensitivity, cost vs
    p, the CFL-ceiling table, CFL-ceiling vs p (with (2p+1)^-1/-2 reference
    scalings), and the error-flatness-until-failure panels.
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from check_iso_vortex import get_sim_cmd
except ImportError:
    get_sim_cmd = None  # fine for `analyze`-only use; `run` will error loudly if needed

# ---------------------------------------------------------------------------
# Fixed problem/test-matrix definitions
# ---------------------------------------------------------------------------

# Named VCJH values from Vincent, Castonguay & Jameson (2011), indexed by
# poly_deg - 2 (i.e. c_vals["cDG"][0] is p=2, c_vals["cDG"][1] is p=3, ...)
C_VALS = {
    "c_": [-2.0 / 45.0, -2.0 / 1575.0, -2.0 / 99225.0, -2.0 / 9823275.0],
    "cSD": [4.0 / 135.0, 1.0 / 1050.0, 8.0 / 496125.0, 1.0 / 5893965.0],
    "cHU": [1.0 / 15.0, 8.0 / 4725.0, 1.0 / 39690.0, 12.0 / 49116375.0],
    "cDG": [0.0, 0.0, 0.0, 0.0],
}
POLY_DEGS = [2, 3, 4, 5]
C_ORDER = ["cDG", "cSD", "cHU", "c_"]
C_LABELS = {
    "cDG": r"$c_{DG}=0$",
    "cSD": r"$c_{SD}$",
    "cHU": r"$c_{HU}$",
    "c_": r"$c_{+}$",
}
ERR_COLS = ["err_rho", "err_rhou", "err_rhov", "err_rhoe"]

CSV_HEADERS = [
    "test",
    "n_els",
    "dt",
    "p",
    "cfl",
    "c_key",
    "c_val",
    "time_to_solution",
    "max_time",
    "err_rho",
    "err_rhou",
    "err_rhov",
    "err_rhoe",
]

# ---------------------------------------------------------------------------
# Simulation execution (unchanged from the original, only bug-fixed)
# ---------------------------------------------------------------------------


def get_dt(cfl, case):
    if case["test"] == "vortex":
        dx = case["box_l"] / case["n_els"]
        return cfl / case["adv_v"] * dx
    raise NotImplementedError(f"get_dt not implemented for test={case['test']!r}")


def run_sim_get_ris(inp_var, inp_fix):
    if get_sim_cmd is None:
        raise RuntimeError(
            "check_iso_vortex.get_sim_cmd is not importable -- "
            "`run` mode needs to be executed where the solver "
            "build/module is available."
        )

    p, dt, c_val = inp_var["p"], inp_var["dt"], inp_var["c_val"]
    out = {}

    exec_cmd, base_args = get_sim_cmd(
        inp_fix["test"],
        inp_fix["n_els"],
        p,
        dt,
        inp_fix["max_t"],
        str(inp_fix["filename"]),
        inp_fix["box_l"],
        inp_fix["ch_var"],
        RADIUS=inp_fix["radius"],
        implicit=inp_fix["is_impl"],
        MPI_PROCS=inp_fix["mpi_proc"],
    )
    cmd = exec_cmd + base_args + ["-radau_corr", str(c_val), "-fp_trap"]
    out["cmd"] = " ".join(cmd)

    print("=" * 60)
    print(f"Running {inp_var}")
    print("=" * 60)
    t0 = perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    out["time_to_solution"] = perf_counter() - t0

    if inp_fix["test"] == "vortex":
        time_matches = re.findall(
            r"Time\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", result.stdout
        )
        out["max_time"] = max(float(t) for t in time_matches) if time_matches else None
        if (
            out["max_time"] is not None
            and np.abs(out["max_time"] - inp_fix["max_t"]) <= 1e-9
        ):
            out["success"] = True
        else:
            out["success"] = False

        fields = {}
        for name in ("Rho", "RhoU", "RhoV", "RhoE"):
            m = re.search(rf"{name}\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
            fields[name] = float(m.group(1)) if m else None
        if out["success"] is True and all(v is not None for v in fields.values()):
            out["err"] = [fields["Rho"], fields["RhoU"], fields["RhoV"], fields["RhoE"]]
        else:
            out["err"] = [None] * 4
            out["success"] = False
        if out["success"]:
            print("Run was successful")
        else:
            print("Run failed")
    else:
        raise NotImplementedError(
            f"output parsing not implemented for test={inp_fix['test']!r}"
        )

    return out


# ---------------------------------------------------------------------------
# Campaign driver: per-level doubling with an alive-set, then bisection
# refinement of each combo's failure bracket
# ---------------------------------------------------------------------------

BISECT_ITERS = 4
BISECT_TOL_REL = 0.02


def resume_lookup(csv_path, test_name):
    """Map every (p, cfl, c_key) point already in the CSV to its actual
    success/failure outcome (using the same criterion as ceiling_table),
    so resuming can reuse known results to drive the alive/bounds state
    machine instead of just silently skipping them."""
    if not Path(csv_path).exists():
        return {}
    df = load(csv_path)
    tgt = target_time(df, test_name)
    if tgt is None:
        return {}
    sub = df[df.test == test_name]
    ok = np.isclose(sub.max_time, tgt) & sub.err_total.notna()
    return {
        (p, cfl, key): bool(o) for p, cfl, key, o in zip(sub.p, sub.cfl, sub.c_key, ok)
    }


def run_campaign(case, cfl0, outcomes, run=True):
    """Coarse per-level doubling sweep with an alive-set (fixes the
    duplicate-append bug), followed by bisection refinement of each combo's
    failure bracket. `outcomes` is a {(p, cfl, c_key): bool} lookup of
    already-known results (from resume_lookup) -- known points are reused
    instead of rerun, but still properly drive the alive/bounds state
    machine. Returns the flat list of {"inp":..., "ris":...} records for
    newly-run combos only."""
    records = []
    alive = {(p, key) for p in POLY_DEGS for key in C_VALS}
    bounds = {combo: {"last_ok": None, "first_fail": None} for combo in alive}

    def do_one(cfl, p, key):
        known = outcomes.get((p, cfl, key))
        if known is not None:
            return known  # reuse a previously-run point, no rerun needed
        c_val = C_VALS[key][p - 2]
        dt = get_dt(cfl, case)
        if dt <= case["max_t"]:
            inp = {"dt": dt, "p": p, "cfl": cfl, "c_val": c_val, "c_key": key}
            ris = run_sim_get_ris(inp, case) if run else {}
            records.append({"inp": inp, "ris": ris})
            return ris.get("max_time") is not None
        else:
            return None

    # --- Phase 1: coarse doubling, per LEVEL not per combo ---
    cfl = cfl0
    while alive:
        any_ok_this_level = False
        for p in POLY_DEGS:
            for key in list(C_VALS):
                combo = (p, key)
                if combo not in alive:
                    continue
                ok = do_one(cfl, p, key)
                if ok:
                    bounds[combo]["last_ok"] = cfl
                    any_ok_this_level = True
                else:
                    bounds[combo]["first_fail"] = cfl
                    alive.discard(combo)
        if not any_ok_this_level:
            break
        cfl *= 2

    # --- Phase 2: bisection refinement, only for combos that actually failed ---
    for (p, key), b in bounds.items():
        if b["first_fail"] is None or b["last_ok"] is None:
            continue
        lo, hi = b["last_ok"], b["first_fail"]
        for _ in range(BISECT_ITERS):
            if (hi - lo) / hi < BISECT_TOL_REL:
                break
            mid = 0.5 * (lo + hi)
            ok = do_one(mid, p, key)
            if ok:
                lo = mid
            else:
                hi = mid
        b["ceiling_estimate"] = lo

    return records, bounds


def write_csv(records, case, csv_path, append):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (append and csv_path.exists()) else "w"
    write_header = not (mode == "a")

    with open(csv_path, mode=mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        for entry in records:
            i, r = entry["inp"], entry["ris"]
            err = r.get("err", [None] * 4)
            writer.writerow(
                {
                    "test": case["test"],
                    "n_els": case["n_els"],
                    "dt": i.get("dt"),
                    "p": i.get("p"),
                    "cfl": i.get("cfl"),
                    "c_key": i.get("c_key"),
                    "c_val": i.get("c_val"),
                    "time_to_solution": r.get("time_to_solution"),
                    "max_time": r.get("max_time"),
                    "err_rho": err[0],
                    "err_rhou": err[1],
                    "err_rhov": err[2],
                    "err_rhoe": err[3],
                }
            )


# ---------------------------------------------------------------------------
# Analysis: collapse the (p, cfl, c_key) cube into 1D summaries + plots
# ---------------------------------------------------------------------------


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["c_key"] = pd.Categorical(df["c_key"], categories=C_ORDER, ordered=True)
    df["err_total"] = np.sqrt((df[ERR_COLS].astype(float) ** 2).sum(axis=1))
    return df


def target_time(df, test_name):
    sub = df[df.test == test_name]
    modes = sub.max_time.mode()
    if modes.empty:
        return None
    return modes.iloc[0]


def success_mask(df, test_name):
    tgt = target_time(df, test_name)
    if tgt is None:
        return pd.Series(False, index=df.index)
    return (df.test == test_name) & np.isclose(df.max_time, tgt) & df.err_total.notna()


def lowest_cfl_slice(sub):
    """Restrict to the lowest CFL present, for plots that assume one CFL
    per (p, c_key) -- e.g. the plain convergence-vs-p view."""
    if sub.cfl.nunique() > 1:
        return sub[sub.cfl == sub.cfl.min()]
    return sub


def plot_convergence_vs_p(df, test_name, outdir):
    sub = lowest_cfl_slice(df[df.test == test_name])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in C_ORDER:
        s = sub[sub.c_key == key].sort_values("p")
        if not s.empty:
            ax.semilogy(s.p, s.err_total, "o-", label=C_LABELS[key])
    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel(r"$\|e\|$ (combined conservative-variable norm)")
    ax.set_title(f"{test_name}: error vs $p$ at CFL={sub.cfl.iloc[0]:.4g}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{test_name}_convergence_vs_p.png", dpi=150)
    plt.close(fig)


def plot_c_sensitivity(df, test_name, outdir):
    sub = lowest_cfl_slice(df[df.test == test_name])
    rows = []
    for p, g in sub.groupby("p"):
        base = g.loc[g.c_key == "cDG", "err_total"]
        if base.empty:
            continue
        base = base.iloc[0]
        for _, r in g.iterrows():
            rows.append((p, r.c_key, (r.err_total - base) / base * 100))
    spread = pd.DataFrame(rows, columns=["p", "c_key", "pct_diff_vs_cDG"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in C_ORDER:
        s = spread[spread.c_key == key].sort_values("p")
        if not s.empty:
            ax.plot(s.p, s.pct_diff_vs_cDG, "o-", label=C_LABELS[key])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel(r"error relative to $c_{DG}$ [%]")
    ax.set_title(f"{test_name}: does $c_{{VCJH}}$ matter at CFL={sub.cfl.iloc[0]:.4g}?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{test_name}_c_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_time_to_solution(df, test_name, outdir):
    sub = lowest_cfl_slice(df[df.test == test_name])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in C_ORDER:
        s = sub[sub.c_key == key].sort_values("p")
        if not s.empty:
            ax.plot(s.p, s.time_to_solution, "o-", label=C_LABELS[key])
    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel("time to solution [s]")
    ax.set_title(f"{test_name}: cost vs $p$ at CFL={sub.cfl.iloc[0]:.4g}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{test_name}_time_vs_p.png", dpi=150)
    plt.close(fig)


def ceiling_table(df, test_name):
    sub = df[df.test == test_name].copy()
    sub["ok"] = success_mask(df, test_name)[sub.index]
    rows = []
    for (p, key), g in sub.groupby(["p", "c_key"], observed=True):
        g = g.sort_values("cfl")
        ok, bad = g[g.ok], g[~g.ok]
        last_ok = ok.cfl.max() if not ok.empty else np.nan
        first_bad = (
            bad[bad.cfl > (last_ok if not np.isnan(last_ok) else 0)].cfl.min()
            if not bad.empty
            else np.nan
        )
        rows.append(
            {"p": p, "c_key": key, "cfl_last_ok": last_ok, "cfl_first_fail": first_bad}
        )
    return pd.DataFrame(rows).sort_values(["c_key", "p"])


def plot_ceiling_vs_p(tbl, test_name, outdir):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in C_ORDER:
        s = tbl[tbl.c_key == key].dropna(subset=["cfl_last_ok"]).sort_values("p")
        if not s.empty:
            ax.semilogy(s.p, s.cfl_last_ok, "o-", label=C_LABELS[key])

    base = tbl[tbl.c_key == "cDG"].dropna(subset=["cfl_last_ok"]).sort_values("p")
    if not base.empty:
        p0, cfl0 = base.p.iloc[0], base.cfl_last_ok.iloc[0]
        ps = np.array(sorted(tbl.p.unique()))
        ax.semilogy(
            ps,
            cfl0 * (2 * p0 + 1) / (2 * ps + 1),
            "k--",
            alpha=0.6,
            label=r"$\propto (2p+1)^{-1}$",
        )
        ax.semilogy(
            ps,
            cfl0 * ((2 * p0 + 1) / (2 * ps + 1)) ** 2,
            "k:",
            alpha=0.6,
            label=r"$\propto (2p+1)^{-2}$",
        )

    ax.set_xlabel("polynomial degree $p$")
    ax.set_ylabel("last stable CFL")
    ax.set_title(f"{test_name}: explicit CFL ceiling vs $p$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{test_name}_cfl_ceiling_vs_p.png", dpi=150)
    plt.close(fig)


def plot_error_flatness(df, test_name, outdir):
    sub = df[df.test == test_name]
    ps = sorted(sub.p.unique())
    fig, axes = plt.subplots(1, len(ps), figsize=(4 * len(ps), 4), sharey=True)
    if len(ps) == 1:
        axes = [axes]
    for ax, p in zip(axes, ps):
        s = sub[sub.p == p]
        for key in C_ORDER:
            ss = s[s.c_key == key].sort_values("cfl")
            ok = ss[ss.err_total.notna()]
            ax.loglog(ok.cfl, ok.err_total, "o-", label=C_LABELS[key])
        ax.set_xlabel("CFL")
        ax.set_title(f"$p={p}$")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"$\|e\|$")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"{test_name}: error stays flat until the stability wall")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{test_name}_error_flatness.png", dpi=150)
    plt.close(fig)


def plot_cfl_ceiling_error_vs_cfl(df, test_name, outdir):
    """Once several CFL values exist per (p, c_key): error & general shape
    vs CFL -- kept for completeness/back-compat with the two-CFL-value case."""
    sub = df[df.test == test_name]
    if sub.cfl.nunique() < 2:
        return
    plot_error_flatness(df, test_name, outdir)


def analyze(csv_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(csv_path)

    for test_name in df.test.unique():
        if target_time(df, test_name) is None:
            print(
                f"\n=== {test_name}: no successful runs found -- skipping analysis ==="
            )
            continue

        print(f"\n=== {test_name}: CFL ceiling table ===")
        tbl = ceiling_table(df, test_name)
        print(tbl.to_string(index=False))

        plot_convergence_vs_p(df, test_name, outdir)
        plot_c_sensitivity(df, test_name, outdir)
        plot_time_to_solution(df, test_name, outdir)
        plot_ceiling_vs_p(tbl, test_name, outdir)
        plot_cfl_ceiling_error_vs_cfl(df, test_name, outdir)

    print(f"\nFigures written to {outdir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    p_run = sub.add_parser("run", help="run the simulation sweep, then analyze")
    p_run.add_argument("--outdir", default="outputs")
    p_run.add_argument("--figdir", default="figs")
    p_run.add_argument("--cfl0", type=float, default=0.005)
    p_run.add_argument(
        "--dry",
        action="store_true",
        help="build the campaign structure without invoking the solver",
    )
    p_run.add_argument(
        "--resume",
        action="store_true",
        help="skip (p, cfl, c_key) combos already present in the output CSV",
    )

    p_ana = sub.add_parser(
        "analyze", help="analyze an existing CSV only, no solver needed"
    )
    p_ana.add_argument("csv")
    p_ana.add_argument("--figdir", default="figs")

    args = ap.parse_args()

    if args.mode == "analyze":
        analyze(args.csv, args.figdir)
        return

    # mode == "run"
    case = {
        "test": "vortex",
        "n_els": 10,
        "filename": Path("build/solution"),
        "max_t": 2.0,
        "box_l": 10,
        "ch_var": False,
        "radius": 0.0,
        "adv_v": 5.0,
        "mpi_proc": 4,
        "is_impl": True,
    }
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{case['test']}.csv"

    outcomes = resume_lookup(csv_path, case["test"]) if args.resume else {}
    records, bounds = run_campaign(case, args.cfl0, outcomes, run=not args.dry)

    write_csv(records, case, csv_path, append=args.resume)

    bounds_serializable = {f"{p}|{key}": v for (p, key), v in bounds.items()}
    with open(out_dir / "ceiling_bounds.json", "w") as f:
        json.dump(bounds_serializable, f, indent=2, default=str)

    print(f"\nWrote {len(records)} new rows to {csv_path}")
    analyze(csv_path, args.figdir)


if __name__ == "__main__":
    main()
