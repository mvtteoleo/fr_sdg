"""
Launches the shock-test runs still missing for the quantitative metrics
(stability map, TV/overshoot, grid imprinting) -- i.e. everything in
shock_test_matrix.new_runs_only(): the Group A CFL sweep and the Group C4
budget-capped imprinting run.

Reuses, unmodified, the existing helpers from check_radial.py:
    SimInputs, get_circ_expl, get_problem_data, RiemannIC, get_sol_filename

Every run is logged individually (each entry already gets a run.log and a
meta.json from run_circ_expl -- see check_radial.py), and this script
ADDITIONALLY writes one consolidated JSON, `stability_results.json`, that
records (group, scheme, p, N, CFL, dt, did_end, time_to_sol, run_dir) for
every run in the matrix -- new AND reused -- so the analysis scripts can
read a single file instead of re-globbing meta.json files scattered across
the build tree.

Usage:
    python run_test_matrix.py --dir . --list-only
    python run_test_matrix.py --dir . --group A --run
    python run_test_matrix.py --dir . --group C --run --force
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from check_radial import SimInputs, get_circ_expl, get_problem_data, get_sol_filename

from shock_test_matrix import (
    ALL_GROUPS,
    MatrixEntry,
    check_dof_budget,
    new_runs_only,
)


def build_si(
    entry: MatrixEntry, cfl: float, l_0: float, t_fin: float, c_max: float
) -> SimInputs:
    dx = l_0 / entry.n_els
    dt = cfl * dx / c_max
    return SimInputs(
        is_implicit=entry.is_implicit,
        dt=dt,
        t_final=t_fin,
        n_elems=entry.n_els,
        poly_deg=entry.poly_deg,
        l_0=l_0,
        change_var=False,
    )


def already_ran(si: SimInputs, parent_dir: Path) -> bool:
    circ_dir = Path(parent_dir, "build/circ_expl")
    run_dir = si.get_dir(circ_dir)
    meta_fl = Path(run_dir / "meta.json")
    return meta_fl.exists()


def run_matrix(
    entries, p_dir: Path, ric, l_0, t_fin, radius, do_run: bool, force: bool
):
    results = []
    c_max = ric.c_max()

    for entry in entries:
        # Group B and most of Group C are REUSED runs (new_run=False):
        # they already exist on disk from the CFL-ceiling campaign, very
        # likely under a hand-tuned dt that does NOT equal the exact
        # CFL*dx/c_max value build_si() would compute here. Launching (or
        # even just probing for) them at that exact-CFL path is wrong on
        # both counts -- it can silently create a redundant duplicate run,
        # or crash on a nonexistent meta.json. The tolerant, actual-CFL
        # lookup that handles reused runs correctly lives in
        # radial_ref_utils.find_matching_run_dir(), used by
        # analyze_oscillations.py / analyze_grid_imprinting.py -- this
        # launcher stays out of the way for anything marked new_run=False.
        if not entry.new_run:
            print(
                f"[{entry.group}] reused run (new_run=False) -- "
                f"run_test_matrix.py does not launch or probe these. Use "
                f"analyze_oscillations.py / analyze_grid_imprinting.py, "
                f"which locate reused runs via a CFL-tolerant search "
                f"instead of assuming an exact dt match. Skipping."
            )
            continue

        # Group A/C entries use extract_sol=False for the stability sweep
        # (pass/fail + timing only, no field data needed -- and a crashed
        # run may have written a partial/corrupt solution.bin that we do
        # NOT want to try to parse here). Entries that need the field
        # (Group C imprinting) set extract_sol=True in the matrix itself;
        # honour that instead of hardcoding it here.
        for cfl in entry.cfl_values:
            si = build_si(entry, cfl, l_0, t_fin, c_max)
            run_dir = si.get_dir(Path(p_dir, "build/circ_expl"))

            skip = (not force) and already_ran(si, p_dir)
            status = (
                "SKIP (already on disk)" if skip else ("RUN" if do_run else "PLAN ONLY")
            )
            print(
                f"[{entry.group}] {entry.scheme_name:8s} p={entry.poly_deg} "
                f"N={entry.n_els:4d} CFL={cfl:<6g} dt={si.dt:.3e}  -> {status}"
            )

            if skip:
                # Still record it, reading the meta.json already on disk.
                meta_fl = run_dir / "meta.json"
                with open(meta_fl) as f:
                    meta = json.load(f)
                results.append(_result_row(entry, cfl, si, meta, run_dir))
                continue

            if not do_run and not (run_dir / "meta.json").exists():
                print(
                    f"    [WARN] no meta.json at {run_dir} and --run was not "
                    f"given -- this entry has never been launched. Skipping "
                    f"(re-invoke with --run to launch it)."
                )
                continue

            opts = {
                "extract_sol": entry.extract_sol,
                "other_opt": {"hard_fail": True},
            }
            ris = get_circ_expl(
                si, ric, run=do_run, parent_dir=p_dir, radius=radius, opts=opts
            )
            meta = ris["meta"]
            results.append(_result_row(entry, cfl, si, meta, run_dir))

    return results


def _result_row(
    entry: MatrixEntry, cfl: float, si: SimInputs, meta: dict, run_dir: Path
) -> dict:
    return {
        "group": entry.group,
        "purpose": entry.purpose,
        "scheme": entry.scheme_name,
        "poly_deg": entry.poly_deg,
        "n_els": entry.n_els,
        "cfl": cfl,
        "dt": si.dt,
        "did_end": meta.get("did_end", False),
        "time_to_sol": meta.get("time_to_sol", None),
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./")
    parser.add_argument(
        "--group",
        default="A",
        choices=["A", "B", "C", "all"],
        help="Which matrix group to process. B never launches anything "
        "(new_run=False for all its entries) but is still readable here "
        "for completeness.",
    )
    parser.add_argument(
        "--run", action="store_true", help="Actually launch missing runs."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if meta.json already exists."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the plan and exit, no DOF check side effects.",
    )
    parser.add_argument("--out", default="build/circ_expl/stability_results.json")
    args = parser.parse_args()

    p_dir = Path(args.dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    sim_data = get_problem_data("Zouh")
    l_0 = sim_data["l_0"]
    t_fin = sim_data["t_fin"]
    radius = sim_data["radius"]
    ric = sim_data["ric"]

    entries = (
        ALL_GROUPS["A"] + ALL_GROUPS["B"] + ALL_GROUPS["C"]
        if args.group == "all"
        else ALL_GROUPS[args.group]
    )

    to_launch = [e for e in entries if e.new_run]
    if to_launch:
        print("DOF budget check for entries that will actually be launched:")
        check_dof_budget(to_launch)
    else:
        print(
            f"Group {args.group}: nothing new to launch (all entries are reused runs)."
        )

    if args.list_only:
        raise SystemExit(0)

    results = run_matrix(
        entries, p_dir, ric, l_0, t_fin, radius, do_run=args.run, force=args.force
    )

    out_path = Path(p_dir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with whatever is already recorded there, keyed by (group, cfl),
    # so re-running a subset of groups doesn't clobber previously-collected
    # results from other groups/sessions.
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
    key = lambda r: (r["group"], r["cfl"])
    merged = {key(r): r for r in existing}
    merged.update({key(r): r for r in results})

    with open(out_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2)

    print(f"\nWrote {len(merged)} result rows to {out_path}")
