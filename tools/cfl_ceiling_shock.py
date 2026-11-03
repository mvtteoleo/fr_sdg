"""
find_cfl_ceiling.py

Standalone ceiling-finder: for a fixed N (default 100), sweeps P and
scheme (implicit/explicit) and does a top-down geometric search for the
highest CFL that still completes a run. Meant to run unattended overnight
and log results, so the missing N=100 P=3/P=4 cells can get filled in
with a *measured* ceiling instead of assuming the N=50 table holds.

Reuses the real simulation plumbing (get_problem_data, SimInputs,
get_circ_expl) imported from check_radial.py, rather than reimplementing
run logic or importing the module's `run_case` — that function reads
`run_2d` / `p_dir` from module-level globals that are only set inside
check_radial.py's own `if __name__ == "__main__":` block, so calling it
after a plain `import` raises NameError. Everything here is self-contained.

Usage:
    python find_cfl_ceiling.py --n_els 100
    python find_cfl_ceiling.py --n_els 100 --p_list 3 4   # just the missing cells
"""

import argparse
import json
from pathlib import Path

from check_radial import get_problem_data, SimInputs, get_circ_expl


# Measured at N=50 — used only as the STARTING GUESS for the search
# (with a buffer), never assumed to hold at N=100. That assumption is
# exactly what this script exists to stop trusting blindly.
CFL_CEILING_N50 = {
    False: {1: 0.40, 2: 0.20, 3: 0.20, 4: 0.05},  # Explicit
    True: {1: 1.20, 2: 0.40, 3: 0.30, 4: 0.16},  # Implicit
}
GUESS_BUFFER = 1.3  # start 30% above the N=50 value, search down from there


def run_one(p_deg, n_els, cfl, is_impl, p_dir, radius, l_0, t_fin, ric):
    """Launch one sim via get_circ_expl and report whether it completed."""
    dx = l_0 / n_els
    dt = cfl * dx / ric.c_max()

    si = SimInputs(
        is_implicit=is_impl,
        dt=dt,
        t_final=t_fin,
        n_elems=n_els,
        poly_deg=p_deg,
        l_0=l_0,
        change_var=False,
    )

    opts = {
        "extract_sol": False,
        "other_opt": {"hard_fail": True},  # crash fast on step failure
    }

    print(
        f"  Trying N={n_els} P={p_deg} {'Impl' if is_impl else 'Expl'} "
        f"CFL={cfl:.4f} (dt={dt:.3e}) ..."
    )

    ris = get_circ_expl(si, ric, run=True, parent_dir=p_dir, radius=radius, opts=opts)

    worked = bool(ris["meta"]["did_end"])
    t_sol = ris["meta"].get("time_to_sol")
    status = "OK" if worked else "FAILED"
    extra = f" ({t_sol:.1f}s)" if worked and t_sol is not None else ""
    print(f"    -> {status}{extra}")
    return worked, t_sol


def find_ceiling(
    p_deg,
    n_els,
    is_impl,
    guess,
    p_dir,
    radius,
    l_0,
    t_fin,
    ric,
    shrink=0.75,
    min_cfl=1e-4,
    max_tries=25,
):
    """Top-down geometric search: start at `guess`, shrink by `shrink`
    after each failure, stop at the first success — that CFL is the
    ceiling. "Top-down" means every failure narrows the search from
    above, so we never waste time creeping up from a too-low guess."""
    cfl = guess
    attempts = []

    for _ in range(max_tries):
        if cfl < min_cfl:
            print(
                f"  !! bottomed out below {min_cfl} for N={n_els} P={p_deg} impl={is_impl}"
            )
            return None, attempts

        worked, t_sol = run_one(
            p_deg, n_els, cfl, is_impl, p_dir, radius, l_0, t_fin, ric
        )
        attempts.append({"cfl": cfl, "worked": worked, "time_to_sol": t_sol})

        if worked:
            return cfl, attempts
        cfl *= shrink

    print(f"  !! exhausted {max_tries} tries for N={n_els} P={p_deg} impl={is_impl}")
    return None, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", default="./", help="Parent dir passed through to get_circ_expl."
    )
    parser.add_argument("--n_els", type=int, default=100)
    parser.add_argument("--p_list", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--shrink", type=float, default=0.75, help="CFL multiplier after each failure."
    )
    parser.add_argument("--out", default="cfl_ceiling_log.json")
    args = parser.parse_args()

    p_dir = Path(args.dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    sim_data = get_problem_data("Zouh")
    l_0 = sim_data["l_0"]
    t_fin = sim_data["t_fin"]
    radius = sim_data["radius"]
    ric = sim_data["ric"]

    ceiling_log = []

    for p_deg in args.p_list:
        for is_impl in [False, True]:
            guess = CFL_CEILING_N50[is_impl][p_deg] * GUESS_BUFFER
            print(
                f"\n=== N={args.n_els} P={p_deg} "
                f"{'Implicit' if is_impl else 'Explicit'} — starting from CFL={guess:.4f} ==="
            )
            ceiling, attempts = find_ceiling(
                p_deg,
                args.n_els,
                is_impl,
                guess,
                p_dir,
                radius,
                l_0,
                t_fin,
                ric,
                shrink=args.shrink,
            )
            ceiling_log.append(
                {
                    "n": args.n_els,
                    "p": p_deg,
                    "impl": is_impl,
                    "ceiling": ceiling,
                    "attempts": attempts,
                }
            )

            # Save after every cell, not just at the end — if this gets
            # killed partway through an unattended run, everything found
            # so far is still on disk.
            with open(args.out, "w") as f:
                json.dump(ceiling_log, f, indent=2)

    print(f"\nDone. Ceilings saved to {args.out}")
    for row in ceiling_log:
        tag = f"N={row['n']} P={row['p']} {'Impl' if row['impl'] else 'Expl'}"
        if row["ceiling"] is not None:
            print(f"  {tag}: ceiling = {row['ceiling']:.4f}")
        else:
            print(f"  {tag}: NOT FOUND")


if __name__ == "__main__":
    main()
