import argparse
import json
from pathlib import Path
from time import perf_counter
import re
import subprocess

import numpy as np
from check_iso_vortex import exact_isentropic_vortex, get_sim_cmd, extract_sol
from check_radial import extract_times

# Vals from A_New_...
__c_vals__ = {
    "c_": [-2.0 / 45.0, -2.0 / 1575.0, -2.0 / 99225.0, -2.0 / 9823275.0],
    "cSD": [24.0 / 135.0, 31.0 / 1050.0, 48.0 / 496125.0, 51.0 / 5893965.0],
    "cHU": [21.0 / 15.0, 38.0 / 4725.0, 41.0 / 39690.0, 512.0 / 49116375.0],
    "cDG": [0.0, 0.0, 0.0, 0.0],
}

c_its = {"poly_deg": [2, 3, 4, 5], "c_vals": __c_vals__}

import re
import statistics


def parse_solver_stats(log_text):
    """
    Parses a log file string to extract linear and nonlinear solver iterations,
    returning the mean, min, and max for each.
    """
    linear_iters = []
    nonlinear_iters = []

    # Regular expressions to capture the integer after "iterations "
    lin_pattern = re.compile(r"Linear solve converged due to .* iterations (\d+)")
    nonlin_pattern = re.compile(r"Nonlinear solve converged due to .* iterations (\d+)")

    # Process the log line by line
    for line in log_text.splitlines():
        lin_match = lin_pattern.search(line)
        if lin_match:
            linear_iters.append(int(lin_match.group(1)))
            continue

        nonlin_match = nonlin_pattern.search(line)
        if nonlin_match:
            nonlinear_iters.append(int(nonlin_match.group(1)))

    # Helper function to compute statistics safely
    def calculate_stats(iters):
        if not iters:
            return {"mean": 0, "min": 0, "max": 0}
        return {"mean": statistics.mean(iters), "min": min(iters), "max": max(iters)}

    return {
        "linear": calculate_stats(linear_iters),
        "nonlinear": calculate_stats(nonlinear_iters),
    }


def run_sim_get_ris(inp_var: dict, inp_fix: dict):

    POLY_DEG = inp_var["p"]
    DT = inp_var["dt"]
    c_val = inp_var["c_val"]

    out = {
        "p": POLY_DEG,
        "dt": DT,
    }

    EXEC_CMD, BASE_ARGS = get_sim_cmd(
        inp_fix["test"],
        inp_fix["n_els"],
        POLY_DEG,
        DT,
        inp_fix["max_t"],
        inp_fix["filename"],
        inp_fix["box_l"],
        inp_fix["ch_var"],
        RADIUS=inp_fix["radius"],
        implicit=inp_var["is_impl"],
        MPI_PROCS=1,
        WRITE_SOL=False,
    )

    cmd = (
        EXEC_CMD
        + BASE_ARGS
        + [
            "-radau_corr",
            str(c_val),
            "" if inp_var["is_impl"] else "-fp_trap",
            "-ksp_converged_reason",
            "-snes_converged_reason",
            "-ksp_monitor",
            "-snes_monitor",
            "-ts_exact_final_time",
            "matchstep",
            "-ts_max_snes_failures",
            "1",
            "-ts_max_reject",
            "0",
            "-ts_error_if_step_fails",
            "true",
            "-ts_adapt_type",
            "none",
            "-pc_type",
            "bjacobi",
            "-sub_pc_type",
            "ilu",
            "-sub_pc_factor_levels",
            "1",
            "-snes_rtol",
            "1e-11",
            "-snes_atol",
            "1e-11",
            "-snes_stol",
            "1e-12",
            "-snes_max_it",
            "50",
            "-ksp_rtol",
            "1e-9",
            "-ksp_atol",
            "1e-12",
            "-ksp_max_it",
            "500",
        ]
    )  # , "-pc_type", "none"]
    print("=" * 60)
    print(f"LAUNCHING SOLVER VIA: {' '.join(cmd)}")
    print("=" * 60)
    t0 = perf_counter()
    out["cmd"] = "".join(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    out["time_to_solution"] = perf_counter() - t0

    if inp_fix["test"] == "vortex":
        fields = {}
        for name in ("Rho", "RhoU", "RhoV", "RhoE"):
            m = re.search(rf"{name}\s*:\s*([a-zA-Z0-9\.eE\+\-]+)", result.stdout)
            fields[name] = float(m.group(1)) if m else None
        if all(v is not None for v in fields.values()):
            out["err"] = [fields["Rho"], fields["RhoU"], fields["RhoV"], fields["RhoE"]]
        else:
            out["err"] = [None] * 4

    ksp_its = re.findall(
        r"Time\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", result.stdout
    )
    time_matches = re.findall(
        r"Time\s*:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", result.stdout
    )

    stats = parse_solver_stats(result.stdout)
    out["stats"] = stats

    if time_matches:
        times = [float(t) for t in time_matches]
        if len(times) > 1:
            times = times[:-1:]
            # Calculate dt for each step (t_current - t_previous)
            dts = [times[i] - times[i - 1] for i in range(1, len(times))]
            # Include the first step (t_0 to t_1) assuming it started at 0.0
            dts.insert(0, times[0])
            out["mean_dt"] = statistics.mean(dts)
        else:
            out["mean_dt"] = times[0]
    else:
        out["mean_dt"] = None

    out["max_time"] = max(float(t) for t in time_matches) if time_matches else None
    if (
        out["max_time"] is not None
        and np.abs(out["max_time"] - inp_fix["max_t"]) <= 1e-9
    ):
        out["success"] = True
    else:
        out["success"] = False
    return out


def get_dt(cfl, case: dict):
    if case["test"] == "vortex":
        dx = case["box_l"] / case["n_els"]
        return cfl / case["adv_v"] * dx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ceiling analysis")
    parser.add_argument(
        "--out_dir", type=str, default="./outputs", help="Base directory for output files"
    )
    parser.add_argument(
        "--cfl0", type=float, default=0.1, help="Initial CFL number to start the search"
    )

    args = parser.parse_args()

    cfl_0 = args.cfl0
    out_file = Path(args.out_dir) / "after_check.json"  # "outputs/cfl_ceiling/strict_tols.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    vtx_test = {
        "test": "vortex",
        "n_els": 30,
        "filename": Path("build/solution"),
        "max_t": 2.0,
        "box_l": 10,
        "ch_var": False,
        "radius": 0.0,
        "adv_v": 5.0,
    }
    """
    shock_test = {"test" : "shock", "n_els" : 30, "filename" :
                Path("build/solution"), "max_t" : 2.0, "box_l" : 10,
                "ch_var":False, "radius" : 0.0, "adv_v" : 5.0,
                }
    """

    case = vtx_test

    print(case)
    poly_degs = [x for x in range(1, 4)]  # c_its["poly_deg"]

    if out_file.exists():
        with open(out_file, "r") as f:
            campaign = json.load(f)
        print("Loaded existing campaign data to resume/append!")
    else:
        campaign = {}

    for is_impl in [True]:  # [False, True]:
        case["is_impl"] = is_impl
        for p in poly_degs:
            print(f"Running {p=} | is_impl={is_impl}")

            # 2. SAFELY INITIALIZE DICT ONLY IF IT DOESN'T EXIST
            if str(p) not in campaign:
                campaign[str(p)] = {"runs": []}

            camp_p = campaign[str(p)]["runs"]

            cfl = cfl_0
            safe_cfl = cfl_0
            unstable_cfl = None
            tolerance = 0.1

            while True:
                dt = get_dt(cfl, case)
                if dt > case["max_t"]:
                    break

                inps = {
                    "dt": dt,
                    "p": p,
                    "c_val": 0.0,
                    "c_type": "c_DG",
                    "is_impl": is_impl,
                    "cfl": cfl,
                }

                # Make sure your run_sim_get_ris has the NaN check we discussed!
                ris = run_sim_get_ris(inps, case)
                camp_p.append({"in": inps, "out": ris})

                # Dump safely inside the loop
                with open(out_file, "w") as f:
                    json.dump(campaign, f, indent=2)

                if ris["success"]:
                    print(f"Success at CFL={cfl}")
                    safe_cfl = cfl

                    if unstable_cfl is None:
                        cfl *= 2.0
                    else:
                        if (unstable_cfl - safe_cfl) < tolerance:
                            break
                        cfl = (safe_cfl + unstable_cfl) / 2.0
                else:
                    print(f"Failed at CFL={cfl}")
                    unstable_cfl = cfl

                    if (unstable_cfl - safe_cfl) < tolerance:
                        break
                    cfl = (safe_cfl + unstable_cfl) / 2.0

            # 3. SAVE MAX_CFL UNIQUELY FOR EXPLICIT AND IMPLICIT
            cfl_key = "max_cfl_impl" if is_impl else "max_cfl_expl"
            campaign[str(p)][cfl_key] = safe_cfl
            print(f"{cfl_key} for p={p} is {safe_cfl}")

            with open(out_file, "w") as f:
                json.dump(campaign, f, indent=2)

        """
        for c_key, c_val in c_its["c_vals"]:
            c = c_val[p-2]
            in = {"dt" : dt, "p" : p, "c_val" : c, "c_type"  : c_key}
            ris = run_sim_get_ris(case)
        """
