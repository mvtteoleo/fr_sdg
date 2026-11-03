import os
from os.path import exists
import argparse
import re
import json
import subprocess
from time import perf_counter
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Assuming these exist in your directory based on your original script
from post_proc import run_sim_get_hyst, RiemannIC, _make_run_dir
from check_iso_vortex import get_sim_cmd, extract_sol

a_vdw = 0.0
b_vdw = 0.001265


@dataclass
class SimInputs:
    dt: float
    n_elems: int
    poly_deg: int
    t_final: float = 0.13
    l_0: float = 1.0
    is_implicit: bool = False
    change_var: bool = False

    def get_dir(self, parent_dir: Path) -> Path:
        n_p_dir = Path(f"n_{self.n_elems}_p_{self.poly_deg}")
        rk_bdf_dir = Path("BDF2") if self.is_implicit else Path("SSP3")
        dt_dir = Path(f"{self.dt:3e}")
        return Path(parent_dir, n_p_dir, rk_bdf_dir, dt_dir)

    def get_cfl_s(self, c_max: float):
        dx = self.l_0 / self.n_elems
        min_dt = dx / (2 * self.poly_deg + 1) / c_max
        CFL = self.dt * c_max / dx
        CFL_DG = CFL * (2 * self.poly_deg + 1)
        return CFL, CFL_DG, min_dt


import math


def get_vdw_c_max(ric, a, b):
    """
    Calculates the maximum initial wave speed for a Van der Waals gas.
    ric: The RiemannIC object containing left and right states.
    a: VdW attraction parameter.
    b: VdW co-volume parameter.
    """
    gamma = ric.gamma

    def calc_c_vdw(rho, p):
        # The analytical Van der Waals speed of sound
        term1 = (gamma * (p + a * rho**2)) / (rho * (1.0 - b * rho))
        term2 = 2.0 * a * rho

        # Ensure we don't take the square root of a negative number
        # if unphysical states are passed
        c_squared = term1 - term2
        if c_squared <= 0:
            raise ValueError(f"Unphysical state! c^2 = {c_squared}")

        return math.sqrt(c_squared)

    # Calculate sound speed for Left and Right states
    c_L = calc_c_vdw(ric.rho_L, ric.pres_L)
    c_R = calc_c_vdw(ric.rho_R, ric.pres_R)

    # Maximum characteristic wave speed S_max = |u| + c
    s_L = abs(ric.vel_L) + c_L
    s_R = abs(ric.vel_R) + c_R

    return max(s_L, s_R)


def extract_times(log_f):
    times = []
    if not os.path.exists(log_f):
        return times
    for line in open(log_f, "r"):
        match = re.search(rf"Time\s:\s*([a-za-z0-9\.ee\+\-]+)+", line)
        if match:
            times.append(float(match.group(1)))
    return times


def check_sim_ended_ker(run_dir: Path, t_f: float):
    target_t_f = t_f
    log_f = Path(run_dir, Path("run.log"))
    times = extract_times(log_f)

    if len(times) == 0:
        return False, [0.0]

    if target_t_f >= times[-1]:
        if np.abs(target_t_f - times[-1]) <= 1e-9:
            return True, times
        else:
            return False, times
    else:
        print("Simulation over-run!")
        print(f"Reached {float(times[-1])}, instead of {target_t_f}")
        return False, times


def check_sim_ended(run_dir: Path, si: SimInputs):
    check, times = check_sim_ended_ker(run_dir, si.t_final)
    return check, times


def check_dts(time_arr, dt):
    if len(time_arr) <= 3:
        return False, [0.0]
    dt_arr = np.zeros(len(time_arr) - 3)
    for i in range(np.size(dt_arr)):
        dt_arr[i] = time_arr[i + 1] - time_arr[i]

    mean_dt = np.mean(dt_arr)
    succ = False
    if abs(mean_dt - dt) / dt <= 1e-2:
        succ = True

    return succ, mean_dt


def get_sol_filename(run_dir: Path):
    return Path(Path(run_dir) / "solution.bin")


def get_problem_data(name: str):
    if name == "VdW_Dense_Explosion":
        # Dense gas setup: rho_L is pushed very close to the excluded volume limit (790.5)
        ric = RiemannIC(
            rho_L=785.0,  # Very close to the excluded volume limit (790.5)
            vel_L=0.0,
            pres_L=1000000.0,  # 10 bar
            rho_R=1.225,  # Standard air density
            vel_R=0.0,
            pres_R=100000.0,  # 1 bar
            gamma=1.4,
        )
        return {"ric": ric, "t_fin": 0.005, "radius": 0.13, "l_0": 1.0}

    # Fallback to original Zouh
    ric = RiemannIC(
        rho_L=2.0, vel_L=0.0, pres_L=15.0, rho_R=1.0, vel_R=0.0, pres_R=2.0, gamma=1.4
    )
    return {"ric": ric, "t_fin": 0.13, "radius": 0.13, "l_0": 1.0}


def run_circ_expl(
    si: SimInputs, ric: RiemannIC, run_dir: Path, radius: float, other_opts: dict
) -> None:
    sol_filename = get_sol_filename(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    EXEC_CMD, BASE_ARGS = get_sim_cmd(
        "circ_explosion",
        si.n_elems,
        si.poly_deg,
        si.dt,
        si.t_final,
        sol_filename,
        si.l_0,
        si.change_var,
        RADIUS=radius,
        implicit=si.is_implicit,
    )

    # ---------------------------------------------------------
    # Van der Waals Gas Model Flags from sod_shock.opts
    # ---------------------------------------------------------
    vdw_args = [
        "-gas_mod_law",
        "van_der_waals",
        "-gas_mod_R",
        "287.5",
        "-gas_mod_gamma",
        "1.4",
        "-gas_mod_VdW_a",
        f"{str(a_vdw)}",
        "-gas_mod_VdW_b",
        f"{str(b_vdw)}",
    ]

    # ---------------------------------------------------------
    # Time Stepping & SNES/KSP Flags from sod_shock.opts
    # ---------------------------------------------------------
    solver_args = []
    if si.is_implicit:
        solver_args = [
            # "-jac_type", "complexfd",
            "-snes_mf",
            "-ts_type",
            "bdf",
            "-ts_bdf_order",
            "2",
            "-ts_max_snes_failures",
            "20",
            "-ts_max_reject",
            "50",
            "-snes_max_funcs",
            "unlimited",
            "-snes_max_it",
            "20",
            "-snes_monitor",
            "-snes_converged_reason",
            "-ksp_type",
            "gmres",
            "-ksp_gmres_restart",
            "100",
            "-ksp_pc_side",
            "right",
            "-ksp_max_it",
            "100",
            "-ksp_monitor",
            "-ksp_converged_reason",
            "-pc_type",
            "bjacobi",
            "-sub_pc_type",
            "ilu",
            "-ts_adapt_type",
            "none",
            "-ts_atol",
            "1e-3",  # Absolute error tolerance
            "-ts_rtol",
            "1e-3",  # Relative error tolerance
            "-ts_adapt_clip",
            "0.1,10.0",
        ]
    else:
        solver_args = ["-ts_type", "ssp", "-ts_ssp_type", "rks3"]

    cut_off = 0

    cmd_list = (
        EXEC_CMD
        + BASE_ARGS
        + ric.as_cli_args()
        + vdw_args
        + solver_args
        + [
            "-apply_filter",
            "false",
            "-filt_strength",
            "32",
            "-filt_cut_off_deg",
            str(cut_off),
            "-apply_weno",
            "true",
            "-weno_gamma",
            "0.998",
            "-fp_trap",
            "off" if si.is_implicit else "on",
            "-shock_sens_activation_value",
            "0.8",
            "-shock_sens_theta",
            "1",
            "-ts_monitor_solution_interval",
            "-1",
        ]
    )

    cmd_str = " ".join(cmd_list)
    log_path = f"{run_dir}/run.log"
    full_cmd = f"{cmd_str} > {log_path} 2>&1"

    type_str = "IMPLICIT" if si.is_implicit else "EXPLICIT"
    print("=" * 60)
    print(f"LAUNCHING {type_str} VDW SOLVER")
    print(f"LOGGING TO: {log_path}")
    print("=" * 60)

    t0 = perf_counter()
    subprocess.run(full_cmd, shell=True, text=True)
    elapsed = perf_counter() - t0

    meta = asdict(si)
    meta["command"] = cmd_str
    meta["time_to_sol"] = elapsed
    meta["did_end"], times = check_sim_ended(run_dir, si)

    is_correct, dt_mean = check_dts(times, si.dt)
    if is_correct:
        print("Dt correctly imposed!")
    else:
        print(f"Dt CHANGED to {dt_mean}!")

    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def get_circ_expl(
    si: SimInputs,
    ric: RiemannIC,
    run=False,
    parent_dir=Path("./"),
    radius=0.13,
    opts={},
):
    circ_dir = Path(parent_dir, f"build/vdw_expl")
    run_dir = si.get_dir(circ_dir)
    sol_filename = get_sol_filename(run_dir)

    # Note: RiemannIC c_max might calculate using ideal gas logic,
    # so the reported CFL here will be vastly underestimating the true acoustic CFL.
    c_max = ric.c_max()
    CFL, CFL_safe, min_dt = si.get_cfl_s(c_max)

    if run:
        meta = run_circ_expl(si, ric, run_dir, radius, opts["other_opt"])
    else:
        meta_fl = Path(run_dir / "meta.json")
        with open(meta_fl) as json_file:
            meta = json.load(json_file)

    return {"meta": meta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./")
    parser.add_argument("--cfl", default=1.0)
    parser.add_argument("--n_els", default=50)
    parser.add_argument("--p_deg", default=1)
    parser.add_argument("--is_impl", action="store_true")
    args = parser.parse_args()

    cfl_0 = float(args.cfl)
    n_els_0 = int(args.n_els)
    p_deg_0 = int(args.p_deg)

    p_dir = Path(str(args.dir))
    p_dir.mkdir(parents=True, exist_ok=True)

    is_impl = args.is_impl

    sim_data = get_problem_data("VdW_Dense_Explosion")
    l_0 = sim_data["l_0"]
    t_fin = sim_data["t_fin"]
    radius = sim_data["radius"]
    ric = sim_data["ric"]

    sims = []

    c_max = get_vdw_c_max(ric, a_vdw, b_vdw)
    print(f"Corrected VdW c_max: {c_max:.2f} m/s")

    # Run both Explicit and Implicit to compare
    print(f"\nStarting N: {n_els_0}, P: {p_deg_0}, Implicit: {is_impl}")

    dx = l_0 / n_els_0
    dt = cfl_0 * dx / c_max

    si = SimInputs(
        is_implicit=is_impl,
        dt=dt,
        t_final=t_fin,
        n_elems=n_els_0,
        poly_deg=p_deg_0,
        l_0=l_0,
        change_var=False,
    )

    opts = {
        "extract_sol": False,
        "other_opt": {"hard_fail": False},
    }

    ris = get_circ_expl(si, ric, run=True, parent_dir=p_dir, radius=radius, opts=opts)

    meta = ris["meta"]
    if meta["did_end"]:
        print(f"Finished run successfully in {meta['time_to_sol']:.2f} seconds.")
    else:
        print(f"Failed run! Time to crash/stop: {meta['time_to_sol']:.2f} seconds.")

    sims.append({"in": asdict(si), "out": ris})

    print("\n" + "=" * 40)
    print("PERFORMANCE SUMMARY")
    print("=" * 40)
    for sim in sims:
        impl_str = "Implicit" if sim["in"]["is_implicit"] else "Explicit"
        status = "SUCCESS" if sim["out"]["meta"]["did_end"] else "FAILED"
        print(
            f"{impl_str} Solver | Status: {status} | Wall Time: {sim['out']['meta']['time_to_sol']:.2f} s"
        )
