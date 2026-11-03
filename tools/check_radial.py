from os.path import exists
import argparse
import re
import json
import subprocess
from time import perf_counter
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict

import numpy as np

from plot_zouh import get_data_plots
from sod_radial_moc import read_results_binary
from plot_func import plot_flow_fields
from plot import csi_ws, get_mesh_on_gll

from post_proc import run_sim_get_hyst, RiemannIC, Path, _make_run_dir
from check_iso_vortex import get_sim_cmd, extract_sol


def my_set_title(ax, title):
    ax.set_title(title, fontsize=11, fontweight="medium")


def get_radial_sol(
    run=False,
    parent_dir=Path("./"),
    ric=RiemannIC(
        rho_L=2.0,
        vel_L=0.0,
        pres_L=15,
        rho_R=1.0,
        vel_R=0.0,
        pres_R=2.0,
        gamma=1.4,
    ),
    t_max=0.13,
    l_0=0.5,
    radius=0.13,
):
    inps = dict()
    inps["n_elems"] = 1000
    inps["poly_deg"] = 1
    inps["l_0"] = l_0
    n_elems = inps["n_elems"]
    poly_deg = inps["poly_deg"]
    dt = 4e-6
    run_dir = Path(parent_dir, "build/radial/")
    run_dir.mkdir(parents=True, exist_ok=True)
    solver_key = "csd"
    extra_args = [
        "-ts_adapt_type",
        "none",
        "-apply_filter",
        "true",
        "-filt_strength",
        "32",
        "-filt_cut_off_deg",
        "0",
        "-apply_weno",
        "true",
        "-weno_gamma",
        "0.98",
        "-jac_type",
        "complexfd",
        "-ts_type",
        "bdf",
        "-ts_bdf_order",
        "2",
        # , ] "-ts_type", "ssp", "-ts_ssp_type", "rks3"]
        "-ts_monitor_solution_interval",
        "-1",
        "-radius",
        str(radius),
    ]
    EXEC_CMD = ["mpirun", "./build/test/radial_shock"]

    CFL = ric.c_max() * dt / (l_0 / n_elems)
    CFL_dg = CFL * (poly_deg**2 + 1)
    print(f"{CFL=}")
    print(f"{CFL_dg=}")

    mesh_x = get_mesh_on_gll(n_elems, poly_deg, l_0)
    rho, rhoU, rhoE, time = run_sim_get_hyst(
        l_0,
        n_elems,
        poly_deg,
        dt,
        t_max,
        run_dir,
        solver_key,
        extra_args,
        ric,
        run,
        EXEC_CMD,
    )
    out = {"inp": inps, "mesh": mesh_x, "data": [rho, rhoU, rhoE]}
    return out


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
        rk_bdf_dir = Path("BFD2") if self.is_implicit else Path("SSP3")
        dt_dir = Path(f"{self.dt:3e}")
        return Path(parent_dir, n_p_dir, rk_bdf_dir, dt_dir)

    def get_cfl_s(self, c_max: float):
        dx = self.l_0 / self.n_elems
        min_dt = dx / (2 * self.poly_deg + 1) / c_max
        CFL = self.dt * c_max / dx
        CFL_DG = CFL * (2 * self.poly_deg + 1)
        return CFL, CFL_DG, min_dt


def extract_times(log_f):

    times = []
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
    # -2 cause discard the last dt since it needs to match the final time
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


def run_circ_expl(
    si: SimInputs, ric: RiemannIC, run_dir: Path, radius: float, other_opts: dict
) -> None:
    # TODO Write the data here !!
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

    cut_off = 0
    """
    if si.poly_deg >= 2:
        cut_off = 1
    """

    str_oth = []
    if other_opts["hard_fail"]:
        str_oth = [
            "-ts_error_if_step_fails",
            "true",
            "-ts_max_snes_failures",
            "1",
            "-ts_adapt_type",
            "none",
        ]
    # 1. Gather your Python list of arguments normally
    cmd_list = (
        EXEC_CMD
        + BASE_ARGS
        + ric.as_cli_args()
        + [
            "-apply_filter",
            "false",  # if si.is_implicit == True else "true",
            "-filt_strength",
            "32",
            "-filt_cut_off_deg",
            str(cut_off),
            "-apply_weno",
            "true",
            "-weno_gamma",
            "0.998",
            "-fp_trap",
            "off" if si.is_implicit == True else "on",
            "-shock_sens_activation_value",
            "0.8",
            "-shock_sens_theta",
            "1",
            "-ts_monitor_solution_interval",
            "-1",
            "-pc_type",
            "none",
        ]
        + str_oth
        # "-snes_type", "ksponly",
        # "-pc_type", "none",
        # "-memory_view", "-malloc_view",
    )

    # 2. Convert the list into a single space-separated string
    cmd_str = " ".join(cmd_list)

    # 3. Redirect output ONLY to the log file
    log_path = f"{run_dir}/run.log"

    # > redirects standard output to the file.
    # 2>&1 redirects standard error to the same place, so crashes don't bleed into your CLI.
    full_cmd = f"{cmd_str} > {log_path} 2>&1"
    print(full_cmd)

    print("=" * 60)
    print(f"LAUNCHING SOLVER AND LOGGING TO: {log_path}")
    print("=" * 60)

    # 4. Run as a STRING with shell=True
    t0 = perf_counter()
    result = subprocess.run(full_cmd, shell=True, text=True)
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

    input_file = run_dir / "meta.json"
    with open(input_file, "w") as f:
        json.dump(meta, f, indent=2)
    return


def get_circ_expl(
    si: SimInputs,
    ric: RiemannIC,
    run=False,
    parent_dir=Path("./"),
    radius=0.13,
    opts={},
):
    circ_dir = Path(parent_dir, f"build/circ_expl")
    run_dir = si.get_dir(circ_dir)
    sol_filename = get_sol_filename(run_dir)

    c_max = ric.c_max()
    print(f"{c_max=}")
    CFL, CFL_safe, min_dt = si.get_cfl_s(c_max)
    print(f"{CFL_safe=}")
    print(f"{CFL=}")
    if CFL < 2:
        print(f"Suggest to move to dt: {min_dt}")
    if run:
        run_circ_expl(si, ric, run_dir, radius, opts["other_opt"])

    print(f"Extracting solution from {sol_filename}")

    meta_fl = Path(run_dir / "meta.json")
    data = {}
    with open(meta_fl) as json_file:
        data = json.load(json_file)
    if opts.get("extract_sol", True):
        X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
            sol_filename, si.n_elems, si.poly_deg, si.l_0, si.change_var
        )
        out = {
            "mesh": [X_glob, Y_glob],
            "num_ris": [rho_num, rhou_num, rhov_num, rhoE_num],
            "meta": data,
        }
    else:
        out = {
            "meta": data,
        }
    return out


def get_moc_radial_sol():
    file_path = "build/moc_radial/zouh_moc_results.bin"
    saved_data = read_results_binary(file_path)
    final_t, r, rho, rhou, rhoE = saved_data[-1]
    print(f"MOC solution reached {final_t}")
    out = {
        "mesh": r,
        "num_ris": [rho, rhou, rhoE],
    }
    return out


def get_problem_data(name: str):
    if name == "Toro":
        return {"ric": RiemannIC(), "t_fin": 0.25, "radius": 0.4, "l_0": 2.0}
    if name == "Zouh":
        ric = RiemannIC(
            rho_L=2.0,
            vel_L=0.0,
            pres_L=15,
            rho_R=1.0,
            vel_R=0.0,
            pres_R=2.0,
            gamma=1.4,
        )
        return {"ric": ric, "t_fin": 0.13, "radius": 0.13, "l_0": 1.0}


def get_hyperref_1d():
    run_1d = False
    sim_data = get_problem_data("Zouh")
    ric = sim_data["ric"]
    ris_1d = get_radial_sol(
        run_1d,
        parent_dir=Path("./"),
        ric=sim_data["ric"],
        t_max=sim_data["t_fin"],
        l_0=sim_data["l_0"] * 0.5,
        radius=sim_data["radius"],
    )
    return ris_1d


def run_case(p_deg, n_els, cfl, is_impl):
    sims = []
    sim_data = get_problem_data("Zouh")

    pos_cfl = cfl > 0.0

    l_0 = sim_data["l_0"]
    t_fin = sim_data["t_fin"]
    radius = sim_data["radius"]
    ric = sim_data["ric"]
    print(f"Starting N : {n_els}, P : {p_deg} run.")

    dx = l_0 / n_els

    dt = dx / ric.c_max()
    if pos_cfl:
        dt *= cfl
    else:
        dt *= 2.0

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
        "other_opt": {
            "hard_fail": pos_cfl,
        },
    }
    ris = get_circ_expl(si, ric, run=run_2d, parent_dir=p_dir, radius=radius, opts=opts)
    if opts["extract_sol"]:
        X_glob, Y_glob = ris["mesh"]
        rho_num, rhou_num, rhov_num, rhoE_num = ris["num_ris"]

        scalars = {
            "rho": rho_num,
            "rhou": rhou_num,
            "rhov": rhov_num,
            "rhoE": rhoE_num,
        }
    # plot_flow_fields( X_glob, Y_glob, scalars, l_0, save_path=None, show=True,)
    """
    """
    meta = ris["meta"]
    if meta["did_end"]:
        print(f"Finisched run in {meta["time_to_sol"]}.")
    else:
        print(f"Failed run !!.")

    sims.append({"in": asdict(si), "out": ris})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./")
    parser.add_argument("--out_im_dir", default=f"build/circ_expl")
    parser.add_argument("--cfl", default=1.0)
    parser.add_argument("--n_els", default=250)
    parser.add_argument("--p_deg", default=1)
    parser.add_argument("--run", default=False)
    args = parser.parse_args()

    cfl_0 = float(args.cfl)
    n_els_0 = int(args.n_els)
    p_deg_0 = int(args.p_deg)

    out_im_dir = Path(str(args.out_im_dir))
    out_im_dir.mkdir(parents=True, exist_ok=True)
    p_dir = Path(str(args.dir))
    p_dir.mkdir(parents=True, exist_ok=True)
    run_2d = bool(args.run)

    # sim_data = get_problem_data("Toro")
    sim_data = get_problem_data("Zouh")

    l_0 = sim_data["l_0"]
    t_fin = sim_data["t_fin"]
    radius = sim_data["radius"]
    ric = sim_data["ric"]

    fil, ___ = get_data_plots()
    ris_1d = get_hyperref_1d()
    moc_1d = get_moc_radial_sol()

    run_case(2, 50, 2, True)

    exit()

    """
    # IMPL_RUNS_TARGETS
    for p_deg in [p for p in range(1, 3)]:
        for n_els in [100, 250]:
            for is_impl in [False, True]:
    for p_deg in [p_deg_0]:
        for n_els in [n_els_0]:
            is_impl = True
            # for cfl in [0.5, 1.0, 2.0] :
            cfl = cfl_0
            """

    # ============================================================
    # Ceiling data (measured at N=50) — assumed ~N-independent since
    # CFL is already normalized by dh=1/N. If a "Worked: No" shows up
    # at a different N, that assumption broke and the table needs a
    # per-N entry instead.
    # ============================================================
    CFL_CEILING = {
        False: {1: 0.40, 2: 0.20, 3: 0.20, 4: 0.05},  # Explicit
        True: {1: 1.20, 2: 0.40, 3: 0.30, 4: 0.16},  # Implicit
    }

    def safe_cfl(is_impl: bool, p_deg: int, frac: float = 1.0) -> float:
        """frac of this scheme/P's stability ceiling."""
        return frac * CFL_CEILING[is_impl][p_deg]

    # --------------------------------------------------------
    # Cube 1 — ceiling / half / quarter, N=50, all P, both schemes
    # (the 24-run grid we just laid out)
    # --------------------------------------------------------
    for p_deg in [1, 2, 3, 4]:
        for is_impl in [False, True]:
            for frac in [1.0, 0.5, 0.25, -1.0]:
                n_els = 50
                cfl = safe_cfl(is_impl, p_deg, frac)
                if cfl <= 0 and is_impl == False:
                    continue
                run_case(p_deg, n_els, cfl, is_impl)

    """
    # --------------------------------------------------------
    # Cube 2 — same ceiling/half/quarter pattern, N=100
    # fills the sparse P=3/P=4 gap at N=100 and gives a second
    # N to check whether the ceilings actually hold there
    # --------------------------------------------------------
    for p_deg in [1, 2, 3, 4]:
        for is_impl in [False, True]:
            for frac in [1.0, 0.5, 0.25]:
                n_els = 100
                cfl = safe_cfl(is_impl, p_deg, frac)
                run_case(p_deg, n_els, cfl, is_impl)
    """

    # --------------------------------------------------------
    # Priority A — L2 resolution study, P=1 fixed, N sweeps up
    # CFL=0.3 is below explicit's P=1 ceiling (0.40) and well
    # inside implicit's (1.20), so both schemes stay stable at
    # every N -> a genuinely apples-to-apples refinement study.
    # --------------------------------------------------------
    for n_els in [50, 100, 150, 200]:
        for is_impl in [False, True]:
            p_deg = 1
            cfl = 0.3
            run_case(p_deg, n_els, cfl, is_impl)

    exit()

    moc_mesh = moc_1d["mesh"]
    rho_moc, rhoU_moc, rhoE_moc = moc_1d["num_ris"]

    mesh_x = ris_1d["mesh"]
    rho, rhoU, rhoE = ris_1d["data"]

    def get_name_image(descr, format="png"):
        if out_im_dir == f"build/circ_expl":
            run_dir = si.get_dir(f"build/circ_expl")
        else:
            run_dir = out_im_dir

        file_name_descr = str(descr + "." + str(format))
        return Path(run_dir, file_name_descr)

    ntargs = [100, 50, 25]

    for ntarg in ntargs:
        fig, ax = plt.subplots(1, 3, figsize=(10, 8))
        l_half = l_0 * 0.5
        ax[0].plot(
            mesh_x + l_half, rho[-1, :], "k", lw=1.5, label="1D radial reference"
        )
        ax[1].plot(mesh_x + l_half, rhoU[-1, :], "k", lw=1.5)
        ax[2].plot(mesh_x + l_half, rhoE[-1, :], "k", lw=1.5)

        for sim in sims:
            n = sim["in"]["n_elems"]
            p = sim["in"]["poly_deg"]
            if n != ntarg:
                continue
            num_ris = sim["out"]["num_ris"]
            rho_num, rhou_num, rhov_num, rhoE_num = num_ris

            X_glob, Y_glob = sim["out"]["mesh"]
            scalars = {
                "rho": rho_num,
                "rhou": rhou_num,
                "rhov": rhov_num,
                "rhoE": rhoE_num,
            }
            save_dir = Path("img")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir / f"cont_plot_{n}_{p}")
            # plot_flow_fields( X_glob, Y_glob, scalars, l_0, save_path=str(save_path), show=True,)

            y_half_idx = np.argmin(np.abs(Y_glob[:, 0] - l_half))
            ax[0].plot(
                X_glob[0, :], rho_num[y_half_idx, :], label=f"N : {n}  P : {p} slice"
            )
            ax[1].plot(
                X_glob[0, :],
                rhou_num[y_half_idx, :],
            )
            ax[2].plot(
                X_glob[0, :],
                rhoE_num[y_half_idx, :],
            )

        my_set_title(ax[0], r"$\rho$ profile")
        my_set_title(ax[1], r"$\rho u$ profile")
        my_set_title(ax[2], r"$\rho E$ profile")

        fig.legend(loc="upper right", fontsize=14)
        fig.suptitle(r"2D shock test. Slice solution", fontsize=14, fontweight="bold")
        for axis in ax.flat:
            axis.set_xlim(l_half, l_0)

        slice_name = f"imgs/shocks-test/slice_{ntarg}_theta_2.pdf"  # get_name_image(f"slice_confront", format="pdf")
        print(f"Image save at: {slice_name}")
        plt.savefig(slice_name, format="pdf")

        # plt.show()

    # Move to radial 1D the data
    r_2d = np.sqrt((X_glob - l_half) ** 2 + (Y_glob - l_half) ** 2).ravel()
    rho_2d = rho_num.ravel()

    """
    # Remove any points outside [0, 2]
    mask = r_2d <= l_half
    r_2d = r_2d[mask]
    rho_2d = rho_2d[mask]
    plt.plot(r_2d, rho_2d, label="Coarse 2D data, in polar form")

    plt.plot(mesh_x, rho[-1, :], label="High resolution radial solution")
    # plt.show()
    """

    fig, ax = plt.subplots(1, figsize=(10, 8))
    linestyle = ".-"
    ax.plot(
        moc_mesh + l_half,
        rho_moc,
        linestyle,
        color="blue",
        lw=1.0,
        label="MOC solution",
    )
    ax.plot(fil[:, 0], fil[:, 1], linestyle, label="Zouh (from png)", lw=1.0)
    ax.plot(
        mesh_x + l_half,
        rho[-1, :],
        linestyle,
        color="k",
        lw=1.0,
        label="1D radial reference",
    )
    ax.plot(
        X_glob[0, :],
        rho_num[y_half_idx, :],
        linestyle,
        color="r",
        lw=1.0,
        label="2D solution slice",
    )
    ax.set_xlim(l_half, l_0)
    fig.legend(loc="upper left", fontsize=14)
    fig.suptitle(
        r"$\rho$ profile of different comparison methods",
        fontsize=14,
        fontweight="bold",
    )
    comp_name = get_name_image("comp_methods", format="pdf")
    print(f"Image save at: {comp_name}")
    plt.savefig(comp_name, format="pdf")

    plt.show()
    plt.plot(mesh_x, rho[-1, :], "k", lw=1.5, label="1D radial reference")
    plt.show()
    # Binned mean and std
    n_bins = int(si.n_elems / 5)
    bins = np.linspace(0, l_half, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    rho_mean = np.zeros(n_bins)
    rho_std = np.zeros(n_bins)

    for i in range(n_bins):
        m = (r_2d >= bins[i]) & (r_2d < bins[i + 1])
        if m.sum() > 0:
            rho_mean[i] = rho_2d[m].mean()
            rho_std[i] = rho_2d[m].std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mesh_x, rho[-1, :], "k", lw=1.5, label="1D radial reference")
    # ax.scatter(r_2d, rho_2d, s=0.5, c="r", alpha=0.3, label="2D nodes")
    ax.plot(bin_centers, rho_mean, "r", lw=1.5, label="2D azimuthal mean")
    ax.fill_between(
        bin_centers,
        rho_mean - rho_std,
        rho_mean + rho_std,
        color="r",
        alpha=0.3,
        label=r"$\pm 1\sigma$ azimuthal",
    )
    ax.set_xlim(0, l_half)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho$")
    ax.legend()
    plt.tight_layout()
    plt.show()

    """
    """

    ###############################

    from scipy.interpolate import BarycentricInterpolator

    # Pre-reshape arrays once so we don't do it per-quadrature point
    _inps = ris_1d["inp"]
    _n_ele = _inps["n_elems"]
    _p_deg = _inps["poly_deg"]

    _mesh_n_p = ris_1d["mesh"].reshape((_n_ele, _p_deg + 1))
    _rho_1d_el = rho[-1, :].reshape((_n_ele, _p_deg + 1))
    _rhoU_1d_el = rhoU[-1, :].reshape((_n_ele, _p_deg + 1))
    _rhoE_1d_el = rhoE[-1, :].reshape((_n_ele, _p_deg + 1))

    _el_boundaries = _mesh_n_p[:, 0]
    _max_r = _mesh_n_p[-1, -1]

    def extract_ref_vals(r_eq):
        out = {"rho": 0.0, "rhoU": 0.0, "rhoE": 0.0}

        # Out of bounds clamping
        if r_eq <= (sim_data["l_0"] * 0.5):
            return {
                "rho": _rho_1d_el[0, 0],
                "rhoU": _rhoU_1d_el[0, 0],
                "rhoE": _rhoE_1d_el[0, 0],
            }
        elif r_eq >= _max_r:
            return {
                "rho": _rho_1d_el[-1, -1],
                "rhoU": _rhoU_1d_el[-1, -1],
                "rhoE": _rhoE_1d_el[-1, -1],
            }

        # searchsorted instantly finds the containing element (O(log N) instead of O(N))
        el_idx = np.searchsorted(_el_boundaries, r_eq, side="right") - 1

        # If it is on a shared node => mean
        if el_idx < _n_ele - 1 and abs(r_eq - _mesh_n_p[el_idx, -1]) < 1e-10:
            out["rho"] = 0.5 * (_rho_1d_el[el_idx, -1] + _rho_1d_el[el_idx + 1, 0])
            out["rhoU"] = 0.5 * (_rhoU_1d_el[el_idx, -1] + _rhoU_1d_el[el_idx + 1, 0])
            out["rhoE"] = 0.5 * (_rhoE_1d_el[el_idx, -1] + _rhoE_1d_el[el_idx + 1, 0])
        else:
            # Else reconstruct Lagrange polynomial using scipy (Barycentric is the stable Lagrange)
            x_nodes = _mesh_n_p[el_idx, :]
            out["rho"] = float(
                BarycentricInterpolator(x_nodes, _rho_1d_el[el_idx, :])(r_eq)
            )
            out["rhoU"] = float(
                BarycentricInterpolator(x_nodes, _rhoU_1d_el[el_idx, :])(r_eq)
            )
            out["rhoE"] = float(
                BarycentricInterpolator(x_nodes, _rhoE_1d_el[el_idx, :])(r_eq)
            )

        return out

    for sim in sims:
        inps = sim["in"]
        ris = sim["out"]
        if ris["meta"]["did_end"] is False:
            print("This run did NOT reach final time!")
            print(inps)
            break
        n_els = inps["n_elems"]
        p_deg = inps["poly_deg"]
        x_mesh, y_mesh = ris["mesh"]
        rho, rhou, rhov, rhoE = ris["num_ris"]
        csi, ws = csi_ws(p_deg)
        dh = inps["l_0"] / inps["n_elems"]

        rho_2d = rho.reshape((inps["n_elems"], inps["n_elems"], p_deg + 1, p_deg + 1))
        rhou_2d = rho.reshape((inps["n_elems"], inps["n_elems"], p_deg + 1, p_deg + 1))
        rhov_2d = rho.reshape((inps["n_elems"], inps["n_elems"], p_deg + 1, p_deg + 1))
        rhoE_2d = rho.reshape((inps["n_elems"], inps["n_elems"], p_deg + 1, p_deg + 1))

        err_rho = 0
        err_rhou = 0
        err_rhov = 0
        err_rhoE = 0

        def map(xi, a, b):
            return xi * (b - a) / 2 + (a + b) / 2

        for i in range(inps["n_elems"]):
            for j in range(inps["n_elems"]):
                """
                In this case the 1D IS hyper-resolved and is the reference.
                This implies that the less resolved needs to be projected in
                the hyper-resolved solution space.
                Then integration to compute error is made on the reference
                element in the hyper-resolved space.
                """
                # Move the radial data to 2D
                #  (x, y) => (r, ϴ)
                #  (u, v) => (u_r, u_ϴ)

                # Find the interval of reference on the 1D radial sol
                # Compute the error
                x = i * dh
                y = j * dh
                for k in range(p_deg + 1):
                    x_k = map(csi[k], x, x + dh)
                    for l in range(p_deg + 1):
                        y_l = map(csi[l], x, x + dh)

                        l_half = l_0 * 0.5
                        dx = x_k - l_half
                        dy = y_l - l_half
                        r_eq = np.sqrt(dx**2 + dy**2)
                        ref_vals = extract_ref_vals(r_eq)
                        theta_rad = np.atan2(dx, dy)
                        vel_ref = [
                            ref_vals["rhoU"] * np.cos(theta_rad),
                            ref_vals["rhoU"] * np.sin(theta_rad),
                        ]

                        rho_num_val = rho_2d[i, j, k, l]
                        rhou_num_val = rhou_2d[i, j, k, l]
                        rhov_num_val = rhov_2d[i, j, k, l]
                        rhoE_num_val = rhoE_2d[i, j, k, l]

                        # Integration weight for 2D (w_k * w_l * detJ)
                        detJ = (dh / 2.0) ** 2
                        weight = ws[k] * ws[l] * detJ

                        # Accumulate squared error
                        err_rho += weight * np.abs(rho_num_val - ref_vals["rho"]) ** 1
                        err_rhou += weight * np.abs(rhou_num_val - vel_ref[0]) ** 1
                        err_rhov += weight * np.abs(rhov_num_val - vel_ref[1]) ** 1
                        err_rhoE += (
                            weight * np.abs(rhoE_num_val - ref_vals["rhoE"]) ** 1
                        )

        """
        err_rho = np.sqrt(err_rho)
        err_rhou = np.sqrt(err_rhou)
        err_rhov = np.sqrt(err_rhov)
        err_rhoE = np.sqrt(err_rhoE)
        """

        sim["err"] = [err_rho, err_rhou, err_rhov, err_rhoE]
        print(f"--- Errors for N={n_els}, P={p_deg} ---")
        print(f"L2 Error Rho  : {err_rho:e}")
        print(f"L2 Error RhoU : {err_rhou:e}")
        print(f"L2 Error RhoV : {err_rhov:e}")
        print(f"L2 Error RhoE : {err_rhoE:e}")
