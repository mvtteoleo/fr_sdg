from os.path import exists
import argparse
import re
import json
import subprocess
from time import perf_counter
from dataclasses import dataclass, field, asdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

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
    return check_sim_ended_ker(run_dir, si.t_final)


def check_dts(time_arr, dt):
    if len(time_arr) <= 3:
        return False, [0.0]
    dt_arr = np.zeros(len(time_arr) - 3)
    for i in range(np.size(dt_arr)):
        dt_arr[i] = time_arr[i + 1] - time_arr[i]

    mean_dt = np.mean(dt_arr)
    succ = abs(mean_dt - dt) / dt <= 1e-2
    return succ, mean_dt


def get_sol_filename(run_dir: Path):
    return Path(Path(run_dir) / "solution.bin")


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

    cut_off = 0
    str_oth = []
    if other_opts.get("hard_fail", False):
        str_oth = [
            "-ts_error_if_step_fails",
            "true",
            "-ts_max_snes_failures",
            "1",
            "-ts_adapt_type",
            "none",
        ]

    cmd_list = (
        EXEC_CMD
        + BASE_ARGS
        + ric.as_cli_args()
        + [
            "-apply_filter",
            "false" if si.is_implicit else "true",
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
        ]
        + str_oth
    )

    cmd_str = " ".join(cmd_list)
    log_path = f"{run_dir}/run.log"
    full_cmd = f"{cmd_str} > {log_path} 2>&1"

    print("=" * 60)
    print(f"LAUNCHING SOLVER AND LOGGING TO: {log_path}")
    print("=" * 60)

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
    CFL, CFL_safe, min_dt = si.get_cfl_s(c_max)
    print(f"{CFL_safe=}\n{CFL=}")
    if CFL < 2:
        print(f"Suggest to move to dt: {min_dt}")

    if run:
        run_circ_expl(si, ric, run_dir, radius, opts.get("other_opt", {}))

    print(f"Extracting solution from {sol_filename}")
    meta_fl = Path(run_dir / "meta.json")
    with open(meta_fl) as json_file:
        data = json.load(json_file)

    if opts.get("extract_sol", True):
        X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
            sol_filename, si.n_elems, si.poly_deg, si.l_0, si.change_var
        )
        return {
            "mesh": [X_glob, Y_glob],
            "num_ris": [rho_num, rhou_num, rhov_num, rhoE_num],
            "meta": data,
        }
    return {"meta": data}


def get_moc_radial_sol():
    file_path = "build/moc_radial/zouh_moc_results.bin"
    saved_data = read_results_binary(file_path)
    final_t, r, rho, rhou, rhoE = saved_data[-1]
    print(f"MOC solution reached {final_t}")
    return {"mesh": r, "num_ris": [rho, rhou, rhoE]}


def get_problem_data(name: str):
    if name == "Toro":
        return {"ric": RiemannIC(), "t_fin": 0.25, "radius": 0.4, "l_0": 2.0}
    if name == "Zouh":
        return {
            "ric": RiemannIC(
                rho_L=2.0,
                vel_L=0.0,
                pres_L=15,
                rho_R=1.0,
                vel_R=0.0,
                pres_R=2.0,
                gamma=1.4,
            ),
            "t_fin": 0.13,
            "radius": 0.13,
            "l_0": 1.0,
        }


def get_hyperref_1d(sim_data):
    return get_radial_sol(
        run=False,
        parent_dir=Path("./"),
        ric=sim_data["ric"],
        t_max=sim_data["t_fin"],
        l_0=sim_data["l_0"] * 0.5,
        radius=sim_data["radius"],
    )


class ReferenceInterpolator1D:
    def __init__(self, ris_1d, l_0):
        self.l_0 = l_0
        self.l_half = l_0 * 0.5
        inps = ris_1d["inp"]
        self.n_ele = inps["n_elems"]
        self.p_deg = inps["poly_deg"]

        mesh_x = ris_1d["mesh"]
        rho, rhoU, rhoE = ris_1d["data"]

        self.mesh_n_p = mesh_x.reshape((self.n_ele, self.p_deg + 1))
        self.rho_1d_el = rho[-1, :].reshape((self.n_ele, self.p_deg + 1))
        self.rhoU_1d_el = rhoU[-1, :].reshape((self.n_ele, self.p_deg + 1))
        self.rhoE_1d_el = rhoE[-1, :].reshape((self.n_ele, self.p_deg + 1))

        self.el_boundaries = self.mesh_n_p[:, 0]
        self.max_r = self.mesh_n_p[-1, -1]

    def extract_ref_vals(self, r_eq):
        out = {"rho": 0.0, "rhoU": 0.0, "rhoE": 0.0}

        # -----------------------------------------------------
        # THE FIX IS HERE: Change self.l_half to 0.0
        # -----------------------------------------------------
        if r_eq <= 0.0:
            return {
                "rho": self.rho_1d_el[0, 0],
                "rhoU": self.rhoU_1d_el[0, 0],
                "rhoE": self.rhoE_1d_el[0, 0],
            }
        elif r_eq >= self.max_r:
            return {
                "rho": self.rho_1d_el[-1, -1],
                "rhoU": self.rhoU_1d_el[-1, -1],
                "rhoE": self.rhoE_1d_el[-1, -1],
            }

        # searchsorted instantly finds the containing element
        el_idx = np.searchsorted(self.el_boundaries, r_eq, side="right") - 1

        # If it is exactly on a shared node
        if el_idx < self.n_ele - 1 and abs(r_eq - self.mesh_n_p[el_idx, -1]) < 1e-10:
            out["rho"] = 0.5 * (
                self.rho_1d_el[el_idx, -1] + self.rho_1d_el[el_idx + 1, 0]
            )
            out["rhoU"] = 0.5 * (
                self.rhoU_1d_el[el_idx, -1] + self.rhoU_1d_el[el_idx + 1, 0]
            )
            out["rhoE"] = 0.5 * (
                self.rhoE_1d_el[el_idx, -1] + self.rhoE_1d_el[el_idx + 1, 0]
            )
        else:
            # Reconstruct Lagrange polynomial
            x_nodes = self.mesh_n_p[el_idx, :]
            out["rho"] = float(
                BarycentricInterpolator(x_nodes, self.rho_1d_el[el_idx, :])(r_eq)
            )
            out["rhoU"] = float(
                BarycentricInterpolator(x_nodes, self.rhoU_1d_el[el_idx, :])(r_eq)
            )
            out["rhoE"] = float(
                BarycentricInterpolator(x_nodes, self.rhoE_1d_el[el_idx, :])(r_eq)
            )

        return out


def map_xi(xi, a, b):
    return xi * (b - a) / 2 + (a + b) / 2


def compute_2d_errors(sim, ref_interpolator, l_0):
    inps = sim["in"]
    ris = sim["out"]
    if not ris["meta"]["did_end"]:
        print("This run did NOT reach final time!")
        return None

    n_els, p_deg = inps["n_elems"], inps["poly_deg"]

    # These are global 2D arrays of shape (N_nodes_y, N_nodes_x)
    rho, rhou, rhov, rhoE = ris["num_ris"]
    csi_pts, ws = csi_ws(p_deg)
    dh = l_0 / n_els
    l_half = l_0 * 0.5

    l1_err = np.zeros(4)
    l2_err = np.zeros(4)
    linf_err = np.zeros(4)

    for i in range(n_els):  # x-element
        for j in range(n_els):  # y-element
            x = i * dh
            y = j * dh

            # Slice the global 2D array to get the local (p+1) x (p+1) block
            # Axis 0 is Y (j), Axis 1 is X (i)
            y_start, y_end = j * (p_deg + 1), (j + 1) * (p_deg + 1)
            x_start, x_end = i * (p_deg + 1), (i + 1) * (p_deg + 1)

            rho_loc = rho[y_start:y_end, x_start:x_end]
            rhou_loc = rhou[y_start:y_end, x_start:x_end]
            rhov_loc = rhov[y_start:y_end, x_start:x_end]
            rhoE_loc = rhoE[y_start:y_end, x_start:x_end]

            for k in range(p_deg + 1):  # x-node index
                x_k = map_xi(csi_pts[k], x, x + dh)
                for l in range(p_deg + 1):  # y-node index
                    y_l = map_xi(csi_pts[l], y, y + dh)

                    dx, dy = x_k - l_half, y_l - l_half
                    r_eq = np.sqrt(dx**2 + dy**2)

                    # TARGETED SMOOTH REGION
                    if not (0.15 <= r_eq <= 0.42):
                        continue

                    ref_vals = ref_interpolator.extract_ref_vals(r_eq)
                    theta_rad = np.atan2(dy, dx)

                    # Note: Local arrays are accessed as [y_index, x_index] -> [l, k]
                    num_vals = np.array(
                        [rho_loc[l, k], rhou_loc[l, k], rhov_loc[l, k], rhoE_loc[l, k]]
                    )

                    ref_vals_arr = np.array(
                        [
                            ref_vals["rho"],
                            ref_vals["rhoU"] * np.cos(theta_rad),
                            ref_vals["rhoU"] * np.sin(theta_rad),
                            ref_vals["rhoE"],
                        ]
                    )

                    detJ = (dh / 2.0) ** 2
                    weight = ws[k] * ws[l] * detJ

                    abs_diff = np.abs(num_vals - ref_vals_arr)
                    l1_err += weight * abs_diff
                    l2_err += weight * (abs_diff**2)
                    linf_err = np.maximum(linf_err, abs_diff)

    l2_err = np.sqrt(l2_err)

    return {"L1": l1_err.tolist(), "L2": l2_err.tolist(), "Linf": linf_err.tolist()}


def plot_slice_results(sims, ris_1d, l_0):
    mesh_x = ris_1d["mesh"]
    rho, rhoU, rhoE = ris_1d["data"]
    l_half = l_0 * 0.5
    ntargs = [100, 50, 25]

    for ntarg in ntargs:
        fig, ax = plt.subplots(1, 3, figsize=(10, 8))
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

            rho_num, rhou_num, rhov_num, rhoE_num = sim["out"]["num_ris"]
            X_glob, Y_glob = sim["out"]["mesh"]

            y_half_idx = np.argmin(np.abs(Y_glob[:, 0] - l_half))
            ax[0].plot(
                X_glob[0, :], rho_num[y_half_idx, :], label=f"N: {n}  P: {p} slice"
            )
            ax[1].plot(X_glob[0, :], rhou_num[y_half_idx, :])
            ax[2].plot(X_glob[0, :], rhoE_num[y_half_idx, :])

        my_set_title(ax[0], r"$\rho$ profile")
        my_set_title(ax[1], r"$\rho u$ profile")
        my_set_title(ax[2], r"$\rho E$ profile")

        fig.legend(loc="upper right", fontsize=14)
        fig.suptitle(r"2D shock test. Slice solution", fontsize=14, fontweight="bold")
        for axis in ax.flat:
            axis.set_xlim(l_half, l_0)

        slice_name = f"imgs/shocks-test/slice_{ntarg}_theta_2.pdf"
        Path(slice_name).parent.mkdir(parents=True, exist_ok=True)
        print(f"Image saved at: {slice_name}")
        plt.savefig(slice_name, format="pdf")
        plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./")
    parser.add_argument("--out_im_dir", default="build/circ_expl")
    parser.add_argument("--cfl", default=1.0, type=float)
    parser.add_argument("--n_els", default=250, type=int)
    parser.add_argument("--p_deg", default=1, type=int)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()

    out_im_dir = Path(args.out_im_dir)
    out_im_dir.mkdir(parents=True, exist_ok=True)
    p_dir = Path(args.dir)
    p_dir.mkdir(parents=True, exist_ok=True)

    sim_data = get_problem_data("Zouh")
    l_0, t_fin, radius, ric = (
        sim_data["l_0"],
        sim_data["t_fin"],
        sim_data["radius"],
        sim_data["ric"],
    )

    fil, ___ = get_data_plots()
    ris_1d = get_hyperref_1d(sim_data)
    moc_1d = get_moc_radial_sol()

    sims = []

    # Run over Target P and N configurations
    for p_deg in [1, 2, 3]:
        for n_els in [50, 100]:
            print(f"Starting N : {n_els}, P : {p_deg} run.")
            dx = l_0 / n_els
            dt = args.cfl * dx / ric.c_max()

            # Fixed temporal steps for comparison
            if n_els == 50:
                dt = 1.851640e-03
            if n_els == 100:
                dt = 9.258201e-04

            si = SimInputs(
                is_implicit=True,
                dt=dt,
                t_final=t_fin,
                n_elems=n_els,
                poly_deg=p_deg,
                l_0=l_0,
                change_var=False,
            )

            opts = {"extract_sol": True, "other_opt": {"hard_fail": True}}
            ris = get_circ_expl(
                si, ric, run=args.run, parent_dir=p_dir, radius=radius, opts=opts
            )

            if ris["meta"]["did_end"]:
                print(f"Finished run in {ris['meta']['time_to_sol']}.")
            else:
                print(f"Failed run !!")

            sims.append({"in": asdict(si), "out": ris})

    # Generate plotting slices
    plot_slice_results(sims, ris_1d, l_0)

    # Setup the interpolator object only once
    ref_interpolator = ReferenceInterpolator1D(ris_1d, l_0)

    # Compute errors for all runs
    for sim in sims:
        errors = compute_2d_errors(sim, ref_interpolator, l_0)
        if errors:
            sim["err"] = errors
            n_els, p_deg = sim["in"]["n_elems"], sim["in"]["poly_deg"]

            print(f"\n--- Errors for N={n_els}, P={p_deg} ---")
            variables = ["Rho ", "RhoU", "RhoV", "RhoE"]

            print(
                f"{'Variable':<10} | {'L1 Error':<13} | {'L2 Error':<13} | {'L_inf Error':<13}"
            )
            print("-" * 57)
            for idx, var in enumerate(variables):
                print(
                    f"{var:<10} | "
                    f"{errors['L1'][idx]:.6e} | "
                    f"{errors['L2'][idx]:.6e} | "
                    f"{errors['Linf'][idx]:.6e}"
                )
