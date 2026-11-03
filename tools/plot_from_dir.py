from datetime import time
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import sys
from check_iso_vortex import extract_sol
from check_radial import (
    check_sim_ended_ker,
    get_hyperref_1d,
    get_sol_filename,
    SimInputs,
)
import argparse

from plot_func import plot_flow_fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./")
    parser.add_argument("--n_dim", default=2)
    parser.add_argument("--n_elems", default=10)
    parser.add_argument("--poly_deg", default=1)
    parser.add_argument("--change_var", default=False)
    parser.add_argument("--l_0", default=1.0)
    parser.add_argument("--from_json", default=False)

    args = parser.parse_args()
    dir = Path(str(args.dir))

    from_json = bool(args.from_json)
    n_elems = int(args.n_elems)
    poly_deg = int(args.poly_deg)
    change_var = bool(args.change_var)
    l_0 = float(args.l_0)

    if from_json:
        file_name = Path(Path(dir) / "meta.json")
        with open(file_name) as json_file:
            data = json.load(json_file)
        # TODO override the data above with the json ones

        n_elems = data["n_elems"]
        poly_deg = data["poly_deg"]
        change_var = data["change_var"]
        l_0 = data["l_0"]
        t_final = data["t_final"]
        is_implicit = data["is_implicit"]
        time_to_sol = data["time_to_sol"]
        did_end = data["did_end"]

    sol_filename = get_sol_filename(dir)
    check, times = check_sim_ended_ker(dir, t_f=0.13)
    if check:
        print("Final time reached!")
    else:
        print(f"Final time not reached! t_final : {float(times[-1])}")

    ris_1d = get_hyperref_1d()
    mesh_x = ris_1d["mesh"]
    rho, rhoU, rhoE = ris_1d["data"]

    X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
        sol_filename, n_elems, poly_deg, l_0, change_var
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(X_glob, Y_glob, rho_num, color="grey")
    plt.show()

    plt.figure(figsize=(12, 5))
    contour = plt.contourf(X_glob, Y_glob, rho_num, levels=100, cmap="viridis")
    plt.title("Contour plot of the final time step")
    plt.show()

    l_half = l_0 / 2
    y_half_idx = np.argmin(np.abs(Y_glob[:, 0] - l_half))
    plt.plot(
        X_glob[0, :],
        rho_num[y_half_idx, :],
        ".-",
        color="r",
        label="2D solution slice at half plane",
    )

    plt.plot(
        mesh_x + l_half, rho[-1, :], "-", color="k", lw=1.0, label="1D radial reference"
    )

    plt.legend()
    plt.show()
