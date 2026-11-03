from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np

from check_radial import get_hyperref_1d
from check_iso_vortex import extract_sol


n_25_p_2 = "build/circ_expl/n_25_p_2/BFD2/1.371585e-05/"
n_25_p_1 = "build/circ_expl/n_25_p_1/BFD2/3.086067e-03/"

n_50_p_1 = "build/circ_expl/n_50_p_1/BFD2/6.172134e-03/"
n_50_p_2 = "build/circ_expl/n_50_p_2/BFD2/1.000000e-03/"

n_100_p_1 = "build/circ_expl/n_100_p_1/BFD2/3.641559e-03/"
n_100_p_1_l = "build/circ_expl/n_100_p_1/BFD2/1.543033e-03/"

n_250_p_1 = "build/circ_expl/n_250_p_1/BFD2/1.000000e-03/"


configs = [
    {"n": 25, "p": 1, "fn": n_25_p_1},
    {"n": 25, "p": 2, "fn": n_25_p_2},
    {"n": 50, "p": 1, "fn": n_50_p_1},
    {"n": 50, "p": 2, "fn": n_50_p_2},
    {"n": 100, "p": 1, "fn": n_100_p_1_l},
    {"n": 250, "p": 1, "fn": n_250_p_1},
]

impl_expl_100 = [
    {
        "n": 100,
        "p": 1,
        "dt": 3.09e-06,
        "tts": 3248.0917,
        "fn": "build/circ_expl_gamma98/n_100_p_1/SSP3/3.086067e-06/",
        "is_impl": False,
        "simbol": "^-",
    },
    {
        "n": 100,
        "p": 1,
        "dt": 3.09e-05,
        "tts": 332.5822,
        "fn": "build/circ_expl_gamma98/n_100_p_1/SSP3/3.086067e-05/",
        "is_impl": False,
    },
    {
        "n": 100,
        "p": 1,
        "dt": 3.09e-03,
        "tts": 314.6578,
        "fn": "build/circ_expl_gamma98/n_100_p_1/BFD2/3.086067e-03/",
        "is_impl": True,
    },
    {
        "n": 100,
        "p": 2,
        "dt": 9.26e-04,
        "tts": 2540.7093,
        "fn": "build/circ_expl_gamma98/n_100_p_2/BFD2/9.258201e-04/",
        "is_impl": True,
    },
]


if __name__ == "__main__":

    ris_1d = get_hyperref_1d()
    mesh_x = ris_1d["mesh"]
    rho, rhoU, rhoE = ris_1d["data"]

    l_0 = 1.0
    l_half = 0.5

    fig, ax = plt.subplots(1, figsize=(10, 8))
    linestyle_ = ".-"

    ax.plot(
        mesh_x + l_half, rho[-1, :], "-", color="k", lw=1.0, label="1D radial reference"
    )

    for conf in impl_expl_100:
        file_name = str(conf["fn"] + "solution.bin")
        X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
            file_name,
            conf["n"],
            conf["p"],
            l_0,
            False,
        )
        y_half_idx = np.argmin(np.abs(Y_glob[:, 0] - l_half))

        label = f"N : {conf["n"]} | P : {conf["p"]}"

        if "dt" in conf.keys():
            label += f" | dt : {conf["dt"]:.2e}"
        if "is_impl" in conf.keys():
            label += " impl " if conf["is_impl"] == True else " expl "
        if "simbol" in conf.keys():
            linestyle = conf["simbol"]
        else:
            linestyle = linestyle_

        ax.plot(
            X_glob[0, :],
            rho_num[y_half_idx, :],
            linestyle,  # color="r",
            lw=1.0,
            label=label,
        )

    ax.set_xlim(l_half, l_0)
    fig.legend(loc="upper left", fontsize=14)
    fig.suptitle(
        r"$\rho$ profile of different solutions.",
        fontsize=14,
        fontweight="bold",
    )
    comp_name = "imgs/shocks.pdf"
    print(f"Image save at: {comp_name}")
    plt.savefig(comp_name, format="pdf")

    plt.show()
