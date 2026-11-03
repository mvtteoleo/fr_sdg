"""Post-processing for boundary_conforming runs (bound_conf.xlsx).

Three questions:
  1. CSD vs FD Jacobian          -> csd_vs_fd()
  2. Positive-preserving vs conservative variables -> pp_vs_conservative()
  3. Preconditioners at different dt -> preconditioners()
plus an explicit-RK summary -> rk_summary().

Entry point: post_proc(df_csd, df_fd, df_rk)
"""
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUT_DIR = "outputs/wedge_post"
KEY_COLS = ["poly_deg", "dt"]


import matplotlib.pyplot as plt
import pandas as pd

def csd_vs_fd(df_csd: pd.DataFrame, df_fd: pd.DataFrame):
    fig, ax = plt.subplots(1, 1 , figsize=(12, 6))

    ax1 = ax
    ax2 = ax
    preconditioners = ["none", "jacobi", "bjacobi"] # "ilu", 
    
    # Assign distinct colors per preconditioner
    colors = {
        "none": "tab:blue",
        "jacobi": "tab:orange",
        "bjacobi": "tab:green"
    }

    for pre in preconditioners:
        # Filter and sort by dt to ensure connected line plots are ordered correctly
        csd_raw = df_csd[ (df_csd["pc_type"] == pre) & ( 0.004 < df_csd["dt"]) & (df_csd["dt"] < 0.03) ]
        fd_raw  = df_fd[ (df_fd["pc_type"] == pre)   & ( 0.004 < df_fd["dt"] ) & (df_fd["dt"] < 0.03) ]

        csd_sub = ( csd_raw.groupby("dt", as_index=False)["wall_time_sec"] .min() .sort_values(by="dt"))
        fd_sub = ( fd_raw.groupby("dt", as_index=False)["wall_time_sec"] .min() .sort_values(by="dt"))

        print(pre )
        # 1. Plot CSD line (solid line, circle markers)
        if not csd_sub.empty:
            print("CSD : ")
            print(csd_sub )
            ax2.semilogy(
                csd_sub["dt"], 
                csd_sub["wall_time_sec"], 
                label=f"CSD ({pre})", 
                color=colors[pre], 
                linestyle="-", 
                marker="o", 
                linewidth=2
            )

        # 2. Plot FD line (dashed line, square markers)
        if not fd_sub.empty:
            print("FD : ")
            print(fd_sub )
            ax1.semilogy(
                fd_sub["dt"], 
                fd_sub["wall_time_sec"], 
                label=f"FD ({pre})", 
                color=colors[pre], 
                linestyle="--", 
                marker="s", 
                linewidth=2
            )

    for ax in [ax1, ax2]:
        ax.set_xlabel("Time Step [s]", fontsize=12)
        ax.set_ylabel("Wall-Clock Time [s]", fontsize=12)
        ax.set_title("CSD vs FD: Wall Time vs. Time Step per Preconditioner", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)

    # Handles unificati con intestazioni in grassetto
    combined_handles = [
        Line2D([], [], color="none", label=r"$\bf{Preconditioner}$"),
        Line2D([0], [0], color=colors["none"], lw=2, label="none"),
        Line2D([0], [0], color=colors["jacobi"], lw=2, label="jacobi"),
        Line2D([0], [0], color=colors["bjacobi"], lw=2, label="bjacobi"),
        Line2D([], [], color="none", label=""),  # spazio vuoto
        Line2D([], [], color="none", label=r"$\bf{Method}$"),
        Line2D([0], [0], color="dimgray", linestyle="-", marker="o", label="CSD"),
        Line2D([0], [0], color="dimgray", linestyle="--", marker="s", label="FD"),
    ]

    ax.legend(handles=combined_handles, loc="lower left", framealpha=0.9, fontsize=9)

    dts = [0.005, 0.010, 0.015, 0.020]
    plt.xticks(dts, [str(i) for i in dts])
    ytcks = [200 * 2**n for n in  range(5)]
    plt.yticks(ytcks, [str(i) for i in ytcks])
    
    
    
    plt.tight_layout()
    plt.savefig("outputs/wedge_post/times_pcs.pdf", format="pdf")
    plt.savefig("../../tesi_leone_matteo/imgs/wedge/pc_times.pdf", format="pdf")
    plt.show()
    return fig, ax

def post_proc(df_csd, df_fd, df_rk):
    os.makedirs(OUT_DIR, exist_ok=True)
    csd_vs_fd(df_csd, df_fd)
    """
    pp_vs_conservative(df_csd, df_fd)
    preconditioners(df_csd, df_fd)
    rk_summary(df_rk, df_csd, df_fd)
    """


if __name__ == "__main__":
    df = pd.read_excel("outputs/lock_bound_conf.xlsx")

    df_csd = df[ (df["solver"] == "bdf2") & (df["csd_jac"] ==  True ) ]
    df_fd = df[ (df["solver"] == "bdf2") & (df["csd_jac"] == False) ]
    df_rk = df[ (df["solver"] == "rk")  ]

    post_proc(df_csd, df_fd, df_rk)

