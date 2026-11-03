"""
Plot a single wedge run's final-time solution.

Usage:
    python plot_wedge.py --dir build/wedge/bdf2/n_50_p_1_dt_0.01
    python plot_wedge.py --dir build/wedge/rk/n_50_p_1_dt_0.0002 --show

Reads meta.json (written by wedge_runner.py) for the run parameters and
solution.bin path, so nothing needs to be re-specified on the command line
besides which run directory to look at.
"""

from __future__ import annotations

from exact_wedge import get_wedge_sol

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Adjust import to match your project directory tree
from check_iso_vortex import extract_sol

GAMMA = 1.4


# ====================================================================
# GEOMETRY MAPPING -- must stay an EXACT match to the C++ GeometryWedge
# used to produce the run. Defaults match the parabolic-to-linear ramp
# from the original curved_wedge script; override via CLI if you change
# the profile on the C++ side.
# ====================================================================
def phys_coords(
    csi: np.ndarray,
    eta: np.ndarray,
    l0: float,
    ly: float,
    x_start: float,
    x_straight: float,
    wedge_slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = l0 * csi
    A = wedge_slope / (2.0 * (x_straight - x_start))

    h_x = np.zeros_like(x)

    mask_parabola = (x > x_start) & (x <= x_straight)
    dx_p = x[mask_parabola] - x_start
    h_x[mask_parabola] = A * dx_p**2

    mask_linear = x > x_straight
    dx_trans = x_straight - x_start
    h_base = A * dx_trans**2
    dx_l = x[mask_linear] - x_straight
    h_x[mask_linear] = h_base + wedge_slope * dx_l

    y = h_x * (1.0 - eta) + ly * eta
    return x, y


def load_run(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No meta.json in {run_dir} -- point --dir at a run directory"
        )

    meta = json.loads(meta_path.read_text())
    if not meta.get("solution_path"):
        raise RuntimeError(
            f"{run_dir}: run has no saved solution (success={meta.get('success')})"
        )

    return meta


def plot_run(
    run_dir: Path,
    l0: float | None,
    ly: float,
    x_start: float | None,
    x_straight: float | None,
    wedge_slope: float,
    out_dir: Path | None,
    show: bool,
) -> None:
    meta = load_run(run_dir)
    n_elems, poly_deg = meta["n_elems"], meta["poly_deg"]
    l0 = l0 if l0 is not None else meta["box_l"]
    x_start = x_start if x_start is not None else 0.3 * l0
    x_straight = x_straight if x_straight is not None else 0.5 * l0

    print(f"Loading solution: {meta['solution_path']}")
    csi_glob, eta_glob, rho, rhou, rhov, rhoE = extract_sol(
        meta["solution_path"],
        n_elems,
        poly_deg,
        l0,
        CHANGE_VAR=False,
    )

    X, Y = phys_coords(csi_glob, eta_glob, l0, ly, x_start, x_straight, wedge_slope)

    u = rhou / rho
    v = rhov / rho
    pressure = (GAMMA - 1.0) * (rhoE - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(GAMMA * pressure / rho)
    mach = np.sqrt(u**2 + v**2) / c

    title_tag = (
        f"{meta['solver']}  N={n_elems} p={poly_deg} dt={meta['dt']:g} "
        f"t_max={meta['t_max']:g}  ({meta.get('wall_time_sec', 0):.1f}s)"
    )

    out_dir = out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── PLOT 1: 2D Mach contour with wedge wall traced in black ──
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    contour = ax1.contourf(X, Y, mach)
    ax1.plot(X[0, :], Y[0, :], color="black", linewidth=3, label="Wedge Wall")

    cbar = fig1.colorbar(contour, ax=ax1, orientation="horizontal", pad=0.15)
    cbar.set_label("Local Mach Number ($M$)")

    ax1.set_title(f"Supersonic Flow Over the Wedge\n{title_tag}")
    ax1.set_xlabel("Axial Location ($x$)")
    ax1.set_ylabel("Transverse Height ($y$)")
    ax1.axis("equal")
    ax1.legend(loc="upper left")
    fig1.tight_layout()
    fig1.savefig(out_dir / "mach_contour.png", dpi=150)

    # ── PLOT 2: density + Mach along the wedge surface (eta = 0) ──
    fig2, ax_rho = plt.subplots(figsize=(10, 5))
    ax_rho.set_xlabel("Axial Coordinate ($x$)")
    ax_rho.set_ylabel(r"Density ($\rho$)", color="tab:blue")

    moc = get_wedge_sol()

    ax_rho.plot(moc["mesh"], moc["rho"], "k-", lw=1.5, label="MOC density")

    ax_rho.plot(X[0, :], rho[0, :], color="tab:blue", lw=2, label="Density")
    ax_rho.tick_params(axis="y", labelcolor="tab:blue")

    ax_mach = ax_rho.twinx()
    ax_mach.set_ylabel("Mach Number ($M$)", color="tab:red")

    ax_mach.plot(moc["mesh"], moc["mach"], "k-", lw=1.5, label="MOC Mach")
    ax_mach.plot(
        X[0, :], mach[0, :], color="tab:red", lw=2, linestyle="--", label="Mach"
    )
    ax_mach.axvline(
        x=x_start, color="gray", linestyle=":", alpha=0.7, label="Ramp Start"
    )
    ax_mach.tick_params(axis="y", labelcolor="tab:red")

    ax_rho.set_title(f"Wedge Surface State\n{title_tag}")
    fig2.tight_layout()
    fig2.savefig(out_dir / "wall_state.png", dpi=150)

    print(f"Saved figures to {out_dir}")
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Run directory containing meta.json + solution.bin",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to save PNGs (default: same as --dir)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open interactive windows"
    )
    parser.add_argument(
        "--l0",
        type=float,
        default=None,
        help="Override L_x (default: box_l from meta.json)",
    )
    parser.add_argument(
        "--ly", type=float, default=1.0, help="L_y (must match main.cpp)"
    )
    parser.add_argument(
        "--x-start",
        type=float,
        default=None,
        help="Ramp parabola start (default: 0.3*l0)",
    )
    parser.add_argument(
        "--x-straight",
        type=float,
        default=None,
        help="Where the ramp becomes linear (default: 0.5*l0)",
    )
    parser.add_argument("--wedge-slope", type=float, default=0.2)
    args = parser.parse_args()

    plot_run(
        args.dir,
        args.l0,
        args.ly,
        args.x_start,
        args.x_straight,
        args.wedge_slope,
        args.out,
        args.show,
    )


if __name__ == "__main__":
    main()
