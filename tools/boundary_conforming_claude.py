"""
Wedge problem run harness: implicit (BDF2) vs explicit (RK) time-stepping.

Each run lives under:
    build/wedge/<solver>/n_<n_elems>_p_<poly_deg>_dt_<dt>/
        run.log        -- full solver stdout+stderr
        solution.bin    -- PETSc binary Vec, FINAL time step only
        meta.json       -- params, wall-clock time, convergence diagnostics

`-ts_type pseudo` is never used -- both configs march real physical time,
which is what you want to compare for steady-state-reaching behavior.

Adjust imports to match your project's directory tree.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import json
import re
import subprocess
from sys import exception
import time
import hashlib  
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from check_iso_vortex import extract_sol

EXECUTABLE = "./build/test/boundary_conforming"  # point at your wedge binary
RUN_ROOT = Path("build/wedge")

SolverTag = Literal["bdf2", "rk"]


# ====================================================================
# RUN CONFIGURATION
# ====================================================================

@dataclass
class RunConfig:
    solver: SolverTag
    n_elems: int
    poly_deg: int
    dt: float
    t_max: float
    box_l: float = 1.0
    n_ranks: int = 6
    weno_gamma: float = 0.998
    apply_weno: bool = True
    apply_filter: bool = False

    solve_pp: bool = False
    pc_type: str = "none"
    csd_jac: bool = True

    ramp_cfl: bool = False
    ts_pseudo_increment: float = 1.1

   # --- SNES (Nonlinear Solver) Tolerances (PETSc Defaults) ---
    snes_rtol: float = 1e-08
    snes_atol: float = 1e-50
    snes_stol: float = 1e-08
    snes_max_it: int = 40

    # --- KSP (Linear Solver) Tolerances (PETSc Defaults) ---
    ksp_rtol: float = 1e-05
    ksp_atol: float = 1e-50
    ksp_dtol: float = 10000.0  # Divergence tolerance
    ksp_max_it: int = 10000

    hard_fail : bool = True

    @property
    def _param_signature(self) -> str:
        d = asdict(self)
        d.pop("n_ranks", None)
        d.pop("t_max", None)
        return "_".join(f"{k}={v}" for k, v in sorted(d.items()))

    @property
    def tag(self) -> str:
        sig_hash = hashlib.sha1(self._param_signature.encode()).hexdigest()[:8]
        return f"{self.solver}_n{self.n_elems}_p{self.poly_deg}_{sig_hash}"

    @property
    def run_dir(self) -> Path:
        return RUN_ROOT / self.tag

    def _tolerance_args(self) -> list[str]:
        """Emits explicit tolerance flags for every run to ensure full solver control."""
        return [
            # SNES (Nonlinear)
            "-snes_rtol", str(self.snes_rtol),
            "-snes_atol", str(self.snes_atol),
            "-snes_stol", str(self.snes_stol),
            "-snes_max_it", str(self.snes_max_it),
            
            # KSP (Linear)
            "-ksp_rtol", str(self.ksp_rtol),
            "-ksp_atol", str(self.ksp_atol),
            "-ksp_divtol", str(self.ksp_dtol),
            "-ksp_max_it", str(self.ksp_max_it),
        ]

    def _time_stepping_args(self) -> list[str]:
        if self.ramp_cfl:
            return [
                "-ts_type", "pseudo",
                "-ts_pseudo_increment", str(self.ts_pseudo_increment),
                "-ts_pseudo_increment_dt_from_initial_dt",
                "-ts_pseudo_fatol", "0",
                "-ts_pseudo_frtol", "0",
                "-ts_dt", str(self.dt),
                "-ts_max_time", str(self.t_max),
                "-ts_exact_final_time", "matchstep",
            ]
        return [
            "-ts_dt", str(self.dt),
            "-ts_max_time", str(self.t_max),
            "-ts_exact_final_time", "stepover",
        ]

    def _solver_args(self) -> list[str]:
        out = []
        out += ["-solve_positive_preserving", "true" if self.solve_pp else "false"]
        out += ["-pc_type", str(self.pc_type)]

        if self.csd_jac is True:
            out += ["-jac_type", "complexfd"]
        else:
            out += ["-jac_type", "fd"]

        if self.solver == "bdf2":
            out += [
                "-ts_type", "bdf", "-ts_bdf_order", "2",
                "-snes_monitor", "-ksp_monitor",
                "-snes_converged_reason", "-ksp_converged_reason",
            ]
        elif self.solver == "rk":
            out += ["-ts_type", "rk", "-ts_rk_type", "3bs"]
        else:
            raise ValueError(f"Unknown solver tag: {self.solver!r}")

        return out

    def command(self, sol_path: Path) -> list[str]:
        command = [
            "mpirun", "-n", str(self.n_ranks), EXECUTABLE,
            "-n_elems", str(self.n_elems),
            "-poly_deg", str(self.poly_deg),
            "-box_l", str(self.box_l),]

        if self.hard_fail:
            command += ["-ts_adapt_type", "none",
            "-ts_max_snes_failures", "1",
            "-ts_max_reject", "0",
            "-ts_error_if_step_fails", "true",]

        command += [
            "-apply_filter", "true" if self.apply_filter else "false",
            "-apply_weno", "true" if self.apply_weno else "false",
            "-weno_gamma", str(self.weno_gamma),
            "-fp_trap", "off",
            "-ts_monitor", "-ts_monitor_solution", f"binary:{sol_path}",
            "-ts_monitor_solution_interval", "-1",
            "-ts_monitor_solution_skip_initial",
            *self._solver_args(),
            *self._time_stepping_args(),  # appended LAST -- see note below
            "-snes_monitor", 
            "-ksp_monitor", 
            "-ts_monitor",
            "-snes_view", 
            "-ksp_view", 
            "-ts_view",
        ]

        return command

# ====================================================================
# LOG PARSING (best-effort -- adjust the regexes to your actual log format)
# ====================================================================
_SNES_NORM_RE = re.compile(r"^\s*\d+\s+SNES Function norm\s+([\d.eE+-]+)", re.MULTILINE)
_KSP_ITS_RE = re.compile(r"^\s*(\d+)\s+KSP Residual norm", re.MULTILINE)
_CONVERGED_RE = re.compile(r"(Nonlinear|Linear) solve converged due to (\w+)")
_DIVERGED_RE = re.compile(r"DIVERGED_\w+")


def _parse_log(log_text: str) -> dict:
    snes_norms = [float(x) for x in _SNES_NORM_RE.findall(log_text)]
    ksp_its = _KSP_ITS_RE.findall(log_text)
    return {
        "diverged": bool(_DIVERGED_RE.search(log_text)),
        "n_snes_evals": len(snes_norms),
        "final_snes_residual": snes_norms[-1] if snes_norms else None,
        "total_ksp_its": int(ksp_its[-1]) if ksp_its else None,
        "converged_reasons": [f"{k}: {v}" for k, v in _CONVERGED_RE.findall(log_text)][
            -5:
        ],
    }

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def prandtl_meyer(M, gamma=1.4):
    """Calculates the Prandtl-Meyer angle (in radians) for a given Mach number."""
    term1 = np.sqrt((gamma + 1.0) / (gamma - 1.0))
    term2 = np.arctan(np.sqrt((gamma - 1.0) / (gamma + 1.0) * (M**2 - 1.0)))
    term3 = np.arctan(np.sqrt(M**2 - 1.0))
    return term1 * term2 - term3

def inv_prandtl_meyer(nu_target, gamma=1.4, M_guess=2.0):
    """Inverts the Prandtl-Meyer function to find Mach number given a PM angle (in rad)."""
    func = lambda M: prandtl_meyer(M, gamma) - nu_target
    return fsolve(func, M_guess)[0]

def get_exp_wall_slope(x, l_0=2.0):
    """Returns dy/dx of the bottom wall for the EXP_FAN case."""
    x_start = 0.3 * l_0
    x_straight = 0.5 * l_0
    wedge_slope = 0.2
    A = wedge_slope / (2.0 * (x_straight - x_start))
    
    if x <= x_start:
        return 0.0
    elif x <= x_straight:
        return 2.0 * A * (x - x_start)
    else:
        return wedge_slope
def load_sol(dir, meta):
    l_0 = 1.0
    n_elems, poly_deg = meta["n_elems"], meta["poly_deg"]
    csi_glob, eta_glob, rho_num, rhou_num, rhov_num, rhoE_num = extract_sol(
        Path(dir, "solution.bin"), meta["n_elems"], meta["poly_deg"], l_0, meta["solve_pp"])

    Y_glob = np.zeros_like(csi_glob)
    X_glob = np.zeros_like(eta_glob)

    # -------------------------------------------------------------
    # EXACT MATCH TO C++ GEOMETRY MAPPING
    # -------------------------------------------------------------
    def wedge_test_phys_coord(csi, eta):
        x = l_0 * csi
        x_start = 0.3 * l_0
        x_straight = 0.5 * l_0
        wedge_slope = 0.2
        A = wedge_slope / (2.0 * (x_straight - x_start))

        if x <= x_start:
            h_x = 0.0
        elif x <= x_straight:
            dx = x - x_start
            h_x = A * dx * dx
        else:
            dx_trans = x_straight - x_start
            h_base = A * dx_trans * dx_trans
            dx = x - x_straight
            h_x = h_base + wedge_slope * dx

        y = h_x * (1.0 - eta) + l_0 * eta
        return x, y

    for i in range(n_elems * (poly_deg + 1)):
        for j in range(n_elems * (poly_deg + 1)):
            csi = csi_glob[i, j]
            eta = eta_glob[i, j]
            x, y = wedge_test_phys_coord(csi, eta)
            Y_glob[i, j] = y
            X_glob[i, j] = x
    return X_glob, Y_glob, rho_num, rhou_num, rhov_num, rhoE_num


def moc_wall_profile(x_wall, M_inf, rho_inf, gamma=1.4, l_0=1.0):
    """MOC (simple-wave) rho and Mach at the bottom wall for the EXP_FAN case."""
    nu_inf = prandtl_meyer(M_inf, gamma)
    stag_inf = 1.0 + 0.5 * (gamma - 1.0) * M_inf**2

    mach_ref = np.empty_like(x_wall)
    rho_ref = np.empty_like(x_wall)

    M_guess = M_inf
    for k, x in enumerate(x_wall):
        theta = np.arctan(get_exp_wall_slope(x, l_0))
        nu_wall = nu_inf - theta
        M_wall = inv_prandtl_meyer(nu_wall, gamma, M_guess)
        M_guess = M_wall  # warm start: Mach increases monotonically along the wall

        stag_wall = 1.0 + 0.5 * (gamma - 1.0) * M_wall**2
        mach_ref[k] = M_wall
        rho_ref[k] = rho_inf * (stag_inf / stag_wall) ** (1.0 / (gamma - 1.0))

    return rho_ref, mach_ref

def ref_vs_vals(x_s, rho_vals, mach_vals, dir, title=None):
    if title is None:
        title = "Fluid State Distribution Along the Wedge Surface"

    rho_ref, mach_ref = moc_wall_profile(x_s, M_inf=1.69,
                                         rho_inf=1.0, gamma=1.4,
                                         l_0=1.0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: Density tracking
    color = "tab:blue"
    ax1.set_xlabel("Axial Coordinate ($x$)")
    ax1.set_ylabel("Density ($\\rho$)", color=color)
    ax1.plot(x_s, rho_ref, color="black", lw=1, alpha=0.7, label="MOC Density")
    ax1.plot(x_s, rho_vals, color=color, lw=2, label="Density")
    ax1.tick_params(axis="y", labelcolor=color)

    # Right axis: Mach number tracking
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Mach Number ($M$)", color=color)
    ax2.plot(x_s, mach_ref,  color="black",lw=1, alpha=0.7, label="MOC Mach")
    ax2.plot(
        x_s,
        mach_vals,
        color=color,
        lw=2,
        linestyle="--",
        label="Mach",
    )

    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(Path(dir,"mach_dens.pdf"), format="pdf")

def save_plots(out_dir, meta, m_min=1.25, m_max=1.70):
    X, Y, rho, rhoU, rhoV, rhoE = load_sol(out_dir, meta)

    gamma = 1.4
    u = rhoU / rho
    v = rhoV / rho
    vel_mag = np.sqrt(u**2 + v**2)

    pressure = (gamma - 1.0) * (rhoE - 0.5 * rho * vel_mag**2)
    c = np.sqrt(gamma * pressure / rho)
    mach = vel_mag / c

    # ====================================================================
    # PLOT 1: 2D CURVILINEAR FLOW FIELD CONTOUR
    # ====================================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    # 1. Fixed, identical levels across both BDF2 and RK runs
    levels = np.linspace(m_min, m_max, 61)

    # 2. extend="both" ensures values below m_min or above m_max are colored
    #    with the under/over colormap values rather than left blank
    contour = ax.contourf(
        X, Y, mach,
        levels=levels,
        cmap="viridis",
        vmin=m_min,
        vmax=m_max,
        extend="both"
    )

    # Bottom wall contour (eta = 0 line)
    bottom_wall_x = X[0, :]
    bottom_wall_y = Y[0, :]
    ax.plot(
        bottom_wall_x,
        bottom_wall_y,
        color="black",
        linewidth=2.2,
        label="Wedge Wall",
    )

    # Tighten bounding box
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"Axial Coordinate $x$")
    ax.set_ylabel(r"Transverse Coordinate $y$")
    ax.set_title("2D Mach Field Over Curved Wedge", pad=10)

    # 3. Clean colorbar with explicit, synchronized tick marks
    cbar = fig.colorbar(
        contour,
        ax=ax,
        # orientation="horizontal",
         # pad=0.10,
         fraction=0.06,
         aspect=28,
        ticks=np.linspace(m_min, m_max, 10),
    )
    cbar.set_label(r"Local Mach Number $M$")
    cbar.ax.tick_params(labelsize=9)

    ax.legend(loc="upper left", framealpha=0.85)

    plt.savefig(
        Path(out_dir, "mach_cont.pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)

    # Profile plot
    lowest = (X[0, :].ravel(), rho[0, :].ravel(), mach[0, :].ravel())
    ref_vs_vals(*lowest, dir=out_dir)


# ====================================================================
# SINGLE-RUN EXECUTION
# ====================================================================
def run_case(cfg: RunConfig, overwrite: bool = False) -> dict:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    sol_path = cfg.run_dir / "solution.bin"
    log_path = cfg.run_dir / "run.log"
    meta_path = cfg.run_dir / "meta.json"

    print("= "*30)

    if meta_path.exists() and not overwrite:
        print("Running cfg: ")
        print(cfg)
        meta = json.loads(meta_path.read_text())
        if sol_path.exists() and sol_path.stat().st_size > 0:
            try:  
                save_plots(cfg.run_dir, meta)
            except Exception as e:
                print(f"[skip] {cfg.tag} (meta.json exists, but solution.bin is unreadable/corrupted: {e}) || No plot available")
        else:
            print(f"[skip] {cfg.tag} (meta.json exists, overwrite=True to redo)")
        return meta

    if sol_path.exists():
        sol_path.unlink()

    cmd_list = cfg.command(sol_path)
    cmd_str = " ".join(cmd_list)

    full_cmd = f"{cmd_str} > {log_path} 2>&1"
    print(full_cmd )
    print(f"[run ] {cfg.tag}")

    print("Running cfg: ")
    print(cfg)

    t0 = time.perf_counter()
    subprocess.run(full_cmd, shell=True, text=True)
    wall_time_sec = time.perf_counter() - t0

    # Read the text back from the file to feed your regex parser
    log_text = log_path.read_text()

    diagnostics = _parse_log(log_text)
    success = ( sol_path.exists() and not diagnostics["diverged"])

    meta = {
        **asdict(cfg),
        "command" : full_cmd,
        "wall_time_sec": wall_time_sec,
        "success": success,
        "solution_path": str(sol_path) if sol_path.exists() else None,
        "log_path": str(log_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **diagnostics,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[done] {cfg.tag}: {'OK' if success else 'FAILED'} in {wall_time_sec:.1f}s")

    if success:
        try:
            save_plots(cfg.run_dir, meta)
        except Exception as e:
            print(f"[skip] {cfg.tag} (meta.json exists, but solution.bin is unreadable/corrupted: {e}) || No plot available")
        

    return meta


def run_sweep(configs: list[RunConfig], overwrite: bool = False) -> pd.DataFrame:
    return pd.DataFrame([run_case(cfg, overwrite=overwrite) for cfg in configs])


def load_database(root: Path = RUN_ROOT) -> pd.DataFrame:
    """Rebuild the comparison table from meta.json files already on disk --
    no re-running needed."""
    rows = [json.loads(p.read_text()) for p in root.rglob("meta.json")]
    return pd.DataFrame(rows)

def build_jacobian_study_configs(
    n_elems: int = 80, t_max: float = 2.0
) -> list[RunConfig]:
    """
    Study: finite-difference vs. complex-step Jacobian, at the wedge's
    working config. The shock test's own FD/CSD comparison ran at loose
    tolerances where the two were indistinguishable; this compares them
    directly where gradients are steepest (near the shock), where FD's
    cancellation error would matter most if it matters at all.

    Primary point: p=1. Secondary point: p=2.
    Solver fixed to bdf2 -- Jacobian strategy only applies to the implicit
    solve; RK has no SNES/KSP step to vary.
    """
    configs = []
    for poly_deg in [1]:
        for csd_jac in [ False ]:
            configs.append(
                RunConfig(
                    solver="bdf2",
                    n_elems=n_elems,
                    poly_deg=poly_deg,
                    dt=1e-2,
                    t_max=t_max,
                    csd_jac=csd_jac,
                )
            )
    return configs


def build_variable_formulation_study_configs(
    n_elems: int = 80, t_max: float = 2.0
) -> list[RunConfig]:
    """
    Study: positive-preserving (log) vs. conservative variables, at the
    known-working baseline IC.

    Primary point: p=1. Secondary point: p=2.
    Solver fixed to bdf2 -- this question is specifically about the
    implicit Newton iterate's behavior; RK has no such iterate to stress.
    """
    configs = []
    for poly_deg in [ 1 ]:
        for solve_pp in [ True ]:
            configs.append(
                RunConfig(
                    solver="bdf2",
                    n_elems=n_elems,
                    poly_deg=poly_deg,
                    dt=1e-2,
                    csd_jac=False,
                    t_max=t_max,
                    solve_pp=solve_pp,
                )
            )
    return configs


def build_preconditioner_study_configs(
    n_elems: int = 80,
    t_max: float = 2.0,
    cfl_ceiling_dt: float = 1e-2,
) -> list[RunConfig]:
    """
    Study: preconditioner on/off, and choice of preconditioner, crossed
    against CFL fraction -- tests whether a preconditioner's benefit grows
    as the linear system stiffens near the stability ceiling, not just
    whether it helps at all.

    `cfl_ceiling_dt` should be the largest known-stable dt for this
    (n_elems, poly_deg) -- pass the actual measured ceiling, not the
    1e-2 default.

    Primary point: p=1. Secondary point: p=2.
    Solver fixed to bdf2 -- preconditioning is specific to the implicit
    linear solve; RK has no linear system to precondition.
    """
    pc_types = ["none", "jacobi", "bjacobi", ]
    cfl_fractions = [1.00, 0.5, 1.50, 2.0]

    configs = []
    for cd in [False, True]:
      for p in [1]: #, 2, 3]:
        for pc_type in pc_types:
              hf = True
            # for hf in (False, True):

              for frac in cfl_fractions:
                # frac = 1.0
                configs.append(
                    RunConfig(
                        hard_fail = hf,
                        solver="bdf2",
                        n_elems=n_elems,
                        poly_deg=p,
                        dt=cfl_ceiling_dt * frac,
                        csd_jac=cd,
                        t_max=t_max,
                        pc_type=pc_type,
                    )
                )
    return configs


def build_closing_confirmation_configs(
    best_solve_pp: bool,
    best_pc_type: str,
    best_csd_jac: bool,
    t_max: float = 2.0,
) -> list[RunConfig]:
    """
    Closing run: the single best configuration found across the three
    studies above, re-run at the baseline resolution and at one higher
    resolution, to check the result survives scaling up and ties back
    into the shock-test chapter's resolution-scaling story.
    """
    configs = []
    for n_elems in (80, 150):
        configs.append(
            RunConfig(
                solver="bdf2",
                n_elems=n_elems,
                poly_deg=1,
                dt=1e-2,
                t_max=t_max,
                solve_pp=best_solve_pp,
                pc_type=best_pc_type,
                csd_jac=best_csd_jac,
            )
        )
    return configs

def get_init_configs():
    n_elems = 80
    t_max = 2.0
    loose = dict(
        snes_rtol=1e-08, snes_atol=1e-12, snes_stol=1e-08, snes_max_it=40,
        ksp_rtol=1e-05, ksp_atol=1e-50, ksp_dtol=10000.0, ksp_max_it=10000
    )
    configs = []
    for poly_deg in [1, 2]:
        # Impicit run
        for hf  in (False, True):
            configs.append(RunConfig(
                solver="bdf2", n_elems=n_elems, poly_deg=poly_deg, 
                hard_fail=hf,
                dt=1e-2, t_max=t_max, **loose
            ))
        
        # Explicit run
        configs.append(RunConfig(
            solver="rk", n_elems=n_elems, poly_deg=poly_deg, 
            dt=1e-4, t_max=t_max
        ))

    return configs

def all_confs():
    n_elems = 80
    t_max = 2.0
    loose = dict(
        snes_rtol=1e-08, snes_atol=1e-12, snes_stol=1e-08, snes_max_it=40,
        ksp_rtol=1e-05, ksp_atol=1e-50, ksp_dtol=10000.0, ksp_max_it=10000
    )
    configs = []
    pc_types = ["none", "jacobi", "bjacobi", "ilu"]
    for pc in pc_types:
     for csd_jac in (True, False):
      for solve_pp in (True, False):
       # Impicit run
       for hf  in (False, True):
        configs.append(RunConfig(
                solver="bdf2", n_elems=n_elems, poly_deg=poly_deg, 
                hard_fail=hf,
                solve_pp=solve_pp,
                csd_jac=csd_jac,
                pc_type=pc,
                dt=1e-2, t_max=t_max, **loose
            ))
        
    return configs

def rerun_confs(n_elems: int = 80, t_max: float = 2.0) -> list[RunConfig]:
    """Targeted rerun suite to fix data anomalies and complete the thesis comparison:
    
    1. Fixes the machine-load spike of dt=0.010 for matrix-free (-snes_mf).
    2. Re-runs the failed 1.62s CSD + none abort at dt=0.010 (Row 12).
    3. Sweeps dt in [0.005, 0.010, 0.015, 0.020] for p=2 and p=3 under -snes_mf.
    4. Omits raw parallel ILU (unsupported in PETSc across 6 MPI ranks).
    """
    configs: list[RunConfig] = []
    dt_sweep = [0.005, 0.010, 0.015, 0.020]

    # --- 1. RERUN CORRUPTED/ANOMALOUS CSD RUNS (p=1, hard_fail=True) ---
    # Replaces Row 12 (1.62s abort) to get the true ~1000s baseline
    configs.append(
        RunConfig(
            solver="bdf2",
            n_elems=n_elems,
            poly_deg=1,
            dt=0.010,
            t_max=t_max,
            csd_jac=True,
            pc_type="none",
            hard_fail=True,
        )
    )

    # --- 2. FAST MATRIX-FREE (SNES_MF) SWEEPS ACROSS p=1, 2, 3 ---
    # At p=1: re-runs dt=0.010 to resolve the 251s load spike and confirm the win over RK (193s)
    # At p=2, 3: completes the missing dt grid to compare against explicit p=2 (392s)
    for p in [1, 2, 3]:
        for dt_val in dt_sweep:
            # We enforce hard_fail=True for p=1,2; for p=3 we also include hard_fail=False
            # because p=3 requires adaptive step-back to survive the shock transient
            hf_options = [True] if p < 3 else [True, False]
            for hf in hf_options:
                configs.append(
                    RunConfig(
                        solver="bdf2",
                        n_elems=n_elems,
                        poly_deg=p,
                        dt=dt_val,
                        t_max=t_max,
                        csd_jac=False,  # -snes_mf
                        pc_type="none",
                        hard_fail=hf,
                    )
                )

    return configs

def run_tests() :
    # 

    all_configs = ( get_init_configs() + rerun_confs() 
                     + build_jacobian_study_configs()
                     + build_variable_formulation_study_configs()  
                     + build_preconditioner_study_configs()
                    )
                  
    
    base = RunConfig(solver="bdf2", n_elems=80, poly_deg = 1, dt=1e-2, t_max=2.0)
    # [base ]
    df = run_sweep(build_preconditioner_study_configs(), overwrite=True)  
    return df


def extract_all_runs(
    root_dir: str | Path = "build/wedge"
) -> pd.DataFrame:
    """Recursively walks `root_dir` for all `meta.json` files, extracts run data,
    fills missing keys with np.nan, returns a DataFrame matching the harness output,
    and optionally writes it to an Excel spreadsheet.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"[warning] Directory {root_path} does not exist.")
        return pd.DataFrame()

    records = []
    # Find all meta.json files recursively
    for meta_file in sorted(root_path.rglob("meta.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Ensure None values in the JSON are converted to np.nan
            cleaned = {k: (np.nan if v is None else v) for k, v in data.items()}
            records.append(cleaned)
        except Exception as e:
            print(f"[warning] Could not read {meta_file}: {e}")

    if not records:
        print(f"[warning] No meta.json files found under {root_path}.")
        return pd.DataFrame()

    # Define standard column ordering matching the harness output
    expected_cols = [
        "solver", "n_elems", "poly_deg", "dt", "t_max", "box_l", "n_ranks",
        "weno_gamma", "apply_weno", "apply_filter", "solve_pp", "pc_type",
        "csd_jac", "ramp_cfl", "ts_pseudo_increment", "snes_rtol", "snes_atol",
        "snes_stol", "snes_max_it", "ksp_rtol", "ksp_atol", "ksp_dtol",
        "ksp_max_it", "hard_fail", "command", "wall_time_sec", "success",
        "solution_path", "log_path", "timestamp", "diverged", "n_snes_evals",
        "final_snes_residual", "total_ksp_its", "converged_reasons"
    ]

    df = pd.DataFrame(records)

    # Fill any missing expected columns with np.nan
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Place expected columns first, followed by any extra diagnostic fields
    ordered_cols = expected_cols + [c for c in df.columns if c not in expected_cols]
    df = df[ordered_cols]

    # Replace remaining None/NA objects with np.nan
    df = df.fillna(value=np.nan)

    return df

def write_filter_excel( df , excel_file) :

    bdf2_cond = ( (df["solver"] == "bdf2") & (df["success"] == True) & (df["final_snes_residual"] < 1.0))
    rk_cond = df["solver"] == "rk"

    df_filt = ( df[bdf2_cond | rk_cond] .sort_values(by="wall_time_sec", ascending=True))

    df_filt.to_excel(excel_file)


if __name__ == "__main__":
    # run_tests()
    df = extract_all_runs("build/wedge")

    df.to_excel("outputs/no_filt_bound_conf.xlsx")
    write_filter_excel(df, "outputs/bound_conf.xlsx")

