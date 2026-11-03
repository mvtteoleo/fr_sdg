"""
radial_ref_utils.py

Shared post-processing utilities for the shock-test quantitative metrics
(sec:shock-metrics). Used by analyze_oscillations.py and
analyze_grid_imprinting.py so the interpolation / shock-detection / metric
logic lives in exactly one place.

No PETSc / solver dependency -- pure numpy/scipy on already-extracted
fields, so it is safe and fast to import from every analysis script.
"""

from pathlib import Path
import json

import numpy as np
from scipy.interpolate import BarycentricInterpolator, RegularGridInterpolator


# ---------------------------------------------------------------------------
# 1D radial reference interpolation
# ---------------------------------------------------------------------------
def build_ref_interpolator(ris_1d: dict):
    """
    Build a callable r -> {"rho": ..., "rhoU": ..., "rhoE": ...} that
    interpolates the (final-time) 1D hyper-resolved radial reference at an
    arbitrary radius r, using the same per-element GLL/Lagrange
    reconstruction as the inline `extract_ref_vals` in check_radial.py's
    __main__ block.

    NOTE -- bug fix relative to the original inline version: that version
    clamped to the r=0 node whenever `r_eq <= sim_data["l_0"] * 0.5`, i.e.
    against the *outer* domain half-width rather than the mesh's own
    minimum radius. Since sim_data["l_0"] is the FULL 2D domain length,
    that condition is true for almost the entire radial range and would
    silently clamp most interior points to the innermost reference value.
    Here we clamp against the mesh's own [min, max] radius instead, which
    is what the comment ("Out of bounds clamping") actually intended.
    """
    inps = ris_1d["inp"]
    n_ele = inps["n_elems"]
    p_deg = inps["poly_deg"]

    mesh_n_p = ris_1d["mesh"].reshape((n_ele, p_deg + 1))
    rho, rhoU, rhoE = ris_1d["data"]
    rho_el = rho[-1, :].reshape((n_ele, p_deg + 1))
    rhoU_el = rhoU[-1, :].reshape((n_ele, p_deg + 1))
    rhoE_el = rhoE[-1, :].reshape((n_ele, p_deg + 1))

    el_boundaries = mesh_n_p[:, 0]
    min_r = mesh_n_p[0, 0]
    max_r = mesh_n_p[-1, -1]

    def extract_ref_vals(r_eq: float) -> dict:
        if r_eq <= min_r:
            return {"rho": rho_el[0, 0], "rhoU": rhoU_el[0, 0], "rhoE": rhoE_el[0, 0]}
        if r_eq >= max_r:
            return {
                "rho": rho_el[-1, -1],
                "rhoU": rhoU_el[-1, -1],
                "rhoE": rhoE_el[-1, -1],
            }

        el_idx = np.searchsorted(el_boundaries, r_eq, side="right") - 1
        el_idx = int(np.clip(el_idx, 0, n_ele - 1))

        if el_idx < n_ele - 1 and abs(r_eq - mesh_n_p[el_idx, -1]) < 1e-10:
            return {
                "rho": 0.5 * (rho_el[el_idx, -1] + rho_el[el_idx + 1, 0]),
                "rhoU": 0.5 * (rhoU_el[el_idx, -1] + rhoU_el[el_idx + 1, 0]),
                "rhoE": 0.5 * (rhoE_el[el_idx, -1] + rhoE_el[el_idx + 1, 0]),
            }

        x_nodes = mesh_n_p[el_idx, :]
        return {
            "rho": float(BarycentricInterpolator(x_nodes, rho_el[el_idx, :])(r_eq)),
            "rhoU": float(BarycentricInterpolator(x_nodes, rhoU_el[el_idx, :])(r_eq)),
            "rhoE": float(BarycentricInterpolator(x_nodes, rhoE_el[el_idx, :])(r_eq)),
        }

    return extract_ref_vals


def ref_rho_array(ris_1d: dict):
    """Return (mesh_x, rho(final time)) flat arrays for the 1D reference."""
    return ris_1d["mesh"], ris_1d["data"][0][-1, :]


# ---------------------------------------------------------------------------
# Shock / contact detection (used for TV/overshoot band + I(r) radii)
# ---------------------------------------------------------------------------
def find_shock_contact_radii(
    mesh_x, rho_ref, n_features: int = 2, min_separation: float = 0.02
):
    """
    Locate the `n_features` points of steepest |d(rho)/dr| along the 1D
    radial reference profile -- for the circular explosion test these are
    the shock and the contact discontinuity. Returns radii sorted
    ascending. `min_separation` (in the same units as mesh_x) prevents
    picking two neighbouring GLL nodes of the same physical discontinuity
    as two separate "features".
    """
    mesh_x = np.asarray(mesh_x)
    rho_ref = np.asarray(rho_ref)
    grad = np.abs(np.gradient(rho_ref, mesh_x))

    order = np.argsort(grad)[::-1]  # steepest first
    picked = []
    for idx in order:
        r = mesh_x[idx]
        if all(abs(r - p) > min_separation for p in picked):
            picked.append(r)
        if len(picked) == n_features:
            break
    return sorted(picked)


def locate_steepest_gradient(x, y):
    """Radius of steepest |dy/dx| in a single 1D profile (used for the
    shock-location error: compare this on the numerical slice vs the
    reference)."""
    x = np.asarray(x)
    y = np.asarray(y)
    grad = np.abs(np.gradient(y, x))
    return float(x[np.argmax(grad)])


# ---------------------------------------------------------------------------
# Total variation / overshoot-undershoot
# ---------------------------------------------------------------------------
def total_variation(y):
    y = np.asarray(y)
    return float(np.sum(np.abs(np.diff(y))))


def overshoot_undershoot(y, rho_out: float, rho_in: float):
    """Max overshoot/undershoot relative to the admissible [rho_out, rho_in]
    range, normalized by the shock jump height (rho_in - rho_out). Returns
    (overshoot_pct, undershoot_pct), both >= 0."""
    y = np.asarray(y)
    jump = rho_in - rho_out
    lo, hi = min(rho_out, rho_in), max(rho_out, rho_in)
    overshoot = max(0.0, float(np.max(y) - hi))
    undershoot = max(0.0, float(lo - np.min(y)))
    return 100.0 * overshoot / jump, 100.0 * undershoot / jump


def tv_oscillation_metrics(
    x_num,
    rho_num,
    ref_interp,
    rho_out,
    rho_in,
    exclude_radii=None,
    exclude_width=None,
):
    """
    Compute TV ratio + overshoot/undershoot + shock-location error for a
    numerical radial slice (x_num, rho_num) against the reference
    interpolator `ref_interp`.

    If `exclude_radii` + `exclude_width` are given, points within
    `exclude_width` of any radius in `exclude_radii` are dropped before
    computing TV / overshoot -- use this to exclude a thin band around the
    shock/contact if you want a metric that isolates *spurious* oscillation
    away from the genuine, physically-discontinuous jump. Left as None by
    default (TV is computed on the full slice), since for this scheme the
    oscillation of interest is often concentrated near the discontinuity
    itself -- adjust based on what you want to isolate.
    """
    x_num = np.asarray(x_num)
    rho_num = np.asarray(rho_num)

    mask = np.ones_like(x_num, dtype=bool)
    if exclude_radii and exclude_width:
        for r in exclude_radii:
            mask &= np.abs(x_num - r) > exclude_width

    rho_ref_on_num = np.array([ref_interp(r)["rho"] for r in x_num])

    tv_num = total_variation(rho_num[mask])
    tv_ref = total_variation(rho_ref_on_num[mask])
    tv_ratio = tv_num / tv_ref if tv_ref > 0 else np.nan

    overshoot_pct, undershoot_pct = overshoot_undershoot(rho_num, rho_out, rho_in)

    shock_loc_num = locate_steepest_gradient(x_num, rho_num)
    shock_loc_ref = locate_steepest_gradient(x_num, rho_ref_on_num)
    shock_loc_error = shock_loc_num - shock_loc_ref

    return dict(
        tv_num=tv_num,
        tv_ref=tv_ref,
        tv_ratio=tv_ratio,
        overshoot_pct=overshoot_pct,
        undershoot_pct=undershoot_pct,
        shock_loc_num=shock_loc_num,
        shock_loc_ref=shock_loc_ref,
        shock_loc_error=shock_loc_error,
    )


# ---------------------------------------------------------------------------
# Angular non-uniformity I(r)
# ---------------------------------------------------------------------------
def build_grid_interpolator(X_glob, Y_glob, field):
    """
    Build a scipy RegularGridInterpolator for a field given on the
    solver's structured (tensor-product) GLL grid. Assumes X_glob, Y_glob
    are 2D meshgrid-style arrays where X_glob[0, :] gives the unique x
    coordinates and Y_glob[:, 0] gives the unique y coordinates -- this is
    the same assumption already used in check_radial.py to extract a
    horizontal slice (`Y_glob[:, 0]` to find `y_half_idx`).
    """
    x_unique = np.asarray(X_glob[0, :])
    y_unique = np.asarray(Y_glob[:, 0])
    # RegularGridInterpolator wants field indexed [y, x] matching (y_unique, x_unique)
    return RegularGridInterpolator(
        (y_unique, x_unique), field, bounds_error=False, fill_value=None
    )


def angular_uniformity(interp, center, r, n_theta: int = 180):
    """
    Sample `interp` on a circle of radius r around `center = (xc, yc)` and
    return (mean, std, I) where I = std/mean is the angular
    coefficient-of-variation non-uniformity index.
    """
    xc, yc = center
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    xs = xc + r * np.cos(theta)
    ys = yc + r * np.sin(theta)
    pts = np.column_stack([ys, xs])  # (y, x) order, matching build_grid_interpolator
    vals = interp(pts)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    I = std / mean if mean != 0 else np.nan
    return mean, std, I


# ---------------------------------------------------------------------------
# Tolerant run-directory lookup
# ---------------------------------------------------------------------------
def find_matching_run_dir(
    circ_dir: Path,
    n_els: int,
    p: int,
    is_implicit: bool,
    target_cfl: float,
    c_max: float,
    l_0: float,
    tol: float = 0.15,
):
    """
    Locate an existing run directory for (n_els, p, scheme) whose stored
    CFL is close to `target_cfl`, WITHOUT assuming the run was originally
    launched with dt computed exactly as CFL*dx/c_max (older runs -- e.g.
    those behind Table 7/8 and Figures 16-18 -- may have used a
    hand-tuned dt). Falls back to the exact CFL-derived path if no
    meta.json is found under the scheme/N/p directory at all.

    Returns (run_dir: Path, actual_cfl: float, meta: dict) or None if
    nothing usable is found.
    """
    n_p_dir = circ_dir / f"n_{n_els}_p_{p}"
    scheme_dir = n_p_dir / ("BFD2" if is_implicit else "SSP3")
    if not scheme_dir.exists():
        return None

    # Two passes: prefer a PASSING run (did_end=True) within tolerance --
    # this matters whenever a Group A/C stability-sweep point happens to
    # land at a CFL close to a Group B/C historical run's (e.g. both near
    # a scheme's ceiling by construction), since the nearest-by-CFL
    # candidate is then not necessarily the one we actually want: TV,
    # overshoot, and I(r) are meaningless on a run that never reached the
    # final time. Only if no passing candidate is within tolerance do we
    # fall back to the closest candidate overall (so the caller can still
    # report "closest run did not reach final time" rather than silently
    # finding nothing).
    dx = l_0 / n_els
    candidates = []
    for dt_dir in scheme_dir.iterdir():
        meta_fl = dt_dir / "meta.json"
        if not meta_fl.exists():
            continue
        try:
            with open(meta_fl) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        dt = meta.get("dt")
        if dt is None:
            continue
        actual_cfl = dt * c_max / dx
        rel_err = abs(actual_cfl - target_cfl) / target_cfl
        candidates.append((rel_err, dt_dir, actual_cfl, meta))

    passing = [c for c in candidates if c[3].get("did_end")]
    best = min(passing, key=lambda c: c[0]) if passing else None
    if best is None or best[0] > tol:
        best = min(candidates, key=lambda c: c[0]) if candidates else None

    if best is None or best[0] > tol:
        return None

    _, run_dir, actual_cfl, meta = best
    return run_dir, actual_cfl, meta
