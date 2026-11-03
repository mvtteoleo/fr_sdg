"""
Definition of the shock-test post-processing campaign (see Section
"Quantitative shock-test metrics" of the thesis).

This module ONLY defines *what* to run -- it has no side effects and does
not import PETSc/solver bindings, so it is safe to import from both the
launcher (run_test_matrix.py) and the analysis scripts
(analyze_stability_map.py, analyze_oscillations.py,
analyze_grid_imprinting.py) without duplicating the matrix in four places.

Groups:
  A  - Stability sweep (Claim 1: "more stable"). Needs only pass/fail per
       (scheme, p, N, CFL); the field itself is NOT extracted to keep these
       runs cheap.
  B  - Oscillation metrics (Claim 2: "less oscillating"). No new runs --
       reuses the ceiling runs already on disk from the CFL-ceiling
       campaign (Table "cfl-ceiling-shock"). Listed here anyway so the
       matrix is a single source of truth for the whole campaign.
  C  - Grid imprinting (Claim 3: "less grid imprinting"). C1-C3 reuse
       existing runs; C4 is the one genuinely new run, sized to sit at the
       ~1M DOF budget so it gives the best available p=3 imprinting
       comparison.

DOF accounting (2D, 4 conserved variables): DOFs ~= 4 * N^2 * (p+1)^2
"""

from dataclasses import dataclass, field
from typing import List, Optional


DOF_BUDGET = 1_000_000


def estimate_dofs(n_els: int, poly_deg: int, n_conserved: int = 4) -> int:
    return n_conserved * n_els**2 * (poly_deg + 1) ** 2


@dataclass
class MatrixEntry:
    group: str  # "A1", "A2", ..., "C4"
    purpose: str  # short human label, goes straight into logs/tables
    is_implicit: bool
    poly_deg: int
    n_els: int
    cfl_values: List[
        float
    ]  # one run per value; last one may be the "ceiling" candidate
    extract_sol: bool = (
        False  # False: pass/fail only (Group A). True: keep the field (Group C).
    )
    new_run: bool = True  # False for Group B / C1-C3: already on disk, just reload.

    @property
    def scheme_name(self) -> str:
        return "implicit" if self.is_implicit else "explicit"

    def dof_estimate(self) -> int:
        return estimate_dofs(self.n_els, self.poly_deg)


# ---------------------------------------------------------------------------
# Group A -- stability sweep. CFL lists bracket the ceilings already found in
# tab:cfl-ceiling-shock (0.079/0.023 explicit, 1.178/0.348/0.338 implicit at
# N=100), plus one point past each ceiling to make sure the "fail" side of
# the map is populated too.
# ---------------------------------------------------------------------------
GROUP_A: List[MatrixEntry] = [
    MatrixEntry(
        group="A1",
        purpose="stability sweep",
        is_implicit=False,
        poly_deg=1,
        n_els=100,
        cfl_values=[0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
        extract_sol=False,
    ),
    MatrixEntry(
        group="A2",
        purpose="stability sweep",
        is_implicit=False,
        poly_deg=2,
        n_els=100,
        cfl_values=[0.005, 0.01, 0.015, 0.02, 0.023, 0.03],
        extract_sol=False,
    ),
    MatrixEntry(
        group="A3",
        purpose="stability sweep",
        is_implicit=True,
        poly_deg=1,
        n_els=100,
        cfl_values=[0.3, 0.6, 1.0, 1.18, 1.4, 1.6],
        extract_sol=False,
    ),
    MatrixEntry(
        group="A4",
        purpose="stability sweep",
        is_implicit=True,
        poly_deg=2,
        n_els=100,
        cfl_values=[0.1, 0.2, 0.3, 0.35, 0.42, 0.5],
        extract_sol=False,
    ),
    MatrixEntry(
        group="A5",
        purpose="stability sweep",
        is_implicit=True,
        poly_deg=3,
        n_els=100,
        cfl_values=[0.1, 0.2, 0.3, 0.33, 0.4, 0.5],
        extract_sol=False,
    ),
]

# ---------------------------------------------------------------------------
# Group B -- oscillation metrics. No new runs: these are the SAME runs
# already produced for tab:cfl-ceiling-shock / Figs 16-18 (the "ceiling" CFL
# for each row). Listed with new_run=False so the launcher skips them and
# analyze_oscillations.py knows where to find them (si.get_dir(...) with
# these exact fields reproduces the existing run directory).
# ---------------------------------------------------------------------------
GROUP_B: List[MatrixEntry] = [
    MatrixEntry(
        "B1",
        "TV/overshoot",
        is_implicit=False,
        poly_deg=1,
        n_els=100,
        cfl_values=[0.079],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "B2",
        "TV/overshoot",
        is_implicit=True,
        poly_deg=1,
        n_els=100,
        cfl_values=[1.178],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "B3",
        "TV/overshoot",
        is_implicit=True,
        poly_deg=2,
        n_els=100,
        cfl_values=[0.348],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "B4",
        "TV/overshoot",
        is_implicit=True,
        poly_deg=3,
        n_els=100,
        cfl_values=[0.338],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "B5",
        "TV/overshoot",
        is_implicit=True,
        poly_deg=1,
        n_els=50,
        cfl_values=[1.179],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "B6",
        "TV/overshoot",
        is_implicit=True,
        poly_deg=2,
        n_els=50,
        cfl_values=[0.353],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "B7",
        "TV/overshoot",
        is_implicit=True,
        poly_deg=3,
        n_els=50,
        cfl_values=[0.313],
        extract_sol=True,
        new_run=False,
    ),
]

# ---------------------------------------------------------------------------
# Group C -- grid imprinting. C1-C3 reuse existing runs; C4 is new, sized to
# sit right at the DOF budget for the best available p=3 comparison.
# ---------------------------------------------------------------------------
GROUP_C: List[MatrixEntry] = [
    MatrixEntry(
        "C1",
        "imprinting baseline",
        is_implicit=False,
        poly_deg=1,
        n_els=100,
        cfl_values=[0.079],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "C2",
        "imprinting finest reachable",
        is_implicit=True,
        poly_deg=1,
        n_els=250,
        cfl_values=[1.0],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "C3",
        "imprinting matched-res control",
        is_implicit=True,
        poly_deg=1,
        n_els=100,
        cfl_values=[1.0],
        extract_sol=True,
        new_run=False,
    ),
    MatrixEntry(
        "C4",
        "imprinting high-p budget-capped",
        is_implicit=True,
        poly_deg=3,
        n_els=125,
        cfl_values=[0.33],
        extract_sol=True,
        new_run=True,
    ),
]

ALL_GROUPS = {"A": GROUP_A, "B": GROUP_B, "C": GROUP_C}


def new_runs_only() -> List[MatrixEntry]:
    """Entries that actually need a solver launch (used by the launcher)."""
    out = []
    for entries in ALL_GROUPS.values():
        out.extend(e for e in entries if e.new_run)
    return out


def check_dof_budget(entries: Optional[List[MatrixEntry]] = None) -> None:
    entries = entries if entries is not None else new_runs_only()
    for e in entries:
        dofs = e.dof_estimate()
        flag = "  <-- AT/OVER BUDGET" if dofs > DOF_BUDGET else ""
        print(
            f"[{e.group}] {e.scheme_name:8s} p={e.poly_deg} N={e.n_els:4d} "
            f"~{dofs/1e6:.2f}M DOFs{flag}"
        )
        if dofs > DOF_BUDGET:
            raise ValueError(
                f"Entry {e.group} requests ~{dofs/1e6:.2f}M DOFs, "
                f"over the {DOF_BUDGET/1e6:.1f}M budget."
            )


if __name__ == "__main__":
    print("New runs required by the test matrix:")
    check_dof_budget()
