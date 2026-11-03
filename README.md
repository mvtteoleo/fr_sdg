# FR/DG Compressible Flow Solver

A C++23 solver for the Euler equations, able to handle generic gas models
(Ideal and Van der Waals as of now, but extensibility is key).

This code accompanies a Master's thesis in Aeronautical Engineering at
Politecnico di Milano, which compares **implicit and explicit time-stepping
methods for compressible CFD** using high-order Flux Reconstruction (FR) on
**curved meshes**. The design choices below (PETSc TS/SNES backend, complex-step
Jacobian) all serve that comparison.

## Table of contents

1. [Overview](#overview)
2. [Feature matrix](#feature-matrix)
3. [Requirements and environment](#requirements-and-environment)
4. [Build and run](#build-and-run)
5. [Usage example](#usage-example)
6. [Problem definition and configuration](#problem-definition-and-configuration)
7. [Validation and test catalog](#validation-and-test-catalog)
8. [Code architecture](#code-architecture)
9. [The tools directory](#the-tools-directory)
10. [References](#references)
11. [Limitations and roadmap](#limitations-and-roadmap)

## Overview

The numerical framework is that of a Flux Reconstruction scheme, which recovers
nodal Discontinuous Galerkin (spectral element) for the appropriate choice of
correction function.

Both implicit and explicit time integration are available by means of PETSc,
which handles the backend for the time stepping and the nonlinear solver.
A note of interest goes for the possibility of using the complex-step derivative
to approximate the Jacobian.

The C++ standard of choice is C++23. To manage the dependencies I use Nix,
the only real dependency.

## Feature matrix

| Area | Available |
|---|---|
| Spatial dimension | 1D (`physics_impl_1D.hpp`), 2D (`physics_impl_2D.hpp`) |
| Gas models | Ideal, Van der Waals (`gas_models.hpp`) |
| Riemann solvers | BANANA (list the ones implemented in `riemann_solver.hpp`) |
| FR correction functions | BANANA (DG? c-VCJH family? the `tools/c-vcjh_*.py` scripts suggest so) |
| Polynomial order | BANANA (compile-time or runtime? any maximum?) |
| Shock handling | Limiter (`flux_limiter.hpp`): BANANA which one (WENO?), filter (`filter.hpp`): BANANA which type |
| Time integration | Explicit and implicit via PETSc TS |
| Nonlinear solver | PETSc SNES, Newton-Krylov; Jacobian by complex step |
| Boundary conditions | BANANA (list the ones in `boundary_conditions.hpp`) |
| Geometry | Curved meshes, boundary-conforming (`geometry_models.hpp`) |
| Parallelism | MPI |
| Output format | PETSc binary, how to load and read them is visible handled in the tools/*.py|

## Requirements and environment

- **Nix** (with flakes enabled) is the only dependency you need to install.
  [flake.nix](./flake.nix) pins the compiler, PETSc and every other package.

If Nix is missing and you cannot or do not want to install it globally (for
example on an HPC cluster where user namespaces are disabled), use
[nix-portable](https://github.com/DavHau/nix-portable).

## Build and run

Enter the development shell with either

```sh
nix develop
```

or, with nix-portable,

```sh
./nix-portable nix develop --extra-experimental-features "nix-command flakes"
```

Once inside the nix-shell:

```sh
make all              # build every executable
make <TEST-NAME>      # build tests/<TEST-NAME>.cpp only
```

All the executables end up in the `build/` directory.

## Usage example

End-to-end run of the Sod shock tube, compared against a method-of-characteristics
reference:

## Problem definition and configuration


Design principle: constraints are enforced at compile time wherever possible.

All options are set by means of PETSc so one can use the PETSc's commands to
have them listed.


| Choice | When it is fixed |
|---|---|
| Gas model | Run time |
| Riemann solver / flux function | Compile time |
| Polynomial order | Run time |
| Dimension | Compile time |
| Mesh / geometry parameters | Compile time |
| Time-stepping scheme and tolerances | Runtime, through PETSc options |

Commonly useful PETSc options:

```sh
-ts_type # e.g. rk, arkimex, beuler, bdf, ...
-ts_dt 
-snes_view 
-ksp_view 
-pc_view 
-snes_monitor 
-ksp_monitor 
-pc_monitor 
```

## Validation and test catalog

The tests have a double purpose: they verify the solver against reference
solutions, and they show the usage of some more advanced and modern C++ features
such as views and spans.


## Code architecture

The library is header-only and lives in `include/`.

- **Physics**: `physics_impl_1D.hpp`, `physics_impl_2D.hpp`, `physics_evaluator.hpp`,
  `gas_models.hpp`, `flux_function.hpp`, `riemann_solver.hpp`,
  `boundary_conditions.hpp`. These define the equations, the gas laws, the
  numerical fluxes and the boundary treatment.
- **Discretization**: `reference_element.hpp`, `residual_functs.hpp`,
  `geometry_models.hpp`, `indexer.hpp`, `vec_dof_view.hpp`,
  `cached_computations.hpp`. These handle the reference element, the FR residual
  assembly, the curved geometry, and the mapping between elements and PETSc
  vectors.
- **Stabilization**: `flux_limiter.hpp`, `filter.hpp`.
- **Solver glue**: `petsc_solver_impl.hpp`, `problem_def.hpp`, `parser.hpp`,
  `factory.hpp`. These connect a problem definition to PETSc TS/SNES.
- **Utilities**: `my_lib.hpp`, `macro_and_libc.hpp`.

Note
the filename typo `phisics`, and a `.cpp` file sitting in `include/`.)

## The tools directory

One note on the `tools/` directory: it contains some `.py` scripts to handle
reference solutions by means of the method of characteristics, and in general
helpers to automate runs and post-process results.

- Reference solutions: `sod_shock_moc.py`, `sod_radial_moc.py`, `moc_vs_dg.py`,
  `exact_wedge.py`, `sedov_taylor.py`
- Convergence studies: `conv_test.py`, `conv_test_2d.py`
- Stability / CFL studies: `cfl_ceiling_plots.py`, `cfl_ceiling_shock.py`
- Run automation: `run_test_matrix.py`, `shock_test_matrix.py`
- Plotting and post-processing: `plot*.py`, `post_proc.py`, `report.py`, `check_*.py`

## References

The bibliography is in [`articles/master_bibliography.bib`](./articles/master_bibliography.bib),
and the papers that the implementation follows are collected in `articles/`.
Main sources:

- Flux Reconstruction and energy-stable correction functions (`A_New_Class_of_High-Order_Energy_Stable_Flux_Recon.pdf`, `deep_dive_FR_vs_DG_DeGrazia_2013.pdf`)
- Implicit vs. explicit FR (`flux_reconstr_methdos_CUDA_JFNK_vs_explicit_ts_Jourdan_Wang_AIAA_2021.pdf`)
- JFNK for hyperbolic flows (`JFNK_for_hyper_flows_AIAA_2023_Damn_et_al.pdf`)
- Multigrid for DG (`FMG_for_DG_methods_reduce_stiffness_Bulgarini_Ghidoni_JFM_2025.pdf`)
- Limiting and filtering (`weno_limiter.pdf`, `shock_limiter.pdf`, `stable_filtering_spectral_dg.pdf`)
- Non-ideal gases (`non_ideal_gas_solver_with_AD_fluids_2023_Mantecca.pdf`)

Note: the PDFs in `articles/` may be copyright-restricted. BANANA (remove them
or keep only the `.bib` if the repository becomes public?)

## Limitations and roadmap

Known limitations:

- 3D is not supported. (true? planned?)
Roadmap: 
- Restructure the code in order to be able to handle generic equations of shape $ U_{/t} + \nabla \cdot \F(U) = S $

