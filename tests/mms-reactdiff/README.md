# consumer_mms_reactdiff — MMS analytical-solution smoke

A small **2-species coupled reaction-diffusion** model with a **manufactured (exact)
solution**, used to smoke-test the header-only emit-app / `exasim::petsc::solve_steady`
path end to end with a verifiable answer. It's a deliberately reduced stand-in for a
reacting-flow model (multiple components + a reaction source), simple enough to have a
closed-form solution.

## The model (`pdemodel.txt`)

Two components `u0, u1` on the unit square, homogeneous Dirichlet:

- Diffusion flux `f_i = kappa_i grad(u_i)`.
- Linear reaction `R = a (u0 - u1)` (source: `-R` on eq 0, `+R` on eq 1).
- A manufactured forcing (function of position) chosen so the exact solution is
  `u0 = sin(pi x) sin(pi y)`, `u1 = 0.5 sin(pi x) sin(pi y)`.

Both components vanish on the boundary, so homogeneous Dirichlet is exact.

## What the test does (`run.sh` + `main.cpp`)

1. Generates `datain/` from the tracked unit-square mesh (`grid.bin`) via `text2code`
   (`gendatain=1`, HDG).
2. Builds `main.cpp` — which constructs `CSolution<PdeModel>` from `datain/`, solves the
   HDG system through the exported PETSc operator (`exasim::petsc::solve_steady`), and
   compares the DG-node field to the exact solution pointwise.
3. Passes iff the solve converges and `max|u - u_exact| < 2e-2` (the HDG p=3 solve
   recovers it to ~1e-6 in practice).

PETSc-gated: `run.sh` exits **77** (clean ctest SKIP) when `text2code`, PETSc, or a
complete Exasim install is unavailable — like the other petsc consumer tests.

`generated/my_model.hpp` is the canonical model header (identical from `pyt2c` and the
C++ `text2code`; see `text2code2/tests/run_model_sweep.sh`).

## Run standalone

```sh
EXASIM_INSTALL=/path/to/petsc-enabled-install EXASIM_ROOT=/path/to/exasim-src \
  KOKKOS_DIR=/path/to/kokkos ./run.sh
```
