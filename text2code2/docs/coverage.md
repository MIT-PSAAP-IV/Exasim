# Model coverage

`tests/coverage_report.sh` categorizes every model in the corpus (examples/ +
backend/Model/BuiltIn + apps/ + the text2code sample — 47 models) by feature
dimension. Every one passes both `tests/run_model_sweep.sh` (structural + numeric
equivalence pyt2c vs C++ text2code) and `tests/run_appgen_sweep.sh` (emit + compile).

## Feature dimensions exercised

| Dimension | Values in the corpus |
| --- | --- |
| spatial dim (nd) | 2D, 3D  (no true 1D model exists — "poisson1d" is a 2D formulation) |
| solution components (ncu) | 1 (scalar Poisson), 2, 4 (compressible NS), 5 (3D NS), 8 (reacting flow) |
| discretization | 43 HDG, 4 LDG |
| external coupling (Fint/Fext) | 10 models |
| matrix ops (inv/det/transpose — Riemann solvers) | 6 models |
| auxiliary w-field (ncw>0) | 2 models (reactingsharpb2, built-in 4) |
| extra outputs | QoIvolume (12), QoIboundary (7), VisTensors (3), EoS (2), Sourcew (2), Initw (2), Initv (1) |

Representative models per type: scalar Poisson (poisson2d/3d, Lshape, Cone, Nonlinear,
Periodic), compressible NS (isoq, naca, sharpb2, nsmach8, orion), 3D NS (isoq3d),
reacting flow (reactingsharpb2 — ncu=8 + w + EoS + Sourcew + VisTensors), Riemann-solver
BC with matrix inverse (built-in 12, naca, text2code sample), coupling (built-ins
1/6/7/8/9/10/11, isoq variants).

Since every feature combination appears in the 47-model corpus and each is byte-identical
between pyt2c and C++ text2code, the codegen is validated across the full type space.

## Runtime scope: `--emit-app` produces **HDG** apps

The emitted app drives the solve through `exasim::petsc::solve_steady`, which operates on
the **condensed HDG trace system** (res.H MatShell + res.K PCShell). So:

- **Model codegen** (`my_model.hpp`) is discretization-agnostic — it is byte-identical for
  HDG and LDG models (the 4 LDG models pass the equivalence sweep).
- **The emitted app** is HDG-driven. This is now enforced explicitly:
  - `text2code --emit-app` **refuses** to emit for an LDG-configured pdeapp
    (`discretization="ldg"`) with a clear error, unless `--allow-ldg` is passed.
  - `exasim::petsc::solve_steady` **raises** (`SETERRABORT`) if handed a non-HDG problem
    (`spatialScheme != 1`), instead of failing cryptically.
  Running an LDG model with the emitted app requires HDG `datain` (`discretization="hdg"` /
  `hybrid=1`). This matches the "latest CHEFSI app" style, which is HDG throughout.

## End-to-end runs (dgx-b, real PETSc 3.25.3)

Verified converging + correct where the steady HDG solve is tractable:

| Model | Type | Result |
| --- | --- | --- |
| built-in consumer | 2D scalar, coupling, manufactured | `u=sin(πx)sin(πy)` exact (mean (2/π)²) |
| poisson2d (as HDG) | 2D scalar, QoI, manufactured | `u=sin(πx)sin(πy)` exact |
| poisson3d | 3D scalar, manufactured | `u∈[0,1]`, mean (2/π)³ (sin³ product) |
| lshape, nonlinear, periodic | 2D scalar (incl. periodic BC, nonlinear) | converged (`Reason=2`) |
| cone3d | 3D scalar (large) | ran; solver diverged (`Reason=-5`) — solver-tuning, not codegen |

Compressible-NS / reacting models (ncu≥4) compile and their codegen is byte-identical, but a
steady solve from a cold start is problem-specific (needs continuation / a physical initial
guess) — the same for a C++-text2code app, since the model header and driver are shared.
