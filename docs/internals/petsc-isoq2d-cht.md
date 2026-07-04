# PETSc-driven isoq2d CHT — design & first increment

Branch: `teoc-petsc-isoq2d-cht` (off `teoc-properly-separate-out` @ e329d0ad).

Goal (task #25): *"Using the same PETSc as summit, add a PETSc version of the
isoq2d CHT (conjugate heat transfer) app with time stepping on Exasim's side.
Should have similar verification to the existing petsc consumers."*

## 1. What isoq2d CHT actually is

Source app: `CHEFSI-apps/FSP-1/ExaSim-SumMIT/isoq2d_cht-with-kitesurf/`.
It is a transient conjugate-heat-transfer problem on the isoq cone:

- **Fluid** (Exasim): axisymmetric compressible Navier–Stokes, 4 conserved
  vars `(r, ru, rv, rE)`, built-in model ID 100 (`pdemodel100.txt`). Run
  **steady** (`dt = [0]`); it re-solves to steady each coupling step.
- **Solid** (Summit today): **transient linear heat conduction** on the cone
  wall — `rho*cp dT/dt = div(k grad T)`, with `rho=280`, `cp=2000`, `k=0.5`
  (`thermal.yaml`). This is *"the SOLID transient step the coupling marches"*
  (`coupling_config.yaml`, `dt=0.1`).
- **Interface coupling**: an outer transient loop wraps an inner
  Dirichlet–Neumann fixed-point iteration (under-relaxation `omega`), exchanging
  a **single field** across the wall: heat flux `qn` solid-ward, wall
  temperature fluid-ward. It does **not** converge to tolerance — it plateaus at
  a residual floor (expected for this hypersonic aeroheating case; matches
  legacy).

So the transient time-stepping in isoq2d CHT lives on the **solid thermal
side**, which is exactly a transient heat-conduction PDE — the same PDE class as
`tests/petsc/petsc_heat`.

## 2. Interpretation of "time stepping on Exasim's side" — decision: **reading (B)**

Two readings were considered:

- **(A)** Exasim owns the transient temporal scheme; PETSc only does the per-step
  KSP/SNES linear/nonlinear solves.
- **(B)** PETSc `TS` drives the time-stepping loop (as in `petsc_heat`), and
  *"Exasim's side"* contrasts with the **external summit/CHEFSI coupler** driving
  it — i.e. a **self-contained** Exasim+PETSc process, not one driven by the
  summit intercomm coupler.

**Chosen: (B), with the hybrid nuance the existing `petsc_heat` already codifies.**
Reasons:

1. **The existing infra is built for (B).** `tests/petsc/petsc_heat/main.cpp`
   documents itself as *"PETSc TS owns the time loop … driving Exasim's exported
   TIME-AUGMENTED HDG operators (Level A loop-driver)."* `exasim::petsc::Operator`
   + `TSSetPreStep/PostStep/IFunction/IJacobian` is the proven pattern. Reading
   (A) would mean re-implementing a temporal integrator on the Exasim side and
   using PETSc only for `KSPSolve`, throwing away the `TS` glue that already
   exists and is tested.

2. **It matches "same PETSc as summit."** Summit's own thermal solver marches the
   solid with PETSc; putting the loop on *Exasim's side* means the Exasim process
   (linked against the identical summit-dev conda PETSc) now owns that transient
   solve in-process, replacing the external summit thermal driver. The contrast
   "Exasim's side" vs. the summit coupler is precisely reading (B).

3. **The hybrid nuance (why it's not pure-A or pure-B).** In HDG the global
   unknown is the algebraic trace `uh` (no `du/dt`); the time derivative lives on
   the condensed volume `udg`, folded into the source + statically condensed.
   So Exasim *owns the time discretization* (backward Euler baked into `H`, `b`
   via `UpdateSource`/`tdfunc` mass), while PETSc `TS` *owns the step control /
   Newton-Krylov loop*. `dF/dUdot = 0`, `J = H`, and `TS` reduces to a clean loop
   driver. We reuse this exact structure.

## 3. Architecture

```
PETSc TS (TSBEULER)                         Exasim exported operators
──────────────────                          ─────────────────────────
PreStep  ── prepareStep() ────────────────▶ PreviousSolutions, UpdateSource
                                             (dtfactor=1/dt, tdfunc mass=rho*cp),
                                             Node2Gauss(sdg->sdgg), hdgGetQ,
                                             hdgAssembleLinearSystem  -> res.H, b
                                             ComputeHDGPreconditioner -> res.K
IFunction  F(U)=H*(U-uh_n)-b  ────────────▶ exasim::petsc::Operator::mat()  (MatShell res.H)
IJacobian  J = H              ────────────▶   "        "        (constant per step)
KSP(GMRES)+PCShell            ────────────▶ Operator::configure_pc  (PCShell res.K)
PostStep ── recoverStep() ────────────────▶ recover_volume (trace->udg), UpdateSolution
```

Who owns what:

- **PETSc** owns: the outer step loop, step number/time, the per-step
  GMRES(+PCShell) linear solve, convergence bookkeeping.
- **Exasim** owns: the spatial HDG discretization, the backward-Euler time term
  (mass `rho*cp` via `tdfunc`, folded into `H`,`b`), the matrix-free `res.H`
  apply, the `res.K` preconditioner, and trace→volume recovery.
- **Operator export reuse**: one `exasim::petsc::Operator<HeatSolid2D>` is built
  once and reused every step (it applies the *current* `res.H`/`res.K`, refreshed
  in `PreStep`). No hand-rolled `MatShell`/`PCShell`. The assembled-`MATAIJ`
  consistency check reuses `exasim::petsc::assemble_matrix`.

This is the *identical* skeleton as `petsc_heat`; the deltas are (a) the model
(`HeatSolid2D` with real solid material props + an interface-flux boundary type),
(b) the isoq-solid material parameters, and (c) the CHT-interface boundary hook.

## 4. First increment (this run) vs. full CHT

### Done this increment
- New model `tests/models/heatsolid2d.hpp`: transient linear heat conduction
  `rho*cp dT/dt = div(k grad T) + f`, HDG, with
  - conductivity `k=mu[0]`, mass `rho*cp` via `tdfunc` (`mu[1]*mu[2]`),
  - **ib=1 Dirichlet** boundary (used for the verified manufactured run),
  - **ib=2 interface heat-flux (Neumann)** boundary — the CHT coupling hook:
    imposes a prescribed wall normal flux `qn=mu[3]` (what the fluid supplies in
    the full coupling),
  - `QoI` (L2 to the manufactured field) + `Vis`.
- New consumer `tests/petsc/petsc_isoq2d_cht/` (auto-picked-up by
  `run-petsc-test.sh`): **PETSc TS owns the transient loop** on the exported
  operators, with the **isoq2d solid material properties** (`rho=280, cp=2000,
  k=0.5`) on a rectangular slab sized to the isoq wall bounding box.
- **Verification (real, self-contained, mirrors `petsc_heat`):**
  1. Manufactured solution on the slab: `phi = sin(pi x/Lx) sin(pi y/Ly)` (a
     Dirichlet Laplacian eigenmode, zero on the slab boundary), source
     `f = k*gamma*phi` with `gamma = pi^2(1/Lx^2 + 1/Ly^2)` so the **steady**
     solution is exactly `phi`. From `T=0` the exact transient is
     `T(x,t) = a(t)*phi`, `a(t) = 1 - e^{-lambda t}`, `lambda = k*gamma/(rho*cp)`.
  2. **Transient/mass check**: the recorded per-step `||U||` ratio must track
     `a(t_k)` — this exercises the `rho*cp` mass term (the *rate*), not just the
     steady state.
  3. **Steady L2 check**: at the final near-steady time, `L2(T - phi)` small
     (`< 1e-3`), as in `petsc_heat`.
  4. **Operator-export consistency**: assembled `MATAIJ` vs matrix-free
     `op.mat()` agree to `< 1e-10` (same `assemble_matrix` primitive as the
     steady petsc_poisson example).
  5. Solution finite; TS took the expected number of steps.
- **CHT-interface hook demo** (behavioral, not MMS): a short second run applies a
  constant wall flux `qn` on the `ib=2` interface boundary and asserts the field
  stays finite and the interface-side temperature *responds* (warms). This
  exercises the coupling boundary path with a **stubbed constant flux** standing
  in for the fluid→solid transfer.

### Deferred to later increments (documented, not built)
- **Real cone mesh**: load `Mesh/isoq_solid.msh` via `exasim::read_mesh` (the API
  exists; needs triangular-HDG validation + mapping the gmsh physical tags
  {nose,fillet,top,right,bottom} to `add_boundary` predicates). The slab stands
  in for the wall geometry this increment; the MMS (Dirichlet = exact, source =
  residual) transfers to any mesh unchanged.
- **Live fluid side**: the compressible-NS fluid (model ID 100) solved to steady
  each coupling step, supplying the wall flux `qn` to the `ib=2` boundary. The
  fluid model is the 12-component NS DSL in `pdemodel100.txt`; porting it to the
  templated `<exasim/model.hpp>` contract (or driving it via the built-in model
  path) is a substantial separate task.
- **Dirichlet–Neumann coupling loop**: the inner under-relaxed fixed-point
  iteration exchanging `qn` (solid-ward) / `T` (fluid-ward) per step, with the
  Aitken/fixed relaxation from `coupling_config.yaml`. In the (B) architecture
  this becomes an inner loop *inside* `PreStep` (or a custom `SNES`), still with
  PETSc owning the solid solve.
- **MPI + GPU parity**: `petsc_poisson_mpi` shows the distributed pattern
  (`make_preprocessed_distributed`, `VecCreateMPIWithArray`); the `Operator` is
  already backend/comm-templated, so the extension is config, not new physics.

## 5. Verification plan / parity with existing petsc consumers

The consumer is a self-checking executable returning nonzero on failure and 77
to SKIP when PETSc is absent — the exact contract `tests/petsc/run-petsc-test.sh`
enforces, so dropping the dir under `tests/petsc/` wires it into the same
harness/CI gate as `petsc_heat`, `petsc_poisson`, `petsc_poisson_mpi`,
`robustness`. Build is out-of-tree `find_package(Exasim COMPONENTS cpu petsc)` +
`PkgConfig::PETSC` + the `find_package(MPI)` fallback for a PETSc whose `mpi.h`
lives outside `PETSC_PREFIX`.

Built + run locally against the **summit-dev conda PETSc** (`PETSC_PREFIX =
~/projects/psaap4/summit/conda-envs/summit-dev`, the toolchain Exasim was built
with). See the run log in the branch's commit message / final report.
