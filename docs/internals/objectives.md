# Objectives of this effort (branch `teoc-properly-separate-out`)

**Headline goal — PETSc operator export.** Turn Exasim from a monolithic HDG/LDG DG solver
into a library whose discretization can be *driven externally by PETSc*: export the residual,
Jacobian/matrix, matrix-free operator, preconditioner, mesh, and DG↔CG maps as callbacks so an
external PETSc app owns the TS/SNES/KSP loop and **never touches Exasim internals** (the
heat-equation / Poisson consumer is the reference). Everything below is in service of this.

## Numbered objectives (task ledger) + status

| # | Objective | Status |
|---|-----------|--------|
| ① | **Build-time precision switch** — `dstype`/`Int` selectable as one global choice (double/float via `EXASIM_FLOAT`, int/long via `EXASIM_INT64`) | ✅ done |
| ② | **PETSc Layer-1 shim** — `Exasim::petsc` library exposing `Operator` / `ShellMat` | ✅ done |
| ③ | **Export surface** — residual + assembled matrix (MATAIJ) + matrix-free MatMult + block matrices as the concrete API a PETSc app calls | 🔶 in progress (the live thread) |
| ④ | **Robustness / equivalence harness** — consistency checks + a harness proving exported operators match the native solve | ✅ done |
| ⑤ | **Precision→template threading** — thread `T=dstype, I=Int` through the whole backend so precision is *type-level*, byte-identical under defaults, frontends untouched | ✅ compute+orchestration complete |
| ⑥A/B | **Coverage** — llvm-cov report + coverage-guided baseline test expansion | ✅ done |
| ⑥C | **Quality gate** — clang-format + clang-tidy | ⚪ pending |

## Two cross-cutting invariants (every objective is held to these)
- **Byte-identical under defaults** — every increment reproduces golden `rel_L2` (proof it changed
  no numerics), verified on all four paths: CPU · CPU-MPI · GPU · GPU-MPI.
- **Frontends untouched** — Matlab/Python/Julia never change; the frozen `ExasimDriverABI`
  fn-pointer seam is the cut, with the concrete-model path below it free to be precision-generic
  (stance A: `static_assert(T==dstype)` in the `EXASIM_DRIVER_CALL` AbiAdapter branch).

## Architectural backdrop (the "separate-out" refactor)
Decompose the frontend into FEM-like layered objects — Model + QoI, split the pde god-dict, extras
as a git submodule, reconcile to Matlab. Precision threading (⑤) is the C++-side foundation that
lets the PETSc consumer pick precision and keeps the operator export clean.

## Precision decision: banked at "one global choice"
⑤ had a stretch goal — **mixed precision** (a `float` solve inside a `double` build: Phase 4 codegen +
Phase 5 cutover + the GPU `blas<T>` trait). **Decision: bank at single-global-precision granularity** —
the whole tree flips together via the one `dstype` choice (`EXASIM_FLOAT`); mixed-precision is deferred.
This keeps ⑤ a completed, verified foundation and leaves **③ (PETSc export surface)** and **⑥C (lint
gate)** as the forward objectives, not the mixed-precision rabbit hole.

**float32 demonstrated as a consumer choice (no rebuild).** `tests/consumers/model_fp32` instantiates
the Poisson2D model + its flux/source/qoi Kokkos kernels at `float` (and `double`) against the
double-default install — proving precision is a *type-level* choice, not `EXASIM_FLOAT` at build time.
PASS on CPU (Serial) and GPU (CUDA V100): `max |float−double|/|double| = 1.6e-07`. A full float32
*solve* additionally needs the export/preprocessing boundary templated (`Preprocessed` holds double
structs) + a float PETSc — deferred to Phase 5 (see `precision-threading.md`).
