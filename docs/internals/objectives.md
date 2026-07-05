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
| ⑤ | **Precision→template threading** — thread `T=dstype, I=Int` through the whole backend so precision is *type-level*, byte-identical under defaults, frontends untouched | ✅ done — full float32 **solve** works as a consumer type-choice (no rebuild) |
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

## Precision: full float32 solve achieved as a *consumer type choice* (no rebuild)
⑤ is done end-to-end: precision is a **type-level choice a consumer makes**, not the `EXASIM_FLOAT`
build macro. Two consumers prove it against the double-default install, no Exasim rebuild, no conversion:
- `tests/consumers/model_fp32` — the Poisson2D model + its flux/source/qoi Kokkos kernels at `float`
  (and `double`), CPU (Serial) + GPU (CUDA V100): `max |float−double|/|double| = 1.6e-07`.
- `tests/consumers/solve_fp32` — a **full HDG Poisson SOLVE**: instantiate the in-memory stack at
  `float`, condense to `H·uh=b`, LAPACK-solve, recover, `eval_qoi`. The float trace solution reproduces
  the double one to `1.01e-4` (fp32 dense-LU accuracy); float QoI `∫u = 0.405249` vs double `0.405285`.

Getting the full solve there required the **preprocessing/in-memory-construction boundary** threaded
conversion-free (`Preprocessed<T,I>`, builders, `make_preprocessed<M,T,I>`, the compiled-core setup
chain) **plus a ~167-site backend-wide precision-mixing cutover** (mostly one trick: `noDeduce_t<T>` so
`double` constants next to `float` buffers convert instead of forcing a conflicting deduction). See
`precision-threading.md` §Phase 5. Along the way it surfaced a latent BLAS bug: `blas<float>::dot`
misread `sdot_`'s ABI return type (fixed).

**Invariant held throughout:** the default `dstype` path is byte-identical — app-regression 12/12,
`petsc_poisson` consumer 9.084e-14 / 0 / 2.651e-16. Frontends untouched.

**Remaining precision tail (deferred):** a PETSc-*driven* float solve still needs a float-built PETSc
(the `Operator`/`ShellMat` zero-copy `Vec` reinterpret is guarded by `sizeof(PetscScalar)==sizeof(Scalar)`);
and the generated-codegen models (vs the hand-written `Poisson2DT<T>`) would need the same struct-template
+ `T`-kernel treatment to let *generated* apps pick precision.
