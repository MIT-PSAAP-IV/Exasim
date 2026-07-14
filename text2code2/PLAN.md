# text2code2 — header-only app generation + Python codegen

Branch: `teoc-text2code-headeronly-and-py` (off `master`).
Goal (from user): three deliverables.

1. **Header-only app generation.** Improve text2code to emit a self-contained,
   header-only C++ app in the style of the latest CHEFSI app
   (`isoq2d_cht-petsc-fluid-with-kitesurf`), so a generated model can be
   auto-converted into a C++-driven program (no runtime-loaded `.so` model ABI).
   Where possible, push more of the app-side PETSc solver glue INTO Exasim's
   `<exasim/petsc.hpp>` so the generated app stays thin.

2. **Python model generator.** Reimplement the model codegen
   (`pdemodel.txt` → `generated/my_model.hpp`) in Python using the `symengine`
   pip package. Goal: `pip install symengine` only (no vendored SymEngine build),
   and ideally faster than the C++ two-stage pipeline. Benchmark required.

3. **(Maybe) Python app generator.** Python version of the app scaffolding from (1).

## Current architecture (as found)

- Frontends (`frontends/{Python,Matlab,Julia}/.../exporttext2code.*`) export
  high-level `pdemodel.txt` + `pdeapp.txt` + `grid.bin` (NOT C++).
- The C++ `text2code` binary (`text2code/text2code/`) then:
  - `gendatain=1`: preprocess mesh → `datain/*.bin`.
  - `gencode=1`: **two-stage** codegen —
    1. `generateCppCode(spec)` emits a SymEngine C++ program
       (`SymbolicFunctions.*`, `SymbolicScalarsVectors.*`, `Code2Cpp`).
    2. `executeCppCode(spec)` compiles + runs it → `generated/my_model.hpp`
       (`struct PdeModel : ModelDefaults<PdeModel>`, all Kokkos-inline kernels).
    3. (non-CMAKE) `buildDynamicLibraries` compiles the `.so` model ABI.
- The concrete header-only path: `CSolution<PdeModel>` built from `datain/` via the
  No-ABI ctor; `<exasim/operators.hpp>` unity backend + `<exasim/petsc.hpp>`
  (`exasim::petsc::Operator` MatShell/PCShell/SNES). The CHEFSI petsc app wraps
  this in `ExasimPetscFluidSolver.hpp` + a `.cc` driver + `CMakeLists.txt`.

## Key insight

The Python rewrite (2) collapses the two-stage C++ pipeline into ONE stage:
parse `pdemodel.txt` → build symengine exprs directly → diff (jacobians) → cse →
C99 print (Kokkos:: math map) → emit `my_model.hpp`. No compile-a-program-then-run.
This is inherently faster and needs only `pip install symengine`.

Deliverables (1) and (3) overlap: the app scaffold (driver `.cc`, solver header,
`CMakeLists.txt`, `build.sh`) is templating over model sizes + interface config.

## Module layout: `Exasim/text2code2/`

- `pyt2c/`            Python package (parser, symengine codegen, app scaffolder)
- `docs/`            design docs
- `tests/`           golden tests vs existing `my_model.hpp`
- `bench/`           benchmark harness (py vs C++ text2code)

## Status

- [x] Branch + module scaffold + plan (this file)
- [x] Design doc: codegen spec (`docs/codegen-spec.md`) + reference intermediates
- [x] pyt2c: pdemodel.txt parser (`pyt2c/parser.py`)
- [x] pyt2c: DSL interpreter (`pyt2c/interp.py`)
- [x] pyt2c: symengine codegen → my_model.hpp (`pyt2c/codegen.py`)
- [x] **Numeric equivalence PROVEN**: pyt2c output is byte-identical to the C++
      text2code golden kernels for poisson2d AND isoq2d compressible-NS model 100
      (`tests/run_equiv.sh`, `tests/equiv_harness.cpp`). Only textual diff is CSE
      temp ordering (pip SymEngine vs vendored SymEngine).
- [x] Benchmark: pyt2c ~32x faster end-to-end than C++ text2code (`bench/RESULTS.md`)
- [x] exasim/petsc.hpp: `SteadyOptions` + `solve_steady(disc,...)` +
      `prepare_steady(CSolution<M>)` + `solve_steady(CSolution<M>,...)` — factors the
      CHEFSI app's ~90 lines of PETSc glue into Exasim. Syntax-checked against the real
      backend + PETSc 3.24 + Kokkos with a pyt2c-generated model (full templated
      instantiation compiles clean).
- [x] App scaffolder (`pyt2c/appgen.py`, `--emit-app`): standalone header-only
      CHEFSI-style app (driver + CMakeLists + build.sh + README + generated model).
      Model-agnostic scaffold; only my_model.hpp is model-specific. Owns NO PETSc glue.
      Both the poisson AND isoq2d model-100 generated drivers syntax-check clean.
- [x] Python CLI wired (`--emit-app`). Part 3 (Python app generation) done.
- [x] End-to-end build+run of a generated app (`tests/run_e2e.sh`): pyt2c model + app
      scaffold + solve_steady, compiled + linked (unity backend, NO Exasim install) +
      RAN to SNES convergence, correct output. Nothing outside the work dir touched.
- [x] `--emit-app` added to the C++ text2code binary (`AppScaffold.hpp`): same scaffold
      as pyt2c; generated/ cleaned to my_model.hpp+model_sizes.hpp; Exasim/lib untouched;
      emitted driver syntax-checks clean; emitted header numerically identical to golden.
- [x] `--emit-app --from-header`: scaffold from an existing my_model.hpp, NO .txt needed.
- [x] Broad test: full built-in-model sweep (all 15, pyt2c vs C++ text2code, in-process
      NaN/Inf-aware) — `tests/run_builtin_sweep.sh`; 15/15 byte-equal after fixing 3
      interp bugs the sweep found.
- [x] CMake wiring: `cmake/ExasimEmitApp.cmake` (`exasim_emit_app()`) + opt-in
      `EXASIM_EMIT_BUILTIN_APPS` → `builtin-apps` target; tested in isolation
      (`tests/cmake-emit-app/`, + models 1/12 emitted via cmake and syntax-checked).
- [x] Docs: `docs/app-generation.md` frames --emit-app vs the existing `exportapp`.

## Summary of what shipped

Three deliverables, all on branch `teoc-text2code-headeronly-and-py`:
1. **Python model codegen (`pyt2c`)** — pip-symengine, single-stage, proven
   NUMERICALLY BYTE-IDENTICAL to C++ text2code (poisson + isoq2d NS model 100) and
   ~32x faster.
2. **Header-only app generation** — `exasim/petsc.hpp` `solve_steady` helpers (the
   app-side PETSc solver code, moved into Exasim) + `pyt2c --emit-app` scaffolder.
3. **Python app generation** — the `--emit-app` scaffolder is pure Python.

## Notes / gotchas

- symengine pip wheel `0.14.1` (abi3, cp311+) works on this Mac's py3.14.
- `ccode` prints `pow(...)`; Exasim wants `Kokkos::pow`/`Kokkos::sqrt`/... — needs
  a printer map. C++ path uses a custom C99CodePrinter subclass.
- Goldens: `CHEFSI-apps/.../isoq2d_cht-petsc-fluid-with-kitesurf/generated/my_model.hpp`
  and its `pdemodel.txt` (model 100); plus poisson2d.
- Do NOT commit/push without user say-so per feedback-no-commit-without-approval...
  BUT user explicitly said "push frequently please" for THIS task → pushing is authorized here.
