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
- [ ] Benchmark harness + numbers (py one-stage vs C++ generate+compile+run)
- [ ] exasim/petsc.hpp: factor a `solve_steady` helper (thin the app)
- [ ] App scaffolder (header-only CHEFSI-style emitter)
- [ ] Wire into C++ text2code (`--emit-app`) and/or Python CLI
- [ ] End-to-end: generate an app, build it, run it

## Notes / gotchas

- symengine pip wheel `0.14.1` (abi3, cp311+) works on this Mac's py3.14.
- `ccode` prints `pow(...)`; Exasim wants `Kokkos::pow`/`Kokkos::sqrt`/... — needs
  a printer map. C++ path uses a custom C99CodePrinter subclass.
- Goldens: `CHEFSI-apps/.../isoq2d_cht-petsc-fluid-with-kitesurf/generated/my_model.hpp`
  and its `pdemodel.txt` (model 100); plus poisson2d.
- Do NOT commit/push without user say-so per feedback-no-commit-without-approval...
  BUT user explicitly said "push frequently please" for THIS task → pushing is authorized here.
