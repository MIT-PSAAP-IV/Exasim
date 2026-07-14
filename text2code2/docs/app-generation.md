# App generation (`--emit-app`) and how it relates to `exportapp`

Exasim already had one way to package a runnable app; `--emit-app` adds a second,
different one. They are complementary — pick by which driver you want.

## `exportapp` (existing, frontend)

`frontends/*/Gencode/exportapp.*` packages a **relocatable data-transfer bundle**:
it ships `datain/`, the generated **kernel `.cpp` set**, a `main.cpp` that drives
the model through the standard `ExasimSolver` (`<exasim/ExasimSolverSetup.hpp>`),
plus `CMakeLists.txt`/`run.sh`/`manifest.json`. The bundle depends on an Exasim
install at run time and runs the model as external builtin model ID via the
runtime provider ABI. This is the right tool when you want the normal Exasim
solver and the portable kernel bundle.

## `--emit-app` (new, this work)

`text2code --emit-app` (and `python -m pyt2c --emit-app`) emits a **standalone,
header-only, C++-driven app**: a small `main.cc` that builds `CSolution<PdeModel>`
directly from `datain/` (the No-ABI concrete-model path) and runs the whole solve
through `exasim::petsc::solve_steady`. There is **no** runtime-loaded `.so` model
ABI and **no** hand-rolled PETSc glue in the app — the concrete model is a single
header (`generated/my_model.hpp`) and the solver lives in `<exasim/petsc.hpp>`.
This is the C++-driven, PETSc-driven form (the "latest CHEFSI app" style).

| | `exportapp` | `--emit-app` |
| --- | --- | --- |
| Model form | generated kernel `.cpp` set | one header `my_model.hpp` (No-ABI) |
| Driver | `ExasimSolver` (runtime provider) | `exasim::petsc::solve_steady` |
| PETSc | not required | drives the solve |
| Layer | Python frontend | C++ `text2code` **and** `pyt2c` |
| Runtime dep | Exasim install (`find_package`) | Exasim install + PETSc |

Both take the same high-level inputs and neither needs the other; they emit
different drivers over the same generated model.

## Quickstart from a CMake install (just run it)

After `cmake --install` of a PETSc-enabled Exasim, everything below is on the prefix —
`bin/text2code` (with `--emit-app`), the `pyt2c` Python package (on the install's
site-packages), the full backend + `<exasim/petsc.hpp>` headers, and the mesh
preprocessing data — so a model goes to a running app in three commands:

```sh
# 1. generate datain (mesh preprocessing) + emit the standalone app
text2code pdeapp.txt                       # gendatain=1  -> datain/
text2code pdeapp.txt --emit-app myapp      # -> myapp/{main.cc, CMakeLists.txt, build.sh, generated/}

# 2. build the app against the install
EXASIM_INSTALL=<prefix> ./myapp/build.sh

# 3. run it
mpirun -np 1 myapp/build/myapp datain/ dataout/out
```

The Python codegen is equally runnable from the install (needs `pip install symengine`):

```sh
python -m pyt2c pdemodel.txt --emit-app myapp     # or --from-header generated/my_model.hpp
```

The CI `consumer_mms_reactdiff` smoke exercises exactly this install→build→run→verify path.

## Ways to invoke `--emit-app`

```sh
# C++ text2code (also produces generated/my_model.hpp)
text2code pdeapp.txt --emit-app myapp --app-name myapp --model-id 100

# Python (single-stage, pip symengine)
python -m pyt2c pdemodel.txt --emit-app myapp

# Python, from an existing model header — NO .txt input at all
python -m pyt2c --emit-app myapp --from-header generated/my_model.hpp
```

## CMake wiring

`cmake/ExasimEmitApp.cmake` provides `exasim_emit_app()`, which drives
`text2code --emit-app` at build time (and cleanly no-ops when text2code is not
available):

```cmake
include(ExasimEmitApp)
exasim_emit_app(NAME myapp_gen PDEAPP ${src}/pdeapp.txt
                DEST ${bin}/apps/myapp MODEL_ID 100 [ALL])
```

The built-in model build (`backend/Model/BuiltIn/CMakeLists.txt`) exposes the
opt-in `EXASIM_EMIT_BUILTIN_APPS` option: with it ON and `text2code` available,
`cmake --build . --target builtin-apps` emits a standalone header-only app for
every built-in model (1–15), reusing the same `text2code` binary the build
already uses to regenerate their kernels.

## Testing

- `tests/run_model_sweep.sh` — pyt2c vs C++ text2code across **all 47 models** with a
  pdemodel+pdeapp (examples/ + backend/Model/BuiltIn + apps/ + the text2code sample):
  a STRUCTURAL check (identical set of generated methods) + an EXTENDED NUMERIC check
  over every value method + Jacobian (in-process, NaN/Inf-aware). 47/47 byte-identical,
  incl. reactingsharpb2 (reacting flow), isoq3d (3D), nsmach8 (hypersonic), Riemann models.
- `tests/run_appgen_sweep.sh` — emit a standalone app for **all 47 models** and compile
  each emitted driver + my_model.hpp against the real backend + PETSc + Kokkos (full
  templated instantiation). 47/47 compile clean.
- `tests/run_builtin_sweep.sh` — the built-in-only subset (kept for a quick check).
- `tests/run_e2e.sh` — generate → build → run a generated app to convergence (local).
- `tests/run_e2e` on dgx-b — a built-in model app built against real PETSc recovers the
  manufactured solution u=sin(pi x)sin(pi y) (u in [0,1], gradient +-pi).
- `tests/cmake-emit-app/` — minimal project proving `exasim_emit_app()` drives the
  emit at build time.

Note: `my_model.hpp` is a pure function of `pdemodel.txt` in **both** the C++ and Python
paths — `pdeapp.txt` switches (platform, discretization, model=ModelC/D, tdep, wave, ...)
do not change the generated model. The sweeps confirm this holds across the whole corpus.
