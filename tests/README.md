# Exasim regression tests

One entry point, one ctest.

```bash
# from the repo root, with the build toolchain on PATH
bash tests/run-tests.sh
```

`run-tests.sh` configures the superbuild with `-DEXASIM_BUILD_TESTS=ON`, builds
(reusing already-built vendored deps), installs to `<build>/install`, and runs
`ctest`.

## What runs

| ctest | What it checks |
|---|---|
| `consumer_builtin_cpu` | CPU-only out-of-tree consumer (NP=1): **B3** the consumer never reaches into the Exasim source tree, **B2** it builds purely via `find_package(Exasim)` against an install prefix, **B4** it runs Poisson 2D and meets the QoI gate. |
| `consumer_builtin_mpi` | Same, CPU+MPI variant (`mpirun -np 2`). |
| `frontend_python` | End-to-end Poisson 2D through the installed `exasim` Python package: SymPy gencode → external-model build via `find_package(Exasim)` → run → QoI gate. |
| `frontend_julia` | Same through the `Exasim.jl` package. **SKIPs** cleanly when `julia` or `SymPy.jl` is missing. |
| `frontend_matlab` | Same through the MATLAB frontend (`exasim_setup.m`). **SKIPs** when MATLAB (PATH or `/Applications/MATLAB_*.app`) or the Symbolic Math Toolbox is missing. |

All numerical gates are `Domain_QoI1` (integral of `(u-uexact)^2`)
`< QOI_TOL` (default `1e-8`; the runs produce ~`5e-13`).

CI (`.github/workflows/smoke-cpu.yml`) runs the consumers plus
`frontend_python`; the julia/matlab tests SKIP there and run wherever those
toolchains exist locally.

Consumers exercised:
- `consumers/builtin/` — out-of-tree built-in-model consumer (Poisson 2D).
- `consumers/facade/` — headers-resolve consumer; builds (no run).

Add a consumer by dropping a `find_package(Exasim)` project under `consumers/`;
the harness picks it up automatically. If it ships a `pdeapp.txt`, B4 runs it
and applies the QoI gate.

## Frontend tests

`frontends/run-frontend-test.sh` (driven by ctest, also runnable by hand):

```bash
EXASIM_ROOT=$PWD FRONTEND=python bash tests/frontends/run-frontend-test.sh
```

Each test copies `tests/frontends/<lang>/pdeapp.*` + `pdemodel.*` into a
scratch directory and runs it there — `datain/`, `dataout/`, and the hidden
`.exasim/` build dir land in the scratch dir, never in the repo. Honours
`EXASIM_INSTALL` (default `$EXASIM_ROOT/build/install`), `QOI_TOL`, `PYTHON3`.
Exit 77 = dependency missing = ctest SKIP.

## Knobs

`run-consumer-tests.sh` honours: `NP` (MPI ranks, default 2), `QOI_TOL`,
`INSTALL_PREFIX`, `KOKKOS_DIR`, `EXASIM_ROOT`, `CC`/`CXX`. `run-tests.sh`
honours `EXASIM_BUILD_DIR` (superbuild dir) and `CMAKE_ARGS` (extra configure
flags), and forwards trailing args to `ctest` (e.g. `-R`, `-V`).

## Archived

Superseded in-tree consumers and bespoke runners are under `archive/` — see
`archive/README.md`. The legacy numerical baseline lives in `../old/baseline/`
(kept only for the historical port notes; superseded by the B4 QoI gate).
