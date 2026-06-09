# Exasim regression tests

One entry point, one ctest.

```bash
# from the repo root, with the build toolchain on PATH
bash tests/run-tests.sh
```

`run-tests.sh` configures the superbuild with `-DEXASIM_BUILD_TESTS=ON`, builds
(reusing already-built vendored deps), and runs `ctest`.

## What runs

| ctest | What it checks |
|---|---|
| `consumer_out_of_tree` | For each project under `consumers/`: **B3** the consumer never reaches into the Exasim source tree, **B2** it builds purely via `find_package(Exasim)` against an install prefix, **B4** it runs and meets the QoI gate. Catches export/packaging regressions an in-tree build would miss. |

Consumers exercised:
- `consumers/builtin/` — out-of-tree built-in-model consumer; builds and runs the
  Poisson 2D problem, gated on `QoI[1]` (L² error²) `< QOI_TOL` (default `1e-8`;
  the run produces ~`5e-13`).
- `consumers/facade/` — headers-resolve consumer; builds (no run).

Add a consumer by dropping a `find_package(Exasim)` project under `consumers/`;
the harness picks it up automatically. If it ships a `pdeapp.txt`, B4 runs it and
applies the QoI gate.

## Knobs

`run-consumer-tests.sh` honours: `NP` (MPI ranks, default 2), `QOI_TOL`,
`INSTALL_PREFIX`, `KOKKOS_DIR`, `EXASIM_ROOT`, `CC`/`CXX`. `run-tests.sh` honours
`BUILD` (superbuild dir) and `CMAKE_ARGS` (extra configure flags), and forwards
trailing args to `ctest` (e.g. `-R`, `-V`).

## Archived

Superseded in-tree consumers and bespoke runners are under `archive/` — see
`archive/README.md`. The legacy numerical baseline lives in `../baseline/`
(kept only for the historical port notes; superseded by the B4 QoI gate).
