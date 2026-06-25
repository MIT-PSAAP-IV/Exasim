# CI and Validation

Exasim validation combines GitHub Actions, local scripts, frontend tests,
consumer package tests, documentation builds, and example-level numerical
checks.

## Current GitHub Actions

| Workflow | Purpose |
| --- | --- |
| `.github/workflows/smoke-cpu.yml` | CPU smoke build, install, CTest, and hygiene checks. |
| `.github/workflows/docs.yml` | Strict MkDocs build and documentation deployment. |
| `.github/workflows/cache-warm.yml` | Scheduled dependency cache warm-up for Kokkos/SymEngine builds. |

## Smoke CPU Workflow

The CPU smoke workflow performs the main repository health check:

1. Run hygiene checks.
2. Install system dependencies on Ubuntu.
3. Configure Exasim with tests enabled.
4. Build the superbuild.
5. Install the package into the build install prefix.
6. Run CTest in the inner Exasim build directory.
7. Upload CTest reports.

The local equivalent is approximately:

```bash
bash tests/check-hygiene.sh
cmake -S Exasim -B Exasim-build -DEXASIM_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX="$PWD/Exasim-build/install"
cmake --build Exasim-build --parallel 2
cmake --install Exasim-build
ctest --test-dir Exasim-build/exasim_build-prefix/src/exasim_build-build --output-on-failure
```

Adjust paths when running from inside the source tree.

## Documentation Workflow

The documentation workflow runs:

```bash
python3 -m pip install -r requirements-docs.txt
python3 -m mkdocs build --strict --clean
```

All navigation entries must resolve. Broken links, missing pages, and invalid
Markdown extensions should be fixed before merging.

## Hygiene Checks

`tests/check-hygiene.sh` rejects common repository mistakes:

- Tracked `.DS_Store` files.
- Tracked prebuilt shared libraries.
- Tracked static libraries under `backend/`.
- Leaked absolute home paths such as `/Users/<name>` or `/home/<name>`.

Run it before pushing documentation, examples, or installation guides.

## Consumer Tests

`tests/run-consumer-tests.sh` validates the installed Exasim package from the
perspective of external CMake users. Use it when changing:

- Installed headers.
- Exported CMake targets.
- `ExasimConfig.cmake` generation.
- Frontend app templates.
- Public package layout.

## Frontend and Application Tests

Frontend tests are primarily driven from MATLAB utilities such as
`frontends/Matlab/Utilities/exasimapptest.m`, with Python and Julia examples
covered through that integration path where configured. Application-level
tests may also run standalone app modes, built-in-library paths, and
Text2Code-generated cases.

## Validation Matrix

| Change area | Minimum validation |
| --- | --- |
| Documentation only | `mkdocs build --strict --clean`, `git diff --check`, hygiene. |
| Frontend option | MATLAB test plus matching Python/Julia smoke if changed. |
| Text2Code parser | Text2Code example and generated app build/run. |
| Installed package | Consumer tests. |
| Backend solver | Representative serial solve and numerical comparison. |
| GPU runtime | CUDA or HIP build and CPU/GPU numerical comparison. |
| MPI runtime | `mpirun` with at least two ranks and rank-local output check. |
| Postprocessing | Solve, then postprocess from saved files without regenerating solve data. |

## Future Validation Goals

Some validation areas are still evolving:

- More deterministic manufactured-solution tests.
- Wider GPU CI coverage for CUDA and HIP systems.
- Partition-invariant MPI numerical gates for more examples.
- Performance regression tracking for representative kernels.

Document limitations clearly when a pull request cannot run a full hardware
matrix locally.

## Related Pages

- [Testing](testing.md)
- [Baselines](baselines.md)
- [Known divergences](known-divergences.md)
- [Installation verification](../install/verification.md)
