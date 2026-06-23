# Testing

Exasim testing is a layered process. Small changes should run targeted local
checks first, while solver, package, GPU, MPI, or frontend changes require a
larger validation matrix.

## Test Categories

| Category | Purpose | Typical command |
| --- | --- | --- |
| Hygiene | Catch forbidden tracked files and leaked home paths. | `bash tests/check-hygiene.sh` |
| Documentation | Verify MkDocs navigation, links, and Markdown configuration. | `python3 -m mkdocs build --strict --clean` |
| CMake/CTest | Run configured build tests for the current build directory. | `ctest --test-dir <build-dir> --output-on-failure` |
| Consumer package | Verify installed headers and CMake targets from an external project. | `bash tests/run-consumer-tests.sh <build> <work> <prefix>` |
| Frontend integration | Run MATLAB-driven tests that exercise MATLAB, Python, Julia, and app modes. | `exasimapptest(...)` |
| Example validation | Build and run representative apps/examples for changed physics or workflow. | App-specific commands |

## Hygiene Tests

`tests/check-hygiene.sh` is intentionally fast and should be run before
committing documentation or example changes.

It checks:

- No tracked `.DS_Store`.
- No tracked prebuilt shared libraries.
- No tracked static libraries under `backend/`.
- No leaked absolute home paths such as `/Users/<name>` or `/home/<name>`.

## Documentation Tests

Run:

```bash
python3 -m mkdocs build --strict --clean
```

Use this for any change under `docs/` or `mkdocs.yml`. Strict builds catch
missing pages and unresolved navigation entries.

## CTest

When Exasim is configured with tests enabled, run:

```bash
ctest --test-dir <inner-exasim-build-dir> --output-on-failure
```

For a superbuild, the inner build directory is commonly under:

```text
Exasim-build/exasim_build-prefix/src/exasim_build-build
```

Use `ctest -N` to list available tests before running a long suite.

## Consumer Tests

Consumer tests validate Exasim as an installed CMake package. They are
important when modifying public headers, exported targets, install layout, or
frontend app templates.

```bash
bash tests/run-consumer-tests.sh Exasim-build /tmp/exasim-consumer-work /tmp/exasim-install
```

The script installs Exasim, configures external consumers with
`CMAKE_PREFIX_PATH`, builds them, and rejects source-tree-only assumptions.

## Frontend Integration Tests

The MATLAB utility `frontends/Matlab/Utilities/exasimapptest.m` is the
primary integration test driver for language frontends and app modes. It can
exercise MATLAB, Julia, Python, built-in-library, and standalone workflows
depending on the selected test set and local environment.

Run from MATLAB with the required frontend dependencies available:

```matlab
exasimapptest()
```

Use narrower arguments when debugging a specific frontend or app mode.

## Adding Tests

When adding coverage, choose the narrowest test that can catch the
regression:

| Regression risk | Good test |
| --- | --- |
| Broken install package | Consumer test. |
| Broken frontend serialization | Frontend integration test. |
| Broken Text2Code keyword | Minimal `pdeapp.txt` app and generated executable run. |
| Broken GPU parameter update | CPU/GPU comparison on a small sweep. |
| Broken MPI ownership | Serial/MPI comparison with partition-invariant metric. |
| Broken postprocessing read path | Solve once, then postprocess saved records. |

## Numerical Test Guidance

Do not rely on raw binary equality across compilers, BLAS libraries, GPU
backends, or MPI partition counts. Prefer:

- Analytical or manufactured-solution errors when available.
- QoI values when they are partition-invariant.
- Element-ID-aligned relative norms.
- Residual convergence and physical sanity checks.

Raw byte equality is useful only for tightly controlled deterministic paths.

## Common Test Failures

| Failure | Likely cause |
| --- | --- |
| `mkdocs build --strict` missing page | `mkdocs.yml` nav references a file that does not exist. |
| Hygiene leaked path | Documentation or scripts contain a user-specific `/Users/...` or `/home/...` path. |
| Consumer cannot find target | Install/export target or package config changed incorrectly. |
| Frontend test passes MATLAB but fails Julia/Python | Feature propagated only through one frontend. |
| MPI test output differs by rank count | Ownership, ghost elements, or output aggregation issue. |

## Related Pages

- [CI and validation](ci-and-validation.md)
- [Baselines](baselines.md)
- [Known divergences](known-divergences.md)
- [Development workflow](development-workflow.md)
