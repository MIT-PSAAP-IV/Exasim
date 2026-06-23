# Internals References

This page collects source files and documentation entry points that are useful
when working on Exasim internals.

## Runtime Entry Points

| File | Purpose |
| --- | --- |
| `include/ExasimSolver.hpp` | Public runtime class declaration and execution-mode-facing API. |
| `include/ExasimSolverSetup.hpp` | Provider selection and top-level solver setup helpers. |
| `backend/Main/ExasimSolver.cpp` | Runtime initialization, solve/postprocess dispatch, parameter sweeps, MPI/Kokkos lifecycle. |
| `backend/Postprocessing/exasim_postprocess.cpp` | Standalone postprocessing executable. |

## Backend Implementation

| Directory | Purpose |
| --- | --- |
| `backend/Common/` | Shared structs, memory helpers, binary I/O, utility functions. |
| `backend/Discretization/` | Residuals, operators, geometry, model-driver calls. |
| `backend/Solution/` | Solve loops, saved solution I/O, output and postprocessing methods. |
| `backend/Solver/` | Newton/GMRES and system vector management. |
| `backend/Preconditioning/` | Preconditioner setup and application. |
| `backend/Preprocessing/` | C++ preprocessing, DMD, mesh/master/app binary generation. |
| `backend/Visualization/` | VTK/PVD and visualization output support. |

## Code Generation and Providers

| Location | Purpose |
| --- | --- |
| `text2code/` | Text2Code parser and code generator. |
| `backend/Model/Text2codeGenerated/` | Text2Code-generated model support. |
| `backend/Model/FrontendGenerated/` | Frontend-generated provider support. |
| `backend/Model/BuiltIn/` | Built-in model library and registry support. |
| `apps/builtinlibrary/` | Built-in dynamic-library ABI app. |
| `apps/sharedlibrary/` | Shared-library ABI app. |
| `cmake/frontend-app/` | Installed frontend-generated app template. |

## Frontend Implementation

| Frontend | Location |
| --- | --- |
| MATLAB | `frontends/Matlab/` |
| Python | `frontends/Python/exasim/` |
| Julia | `frontends/Julia/Exasim/` |

## Tests and CI

| Location | Purpose |
| --- | --- |
| `tests/check-hygiene.sh` | Repository hygiene checks. |
| `tests/run-consumer-tests.sh` | Installed package consumer tests. |
| `frontends/Matlab/Utilities/exasimapptest.m` | Cross-frontend/application integration test entry point. |
| `.github/workflows/smoke-cpu.yml` | CPU smoke and CTest workflow. |
| `.github/workflows/docs.yml` | Documentation build workflow. |

## Related Documentation

- [Quickstart](../getting-started/quickstart.md)
- [Installation](../install/index.md)
- [Frontends](../frontends/index.md)
- [Application Modes](../usage-modes/index.md)
- [Text2Code Applications](../text2code-applications/index.md)
- [Physics Models](../physics-models/index.md)
- [Theory](../theory/index.md)
- [Using AI Tools](../using-ai-tools/index.md)
