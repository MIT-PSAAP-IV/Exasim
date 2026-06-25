# Repository Structure

This page describes the major source-tree directories from a developer
perspective. It is not a user installation guide; see
[Installation](../install/index.md) for build commands and platform setup.

## Top-Level Layout

```text
Exasim/
├── apps/                Standalone application cases and app-mode examples
├── backend/             C++ runtime, solvers, preprocessing, postprocessing
├── cmake/               Installed CMake templates and helper modules
├── deps/                Third-party dependency source/build trees
├── docs/                MkDocs documentation
├── examples/            MATLAB/Python/Julia frontend examples
├── frontends/           MATLAB, Python, and Julia frontend implementations
├── include/             Installed public C++ headers
├── install/             Superbuild, package, and install scripts
├── tests/               Hygiene, consumer, frontend, and integration tests
├── text2code/           Text2Code parser and code-generation tool
└── CMakeLists.txt       Superbuild entry point
```

## Directory Responsibilities

| Directory | Developer responsibility |
| --- | --- |
| `apps/` | Reference standalone applications, built-in-library apps, shared-library apps, and Text2Code input examples. |
| `examples/` | Frontend examples intended to be run from MATLAB, Python, or Julia. |
| `frontends/` | Language-specific frontend APIs, preprocessing helpers, code generation, compile/run wrappers, and setup scripts. |
| `backend/` | Runtime implementation. Changes here affect all frontends, app modes, CPU/GPU/MPI builds, and standalone executables. |
| `include/` | Public installed headers used by downstream applications. Treat these as compatibility-sensitive. |
| `cmake/` | Templates and installed CMake files used by generated standalone apps and consumers. |
| `install/` | Superbuild logic, Exasim package configuration, runtime data installation, and frontend setup generation. |
| `text2code/` | Text input parsing and source generation from `pdeapp.txt` and `pdemodel.txt`. |
| `tests/` | Local and CI validation scripts. Add regression coverage here when behavior changes. |
| `docs/` | User and developer documentation. Keep internals docs implementation-focused. |
| `deps/` | Third-party dependencies. Do not vendor generated build artifacts unless explicitly required. |

## Backend Subdirectories

```text
backend/
├── Common/              Shared structs, memory helpers, binary I/O, utilities
├── Discretization/      Element operators, residuals, Jacobians, DG/HDG helpers
├── Main/                ExasimSolver runtime entry points
├── Model/               Model providers and generated model support
├── PointLocator/        Point search and interpolation utilities
├── Postprocessing/      Standalone postprocessing and extraction utilities
├── Preconditioning/     Preconditioner setup and application
├── Preprocessing/       C++ preprocessing and binary input generation
├── Solution/            Solve loops, output, postprocessing, solution I/O
├── Solver/              Newton, GMRES, system vectors, solver state
└── Visualization/       VTK/PVD writers and visualization field helpers
```

## Frontend Subdirectories

Each language frontend follows the same broad responsibilities but uses
language-specific naming and packaging:

```text
frontends/
├── Matlab/
│   ├── Gencode/         Generated code, compile helpers, exportapp
│   ├── Preprocessing/   initializepde, preprocessing, writeapp paths
│   ├── Postprocessing/  fetchsolution, visualization, frontend workflows
│   ├── Mesh/            Mesh generation helpers
│   └── Utilities/       setup, paths, tests, helpers
├── Python/
│   └── exasim/          Python package mirroring frontend workflow
└── Julia/
    └── Exasim/          Julia package mirroring frontend workflow
```

When adding a user-facing option, check all three frontends plus Text2Code
unless the feature is intentionally language-specific.

## Generated and Build Directories

Generated data commonly appears in:

| Path | Meaning |
| --- | --- |
| `datain/` | Binary input files consumed by standalone C++ executables. |
| `dataout/` | Solver outputs, residuals, QoI files, and visualization data. |
| `build/` | CMake build directory for an app or example. |
| `local/` | Preferred in-tree install prefix for local Exasim development. |
| `backend/Model/Text2codeGenerated/` | Text2Code-generated model files. Some source files are templates; generated artifacts should be ignored. |
| `backend/Model/FrontendGenerated/` | Frontend-generated provider files. Generated artifacts should be ignored. |

Do not assume generated directories are clean. Developer scripts and examples
may overwrite them during normal workflows.

## Where to Start

| Task | Start in |
| --- | --- |
| New `pde` option | `frontends/*/Preprocessing`, `frontends/*/Gencode`, `text2code/`, `backend/Preprocessing`. |
| New runtime flag | `backend/Common/common.h`, binary read/write paths, frontends, Text2Code. |
| Generated app compile issue | `cmake/frontend-app`, `include/ExasimSolverSetup.hpp`, relevant provider CMake file. |
| Postprocessing bug | `backend/Solution`, `backend/Postprocessing`, visualization writers. |
| GPU stale data or crash | `backend/Common`, `backend/Discretization`, `backend/Solution`, Kokkos/CUDA/HIP compile definitions. |
| MPI divergence | `backend/Preprocessing/parmetisexasim.*`, DMD setup, send/receive paths, rank-local output. |
| Package consumer failure | `install/CMakeLists.txt`, exported targets, `tests/consumers/`. |

## Source-Control Expectations

- Track source files, templates, documentation, tests, and curated examples.
- Do not track local build directories, installed libraries, `.DS_Store`, or
  generated model artifacts unless there is a specific reason.
- Run `bash tests/check-hygiene.sh` before opening a pull request.
