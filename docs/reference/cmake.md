# CMake reference

Build options, exported package targets, and the external-model helper. For the
build procedures that use them, see [Installation](../install/index.md); for the
consumer link lines, see [Application Modes](../usage-modes/index.md).

## Build options

The first group is declared by the top-level superbuild (`CMakeLists.txt`) and
forwarded to the solver layer; the second group is declared in
`install/CMakeLists.txt` and the vendored `text2code` build.

| Option | Default | Meaning |
|---|---|---|
| `EXASIM_MPI` | `ON` | Build MPI solver variants (`cpumpi…`, `gpumpi…`). |
| `EXASIM_NOMPI` | `ON` | Build non-MPI variants (`cpu…`, `gpu…`). |
| `EXASIM_CUDA` | `OFF` | Enable the CUDA backend; builds `gpu*` targets and `builtinmodelcuda`. |
| `EXASIM_HIP` | `OFF` | Enable the HIP (AMD) backend; builds `gpu*` targets and `builtinmodelhip`. |
| `EXASIM_LIB` | `ON` | Build/install static libraries only; skip Exasim executables. |
| `WITH_PARMETIS` | `ON` | Link METIS/ParMETIS/GKlib (defines `HAVE_PARMETIS`). |
| `WITH_TEXT2CODE` | `OFF` | Also build `*t2cEXASIM` executables that link text2code-generated dynamic libraries. |
| `WITH_BUILTINMODEL` | `ON` | Forwarded to the solver build (the built-in model library is built regardless). |
| `EXASIM_FRONTENDS` | `ON` | Install the Python / Julia / MATLAB frontends. |
| `EXASIM_BUILD_TESTS` | `OFF` | Register the ctest regression suite. |
| `EXASIM_PIP_INSTALL` | `OFF` | pip-install the `exasim` Python package at install time. |
| `EXASIM_JULIA_DEVELOP` | `OFF` | `Pkg.develop` the installed `Exasim.jl` at install time. |
| `WITH_KOKKOSKERNEL` | `OFF` | (solver layer) build executables using header-only Kokkos kernel providers. |
| `WITH_METIS` | `ON` | (text2code) build text2code with METIS/GKlib. |

!!! note "Not options, but cache variables"
    `EXASIM_TEXT2CODE` is a *path* the superbuild sets to the built text2code
    binary and forwards. `EXASIM_TUOLUMNE` is an undeclared cache variable a
    consumer may pass (`-DEXASIM_TUOLUMNE=ON`) to select a Tuolumne Kokkos build
    under HIP. When configuring `install/` directly (not via the superbuild),
    `EXASIM_LIB` and `WITH_PARMETIS` default to `OFF` — the superbuild forces
    them `ON` with `-D`.

## Exported targets and components

`find_package(Exasim)` always exports `Exasim::headers`. Requesting
`COMPONENTS cpu | cpumpi | gpu | gpumpi` does not create new targets — it
resolves a variant, synthesizes the stable chooser aliases `Exasim::pre` and
`Exasim::builtinmodel` for it, and verifies the variant was built.

| Target | What it is |
|---|---|
| `Exasim::headers` | INTERFACE include target (`include/`, C++20). Required by every consumer. |
| `Exasim::cpulib` / `Exasim::cpumpilib` | CPU solver static libs (non-MPI / MPI). |
| `Exasim::gpulib` / `Exasim::gpumpilib` | GPU solver static libs (non-MPI / MPI). |
| `Exasim::cpuprelib` / `Exasim::cpumpiprelib` | CPU preprocessing static libs. |
| `Exasim::gpuprelib` / `Exasim::gpumpiprelib` | GPU preprocessing static libs. |
| `Exasim::builtinmodelserial` / `…cuda` / `…hip` | Built-in model static libs per backend. |
| `Exasim::pre` *(chooser alias)* | the `*prelib` for the first requested component. |
| `Exasim::builtinmodel` *(chooser alias)* | the built-in model lib for the first requested component. |

| Component | Selects | `Exasim::pre` → | `Exasim::builtinmodel` → |
|---|---|---|---|
| `cpu` | non-MPI CPU | `cpuprelib` | `builtinmodelserial` |
| `cpumpi` | MPI CPU | `cpumpiprelib` | `builtinmodelserial` |
| `gpu` | non-MPI GPU | `gpuprelib` | `builtinmodelcuda` |
| `gpumpi` | MPI GPU | `gpumpiprelib` | `builtinmodelcuda` |

!!! note
    The chooser maps `gpu`/`gpumpi` to `builtinmodelcuda`; a HIP-only install
    still exports `Exasim::builtinmodelhip` as a plain target, so link it
    directly there. The built-in model libs are **static** archives despite a
    "shared" mention in the config header.

## `exasim_add_external_builtin_model()`

Registers an [external built-in model](../usage-modes/external-builtin.md).
Exactly one of `PDEMODEL` / `SOURCES` / `KERNELS` is required.

| Argument | Required? | Meaning |
|---|---|---|
| `TARGET` | yes | Name of the provider library target to create. |
| `ID` | yes | Model ID this provider intercepts; namespace `exasim_model_<ID>`. Other IDs fall through to `Exasim::builtinmodel*`. |
| `PDEMODEL` | one-of | Path to a `pdeapp*.txt`; runs text2code at build time to generate kernels. |
| `SOURCES` | one-of | Hand-written `model.hpp`/`model.cpp` (+ extras) in `exasim_model_<ID>`. |
| `KERNELS` | one-of | Directory of pre-generated kernel `.cpp` files (frontend gencode output). |
| `SHARED` | optional | Build a SHARED provider (frontend dynamic-model path; Kokkos as flags only). Default is STATIC, linking `Kokkos::kokkos`. |

The target links `Exasim::headers` PUBLIC and the appropriate
`Exasim::builtinmodel*` for fallthrough; consumers must **not** also link
`Exasim::builtinmodel` directly.
