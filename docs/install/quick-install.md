# Quick Installation

This is the recommended starting point for most users on CPU systems.

!!! tip "Use this first"
    If this Superbuild completes successfully, you do not need to manually build
    Kokkos, METIS, ParMETIS, SymEngine, or Text2Code. They are handled by the
    Superbuild.

## Prerequisites

Install the basic system tools for your platform:

| Platform | Typical packages |
| --- | --- |
| macOS | `brew install cmake llvm` and optionally `brew install open-mpi` |
| Ubuntu/Debian CPU | `sudo apt install build-essential cmake git libblas-dev liblapack-dev libopenmpi-dev` |
| HPC CPU cluster | Load site compiler, CMake, MPI, BLAS/LAPACK modules. |

See [macOS](macos.md), [Linux CPU](linux-cpu.md), or
[HPC build chain](hpc.md) for platform notes.

## Recommended CPU Superbuild

From the parent directory containing the `Exasim` source tree:

```bash
cmake -S Exasim -B Exasim-build
cmake --build Exasim-build -j8
cmake --install Exasim-build --prefix /path/to/exasim-prefix
```

What this does:

1. Configures the Exasim Superbuild.
2. Builds required dependencies in order.
3. Builds Exasim libraries, model libraries, Text2Code, and runtime data.
4. Installs everything into `/path/to/exasim-prefix`.

## Confirm The Install

```bash
find /path/to/exasim-prefix -name ExasimConfig.cmake
find /path/to/exasim-prefix -name "libbuiltinmodel*"
ls /path/to/exasim-prefix/bin/text2code
```

Then build and run the first example from the
[Quick Start](../getting-started/quickstart.md).

## If This Fails

Do not immediately start manually installing every dependency. First read
[Troubleshooting Superbuild Failures](superbuild-troubleshooting.md). Manual
installation is mainly for systems where:

- the site compiler/MPI stack must be used explicitly;
- network downloads are blocked;
- GPU toolchains need architecture-specific configuration;
- the Superbuild cannot find compatible BLAS/LAPACK, MPI, or compiler tools.

