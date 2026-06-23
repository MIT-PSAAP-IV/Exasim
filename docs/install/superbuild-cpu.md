# CPU Installation Via Superbuild

The CPU Superbuild is the recommended default installation method for Exasim.
It is intended to work on standard CPU workstations, laptops, and many CPU
cluster environments.

## Why Superbuild Is Preferred

The Superbuild:

- builds dependencies in the correct order;
- installs Exasim into one prefix;
- avoids hand-maintained dependency paths;
- provides a CMake package usable by downstream apps;
- installs Text2Code and runtime data when enabled;
- reduces version mismatch errors between Exasim, Kokkos, and generated models.

If it succeeds, manual dependency installation is unnecessary.

## Basic CPU Build

```bash
cd /path/to/workspace
export EXASIM_PREFIX=/path/to/prefix
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

Use a build directory outside the source tree. The examples above assume:

```text
/path/to/workspace/
  Exasim/
  Exasim-build/
```

## Useful CPU Options

| Option | Typical use |
| --- | --- |
| `-DEXASIM_FRONTENDS=ON` | Install MATLAB/Python/Julia frontend setup files. |
| `-DEXASIM_MPI=ON` | Build MPI-capable libraries. |
| `-DEXASIM_NOMPI=ON` | Build non-MPI CPU libraries. |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build on single-config generators. |
| `-DCMAKE_C_COMPILER=...` / `-DCMAKE_CXX_COMPILER=...` | Use a specific compiler. |
| `-DMPI_C_COMPILER=...` / `-DMPI_CXX_COMPILER=...` | Use a specific MPI wrapper. |

Example with explicit MPI wrappers:

```bash
cmake -S Exasim -B Exasim-build \
  -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX \
  -DMPI_C_COMPILER=$(which mpicc) \
  -DMPI_CXX_COMPILER=$(which mpicxx)
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

## Install Prefix

Prefer a user-writable prefix:

```bash
export EXASIM_PREFIX=$HOME/exasim-local
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

Avoid installing into `/usr/local` unless you intentionally want a system-wide
installation and have permission.

## Expected Layout

```text
/path/to/exasim-prefix/
  bin/text2code
  include/
  lib/ or lib64/
  lib/cmake/Exasim/ or lib64/cmake/Exasim/
  share/exasim/
```

The `share/exasim/` directory contains runtime data such as
`masternodes.bin` and `gaussnodes.bin`.

## Next

- [Verification and Testing](verification.md)
- [Quick Start](../getting-started/quickstart.md)
- [Troubleshooting Superbuild Failures](superbuild-troubleshooting.md)
