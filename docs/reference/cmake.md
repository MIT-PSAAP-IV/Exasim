# CMake Reference

This page documents Exasim CMake options, installed targets, package
components, and common build configurations. For step-by-step installation
instructions, see [Installation](../install/index.md).

## Recommended Superbuild Pattern

```bash
export EXASIM_PREFIX=/path/to/prefix
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

The superbuild configures dependencies, builds Text2Code, builds Exasim, and
installs the package into `CMAKE_INSTALL_PREFIX`.

## Top-Level Superbuild Options

Defined in the root `CMakeLists.txt`:

| Option | Default | Valid values | Description |
| --- | --- | --- | --- |
| `EXASIM_MPI` | `ON` | `ON/OFF` | Build MPI variants. |
| `EXASIM_NOMPI` | `ON` | `ON/OFF` | Build non-MPI variants. |
| `EXASIM_CUDA` | `OFF` | `ON/OFF` | Enable CUDA backend. |
| `EXASIM_HIP` | `OFF` | `ON/OFF` | Enable HIP backend. |
| `EXASIM_LIB` | `ON` | `ON/OFF` | Build static libraries; skip executables in solver layer when enabled. |
| `WITH_PARMETIS` | `ON` | `ON/OFF` | Link METIS/ParMETIS/GKlib support. |
| `WITH_TEXT2CODE` | `OFF` | `ON/OFF` | Build additional text2code-linked executable variants. |
| `WITH_BUILTINMODEL` | `ON` | `ON/OFF` | Forwarded to solver build; built-in model library remains part of package flow. |
| `EXASIM_BUILD_TESTS` | `OFF` | `ON/OFF` | Register CTest tests. |
| `EXASIM_FRONTENDS` | `ON` | `ON/OFF` | Install MATLAB/Python/Julia frontends and app templates. |
| `EXASIM_PIP_INSTALL` | `OFF` | `ON/OFF` | Pip-install Python frontend into configured interpreter. |
| `EXASIM_JULIA_DEVELOP` | `OFF` | `ON/OFF` | Run Julia `Pkg.develop` for installed Julia frontend. |

## Solver-Layer Options

Defined in `install/CMakeLists.txt`:

| Option/cache variable | Default | Description |
| --- | --- | --- |
| `EXASIM_LIB` | `OFF` when configuring `install/` directly | Build/install static libraries only. |
| `WITH_TEXT2CODE` | `OFF` | Build `*t2cEXASIM` executables if generated libraries exist. |
| `WITH_KOKKOSKERNEL` | `OFF` | Build header-only Kokkos-kernel provider executables. |
| `KOKKOSKERNEL_HEADER` | empty | Header defining Kokkos kernel provider; enables `WITH_KOKKOSKERNEL`. |
| `WITH_PARMETIS` | `OFF` when configuring `install/` directly | Link METIS/ParMETIS/GKlib. |
| `PARMETIS_ROOT` | source-tree default | Override ParMETIS root. |
| `METIS_ROOT` | source-tree default | Override METIS root. |
| `GKLIB_ROOT` | source-tree default | Override GKlib root. |
| `EXASIM_TUOLUMNE` | unset | HIP/Kokkos layout selector used by Tuolumne-style builds. |
| `EXASIM_FRONTENDS` | `ON` | Install language frontends and app templates. |
| `EXASIM_PIP_INSTALL` | `OFF` | Mutate Python environment with pip install. |
| `EXASIM_JULIA_DEVELOP` | `OFF` | Mutate Julia environment with `Pkg.develop`. |

## Exported Targets

After installation:

```cmake
find_package(Exasim REQUIRED)
```

Exported targets include:

| Target | Meaning |
| --- | --- |
| `Exasim::headers` | Public include target. |
| `Exasim::cpulib` | CPU non-MPI solver library. |
| `Exasim::cpumpilib` | CPU MPI solver library. |
| `Exasim::gpulib` | GPU non-MPI solver library. |
| `Exasim::gpumpilib` | GPU MPI solver library. |
| `Exasim::cpuprelib` | CPU non-MPI preprocessing/runtime app library. |
| `Exasim::cpumpiprelib` | CPU MPI preprocessing/runtime app library. |
| `Exasim::gpuprelib` | GPU non-MPI preprocessing/runtime app library. |
| `Exasim::gpumpiprelib` | GPU MPI preprocessing/runtime app library. |
| `Exasim::builtinmodelserial` | Static built-in model library for serial/CPU provider. |
| `Exasim::builtinmodelcuda` | Static built-in model library for CUDA provider, if built. |
| `Exasim::builtinmodelhip` | Static built-in model library for HIP provider, if built. |

## Package Components

Components select stable aliases:

```cmake
find_package(Exasim REQUIRED COMPONENTS cpumpi)
target_link_libraries(app PRIVATE Exasim::headers Exasim::pre Exasim::builtinmodel)
```

| Component | Alias `Exasim::pre` | Alias `Exasim::builtinmodel` |
| --- | --- | --- |
| `cpu` | `Exasim::cpuprelib` | `Exasim::builtinmodelserial` |
| `cpumpi` | `Exasim::cpumpiprelib` | `Exasim::builtinmodelserial` |
| `gpu` | `Exasim::gpuprelib` | `Exasim::builtinmodelcuda` |
| `gpumpi` | `Exasim::gpumpiprelib` | `Exasim::builtinmodelcuda` |

For HIP-only installs, link `Exasim::builtinmodelhip` directly where needed.

## Common Build Configurations

### CPU with MPI and frontends

```bash
cmake -S Exasim -B Exasim-build \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix \
  -DEXASIM_MPI=ON \
  -DEXASIM_NOMPI=ON \
  -DEXASIM_FRONTENDS=ON
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

### NVIDIA GPU

```bash
cmake -S Exasim -B Exasim-build-cuda \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix \
  -DEXASIM_CUDA=ON \
  -DEXASIM_HIP=OFF
cmake --build Exasim-build-cuda -j8
cmake --install Exasim-build-cuda
```

### AMD GPU

```bash
cmake -S Exasim -B Exasim-build-hip \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix \
  -DEXASIM_HIP=ON \
  -DEXASIM_CUDA=OFF
cmake --build Exasim-build-hip -j8
cmake --install Exasim-build-hip
```

## External Built-In Models

After `find_package(Exasim)`, consumers can call:

```cmake
exasim_add_external_builtin_model(
  TARGET mymodel
  ID 101
  PDEMODEL path/to/pdeapp.txt)
```

Exactly one of `PDEMODEL`, `SOURCES`, or `KERNELS` is required. See
[External built-in library](../usage-modes/external-builtin.md).

## Notes

- CUDA and HIP should not both be enabled in the same build.
- GPU builds rely on a compatible Kokkos backend.
- MPI targets require MPI at package-consume time when MPI variants are
  exported.
- Frontend-generated apps should prefer imported Exasim targets rather than
  manually linking libraries.
