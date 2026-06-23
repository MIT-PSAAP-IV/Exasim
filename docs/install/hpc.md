# HPC builds

On CPU-only HPC systems, try the [CPU Superbuild](superbuild-cpu.md) first with
the site compiler and MPI modules loaded. If the Superbuild cannot satisfy site
requirements, use this manual HPC build chain.

On GPU HPC systems, the dependencies are often built explicitly because system
software, GPU architecture flags, MPI wrappers, and scheduler environments are
site-specific. The procedure is the same on every machine; only the modules, GPU
architecture, and job scheduler change.

This page is the shared manual/HPC reference; the per-machine pages give the
concrete values:

- [Frontier](frontier.md) — AMD MI250X (`gfx90a`), Slurm
- [Tuolumne](tuolumne.md) — AMD MI300A (`gfx942`), Flux
- [Generic HPC](generic-hpc.md) — CUDA / HIP / MPI knobs for any cluster

## Dependency build chain

Use this chain only after deciding not to use the Superbuild, or when a platform
guide explicitly requires manual dependency builds.

Build these in order into a single install prefix `$EXASIM_PREFIX` (Kokkos is the
exception — it installs into its own build trees under `deps/kokkos/`):

| # | Component | Source dir | Build tool | Installs to |
|---|---|---|---|---|
| 1 | GKlib | `deps/metis/GKlib` | CMake | `$EXASIM_PREFIX` |
| 2 | METIS | `deps/metis/METIS` | `make config` | `$EXASIM_PREFIX` |
| 3 | ParMETIS | `deps/metis/ParMETIS` | CMake | `$EXASIM_PREFIX` |
| 4 | Kokkos (GPU + serial) | `deps/kokkos` | Make / CMake | `deps/kokkos/build{hip,cuda,serial}` |
| 5 | GMP | downloaded tarball | Autotools | `$EXASIM_PREFIX` |
| 6 | SymEngine | `deps/symengine` | CMake | `$EXASIM_PREFIX` |
| 7 | text2code | `text2code/text2code` | CMake | `$EXASIM_PREFIX/bin` |
| 8 | Exasim | `Exasim/install` | CMake | `$EXASIM_PREFIX` |

## Exasim configure options

The Exasim configure step (#8) is driven entirely by CMake cache variables. The
ones that matter on HPC:

| Option | Example | Purpose |
|---|---|---|
| `EXASIM_HIP` | `ON` | build the AMD GPU (HIP) backend |
| `EXASIM_CUDA` | `ON` | build the NVIDIA GPU (CUDA) backend |
| `EXASIM_MPI` | `ON` | build the MPI variant |
| `EXASIM_NOMPI` | `ON` | build the serial variant (can be combined with `EXASIM_MPI`) |
| `EXASIM_LIB` | `ON` | build the installable libraries |
| `WITH_PARMETIS` | `ON` | enable ParMETIS partitioning |
| `WITH_TEXT2CODE` | `OFF` | do *not* build text2code in-tree; use a prebuilt binary |
| `EXASIM_TEXT2CODE` | `$EXASIM_PREFIX/bin/text2code` | path to the prebuilt text2code |
| `Kokkos_DIR` | `deps/kokkos/buildhip` | the prebuilt Kokkos tree to link against |
| `CMAKE_CXX_COMPILER` | `hipcc` | GPU compiler (HIP); `clang++` for CUDA |
| `CMAKE_HIP_FLAGS` | `--offload-arch=gfx90a` | GPU target architecture |
| `BLAS_LIBRARIES` / `LAPACK_LIBRARIES` | `libsci_amd.so` | vendor BLAS/LAPACK |
| `METIS_*`, `PARMETIS_*`, `GKLIB_*` `_INCLUDE_DIR` / `_LIBRARY_DIR` | `$EXASIM_PREFIX/{include,lib64}` | locate the deps built above |

## Kokkos architecture macros

The Kokkos GPU build needs the macro for your device. Common values:

| Device | Kokkos macro | HIP/CUDA offload flag |
|---|---|---|
| AMD MI250X | `VEGA90A` | `--offload-arch=gfx90a` |
| AMD MI300A / MI300X | `AMD_GFX942` | `--offload-arch=gfx942` |
| NVIDIA V100 | `VOLTA70` | (CUDA) `sm_70` |
| NVIDIA A100 | `AMPERE80` | (CUDA) `sm_80` |
| NVIDIA H100 | `HOPPER90` | (CUDA) `sm_90` |

See the [Kokkos architecture reference](https://kokkos.org/kokkos-core-wiki/API/core/Macros.html)
for the full list.
