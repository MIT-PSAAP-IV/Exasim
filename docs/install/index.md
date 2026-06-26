# Installation

Most users should start with the **Superbuild**.

!!! tip "Recommended default"
    For most CPU systems, use the Superbuild installation. If the Superbuild
    completes successfully, no further manual dependency installation is
    required. The Superbuild downloads, configures, builds, and installs the
    required dependencies in the correct order.

## Installation Decision Tree

| Your system | Recommended path |
| --- | --- |
| CPU workstation or laptop | [Quick Installation](quick-install.md) |
| CPU cluster | Try [CPU Superbuild](superbuild-cpu.md) first, then use site MPI/compiler settings if needed. |
| NVIDIA GPU system | Read [GPU Installation](gpu.md), then follow [Linux NVIDIA GPU](linux-nvidia.md). |
| AMD GPU system | Read [GPU Installation](gpu.md), then follow [Linux AMD GPU](linux-amd.md). |
| Frontier or Tuolumne | Use the platform-specific guide: [Frontier](frontier.md) or [Tuolumne](tuolumne.md). |
| Superbuild fails | Use [Troubleshooting Superbuild Failures](superbuild-troubleshooting.md), then [Manual Dependency Installation](manual-dependencies.md) if needed. |

## Recommended Installation Hierarchy

1. [Quick Installation](quick-install.md) for the fastest CPU install.
2. [CPU Installation via Superbuild](superbuild-cpu.md) for the recommended
   default workflow and options.
3. [Troubleshooting Superbuild Failures](superbuild-troubleshooting.md) if the
   default build does not complete.
4. [Manual Dependency Installation](manual-dependencies.md) only when the
   Superbuild cannot be used or when a site requires explicit dependency builds.
5. [GPU Installation](gpu.md) for CUDA/HIP overview and backend selection.
6. Platform-specific guides for macOS, Linux CPU, NVIDIA, AMD, Frontier,
   Tuolumne, and generic HPC.
7. [Verification and Testing](verification.md) to confirm the install works.

## What The Superbuild Does

The Superbuild is the top-level CMake workflow:

```bash
export EXASIM_PREFIX=/path/to/prefix
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

It is preferred because it builds the Exasim stack in the correct order and
keeps users from manually managing dependency paths.

Depending on enabled options and what is already available on the system, the
Superbuild handles:

| Component | Purpose |
| --- | --- |
| Kokkos | CPU/GPU portability layer. |
| GKlib, METIS, ParMETIS | Graph partitioning and MPI domain decomposition. |
| SymEngine | Symbolic expression backend for Text2Code. |
| Text2Code | Generates PDE kernels and input data from `pdeapp.txt`/`pdemodel.txt`. |
| Exasim libraries | Solver, preprocessing, postprocessing, and model libraries. |
| Runtime data | Installed master/gauss node tables and package metadata. |
| Frontends | MATLAB/Python/Julia setup when frontend installation is enabled. |

## Platform Guides

| Platform | Page |
| --- | --- |
| macOS CPU | [macOS](macos.md) |
| Linux CPU | [Linux CPU](linux-cpu.md) |
| Linux NVIDIA GPU | [Linux NVIDIA GPU](linux-nvidia.md) |
| Linux AMD GPU | [Linux AMD GPU](linux-amd.md) |
| Shared HPC reference | [HPC build chain and options](hpc.md) |
| Frontier | [Frontier](frontier.md) |
| Tuolumne | [Tuolumne](tuolumne.md) |
| Other clusters | [Generic HPC](generic-hpc.md) |

## Repository Folder Name

The cloned folder should be named `Exasim`, and the source/build paths should
not contain whitespace. Kokkos and downstream build scripts are sensitive to
paths with spaces.

## After Installation

Continue with:

- [Verification and Testing](verification.md)
- [Quick Start](../getting-started/quickstart.md)
- [Application Modes](../usage-modes/index.md)
