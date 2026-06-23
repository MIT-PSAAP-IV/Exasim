# GPU Installation Overview

GPU installations are more complex than CPU installations because the compiler,
Kokkos backend, MPI runtime, GPU architecture, and scheduler environment must
all match the target hardware.

Use this page to choose the right GPU path, then follow the detailed NVIDIA,
AMD, or HPC guide.

## GPU Decision Guide

| System | Go to |
| --- | --- |
| NVIDIA workstation or server | [Linux NVIDIA GPU](linux-nvidia.md) |
| AMD workstation or server | [Linux AMD GPU](linux-amd.md) |
| Frontier | [Frontier](frontier.md) |
| Tuolumne | [Tuolumne](tuolumne.md) |
| Other GPU cluster | [Generic HPC](generic-hpc.md) |

## Why GPU Builds Need More Configuration

GPU builds require:

- CUDA or ROCm/HIP toolchains;
- a Kokkos build configured for the target backend;
- a GPU architecture flag such as `sm_80`, `gfx90a`, or `gfx942`;
- matching Exasim GPU package variants;
- GPU-aware MPI for MPI+GPU runs;
- scheduler allocation of GPUs at runtime.

## NVIDIA CUDA Systems

NVIDIA builds require:

- CUDA toolkit and compiler support;
- Kokkos with CUDA enabled;
- a CUDA architecture matching the GPU;
- CUDA-aware MPI for multi-rank GPU runs.

Start with [Linux NVIDIA GPU](linux-nvidia.md). Common Kokkos/CUDA targets:

| GPU | Typical architecture |
| --- | --- |
| V100 | `VOLTA70` / `sm_70` |
| A100 | `AMPERE80` / `sm_80` |
| H100 | `HOPPER90` / `sm_90` |

## AMD HIP Systems

AMD builds require:

- ROCm and `hipcc`;
- Kokkos with HIP enabled;
- a HIP offload architecture matching the GPU;
- GPU-aware MPI for multi-rank GPU runs.

Start with [Linux AMD GPU](linux-amd.md). For DOE Cray systems, use:

- [Frontier](frontier.md) for MI250X / `gfx90a`;
- [Tuolumne](tuolumne.md) for MI300A / `gfx942`.

## Verification

After a GPU install, verify both package discovery and a small app build:

```bash
find /path/to/exasim-prefix -name "libgpuEXASIM*" -o -name "libgpumpiEXASIM*"
find /path/to/exasim-prefix -name "builtinmodelcuda*" -o -name "builtinmodelhip*"
```

Then use [Verification and Testing](verification.md).

