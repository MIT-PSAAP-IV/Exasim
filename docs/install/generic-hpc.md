# Generic HPC

For a cluster without a dedicated page, follow the shared
[HPC build chain](hpc.md) and substitute your site's modules, compilers, and
scheduler. This page collects the backend-selection knobs that vary by machine.

## Kokkos build per backend

Build Kokkos once per backend you need, into its own tree under `deps/kokkos/`:

| Backend | Key CMake flags | Notes |
|---|---|---|
| Serial (CPU) | `-DKokkos_ENABLE_SERIAL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON` | host compiler via `-DCMAKE_CXX_COMPILER` |
| CUDA (NVIDIA) | `-DCMAKE_CXX_COMPILER=clang++ -DKokkos_ENABLE_CUDA=ON` | cmake ≥ 3.20, clang ≥ 12, CUDA ≥ 11 |
| HIP (AMD) | `-DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_ROCM=ON -DKokkos_ARCH_<DEV>=ON` | set `CRAYPE_LINK_TYPE=dynamic` on Cray |

Set the architecture macro for your device (see the
[Kokkos arch table](hpc.md#kokkos-architecture-macros)). Example HIP build for an
MI250X:

```bash
cd deps/kokkos && mkdir buildhip && cd buildhip
cmake .. -DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_ROCM=ON \
  -DKokkos_ARCH_VEGA90A=ON -DCMAKE_INSTALL_PREFIX=../buildhip
make install
```

## Exasim configure matrix

Select the variants with the `EXASIM_*` options (they compose — you can build
several backends in one configure). See the full
[options table](hpc.md#exasim-configure-options).

| Target variant | Options |
|---|---|
| Serial | `-DEXASIM_NOMPI=ON` |
| MPI | `-DEXASIM_MPI=ON` |
| Serial + MPI | `-DEXASIM_NOMPI=ON -DEXASIM_MPI=ON` |
| Serial + CUDA | `-DEXASIM_NOMPI=ON -DEXASIM_CUDA=ON` |
| MPI + CUDA | `-DEXASIM_MPI=ON -DEXASIM_CUDA=ON` |
| Serial + HIP | `-DEXASIM_NOMPI=ON -DEXASIM_HIP=ON` |
| MPI + HIP | `-DEXASIM_MPI=ON -DEXASIM_HIP=ON` |

CUDA builds use `-DCMAKE_CXX_COMPILER=clang++`; HIP builds use `hipcc` (or
`mpiamdclang++` for MPI+HIP on Cray). Point `-DKokkos_DIR` at the matching Kokkos
tree built above.

## Running

| Variant | Launch |
|---|---|
| Serial | `./cpuEXASIM 1 datain/ dataout/out` |
| CUDA / HIP, one GPU | `./gpuEXASIM 1 datain/ dataout/out` |
| MPI | `mpirun -np <N> ./cpumpiEXASIM 1 datain/ dataout/out` |
| CUDA, many GPUs (LSF) | `jsrun --smpiargs="-gpu" -n1 -a4 -c4 -g4 ./gpumpiEXASIM 1 datain/ dataout/out` |
| HIP, many GPUs (Flux) | `flux run -N2 -n4 -g1 -o gpu-affinity=per-task --exclusive ./gpumpiEXASIM 1 datain/ dataout/out` |

Set `MPICH_GPU_SUPPORT_ENABLED=1` for GPU-aware MPI. Replace `<N>` with the rank
count from your `pdeapp.txt`.

!!! note
    Out-of-tree consumers (the [usage modes](../usage-modes/index.md)) launch
    their own executable — e.g. `mpirun -np <N> build/consumer_builtin pdeapp.txt`
    — rather than the in-tree `cpumpiEXASIM` binaries shown above.
