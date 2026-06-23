# Linux (CPU)

For Linux machines without an NVIDIA or AMD GPU. CPU and MPI variants only.

!!! tip "Recommended path"
    Start with [Quick Installation](quick-install.md). The Superbuild is the
    recommended CPU installation method. If it completes successfully, do not
    manually build the dependencies listed below.

## System packages

```bash
sudo apt install -y build-essential cmake git \
                    libblas-dev liblapack-dev \
                    libopenmpi-dev   # only for MPI builds
```

`libmkl-dev` is optional and faster than the reference LAPACK.

## MPI

System OpenMPI (`apt install libopenmpi-dev`) is sufficient for
CPU+MPI Exasim. `MPI_PREFIX` for use in [`common.md`](common.md):

```bash
MPI_PREFIX=$(dirname $(dirname $(which mpicc)))   # usually /usr
```

## Recommended Superbuild

```bash
export EXASIM_PREFIX=/path/to/prefix
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

See [CPU Installation via Superbuild](superbuild-cpu.md).

## Manual Dependency Fallback

If the Superbuild fails or you need explicit dependency control, follow
[Manual Dependency Installation](manual-dependencies.md) and
[Common Dependencies](common.md) for GKlib, METIS, ParMETIS, Kokkos serial,
SymEngine, and Text2Code.

## Build Exasim

The commands below are manual fallback commands for users who already built
dependencies explicitly.

CPU only:

```bash
cd $EXASIM
cmake -S install -B build_cpu \
    -DEXASIM_CUDA=OFF -DEXASIM_HIP=OFF \
    -DEXASIM_MPI=OFF  -DEXASIM_NOMPI=ON \
    -DKokkos_DIR=$EXASIM/kokkos/buildserial/lib/cmake/Kokkos \
    -DEXASIM_BUILD_LIBRARY_EXAMPLES=ON
cmake --build build_cpu -j16
```

CPU + MPI:

```bash
cmake -S install -B build_mpi \
    -DEXASIM_CUDA=OFF -DEXASIM_HIP=OFF \
    -DEXASIM_MPI=ON   -DEXASIM_NOMPI=OFF \
    -DWITH_PARMETIS=ON \
    -DKokkos_DIR=$EXASIM/kokkos/buildserial/lib/cmake/Kokkos \
    -DEXASIM_BUILD_LIBRARY_EXAMPLES=ON \
    -DMPI_CXX_COMPILER=$(which mpicxx) \
    -DMPI_C_COMPILER=$(which mpicc)
cmake --build build_mpi -j16
```

## Verification

```bash
ctest --test-dir build_cpu -R "poisson2d:.*:cpu" -j 8
ctest --test-dir build_mpi -R "poisson2d:.*:mpi" -j 8
```
