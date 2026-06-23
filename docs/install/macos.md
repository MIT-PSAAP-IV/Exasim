# macOS

Tested on Apple Silicon (M1, M2) with Homebrew. Apple's Accelerate
framework provides BLAS / LAPACK; no system MPI required for CPU
builds.

!!! tip "Recommended path"
    Start with [Quick Installation](quick-install.md). On macOS CPU systems,
    the Superbuild is the preferred installation path.

## System packages

```bash
brew install cmake llvm
brew install open-mpi          # only for MPI builds
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
[Common Dependencies](common.md).

`$MPI_PREFIX` for ParMETIS is `$(brew --prefix open-mpi)` (e.g.
`/opt/homebrew/Cellar/open-mpi/5.0.8`).

## GPU on macOS

Apple Silicon has no CUDA or HIP support. CPU builds are the only
backend on macOS.

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
cmake --build build_cpu -j8
```

CPU + MPI:

```bash
PATH=$(brew --prefix open-mpi)/bin:$PATH \
    cmake -S install -B build_mpi \
    -DEXASIM_CUDA=OFF -DEXASIM_HIP=OFF \
    -DEXASIM_MPI=ON   -DEXASIM_NOMPI=OFF \
    -DWITH_PARMETIS=ON \
    -DKokkos_DIR=$EXASIM/kokkos/buildserial/lib/cmake/Kokkos \
    -DEXASIM_BUILD_LIBRARY_EXAMPLES=ON \
    -DMPI_CXX_COMPILER=$(brew --prefix open-mpi)/bin/mpicxx \
    -DMPI_C_COMPILER=$(brew --prefix open-mpi)/bin/mpicc
PATH=$(brew --prefix open-mpi)/bin:$PATH cmake --build build_mpi -j8
```

## Verification

```bash
ctest --test-dir build_cpu -R "poisson2d:.*:cpu" -j 4
```
