# Tuolumne

Build Exasim on LLNL **Tuolumne** (AMD Instinct MI300A, `gfx942`, HPE Cray EX
with the Flux scheduler). The dependency build is identical to
[Frontier](frontier.md); only the parameters below differ.

!!! warning "Use the platform guide for Tuolumne"
    Tuolumne is an AMD GPU HPC system. Use this guide because the ROCm, Cray
    MPI, GPU-aware MPI, Kokkos/HIP, and `gfx942` settings must match the
    machine.

## Machine parameters

| Parameter | Value |
|---|---|
| GPU | AMD Instinct MI300A |
| Kokkos arch macro | `AMD_GFX942` |
| HIP offload arch | `gfx942` |
| Modules | `PrgEnv-amd`, `cray-mpich`, `amd/6.4.3`, `rocm/6.4.3`, `craype-accel-amd-gfx942` |
| GPU compiler | `hipcc` (host: `CC` / `cc`) |
| BLAS/LAPACK | `$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_amd.so` |
| Scheduler | Flux (`flux alloc` / `flux run`) |
| Extra runtime env | `HSA_XNACK=1` |

## Environment

```bash
module reset
module load PrgEnv-amd
module load cray-mpich
module load amd/6.4.3
module load rocm/6.4.3
module load craype-accel-amd-gfx942

export PROJDIR=/path/to/PSAAP
export EXASIM_ROOT=$PROJDIR/Exasim
export EXASIM_PREFIX=$EXASIM_ROOT/local
export CRAYPE_LINK_TYPE=dynamic
export MPICH_GPU_SUPPORT_ENABLED=1
export EXASIM_GPU_ARCH=gfx942
mkdir -p $EXASIM_PREFIX

CC --version && cc --version && hipcc --version
```

## Build the dependencies

Run the [Frontier dependency chain](frontier.md#build-the-dependencies)
verbatim, with **one change** — build Kokkos HIP for `gfx942`:

```bash
make -f Makefile.builds hip HIP_DIR=$EXASIM_ROOT/deps/kokkos/buildhip KOKKOS_HIP_ARCH=AMD_GFX942
```

## Build Exasim

Same as Frontier, but with `--offload-arch=gfx942` and the `gfx942` GTL
variables:

```bash
cd $PROJDIR && rm -rf Exasim-build
cmake -S Exasim/install -B Exasim-build \
  -DEXASIM_HIP=ON -DEXASIM_MPI=ON -DEXASIM_NOMPI=ON -DEXASIM_LIB=ON \
  -DWITH_PARMETIS=ON -DWITH_TEXT2CODE=OFF \
  -DEXASIM_TEXT2CODE=$EXASIM_PREFIX/bin/text2code \
  -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_DIR=$EXASIM_ROOT/deps/kokkos/buildhip \
  -DMETIS_INCLUDE_DIR=$EXASIM_PREFIX/include -DMETIS_LIBRARY_DIR=$EXASIM_PREFIX/lib64 \
  -DPARMETIS_INCLUDE_DIR=$EXASIM_PREFIX/include -DPARMETIS_LIBRARY_DIR=$EXASIM_PREFIX/lib64 \
  -DGKLIB_INCLUDE_DIR=$EXASIM_PREFIX/include -DGKLIB_LIBRARY_DIR=$EXASIM_PREFIX/lib64 \
  -DBLAS_LIBRARIES=$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_amd.so \
  -DLAPACK_LIBRARIES=$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_amd.so \
  -DCMAKE_HIP_FLAGS="--offload-arch=gfx942" \
  -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include -I${ROCM_PATH}/include -I${PE_MPICH_GTL_DIR_amd_gfx942}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi -L${ROCM_PATH}/lib -lamdhip64 ${PE_MPICH_GTL_LIBS_amd_gfx942}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build Exasim-build -j8 && cmake --install Exasim-build
```

Verify the libraries installed:

```bash
find $EXASIM_PREFIX -name "libgpuEXASIM*" -o -name "libgpumpiEXASIM*" -o -name "builtinmodel*"
```

## Build and run an app

Build `apps/builtinlibrary` as on Frontier (substituting `gfx942` for `gfx90a`
in `CMAKE_HIP_FLAGS` and the GTL variables), then run under Flux:

```bash
flux alloc -N 1 -q pdebug
module reset
module load PrgEnv-amd cray-mpich amd/6.4.3 rocm/6.4.3 craype-accel-amd-gfx942
export MPICH_GPU_SUPPORT_ENABLED=1 HSA_XNACK=1 CRAYPE_LINK_TYPE=dynamic

flux run -N1 -n1 --verbose --exclusive --setopt=mpibind=verbose:1 build/exasimapp ../poisson/lshape/pdeapp.txt
flux run -N1 -n4 --verbose --exclusive --setopt=mpibind=verbose:1 build/exasimapp ../poisson/poisson2d/pdeapp.txt
```

The [shared-library](../usage-modes/shared-library.md) app works the same way:
generate the model with `text2code`, then `flux run` `build/exasimapp` against
its `pdeapp.txt`.
