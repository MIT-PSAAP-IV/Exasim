# Frontier

Build Exasim on OLCF **Frontier** (AMD Instinct MI250X, `gfx90a`, HPE Cray
EX with Slurm). This page follows the shared [HPC build chain](hpc.md); only the
values below are Frontier-specific.

!!! warning "Use the platform guide for Frontier"
    Frontier is an AMD GPU HPC system. Do not start with generic CPU
    Superbuild instructions for production GPU runs. Use this guide because the
    ROCm, Cray MPI, GPU-aware MPI, Kokkos/HIP, and `gfx90a` settings must match
    the machine.

## Machine parameters

| Parameter | Value |
|---|---|
| GPU | AMD Instinct MI250X |
| Kokkos arch macro | `VEGA90A` |
| HIP offload arch | `gfx90a` |
| Modules | `PrgEnv-amd`, `amd`, `cray-mpich`, `rocm` (+ `craype-accel-amd-gfx90a` at run time) |
| GPU compiler | `hipcc` (host: `CC` / `cc`) |
| BLAS/LAPACK | `$CRAY_LIBSCI_PREFIX_DIR/lib/libsci_amd.so` |
| Scheduler | Slurm (`salloc` / `srun`) |

## Environment

```bash
module reset
module load PrgEnv-amd
module load amd
module load cray-mpich
module load rocm

export PROJDIR=/path/to/PSAAP
export EXASIM_ROOT=$PROJDIR/Exasim
export EXASIM_PREFIX=$EXASIM_ROOT/local
mkdir -p $EXASIM_PREFIX

CC --version && cc --version && hipcc --version    # verify the toolchain
```

## Build the dependencies

```bash
# 1. GKlib
cd $EXASIM_ROOT/deps/metis/GKlib && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=amdclang -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX
make -j8 && make install

# 2. METIS
cd $EXASIM_ROOT/deps/metis/METIS && rm -rf build
make config cc=amdclang prefix=$EXASIM_PREFIX gklib_path=$EXASIM_PREFIX
make -j8 && make install

# 3. ParMETIS
cd $EXASIM_ROOT/deps/metis/ParMETIS && rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=cc -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX \
  -DGKLIB_PATH=$EXASIM_PREFIX -DMETIS_PATH=$EXASIM_PREFIX
make -j8 && make install

# 4. Kokkos — HIP (VEGA90A) and serial
export CRAYPE_LINK_TYPE=dynamic
cd $EXASIM_ROOT/deps/kokkos && rm -rf buildhip
make -f Makefile.builds hip HIP_DIR=$EXASIM_ROOT/deps/kokkos/buildhip KOKKOS_HIP_ARCH=VEGA90A
rm -rf buildserial
cmake -S . -B buildserial -DCMAKE_CXX_COMPILER=CC -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DKokkos_ENABLE_SERIAL=ON \
  -DCMAKE_INSTALL_PREFIX=$EXASIM_ROOT/deps/kokkos/buildserial
cmake --build buildserial -j8 && cmake --install buildserial

# 5. GMP
cd $EXASIM_ROOT/deps && wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar -xf gmp-6.3.0.tar.xz && cd gmp-6.3.0
./configure --prefix=$EXASIM_PREFIX CC=cc CXX=CC --enable-cxx --disable-shared --enable-static
make -j8 && make install

# 6. SymEngine (GMP integer class)
cd $PROJDIR && rm -rf SymEngine-build
cmake -S $EXASIM_ROOT/deps/symengine -B SymEngine-build \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX \
  -DBUILD_SHARED_LIBS=OFF -DWITH_GMP=ON -DWITH_MPFR=OFF -DWITH_LLVM=OFF \
  -DWITH_SYMENGINE_THREAD_SAFE=OFF -DINTEGER_CLASS=gmp \
  -DGMP_INCLUDE_DIR=$EXASIM_PREFIX/include -DGMP_LIBRARY=$EXASIM_PREFIX/lib/libgmp.a \
  -DCMAKE_CXX_FLAGS="-std=gnu++17 -fPIC"
cmake --build SymEngine-build -j8 && cmake --install SymEngine-build

# 7. text2code
cd $PROJDIR && rm -rf text2code-build
cmake -S $EXASIM_ROOT/text2code/text2code -B text2code-build \
  -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_INSTALL_PREFIX=$EXASIM_PREFIX \
  -DUSE_CMAKE=ON -DWITH_METIS=ON -DEXASIM_ROOT=$EXASIM_ROOT \
  -DEXASIM_T2C_OUTPUT_DIR=$EXASIM_PREFIX/bin \
  -DEXASIM_SYMENGINE_DIR=$EXASIM_PREFIX -DEXASIM_SYMENGINE_VENDORED=OFF \
  -DMETIS_INCLUDE_DIR=$EXASIM_PREFIX/include -DMETIS_LIBRARY_DIR=$EXASIM_PREFIX/lib64 \
  -DPARMETIS_INCLUDE_DIR=$EXASIM_PREFIX/include -DPARMETIS_LIBRARY_DIR=$EXASIM_PREFIX/lib64 \
  -DGKLIB_INCLUDE_DIR=$EXASIM_PREFIX/include -DGKLIB_LIBRARY_DIR=$EXASIM_PREFIX/lib64
cmake --build text2code-build -j8
```

## Build Exasim

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
  -DCMAKE_HIP_FLAGS="--offload-arch=gfx90a" \
  -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include -I${ROCM_PATH}/include -I${PE_MPICH_GTL_DIR_amd_gfx90a}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi -L${ROCM_PATH}/lib -lamdhip64 ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build Exasim-build -j8 && cmake --install Exasim-build
```

See the [Exasim configure options](hpc.md#exasim-configure-options) table for what
each flag does.

## Build and run an app

Build the [built-in library](../usage-modes/builtin.md) app:

```bash
cd $EXASIM_ROOT/apps/builtinlibrary && rm -rf build
cmake -S . -B build -DCMAKE_CXX_COMPILER=hipcc \
  -DEXASIM_MPI=ON -DEXASIM_HIP=ON -DEXASIM_CUDA=OFF \
  -DExasim_DIR=$EXASIM_PREFIX/lib64/cmake/Exasim \
  -DCMAKE_HIP_FLAGS="--offload-arch=gfx90a" \
  -DCMAKE_CXX_FLAGS="-I${MPICH_DIR}/include -I${ROCM_PATH}/include -I${PE_MPICH_GTL_DIR_amd_gfx90a}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${MPICH_DIR}/lib -lmpi -L${ROCM_PATH}/lib -lamdhip64 ${PE_MPICH_GTL_LIBS_amd_gfx90a}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j8
```

Run it in a Slurm allocation (set `craype-accel-amd-gfx90a` and the GPU
environment first):

```bash
salloc -N 1 -A <account> -q debug -t 00:30:00
module load craype-accel-amd-gfx90a
export CRAYPE_LINK_TYPE=dynamic MPICH_GPU_SUPPORT_ENABLED=1 EXASIM_GPU_ARCH=gfx90a

srun -N1 -n1 --gpus-per-task=1 build/exasimapp ../poisson/lshape/pdeapp.txt
srun -N1 -n4 --gpus-per-task=1 build/exasimapp ../poisson/lshape/pdeapp.txt
srun -N1 -n4 --gpus-per-task=1 build/exasimapp ../navierstokes/reactingsharpb2/pdeapp.txt
```

For the [shared-library](../usage-modes/shared-library.md) app, build
`apps/sharedlibrary` the same way, then generate the model and run:

```bash
$EXASIM_PREFIX/bin/text2code $EXASIM_ROOT/apps/navierstokes/reactingsharpb2/pdeapp.txt
srun -N1 -n4 --gpus-per-task=1 build/exasimapp ../navierstokes/reactingsharpb2/pdeapp.txt
```
