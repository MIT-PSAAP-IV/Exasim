# Common Dependencies For Manual Installs

This page is for **manual dependency installation**. Most users should not
start here.

!!! tip "Use Superbuild first"
    For most CPU systems, use [Quick Installation](quick-install.md) or
    [CPU Installation via Superbuild](superbuild-cpu.md). If the Superbuild
    succeeds, none of the manual dependency steps below are required.

Use this page only when the Superbuild fails, when network/download restrictions
require site-managed dependencies, or when an HPC environment requires explicit
compiler/MPI/GPU module control.

See [Manual Dependency Installation](manual-dependencies.md) for the dependency
order and graph.

## Sources

| Library | Vendored at | Upstream |
|---|---|---|
| Kokkos | `deps/kokkos/` | <https://github.com/kokkos/kokkos> |
| GKlib | `deps/metis/GKlib/` | <https://github.com/KarypisLab/GKlib> |
| METIS | `deps/metis/METIS/` | <https://github.com/KarypisLab/METIS> |
| ParMETIS | `deps/metis/ParMETIS/` | <https://github.com/KarypisLab/ParMETIS> |
| SymEngine | `deps/symengine/` | <https://github.com/symengine/symengine> |

Vendored copies are already in the repo. The upstream URLs are for
reference only.

## Manual Step 1: Build GKlib + METIS + ParMETIS

```bash
EXASIM=$PWD                    # repo root

cd $EXASIM/deps/metis/GKlib
make config prefix=$EXASIM/deps/metis/GKlib
make -j8 install

cd $EXASIM/deps/metis/METIS
make config gklib_path=$EXASIM/deps/metis/GKlib \
            prefix=$EXASIM/deps/metis/METIS \
            shared=1 r64=1 i64=1
make -j8 install
```

ParMETIS is required only for MPI builds:

```bash
cd $EXASIM/deps/metis/ParMETIS
PATH=$MPI_PREFIX/bin:$PATH make config \
    gklib_path=$EXASIM/deps/metis/GKlib \
    metis_path=$EXASIM/deps/metis/METIS \
    prefix=$EXASIM/deps/metis/ParMETIS \
    cc=mpicc cxx=mpicxx
PATH=$MPI_PREFIX/bin:$PATH make -j8 install
```

`$MPI_PREFIX` is the MPI install prefix from the per-platform page.

## Manual Step 2: Build Kokkos Serial

Required on every platform. The CPU and MPI Exasim builds link against
this prefix.

```bash
cd $EXASIM/deps/kokkos
mkdir buildserial && cd buildserial
cmake .. -DCMAKE_INSTALL_PREFIX=$EXASIM/deps/kokkos/buildserial \
         -DKokkos_ENABLE_SERIAL=ON
make -j16 install
```

## Manual Step 3: Build SymEngine + Text2Code

Required for the generated-PDE authoring path (the user describes
the PDE as `pdemodel.txt` in a SymEngine DSL; `text2code` emits
the C++ model header). Skip if you only use the handwritten model
path.

```bash
cd $EXASIM/deps/symengine
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$EXASIM/deps/symengine/build \
         -DBUILD_SHARED_LIBS=OFF -DWITH_GMP=OFF -DINTEGER_CLASS=boostmp
make -j16 install

cd $EXASIM/text2code/text2code
cmake -S . -B build -DEXASIM_ROOT=$EXASIM
cmake --build build -j8
```

The text2code binary is built under the `text2code/text2code/build` tree unless
you pass an explicit install/output prefix.

## Manual Step 4: Install Exasim

After building any backend variant:

```bash
cmake --install build_cpu --prefix /opt/exasim
```

Layout:

```
/opt/exasim/
├── include/exasim/          # public headers
├── include/backend/         # template-impl headers transitively included
├── lib/cmake/Exasim/        # find_package(Exasim) machinery
└── bin/text2code            # codegen tool (if text2code was built)
```

Consumer projects use `find_package(Exasim REQUIRED)`. See the
[CMake reference](../reference/cmake.md) and
[Application Modes](../usage-modes/index.md) for consumer link patterns.
