"""Header-only C++-driven app scaffolder.

Given a parsed model (and its already-generated ``generated/my_model.hpp``), emit a
self-contained standalone app in the style of the latest CHEFSI app: a C++ driver
that builds ``CSolution<PdeModel>`` from ``datain/`` (No-ABI concrete-model path, no
runtime-loaded ``.so``) and runs a steady HDG solve through Exasim's exported PETSc
operator via ``exasim::petsc::solve_steady`` — so the app owns **no** PETSc glue.

Emitted files (into ``<dest>/``):
  * ``main.cc``          — the driver
  * ``CMakeLists.txt``   — find_package(Exasim COMPONENTS petsc) + Kokkos + MPI + PETSc
  * ``build.sh``         — configure+build helper
  * ``README.md``        — how to build/run

``generated/my_model.hpp`` is written separately by the model codegen (``pyt2c -o``).
"""
from __future__ import annotations

import os
import re
import shutil
import stat

from .parser import Spec


def sizes_from_header(header_path: str) -> dict:
    """Read the model sizes from an existing ``my_model.hpp`` static-constexpr block,
    so an app scaffold can be emitted with NO ``.txt`` input at all (works on a header
    produced by either pyt2c or the C++ text2code)."""
    with open(header_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def geti(name: str, default: int = 0) -> int:
        m = re.search(rf"static\s+constexpr\s+int\s+{name}\s*=\s*(-?\d+)", txt)
        return int(m.group(1)) if m else default

    return {
        "nd": geti("nd"), "ncu": geti("ncu"), "nco": geti("nco"),
        "ncw": geti("ncw"), "nparam": geti("nparam"),
        "coupling": "has_external_coupling" in txt,
    }


def _sizes_from_spec(spec: Spec) -> dict:
    v = spec.vectors
    return {
        "nd": v.get("x", 0), "ncu": v.get("uhat", 0), "nco": v.get("v", 0),
        "ncw": v.get("w", 0), "nparam": v.get("mu", 0),
        "coupling": spec.is_output("Fint") or spec.is_output("Fext"),
    }


def _main_cc(app_name: str, model_id: int) -> str:
    return f'''\
// {app_name}.cc — auto-generated standalone header-only Exasim app (pyt2c app scaffolder).
//
// Drives a steady HDG solve on the concrete text2code-generated model `PdeModel`
// (generated/my_model.hpp) via a genuine PETSc SNES + GMRES on Exasim's exported HDG
// operators (exasim::petsc::solve_steady). There is NO runtime-loaded model ABI (.so)
// and NO hand-rolled PETSc solver code in this app: the whole solver lives in
// <exasim/petsc.hpp>. This is the C++-driven form of a text2code-generated model.
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include <exasim/operators.hpp>   // unity Exasim backend (CSolution<M>/CAssembler<M>/CPreconditioner<M>)
#include <exasim/export.hpp>      // recover_volume
#include <exasim/petsc.hpp>       // exasim::petsc::solve_steady (prepare + SNES+GMRES + recover)

// text2code emits `struct PdeModel : ModelDefaults<PdeModel>` unqualified; with the
// operator-export backend that CRTP base is exasim::ModelDefaults, so bring it into scope.
using exasim::ModelDefaults;
#include "generated/my_model.hpp"

int main(int argc, char** argv)
{{
    MPI_Init(&argc, &argv);
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (!Kokkos::is_initialized()) Kokkos::initialize(argc, argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    // Single-model app: the backend's distributed HDG trace exchange (hdgMatVec / the MPI
    // assembly) communicates over EXASIM_COMM_LOCAL, which defaults to MPI_COMM_NULL and is
    // otherwise only set by ExasimSolver. Without this, np>1 hangs on the halo exchange. For a
    // single model it is just MPI_COMM_WORLD.
    EXASIM_COMM_LOCAL = MPI_COMM_WORLD;

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

#if defined(_CUDA)
    // Pin each rank to a distinct GPU on its node (device = node-local rank % deviceCount),
    // mirroring Exasim's native run.hpp. CSolution does not set the device, so without this
    // every rank would share GPU 0 (correct but unscalable).
    {{
        MPI_Comm shmcomm; int shmrank = 0;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
        MPI_Comm_rank(shmcomm, &shmrank); MPI_Comm_free(&shmcomm);
        int deviceCount = 0; cudaGetDeviceCount(&deviceCount);
        if (deviceCount > 0) cudaSetDevice(shmrank % deviceCount);
    }}
#endif

    const std::string filein  = (argc > 1) ? argv[1] : "datain/";
    const std::string fileout = (argc > 2) ? argv[2] : "dataout/out";
    // Backend is chosen at compile time from the Exasim variant macros the build sets:
    // _CUDA -> CUDA device (2), _HIP -> HIP device (3), otherwise host CPU (0, serial or MPI).
    // solve_steady + CSolution then run entirely on that backend (device pointers wrapped
    // zero-copy into PETSc's device Vec when GPU). See CMakeLists EXASIM_GPU.
#if defined(_CUDA)
    const int backend = 2;  // CUDA device
#elif defined(_HIP)
    const int backend = 3;  // HIP device
#else
    const int backend = 0;  // host CPU (serial or MPI)
#endif

    {{
        // No-ABI concrete-model CSolution, built straight from preprocessed datain/.
        CSolution<PdeModel> model(filein, fileout, "", (Int)size, (Int)rank,
                                  (Int)0 /*fileoffset*/, (Int)0 /*gpuid*/,
                                  (Int)backend, (Int){model_id} /*builtinmodelID*/);
        model.disc.common.nomodels = 1;
        std::vector<Int>     ncarr  = {{ model.disc.common.components.nc }};
        std::vector<dstype*> udgarr = {{ &model.disc.sol.udg[0] }};
        model.disc.common.ncarray = ncarr.data();
        model.disc.sol.udgarray   = udgarr.data();

        // The entire solver: prepare (InitSolution + odg->Gauss) + PETSc SNES+GMRES + recover.
        const int reason = exasim::petsc::solve_steady<PdeModel>(model, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[{app_name}] steady solve SNESConvergedReason=" << reason << "\\n";

        model.writer.SaveSolutions(backend);              // dataout/outudg_np*.bin + outuhat
        if (model.vis.savemode > 0)
            model.writer.SaveParaview(backend, "", true); // dataout/*.vtu when vis is enabled
    }}

    Kokkos::finalize();
    PetscFinalize();
    MPI_Finalize();
    return 0;
}}
'''


def _cmakelists(app_name: str) -> str:
    return f'''\
cmake_minimum_required(VERSION 3.16)
project({app_name} C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vendored find-modules (FindPETSc etc.), if shared alongside the app.
if(NOT CHEFSI_FIND_MODULES)
  foreach(_cand "${{CMAKE_CURRENT_SOURCE_DIR}}/cmake/modules"
                "${{CMAKE_CURRENT_SOURCE_DIR}}/../cmake/modules")
    if(IS_DIRECTORY "${{_cand}}")
      get_filename_component(CHEFSI_FIND_MODULES "${{_cand}}" ABSOLUTE)
      break()
    endif()
  endforeach()
endif()
if(CHEFSI_FIND_MODULES)
  list(PREPEND CMAKE_MODULE_PATH "${{CHEFSI_FIND_MODULES}}")
endif()

# Backend selection. CPU (cpu/cpumpi) is the default; EXASIM_GPU switches to the CUDA
# variant (gpu/gpumpi) + the _CUDA unity macro, so the driver's compile-time backend
# becomes 2 (device) and solve_steady wraps device pointers zero-copy into PETSc's CUDA Vec.
# A GPU build requires a GPU-built Exasim install (CUDA Kokkos + CUDA PETSc) AND nvcc_wrapper
# as CMAKE_CXX_COMPILER (build.sh sets this); configure with -DEXASIM_GPU=ON.
option(EXASIM_MPI "Use the MPI-enabled Exasim variant" ON)
option(EXASIM_GPU "Build the CUDA GPU Exasim variant (needs a GPU-built Exasim install + nvcc_wrapper as CMAKE_CXX_COMPILER)" OFF)
set(EXASIM_GPU_ARCH "" CACHE STRING "CUDA arch for the GPU build, e.g. sm_70 / 70 (blank = let Kokkos decide)")

if(EXASIM_GPU)
  if(EXASIM_MPI)
    set(EXASIM_VARIANT gpumpi)
  else()
    set(EXASIM_VARIANT gpu)
  endif()
else()
  if(EXASIM_MPI)
    set(EXASIM_VARIANT cpumpi)
  else()
    set(EXASIM_VARIANT cpu)
  endif()
endif()

find_package(Exasim REQUIRED COMPONENTS ${{EXASIM_VARIANT}})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# PETSc — the app drives the solve via <exasim/petsc.hpp>, so link PETSc directly.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
message(STATUS "PETSc: ${{PETSC_VERSION}}  (Exasim variant: ${{EXASIM_VARIANT}})")

# The unity-compiled backend needs BLAS/LAPACK (normally transitive via the prelib).
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Backend is unity-compiled INTO this TU (no Exasim::pre prelib); concrete No-ABI model.
set(_UNITY_DEFS EXASIM_HAVE_PETSC HAVE_BACKEND_PREPROCESSING)
if(EXASIM_MPI)
  list(APPEND _UNITY_DEFS _MPI)
endif()
if(EXASIM_GPU)
  list(APPEND _UNITY_DEFS _CUDA)
endif()

add_executable(${{PROJECT_NAME}} {app_name}.cc)
target_compile_definitions(${{PROJECT_NAME}} PRIVATE ${{_UNITY_DEFS}})
target_link_libraries(${{PROJECT_NAME}} PRIVATE
    Exasim::headers          # include dirs + parmetis/metis deps (NO prelib)
    Kokkos::kokkos
    MPI::MPI_CXX
    PkgConfig::PETSC
    LAPACK::LAPACK
    BLAS::BLAS)
target_include_directories(${{PROJECT_NAME}} PRIVATE "${{CMAKE_CURRENT_SOURCE_DIR}}")

if(EXASIM_GPU)
  # CUDA runtime + driver (device-management calls) and the arch. Kokkos::kokkos (CUDA)
  # already propagates --expt-extended-lambda / --expt-relaxed-constexpr and the arch when
  # built for CUDA; EXASIM_GPU_ARCH lets you pin it explicitly if the install did not.
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${{PROJECT_NAME}} PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  if(EXASIM_GPU_ARCH)
    string(REGEX REPLACE "^[Ss][Mm]_?" "" _arch_num "${{EXASIM_GPU_ARCH}}")
    target_compile_options(${{PROJECT_NAME}} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-arch=sm_${{_arch_num}}>)
  endif()
endif()
'''


def _build_sh(app_name: str) -> str:
    return f'''\
#!/usr/bin/env bash
# Configure + build the standalone header-only app against a petsc-enabled Exasim install.
#
#   CPU (default):  EXASIM_INSTALL=/path/to/exasim ./build.sh
#   GPU (CUDA):     EXASIM_GPU=1 EXASIM_INSTALL=/path/to/gpu-exasim \\
#                   NVCC_WRAPPER=/path/to/kokkos/bin/nvcc_wrapper EXASIM_GPU_ARCH=sm_70 ./build.sh
# A GPU build needs a GPU-built Exasim install (CUDA Kokkos + CUDA PETSc on PKG_CONFIG_PATH)
# and nvcc_wrapper as the C++ compiler (its host compiler should be your MPI C++ wrapper:
# export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx).
set -eo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

EXASIM_INSTALL="${{EXASIM_INSTALL:?set EXASIM_INSTALL to a petsc-enabled Exasim install prefix}}"
EXASIM_GPU="${{EXASIM_GPU:-0}}"
BUILD="${{BUILD:-$HERE/build}}"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DEXASIM_MPI=ON
  -DCMAKE_PREFIX_PATH="$EXASIM_INSTALL"
  -DExasim_DIR="$EXASIM_INSTALL/lib/cmake/Exasim"
  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON
)

if [ "$EXASIM_GPU" = 1 ]; then
  NVCC_WRAPPER="${{NVCC_WRAPPER:?set NVCC_WRAPPER to kokkos .../bin/nvcc_wrapper for a GPU build}}"
  KOKKOS_DIR="${{KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildcuda}}"
  CMAKE_ARGS+=(
    -DEXASIM_GPU=ON
    -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER"
    -DEXASIM_GPU_ARCH="${{EXASIM_GPU_ARCH:-}}"
    -DKokkos_DIR="$KOKKOS_DIR"
  )
else
  KOKKOS_DIR="${{KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildserial}}"
  CMAKE_ARGS+=( -DKokkos_DIR="$KOKKOS_DIR" )
fi

cmake -S "$HERE" -B "$BUILD" "${{CMAKE_ARGS[@]}}"
cmake --build "$BUILD" -j 4
echo "built: $BUILD/{app_name}"
'''


def _readme(app_name: str, sizes: dict) -> str:
    coupling = sizes["coupling"]
    return f'''\
# {app_name}

Standalone, header-only, C++-driven Exasim app auto-generated by `pyt2c`. It solves
the model as a steady **HDG** problem through Exasim's exported PETSc operator — **no**
runtime-loaded model `.so`, and **no** hand-rolled PETSc solver code in the app (the
whole solver is `exasim::petsc::solve_steady`). The solve is HDG (condensed trace
system), so preprocess the mesh with `discretization = "hdg"` / `hybrid = 1`; the model
header itself is discretization-agnostic.

**Backends:** CPU by default (`cpu`/`cpumpi` Exasim variant, host backend). Pass
`-DEXASIM_GPU=ON` (or `EXASIM_GPU=1 ./build.sh`) to build the CUDA `gpu`/`gpumpi` variant:
the driver then selects the device backend at compile time (`_CUDA` -> backend 2) and
`solve_steady` wraps device pointers zero-copy into PETSc's CUDA `Vec`. A GPU build needs
a GPU-built Exasim install (CUDA Kokkos + CUDA PETSc) and `nvcc_wrapper` as the C++
compiler (build.sh sets this). HIP (`_HIP` -> backend 3) follows the same shape.

Model sizes: nd={sizes["nd"]}, ncu={sizes["ncu"]}, nco={sizes["nco"]},
ncw={sizes["ncw"]}, nparam={sizes["nparam"]}{"  (has external coupling: Fint/Fext)" if coupling else ""}.

## Layout

- `{app_name}.cc`     — the driver (builds `CSolution<PdeModel>` from `datain/`, solves, saves).
- `generated/my_model.hpp` — the concrete model.
- `CMakeLists.txt`, `build.sh` — build against a petsc-enabled Exasim install.
- `datain/`           — preprocessed mesh/solution bundle (from `text2code`/frontends).

## Build & run (no `.txt` needed)

```sh
# CPU
EXASIM_INSTALL=/path/to/petsc-enabled-exasim ./build.sh
mpirun -np 1 build/{app_name} datain/ dataout/out

# GPU (CUDA)
EXASIM_GPU=1 EXASIM_INSTALL=/path/to/gpu-exasim \\
  NVCC_WRAPPER=/path/to/kokkos/bin/nvcc_wrapper EXASIM_GPU_ARCH=sm_70 ./build.sh
mpirun -np 1 build/{app_name} datain/ dataout/out
```

## Regenerate the scaffold from the model header (no `.txt` needed)

```sh
python -m pyt2c --emit-app . --from-header generated/my_model.hpp
```
'''


def _write_scaffold(dest: str, app_name: str, model_id: int, sizes: dict) -> None:
    files = {
        f"{app_name}.cc": _main_cc(app_name, model_id),
        "CMakeLists.txt": _cmakelists(app_name),
        "build.sh": _build_sh(app_name),
        "README.md": _readme(app_name, sizes),
    }
    for name, content in files.items():
        with open(os.path.join(dest, name), "w", encoding="utf-8") as f:
            f.write(content)
    os.chmod(os.path.join(dest, "build.sh"),
             os.stat(os.path.join(dest, "build.sh")).st_mode | stat.S_IEXEC)


def emit_app(spec: Spec, dest: str, app_name: str | None = None,
             model_id: int = 100, write_model: bool = True) -> str:
    """Write the app scaffold (and, by default, ``generated/my_model.hpp``) into *dest*
    from a parsed model spec (needs pdemodel.txt)."""
    from .codegen import generate_header

    dest = os.path.abspath(dest)
    if app_name is None:
        app_name = os.path.basename(dest.rstrip("/")) or "exasim_app"
    os.makedirs(dest, exist_ok=True)
    _write_scaffold(dest, app_name, model_id, _sizes_from_spec(spec))

    if write_model:
        gen_dir = os.path.join(dest, "generated")
        os.makedirs(gen_dir, exist_ok=True)
        with open(os.path.join(gen_dir, "my_model.hpp"), "w", encoding="utf-8") as f:
            f.write(generate_header(spec))

    return dest


def emit_app_from_header(header_path: str, dest: str, app_name: str | None = None,
                         model_id: int = 100) -> str:
    """Write the app scaffold from an EXISTING ``my_model.hpp`` — NO ``.txt`` input at
    all. Sizes are read from the header's static-constexpr block; the header is copied
    into ``<dest>/generated/my_model.hpp`` (a no-op if it is already there)."""
    dest = os.path.abspath(dest)
    if app_name is None:
        app_name = os.path.basename(dest.rstrip("/")) or "exasim_app"
    os.makedirs(dest, exist_ok=True)
    _write_scaffold(dest, app_name, model_id, sizes_from_header(header_path))

    gen_dir = os.path.join(dest, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    target = os.path.join(gen_dir, "my_model.hpp")
    if os.path.abspath(header_path) != os.path.abspath(target):
        shutil.copyfile(header_path, target)
    return dest
