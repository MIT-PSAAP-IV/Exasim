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
import stat

from .parser import Spec


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

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string filein  = (argc > 1) ? argv[1] : "datain/";
    const std::string fileout = (argc > 2) ? argv[2] : "dataout/out";
    const int backend = 0;  // 0/1 = host CPU (set 2 for CUDA / 3 for HIP builds)

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

option(EXASIM_MPI "Use the MPI-enabled Exasim variant" ON)
option(EXASIM_GPU "Use the GPU-enabled Exasim variant" OFF)
if(EXASIM_GPU AND EXASIM_MPI)
  set(EXASIM_VARIANT gpumpi)
elseif(EXASIM_GPU)
  set(EXASIM_VARIANT gpu)
elseif(EXASIM_MPI)
  set(EXASIM_VARIANT cpumpi)
else()
  set(EXASIM_VARIANT cpu)
endif()

find_package(Exasim REQUIRED COMPONENTS ${{EXASIM_VARIANT}})
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)

# PETSc — the app drives the solve via <exasim/petsc.hpp>, so link PETSc directly.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
message(STATUS "PETSc: ${{PETSC_VERSION}}")

# The unity-compiled backend needs BLAS/LAPACK (normally transitive via the prelib).
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Backend is unity-compiled INTO this TU (no Exasim::pre prelib); concrete No-ABI model.
set(_UNITY_DEFS EXASIM_HAVE_PETSC HAVE_BACKEND_PREPROCESSING)
if(EXASIM_MPI)
  list(APPEND _UNITY_DEFS _MPI)
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
'''


def _build_sh(app_name: str) -> str:
    return f'''\
#!/usr/bin/env bash
# Configure + build the standalone header-only app against a petsc-enabled Exasim install.
set -eo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

EXASIM_INSTALL="${{EXASIM_INSTALL:?set EXASIM_INSTALL to a petsc-enabled Exasim install prefix}}"
KOKKOS_DIR="${{KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildserial}}"
BUILD="${{BUILD:-$HERE/build}}"

cmake -S "$HERE" -B "$BUILD" \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DEXASIM_MPI=ON -DEXASIM_GPU=OFF \\
  -DCMAKE_PREFIX_PATH="$EXASIM_INSTALL" \\
  -DExasim_DIR="$EXASIM_INSTALL/lib/cmake/Exasim" \\
  -DKokkos_DIR="$KOKKOS_DIR" \\
  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON

cmake --build "$BUILD" -j 4
echo "built: $BUILD/{app_name}"
'''


def _readme(app_name: str, spec: Spec) -> str:
    v = spec.vectors
    coupling = spec.is_output("Fint") or spec.is_output("Fext")
    return f'''\
# {app_name}

Standalone, header-only, C++-driven Exasim app auto-generated by `pyt2c` from a
`pdemodel.txt`. It solves the model as a steady HDG problem through Exasim's
exported PETSc operator — **no** runtime-loaded model `.so`, and **no** hand-rolled
PETSc solver code in the app (the whole solver is `exasim::petsc::solve_steady`).

Model sizes: nd={v.get("x", 0)}, ncu={v.get("uhat", 0)}, nco={v.get("v", 0)},
ncw={v.get("w", 0)}, nparam={v.get("mu", 0)}{"  (has external coupling: Fint/Fext)" if coupling else ""}.

## Layout

- `{app_name}.cc`     — the driver (builds `CSolution<PdeModel>` from `datain/`, solves, saves).
- `generated/my_model.hpp` — the concrete model (from `pyt2c`).
- `CMakeLists.txt`, `build.sh` — build against a petsc-enabled Exasim install.
- `datain/`           — preprocessed mesh/solution bundle (from `text2code`/frontends).

## Build & run

```sh
EXASIM_INSTALL=/path/to/petsc-enabled-exasim ./build.sh
mpirun -np 1 build/{app_name} datain/ dataout/out
```

## Regenerate

```sh
python -m pyt2c pdemodel.txt -o generated          # regenerate generated/my_model.hpp
python -m pyt2c pdemodel.txt --emit-app .           # regenerate the whole app scaffold
```
'''


def emit_app(spec: Spec, dest: str, app_name: str | None = None,
             model_id: int = 100, write_model: bool = True) -> str:
    """Write the app scaffold (and, by default, ``generated/my_model.hpp``) into *dest*."""
    from .codegen import generate_header

    dest = os.path.abspath(dest)
    if app_name is None:
        app_name = os.path.basename(dest.rstrip("/")) or "exasim_app"
    os.makedirs(dest, exist_ok=True)

    files = {
        f"{app_name}.cc": _main_cc(app_name, model_id),
        "CMakeLists.txt": _cmakelists(app_name),
        "build.sh": _build_sh(app_name),
        "README.md": _readme(app_name, spec),
    }
    for name, content in files.items():
        with open(os.path.join(dest, name), "w", encoding="utf-8") as f:
            f.write(content)
    os.chmod(os.path.join(dest, "build.sh"),
             os.stat(os.path.join(dest, "build.sh")).st_mode | stat.S_IEXEC)

    if write_model:
        gen_dir = os.path.join(dest, "generated")
        os.makedirs(gen_dir, exist_ok=True)
        with open(os.path.join(gen_dir, "my_model.hpp"), "w", encoding="utf-8") as f:
            f.write(generate_header(spec))

    return dest
