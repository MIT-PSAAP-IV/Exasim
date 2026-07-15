// AppScaffold.hpp — emit a standalone header-only, C++-driven app for a generated model.
//
// Mirrors the pyt2c `appgen.py` emitter: given a parsed model spec, write a self-contained
// app (main driver + CMakeLists + build.sh + README) in the style of the latest CHEFSI app.
// The driver builds CSolution<PdeModel> from datain/ and runs the whole solve through
// <exasim/petsc.hpp>'s exasim::petsc::solve_steady -- no runtime-loaded .so model ABI and no
// hand-rolled PETSc glue in the app. The concrete model header (generated/my_model.hpp) is
// produced by the normal text2code codegen (into <appdir>/generated/).
#pragma once

#include <fstream>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "TextParser.hpp"

namespace appscaffold {

inline int vsize(const ParsedSpec& spec, const std::string& name) {
    auto it = spec.vectors.find(name);
    return it == spec.vectors.end() ? 0 : it->second;
}

inline bool isOutput(const ParsedSpec& spec, const std::string& name) {
    for (const auto& o : spec.outputs) if (o == name) return true;
    return false;
}

inline std::string mainCc(const std::string& app, int modelID) {
    std::string s;
    s += "// " + app + ".cc — auto-generated standalone header-only Exasim app (text2code --emit-app).\n";
    s += "//\n";
    s += "// Drives a steady HDG solve on the concrete text2code-generated model `PdeModel`\n";
    s += "// (generated/my_model.hpp) via a genuine PETSc SNES + GMRES on Exasim's exported HDG\n";
    s += "// operators (exasim::petsc::solve_steady). There is NO runtime-loaded model ABI (.so)\n";
    s += "// and NO hand-rolled PETSc solver code in this app: the whole solver lives in\n";
    s += "// <exasim/petsc.hpp>. This is the C++-driven form of a text2code-generated model.\n";
    s += "#include <iostream>\n#include <string>\n#include <vector>\n\n#include <mpi.h>\n\n";
    s += "#include <exasim/operators.hpp>   // unity Exasim backend (CSolution<M>/CAssembler<M>/CPreconditioner<M>)\n";
    s += "#include <exasim/export.hpp>      // recover_volume\n";
    s += "#include <exasim/petsc.hpp>       // exasim::petsc::solve_steady (prepare + SNES+GMRES + recover)\n\n";
    s += "// text2code emits `struct PdeModel : ModelDefaults<PdeModel>` unqualified; with the\n";
    s += "// operator-export backend that CRTP base is exasim::ModelDefaults, so bring it into scope.\n";
    s += "using exasim::ModelDefaults;\n#include \"generated/my_model.hpp\"\n\n";
    s += "int main(int argc, char** argv)\n{\n";
    s += "    MPI_Init(&argc, &argv);\n";
    s += "    PETSC_COMM_WORLD = MPI_COMM_WORLD;\n";
    s += "    PetscInitialize(&argc, &argv, nullptr, nullptr);\n";
    s += "    if (!Kokkos::is_initialized()) Kokkos::initialize(argc, argv);\n";
    s += "    EXASIM_COMM_WORLD = MPI_COMM_WORLD;\n\n";
    s += "    int rank = 0, size = 1;\n";
    s += "    MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n";
    s += "    MPI_Comm_size(MPI_COMM_WORLD, &size);\n\n";
    s += "    const std::string filein  = (argc > 1) ? argv[1] : \"datain/\";\n";
    s += "    const std::string fileout = (argc > 2) ? argv[2] : \"dataout/out\";\n";
    s += "    // Backend is chosen at compile time from the Exasim variant macros the build sets:\n";
    s += "    // _CUDA -> CUDA device (2), _HIP -> HIP device (3), otherwise host CPU (0, serial or MPI).\n";
    s += "    // solve_steady + CSolution then run entirely on that backend (device pointers wrapped\n";
    s += "    // zero-copy into PETSc's device Vec when GPU). See CMakeLists EXASIM_GPU.\n";
    s += "#if defined(_CUDA)\n";
    s += "    const int backend = 2;  // CUDA device\n";
    s += "#elif defined(_HIP)\n";
    s += "    const int backend = 3;  // HIP device\n";
    s += "#else\n";
    s += "    const int backend = 0;  // host CPU (serial or MPI)\n";
    s += "#endif\n\n";
    s += "    {\n";
    s += "        // No-ABI concrete-model CSolution, built straight from preprocessed datain/.\n";
    s += "        CSolution<PdeModel> model(filein, fileout, \"\", (Int)size, (Int)rank,\n";
    s += "                                  (Int)0 /*fileoffset*/, (Int)0 /*gpuid*/,\n";
    s += "                                  (Int)backend, (Int)" + std::to_string(modelID) + " /*builtinmodelID*/);\n";
    s += "        model.disc.common.nomodels = 1;\n";
    s += "        std::vector<Int>     ncarr  = { model.disc.common.components.nc };\n";
    s += "        std::vector<dstype*> udgarr = { &model.disc.sol.udg[0] };\n";
    s += "        model.disc.common.ncarray = ncarr.data();\n";
    s += "        model.disc.sol.udgarray   = udgarr.data();\n\n";
    s += "        // The entire solver: prepare (InitSolution + odg->Gauss) + PETSc SNES+GMRES + recover.\n";
    s += "        const int reason = exasim::petsc::solve_steady<PdeModel>(model, MPI_COMM_WORLD);\n";
    s += "        if (rank == 0)\n";
    s += "            std::cout << \"[" + app + "] steady solve SNESConvergedReason=\" << reason << \"\\n\";\n\n";
    s += "        model.writer.SaveSolutions(backend);              // dataout/outudg_np*.bin + outuhat\n";
    s += "        if (model.vis.savemode > 0)\n";
    s += "            model.writer.SaveParaview(backend, \"\", true); // dataout/*.vtu when vis is enabled\n";
    s += "    }\n\n";
    s += "    Kokkos::finalize();\n    PetscFinalize();\n    MPI_Finalize();\n    return 0;\n}\n";
    return s;
}

inline std::string cmakeLists(const std::string& app) {
    std::string s;
    s += "cmake_minimum_required(VERSION 3.16)\n";
    s += "project(" + app + " C CXX)\n";
    s += "set(CMAKE_CXX_STANDARD 20)\nset(CMAKE_CXX_STANDARD_REQUIRED ON)\n\n";
    s += "if(NOT CHEFSI_FIND_MODULES)\n";
    s += "  foreach(_cand \"${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules\"\n";
    s += "                \"${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules\")\n";
    s += "    if(IS_DIRECTORY \"${_cand}\")\n";
    s += "      get_filename_component(CHEFSI_FIND_MODULES \"${_cand}\" ABSOLUTE)\n";
    s += "      break()\n    endif()\n  endforeach()\nendif()\n";
    s += "if(CHEFSI_FIND_MODULES)\n  list(PREPEND CMAKE_MODULE_PATH \"${CHEFSI_FIND_MODULES}\")\nendif()\n\n";
    // Backend selection. CPU (cpu/cpumpi) is the default; EXASIM_GPU switches to the CUDA
    // variant (gpu/gpumpi) + the _CUDA unity macro, so the driver's compile-time backend
    // becomes 2 (device) and solve_steady wraps device pointers zero-copy into PETSc's CUDA Vec.
    // A GPU build requires a GPU-built Exasim install (CUDA Kokkos + CUDA PETSc) AND nvcc_wrapper
    // as CMAKE_CXX_COMPILER (build.sh sets this); configure with -DEXASIM_GPU=ON.
    s += "option(EXASIM_MPI \"Use the MPI-enabled Exasim variant\" ON)\n";
    s += "option(EXASIM_GPU \"Build the CUDA GPU Exasim variant (needs a GPU-built Exasim install + nvcc_wrapper as CMAKE_CXX_COMPILER)\" OFF)\n";
    s += "set(EXASIM_GPU_ARCH \"\" CACHE STRING \"CUDA arch for the GPU build, e.g. sm_70 / 70 (blank = let Kokkos decide)\")\n\n";
    s += "if(EXASIM_GPU)\n";
    s += "  if(EXASIM_MPI)\n    set(EXASIM_VARIANT gpumpi)\n  else()\n    set(EXASIM_VARIANT gpu)\n  endif()\nelse()\n";
    s += "  if(EXASIM_MPI)\n    set(EXASIM_VARIANT cpumpi)\n  else()\n    set(EXASIM_VARIANT cpu)\n  endif()\nendif()\n\n";
    s += "find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})\n";
    s += "find_package(Kokkos REQUIRED)\nfind_package(MPI REQUIRED)\n\n";
    s += "find_package(PkgConfig REQUIRED)\n";
    s += "pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)\n";
    s += "message(STATUS \"PETSc: ${PETSC_VERSION}  (Exasim variant: ${EXASIM_VARIANT})\")\n\n";
    s += "find_package(BLAS REQUIRED)\nfind_package(LAPACK REQUIRED)\n\n";
    s += "set(_UNITY_DEFS EXASIM_HAVE_PETSC HAVE_BACKEND_PREPROCESSING)\n";
    s += "if(EXASIM_MPI)\n  list(APPEND _UNITY_DEFS _MPI)\nendif()\n";
    s += "if(EXASIM_GPU)\n  list(APPEND _UNITY_DEFS _CUDA)\nendif()\n\n";
    s += "add_executable(${PROJECT_NAME} " + app + ".cc)\n";
    s += "target_compile_definitions(${PROJECT_NAME} PRIVATE ${_UNITY_DEFS})\n";
    s += "target_link_libraries(${PROJECT_NAME} PRIVATE\n";
    s += "    Exasim::headers\n    Kokkos::kokkos\n    MPI::MPI_CXX\n    PkgConfig::PETSC\n";
    s += "    LAPACK::LAPACK\n    BLAS::BLAS)\n";
    s += "target_include_directories(${PROJECT_NAME} PRIVATE \"${CMAKE_CURRENT_SOURCE_DIR}\")\n\n";
    s += "if(EXASIM_GPU)\n";
    s += "  # CUDA runtime + driver (device-management calls) and the arch. Kokkos::kokkos (CUDA)\n";
    s += "  # already propagates --expt-extended-lambda / --expt-relaxed-constexpr and the arch when\n";
    s += "  # built for CUDA; EXASIM_GPU_ARCH lets you pin it explicitly if the install did not.\n";
    s += "  find_package(CUDAToolkit REQUIRED)\n";
    s += "  target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver)\n";
    s += "  if(EXASIM_GPU_ARCH)\n";
    s += "    string(REGEX REPLACE \"^[Ss][Mm]_?\" \"\" _arch_num \"${EXASIM_GPU_ARCH}\")\n";
    s += "    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-arch=sm_${_arch_num}>)\n";
    s += "  endif()\nendif()\n";
    return s;
}

inline std::string buildSh(const std::string& app) {
    std::string s;
    s += "#!/usr/bin/env bash\n";
    s += "# Configure + build the standalone header-only app against a petsc-enabled Exasim install.\n";
    s += "#\n";
    s += "#   CPU (default):  EXASIM_INSTALL=/path/to/exasim ./build.sh\n";
    s += "#   GPU (CUDA):     EXASIM_GPU=1 EXASIM_INSTALL=/path/to/gpu-exasim \\\n";
    s += "#                   NVCC_WRAPPER=/path/to/kokkos/bin/nvcc_wrapper EXASIM_GPU_ARCH=sm_70 ./build.sh\n";
    s += "# A GPU build needs a GPU-built Exasim install (CUDA Kokkos + CUDA PETSc on PKG_CONFIG_PATH)\n";
    s += "# and nvcc_wrapper as the C++ compiler (its host compiler should be your MPI C++ wrapper:\n";
    s += "# export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx).\n";
    s += "set -eo pipefail\n";
    s += "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n";
    s += "EXASIM_INSTALL=\"${EXASIM_INSTALL:?set EXASIM_INSTALL to a petsc-enabled Exasim install prefix}\"\n";
    s += "EXASIM_GPU=\"${EXASIM_GPU:-0}\"\n";
    s += "BUILD=\"${BUILD:-$HERE/build}\"\n\n";
    s += "CMAKE_ARGS=(\n";
    s += "  -DCMAKE_BUILD_TYPE=Release\n";
    s += "  -DEXASIM_MPI=ON\n";
    s += "  -DCMAKE_PREFIX_PATH=\"$EXASIM_INSTALL\"\n";
    s += "  -DExasim_DIR=\"$EXASIM_INSTALL/lib/cmake/Exasim\"\n";
    s += "  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON\n";
    s += ")\n\n";
    s += "if [ \"$EXASIM_GPU\" = 1 ]; then\n";
    s += "  NVCC_WRAPPER=\"${NVCC_WRAPPER:?set NVCC_WRAPPER to kokkos .../bin/nvcc_wrapper for a GPU build}\"\n";
    s += "  KOKKOS_DIR=\"${KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildcuda}\"\n";
    s += "  CMAKE_ARGS+=(\n";
    s += "    -DEXASIM_GPU=ON\n";
    s += "    -DCMAKE_CXX_COMPILER=\"$NVCC_WRAPPER\"\n";
    s += "    -DEXASIM_GPU_ARCH=\"${EXASIM_GPU_ARCH:-}\"\n";
    s += "    -DKokkos_DIR=\"$KOKKOS_DIR\"\n";
    s += "  )\n";
    s += "else\n";
    s += "  KOKKOS_DIR=\"${KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildserial}\"\n";
    s += "  CMAKE_ARGS+=( -DKokkos_DIR=\"$KOKKOS_DIR\" )\n";
    s += "fi\n\n";
    s += "cmake -S \"$HERE\" -B \"$BUILD\" \"${CMAKE_ARGS[@]}\"\n";
    s += "cmake --build \"$BUILD\" -j 4\n";
    s += "echo \"built: $BUILD/" + app + "\"\n";
    return s;
}

inline std::string readme(const std::string& app, const ParsedSpec& spec) {
    const bool coupling = isOutput(spec, "Fint") || isOutput(spec, "Fext");
    std::string s;
    s += "# " + app + "\n\n";
    s += "Standalone, header-only, C++-driven Exasim app auto-generated by `text2code --emit-app`.\n";
    s += "It solves the model as a steady **HDG** problem through Exasim's exported PETSc operator —\n";
    s += "**no** runtime-loaded model `.so`, and **no** hand-rolled PETSc solver code in the app\n";
    s += "(the whole solver is `exasim::petsc::solve_steady`). The solve is HDG (condensed trace\n";
    s += "system): preprocess the mesh with `discretization = \"hdg\"` / `hybrid = 1`. The model\n";
    s += "header itself is discretization-agnostic.\n\n";
    s += "**Backends:** CPU by default (`cpu`/`cpumpi` Exasim variant, host backend). Pass\n";
    s += "`-DEXASIM_GPU=ON` (or `EXASIM_GPU=1 ./build.sh`) to build the CUDA `gpu`/`gpumpi` variant:\n";
    s += "the driver then selects the device backend at compile time (`_CUDA` -> backend 2) and\n";
    s += "`solve_steady` wraps device pointers zero-copy into PETSc's CUDA `Vec`. A GPU build needs\n";
    s += "a GPU-built Exasim install (CUDA Kokkos + CUDA PETSc) and `nvcc_wrapper` as the C++\n";
    s += "compiler (build.sh sets this). HIP (`_HIP` -> backend 3) follows the same shape.\n\n";
    s += "Model sizes: nd=" + std::to_string(vsize(spec, "x")) +
         ", ncu=" + std::to_string(vsize(spec, "uhat")) +
         ", nco=" + std::to_string(vsize(spec, "v")) +
         ", ncw=" + std::to_string(vsize(spec, "w")) +
         ", nparam=" + std::to_string(vsize(spec, "mu"));
    if (coupling) s += "  (has external coupling: Fint/Fext)";
    s += ".\n\n";
    s += "## Build & run\n\n```sh\n# CPU\nEXASIM_INSTALL=/path/to/petsc-enabled-exasim ./build.sh\n";
    s += "mpirun -np 1 build/" + app + " datain/ dataout/out\n\n";
    s += "# GPU (CUDA)\nEXASIM_GPU=1 EXASIM_INSTALL=/path/to/gpu-exasim \\\n";
    s += "  NVCC_WRAPPER=/path/to/kokkos/bin/nvcc_wrapper EXASIM_GPU_ARCH=sm_70 ./build.sh\n";
    s += "mpirun -np 1 build/" + app + " datain/ dataout/out\n```\n";
    return s;
}

inline void writeFile(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    f << content;
}

// Remove the codegen's intermediate files from <destDir>/generated/, keeping only the
// concrete-model headers the app actually includes (my_model.hpp + model_sizes.hpp). The
// normal codegen dumps its whole SymEngine scratch (Symbolic*/Hdg*/Kokkos*/Code2Cpp/...)
// into the model dir; for a self-contained app we don't want that clutter.
inline void cleanupGeneratedDir(const std::string& generatedDir) {
    std::error_code ec;
    if (!std::filesystem::is_directory(generatedDir, ec)) return;
    for (const auto& entry : std::filesystem::directory_iterator(generatedDir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name == "my_model.hpp" || name == "model_sizes.hpp") continue;
        std::filesystem::remove_all(entry.path(), ec);
    }
}

// Write the app scaffold into destDir. The concrete model header is produced separately by
// the normal codegen into destDir/generated/my_model.hpp.
inline void writeAppScaffold(const ParsedSpec& spec, const std::string& destDir,
                             const std::string& appName, int modelID) {
    std::error_code ec;
    std::filesystem::create_directories(destDir, ec);
    std::filesystem::create_directories(destDir + "/generated", ec);
    cleanupGeneratedDir(destDir + "/generated");
    writeFile(destDir + "/" + appName + ".cc", mainCc(appName, modelID));
    writeFile(destDir + "/CMakeLists.txt", cmakeLists(appName));
    writeFile(destDir + "/build.sh", buildSh(appName));
    writeFile(destDir + "/README.md", readme(appName, spec));
    std::filesystem::permissions(destDir + "/build.sh",
        std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec,
        std::filesystem::perm_options::add, ec);
    std::cout << "Emitted standalone app scaffold to " << destDir
              << " (target " << appName << ")\n";
}

}  // namespace appscaffold
