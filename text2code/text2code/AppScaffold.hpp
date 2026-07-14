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
    s += "    const int backend = 0;  // 0/1 = host CPU (set 2 for CUDA / 3 for HIP builds)\n\n";
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
    s += "option(EXASIM_MPI \"Use the MPI-enabled Exasim variant\" ON)\n";
    s += "option(EXASIM_GPU \"Use the GPU-enabled Exasim variant\" OFF)\n";
    s += "if(EXASIM_GPU AND EXASIM_MPI)\n  set(EXASIM_VARIANT gpumpi)\n";
    s += "elseif(EXASIM_GPU)\n  set(EXASIM_VARIANT gpu)\n";
    s += "elseif(EXASIM_MPI)\n  set(EXASIM_VARIANT cpumpi)\nelse()\n  set(EXASIM_VARIANT cpu)\nendif()\n\n";
    s += "find_package(Exasim REQUIRED COMPONENTS ${EXASIM_VARIANT})\n";
    s += "find_package(Kokkos REQUIRED)\nfind_package(MPI REQUIRED)\n\n";
    s += "find_package(PkgConfig REQUIRED)\n";
    s += "pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)\n";
    s += "message(STATUS \"PETSc: ${PETSC_VERSION}\")\n\n";
    s += "find_package(BLAS REQUIRED)\nfind_package(LAPACK REQUIRED)\n\n";
    s += "set(_UNITY_DEFS EXASIM_HAVE_PETSC HAVE_BACKEND_PREPROCESSING)\n";
    s += "if(EXASIM_MPI)\n  list(APPEND _UNITY_DEFS _MPI)\nendif()\n\n";
    s += "add_executable(${PROJECT_NAME} " + app + ".cc)\n";
    s += "target_compile_definitions(${PROJECT_NAME} PRIVATE ${_UNITY_DEFS})\n";
    s += "target_link_libraries(${PROJECT_NAME} PRIVATE\n";
    s += "    Exasim::headers\n    Kokkos::kokkos\n    MPI::MPI_CXX\n    PkgConfig::PETSC\n";
    s += "    LAPACK::LAPACK\n    BLAS::BLAS)\n";
    s += "target_include_directories(${PROJECT_NAME} PRIVATE \"${CMAKE_CURRENT_SOURCE_DIR}\")\n";
    return s;
}

inline std::string buildSh(const std::string& app) {
    std::string s;
    s += "#!/usr/bin/env bash\n";
    s += "# Configure + build the standalone header-only app against a petsc-enabled Exasim install.\n";
    s += "set -eo pipefail\n";
    s += "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n";
    s += "EXASIM_INSTALL=\"${EXASIM_INSTALL:?set EXASIM_INSTALL to a petsc-enabled Exasim install prefix}\"\n";
    s += "KOKKOS_DIR=\"${KOKKOS_DIR:-$EXASIM_INSTALL/../Exasim-build/deps/kokkos/buildserial}\"\n";
    s += "BUILD=\"${BUILD:-$HERE/build}\"\n\n";
    s += "cmake -S \"$HERE\" -B \"$BUILD\" \\\n";
    s += "  -DCMAKE_BUILD_TYPE=Release \\\n";
    s += "  -DEXASIM_MPI=ON -DEXASIM_GPU=OFF \\\n";
    s += "  -DCMAKE_PREFIX_PATH=\"$EXASIM_INSTALL\" \\\n";
    s += "  -DExasim_DIR=\"$EXASIM_INSTALL/lib/cmake/Exasim\" \\\n";
    s += "  -DKokkos_DIR=\"$KOKKOS_DIR\" \\\n";
    s += "  -DPKG_CONFIG_USE_CMAKE_PREFIX_PATH=ON\n\n";
    s += "cmake --build \"$BUILD\" -j 4\n";
    s += "echo \"built: $BUILD/" + app + "\"\n";
    return s;
}

inline std::string readme(const std::string& app, const ParsedSpec& spec) {
    const bool coupling = isOutput(spec, "Fint") || isOutput(spec, "Fext");
    std::string s;
    s += "# " + app + "\n\n";
    s += "Standalone, header-only, C++-driven Exasim app auto-generated by `text2code --emit-app`.\n";
    s += "It solves the model as a steady HDG problem through Exasim's exported PETSc operator —\n";
    s += "**no** runtime-loaded model `.so`, and **no** hand-rolled PETSc solver code in the app\n";
    s += "(the whole solver is `exasim::petsc::solve_steady`).\n\n";
    s += "Model sizes: nd=" + std::to_string(vsize(spec, "x")) +
         ", ncu=" + std::to_string(vsize(spec, "uhat")) +
         ", nco=" + std::to_string(vsize(spec, "v")) +
         ", ncw=" + std::to_string(vsize(spec, "w")) +
         ", nparam=" + std::to_string(vsize(spec, "mu"));
    if (coupling) s += "  (has external coupling: Fint/Fext)";
    s += ".\n\n";
    s += "## Build & run\n\n```sh\nEXASIM_INSTALL=/path/to/petsc-enabled-exasim ./build.sh\n";
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
