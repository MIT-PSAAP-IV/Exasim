/*
 * text2code.cpp
 *
 * This program generates input files and dynamic libraries for the EXASIM framework
 * from a user-provided PDE application specification in a text file.
 *
 * Usage:
 *   ./parseinput <pdeapp.txt>
 *
 * Main Steps:
 *   1. Checks if the input file exists.
 *   2. Parses the input file to extract PDE and mesh parameters.
 *   3. Initializes PDE, mesh, and master structures.
 *   4. Writes binary files required by EXASIM.
 *   5. Generates C++ code for the PDE specification.
 *   6. Optionally compiles and builds dynamic libraries for EXASIM.
 *
 * Dependencies:
 *   - C++17 standard library
 *   - BLAS and LAPACK libraries
 *   - METIS and GKlib (optional, if HAVE_METIS is defined)
 *   - Several local source files: tinyexpr.cpp, helpers.cpp, readpdeapp.cpp, readmesh.cpp,
 *     makemesh.cpp, makemaster.cpp, domaindecomposition.cpp, connectivity.cpp,
 *     writebinaryfiles.cpp, CodeGenerator.cpp, CodeCompiler.cpp
 *
 * Compilation Examples:
 *   See commented lines at the top of the file for various compilation options.
 *
 * Author: Ngoc Cuong Nguyen
 * Date: 07/07/2025
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <vector>
#include <array>
#include <regex>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <cmath>
#include <cctype>

#ifdef HAVE_METIS
#include <metis.h>
#endif

// g++ -O2 -std=c++17 text2bina.cpp -o text2bina -lblas -llapack
// g++ -O2 -DHAVE_METIS -std=c++17 text2bina.cpp -o text2bina -lblas -llapack -I../METIS/build/xinclude -L../METIS/build/libmetis -lmetis -I../GKlib/include -L../GKlib/lib -lGKlib
// clang++ -fsanitize=address -fno-omit-frame-pointer -g -O2 -DHAVE_METIS -std=c++17 text2bina.cpp -o text2bina -lblas -llapack -I../METIS/build/xinclude -L../METIS/build/libmetis -lmetis -I../GKlib/include -L../GKlib/lib -lGKlib
// g++ -O2 -DHAVE_METIS -std=c++17 text2code.cpp -o text2code -lblas -llapack -I../METIS/build/xinclude -L../METIS/build/libmetis -lmetis -I../GKlib/include -L../GKlib/lib -lGKlib
// g++ -O2 -std=c++17 text2code.cpp -o text2code -lblas -llapack 

using namespace std;

#include "TextParser.hpp"

#include "tinyexpr.cpp"
#include "helpers.cpp"
#include "readpdeapp.cpp"
#include "readmesh.cpp"
#include "makemesh.cpp"
#include "makemaster.cpp"
#include "domaindecomposition.cpp"
#include "connectivity.cpp"
#include "writebinaryfiles.cpp"
#include "CodeGenerator.cpp"
#include "CodeCompiler.cpp"
#include "AppScaffold.hpp"

int main(int argc, char* argv[]) 
{
    if (argc < 2) {
        std::cerr << "Usage: ./text2code <pdeapp.txt> [--out-dir <path>] [--emit-app <dir>]\n"
            "  --out-dir overrides where my_model.hpp + libpdemodel*.{so,dylib}\n"
            "    are written. Default: <exasimpath>/backend/Model/Text2codeGenerated/\n"
            "    Per-target dirs let multiple text2code runs avoid clobbering\n"
            "    each other (HOT.7.14 — drops the test-matrix RESOURCE_LOCK).\n"
            "  --emit-app <dir> also writes a standalone header-only, C++-driven app\n"
            "    (driver + CMakeLists.txt + build.sh + README.md) into <dir>, with the\n"
            "    concrete model header at <dir>/generated/my_model.hpp. The app builds\n"
            "    CSolution<PdeModel> from datain/ and solves via exasim::petsc::solve_steady\n"
            "    -- no runtime .so model ABI, no hand-rolled PETSc glue. Optional companions:\n"
            "    --app-name <name> (default: basename of <dir>), --model-id <n> (default: 100).\n";
        return 1;
    }    

    if (std::filesystem::exists(argv[1])) {
        std::cout << "Generating Exasim's input files for this text file ("<< argv[1] << ") ... \n\n";
    } else {
        error("Error: Input file does not exist.\n");        
    }          
           
    // Optional --out-dir override (HOT.7.14).
    std::string out_dir_override;
    // --gen-only emits the generated kernel sources but skips building the
    // dynamic model library. Used by the built-in model CMake build, which
    // compiles the generated kernels into libbuiltinmodel itself.
    bool gen_only = false;
    // --emit-app <dir> also writes a standalone header-only, C++-driven app scaffold
    // (driver + CMakeLists + build.sh + README) into <dir>, with the concrete model
    // header routed to <dir>/generated/my_model.hpp. See AppScaffold.hpp.
    std::string emit_app_dir;
    std::string app_name;
    int app_model_id = 100;
    // The emitted app is HDG-only (its driver solves the condensed HDG trace system via
    // exasim::petsc::solve_steady). Refuse to emit for an LDG-configured model unless the
    // caller explicitly overrides (they must then supply HDG datain to run it).
    bool allow_ldg = false;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--out-dir" && i + 1 < argc) {
            out_dir_override = argv[i + 1];
            ++i;
        } else if (std::string(argv[i]) == "--gen-only") {
            gen_only = true;
        } else if (std::string(argv[i]) == "--emit-app" && i + 1 < argc) {
            emit_app_dir = argv[i + 1];
            ++i;
        } else if (std::string(argv[i]) == "--app-name" && i + 1 < argc) {
            app_name = argv[i + 1];
            ++i;
        } else if (std::string(argv[i]) == "--model-id" && i + 1 < argc) {
            app_model_id = std::stoi(argv[i + 1]);
            ++i;
        } else if (std::string(argv[i]) == "--allow-ldg") {
            allow_ldg = true;
        }
    }
    // --emit-app routes the model header into <appdir>/generated/ (so the app is
    // self-contained) unless the caller also gave an explicit --out-dir.
    if (!emit_app_dir.empty() && out_dir_override.empty()) {
        out_dir_override = (emit_app_dir.back() == '/' ? emit_app_dir : emit_app_dir + "/") + "generated/";
    }
    if (!emit_app_dir.empty() && app_name.empty()) {
        std::string d = emit_app_dir;
        while (!d.empty() && d.back() == '/') d.pop_back();
        auto slash = d.find_last_of('/');
        app_name = (slash == std::string::npos) ? d : d.substr(slash + 1);
        if (app_name.empty()) app_name = "exasim_app";
    }

    InputParams params = parseInputFile(argv[1]);                           
    PDE pde = initializePDE(params);    
     
    ParsedSpec spec = TextParser::parseFile(make_path(pde.datapath, pde.modelfile));        
    spec.exasimpath = pde.exasimpath;
    //spec.modelpath = make_path(pde.exasimpath, "/backend/Model/Text2codeGenerated/");
    spec.modelpath = out_dir_override.empty()
        ? make_path(pde.exasimpath, "/backend/Model/Text2codeGenerated/")
        : (out_dir_override.back() == '/' ? out_dir_override : out_dir_override + "/");
    std::error_code ec;
    std::filesystem::create_directories(spec.modelpath, ec);
    if (ec) {
        std::cerr << "Unable to create output directory " << spec.modelpath << ": " << ec.message() << "\n";
        return 1;
    }
    spec.symenginepath = make_path(pde.exasimpath, "/text2code/symengine");
    
    if (pde.gendatain == 1) {
      Mesh mesh = initializeMesh(params, pde);        
      Master master = initializeMaster(pde, mesh);                                    
      writeBinaryFiles(pde, mesh, master, spec);
    }
    
#ifdef USE_CMAKE
    if (pde.gencode==1) {
        generateCppCode(spec);
        // An explicit --out-dir means the caller wants the concrete-model header
        // (my_model.hpp) emitted into that dir -- e.g. a standalone C++ app that
        // compiles PdeModel. That header is written by running the generated
        // Code2Cpp (executeCppCode). Without this, USE_CMAKE builds only emitted
        // the kernel sources + Code2Cpp and silently skipped my_model.hpp.
        // The builtin-model build (no --out-dir) keeps the generate-only behaviour.
        if (!out_dir_override.empty() && !gen_only) executeCppCode(spec);
        // Emit the model-owned sizes header (PR #33: ncu/ncv/... from the model).
        { CodeGenerator _gen(spec); _gen.generateModelSizesHpp(spec.modelpath); }
    }
#else
    if (pde.gencode==1) {
        generateCppCode(spec);
        { CodeGenerator _gen(spec); _gen.generateModelSizesHpp(spec.modelpath); }
        executeCppCode(spec);
        if (!gen_only) buildDynamicLibraries(spec);
    }
#endif
    
    // Emit the standalone header-only C++-driven app scaffold (driver + CMakeLists +
    // build.sh + README) alongside the generated model header in <appdir>/generated/.
    if (!emit_app_dir.empty()) {
        // The emitted app drives a steady HDG solve (exasim::petsc::solve_steady on the
        // condensed trace system). Emitting it for an LDG-configured model produces an app
        // that cannot run this model's LDG datain (the backend's HDG connectivity build
        // rejects it). Raise explicitly rather than emit a silently-unusable app.
        if (pde.hybrid != 1 && !allow_ldg) {
            std::cerr << "\nError: --emit-app produces an HDG-only app "
                         "(exasim::petsc::solve_steady solves the condensed HDG trace system), "
                         "but this pdeapp is configured discretization=\""
                      << pde.discretization << "\" (LDG). The emitted app could not run this "
                         "model's datain.\n"
                         "  Fix: set discretization=\"hdg\" (hybrid=1) in the pdeapp, or use the "
                         "frontend exportapp path for an LDG/ExasimSolver app.\n"
                         "  Override (advanced): pass --allow-ldg to emit anyway (you must then "
                         "supply HDG datain to run it).\n";
            return 1;
        }
        appscaffold::writeAppScaffold(spec, emit_app_dir, app_name, app_model_id);
    }

    std::cout << "\n******** Done with generating input files and dynamic libraries for EXASIM ********\n";

    return 0;
}
