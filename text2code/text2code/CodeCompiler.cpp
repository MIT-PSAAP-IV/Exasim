
/*
    CodeCompiler.cpp

    This file provides functions for generating, compiling, and building dynamic libraries for C++ code
    based on a parsed model specification. It is designed to support multiple platforms (Windows, macOS, Linux)
    and toolchains (GCC, Clang, MSVC), as well as optional CUDA and HIP backends via Kokkos.

    Main Components:

    1. generateCppCode(PDE& pde)
        - Parses a model file and generates corresponding C++ source files using SymEngine and custom code generators.
        - Handles framework-specific code generation for CUDA and HIP.
        - Generates empty source files for optional outputs if not specified.
        - Outputs generated files to the model directory.

    2. Compiler Detection and Toolchain Utilities
        - Functions to detect available C++ compilers and their kind (GCC, Clang, MSVC).
        - Utilities for quoting paths, running commands silently, and checking compiler support for flags.
        - Toolchain struct encapsulates compiler executable, kind, and relevant flags.

    3. executeCppCode(ParsedSpec& spec)
        - Compiles the generated C++ source file into an executable using the detected toolchain.
        - Links against SymEngine library.
        - Runs the resulting executable to perform code generation tasks.

    4. buildDynamicLibraries(ParsedSpec& spec)
        - Builds shared libraries for the generated model code using Kokkos backends (serial, CUDA, HIP).
        - Handles platform-specific shared library extensions and compiler flags.
        - Compiles and links against Kokkos core and container libraries.
        - Supports detection and usage of nvcc_wrapper and hipcc_wrapper for CUDA and HIP compilation.

    5. Utility Functions
        - getSharedLibExtension(): Returns the appropriate shared library extension for the current platform.
        - with_exe_if_windows(): Appends ".exe" to executable names on Windows.
        - join(): Joins vector of strings with a separator.

    Notes:
        - Error handling is performed via error() calls when compilation or execution fails.
        - The code is designed to be robust across platforms and compilers, with careful quoting and flag selection.
        - Some legacy code for building dynamic libraries is commented out for reference.

*/

// Build-time SymEngine resolution (installed copy preferred, vendored fallback),
// baked by text2code/CMakeLists.txt. Absent in non-CMake builds → fall back to
// the vendored path constructed from spec.symenginepath.
#if defined(__has_include)
#  if __has_include("symengine_config.h")
#    include "symengine_config.h"
#  endif
#endif
#ifndef EXASIM_INSTALL_PREFIX
#  define EXASIM_INSTALL_PREFIX ""
#endif
#ifndef EXASIM_SYMENGINE_FOUND
#  define EXASIM_SYMENGINE_FOUND 0
#endif
#ifndef EXASIM_KOKKOS_DIR
#  define EXASIM_KOKKOS_DIR ""        // empty -> fall back to the exasimpath layout
#endif
#ifndef EXASIM_KOKKOS_BACKEND
#  define EXASIM_KOKKOS_BACKEND "serial"
#endif

void generateCppCode(ParsedSpec spec)
{
    std::cout << "\nGenerating the C++ code for this model file ("<< spec.modelfile << ") ... \n\n";

    // ParsedSpec spec = TextParser::parseFile(make_path(pde.datapath, pde.modelfile));        
    // spec.exasimpath = pde.exasimpath;
    // spec.modelpath = make_path(pde.exasimpath, "/backend/Model/");
    // spec.symenginepath = make_path(pde.exasimpath, "/text2code/symengine");
    
    // Generate SymEngine code
    CodeGenerator gen(spec);
    gen.generateCode2Cpp(make_path(spec.modelpath, "Code2Cpp.cpp"));
    gen.generateSymbolicFunctionsHpp(make_path(spec.modelpath, "SymbolicFunctions.hpp"));
    gen.generateSymbolicFunctionsCpp(make_path(spec.modelpath, "SymbolicFunctions.cpp"));        
    gen.generateSymbolicScalarsVectorsHpp(make_path(spec.modelpath, "SymbolicScalarsVectors.hpp"));
    gen.generateSymbolicScalarsVectorsCpp(make_path(spec.modelpath, "SymbolicScalarsVectors.cpp"));   

    std::string framework = spec.framework;
    if ((framework == "Cuda") || (framework == "CUDA") || (framework == "cuda")) {
      gen.generateCudaHipHpp(make_path(spec.modelpath, "CudaHip.hpp"));
    } 
    else if ((framework == "Hip") || (framework == "HIP") || (framework == "hip")) {
      gen.generateCudaHipHpp(make_path(spec.modelpath, "CudaHip.hpp"));
    } 

    if (spec.isoutput[6]==false) gen.generateEmptySourcewCpp(spec.modelpath);
    if (spec.isoutput[7]==false) gen.generateEmptyOutputCpp(spec.modelpath);
    if (spec.isoutput[8]==false) gen.generateEmptyMonitorCpp(spec.modelpath);
    if (spec.isoutput[9]==false) gen.generateEmptyInituCpp(spec.modelpath);
    if (spec.isoutput[10]==false) gen.generateEmptyInitqCpp(spec.modelpath);
    if (spec.isoutput[11]==false) gen.generateEmptyInitudgCpp(spec.modelpath);
    if (spec.isoutput[12]==false) gen.generateEmptyInitwdgCpp(spec.modelpath);
    if (spec.isoutput[13]==false) gen.generateEmptyInitodgCpp(spec.modelpath);
    if (spec.isoutput[14]==false) gen.generateEmptyAvfieldCpp(spec.modelpath);
    if (spec.isoutput[15]==false) gen.generateEmptyFintCpp(spec.modelpath);
    if (spec.isoutput[16]==false) gen.generateEmptyEoSCpp(spec.modelpath);
    if (spec.isoutput[17]==false) gen.generateEmptyVisScalarsCpp(spec.modelpath);
    if (spec.isoutput[18]==false) gen.generateEmptyVisVectorsCpp(spec.modelpath);
    if (spec.isoutput[19]==false) gen.generateEmptyVisTensorsCpp(spec.modelpath);
    if (spec.isoutput[20]==false) gen.generateEmptyQoIvolumeCpp(spec.modelpath);
    if (spec.isoutput[21]==false) gen.generateEmptyQoIboundaryCpp(spec.modelpath);
    if (spec.isoutput[22]==false) gen.generateEmptyFextCpp(spec.modelpath);
    gen.generateEmptyFhatCpp(spec.modelpath);
    gen.generateEmptyUhatCpp(spec.modelpath);
    gen.generateEmptyStabCpp(spec.modelpath);
    gen.generateLibPDEModelHpp(spec.modelpath);
    gen.generateLibPDEModelCpp(spec.modelpath);

    std::cout << "C++ source files are generated and located in this directory " << spec.modelpath << "\n";
    std::cout << "Finished generating code.\n";
    
    //return spec;
}

namespace fs = std::filesystem;

static std::string quote(const std::string& s) {
#ifdef _WIN32
    // crude but effective quoting for paths with spaces
    if (s.find_first_of(" \t\"") == std::string::npos) return s;
    return "\"" + s + "\"";
#else
    if (s.find_first_of(" \t\"'\\$()") == std::string::npos) return s;
    return "'" + s + "'";
#endif
}

static int run_silently(const std::string& cmd) {
#ifdef _WIN32
    std::string full = cmd + " >nul 2>&1";
#else
    std::string full = cmd + " >/dev/null 2>&1";
#endif
    return std::system(full.c_str());
}

enum class CompilerKind { GCC, Clang, MSVC, Unknown };

static bool probe_compiler(const std::string& exe) {
#ifdef _WIN32
    // cl/clang-cl don’t accept --version; /? is fine
    if (exe.find("cl") != std::string::npos)
        return run_silently(quote(exe) + " /?") == 0;
#endif
    // Most compilers accept --version or -v
    int ok = run_silently(quote(exe) + " --version");
    if (ok != 0) ok = run_silently(quote(exe) + " -v");
    return ok == 0;
}

static std::string first_nonempty(const char* s) { return s ? std::string(s) : std::string(); }

static std::string detect_cxx_exe() {
    // 1) respect CXX env
    if (auto env = first_nonempty(std::getenv("CXX")); !env.empty()) {
        if (probe_compiler(env)) return env;
    }
#ifdef _WIN32
    // 2) common Windows candidates
    const char* candidates[] = {"g++", "clang++", "c++", "cl", "clang-cl", nullptr};
#else
    // 2) common Unix/mac candidates (prefer c++ wrapper)
    const char* candidates[] = {"g++", "clang++", "c++", nullptr};
#endif
    for (const char** p = candidates; *p; ++p) {
        if (probe_compiler(*p)) return *p;
    }
    throw std::runtime_error("No working C++ compiler found in PATH. Set CXX or install clang++/g++.");
}

static CompilerKind detect_kind(const std::string& exe) {
#ifdef _WIN32
    if (exe.find("cl") != std::string::npos) return CompilerKind::MSVC;
    if (exe.find("clang-cl") != std::string::npos) return CompilerKind::MSVC; // via clang driver
#endif
    // Cheap heuristic via -v output
    // (We don’t capture here; kind is mostly for MSVC vs not)
    if (exe.find("clang") != std::string::npos) return CompilerKind::Clang;
    if (exe.find("g++")   != std::string::npos || exe == "c++") return CompilerKind::GCC;
    return CompilerKind::Unknown;
}

static bool compiler_supports_flag(const std::string& exe, const std::string& flag) {
    fs::path tmpdir = fs::temp_directory_path();
    fs::path src = tmpdir / "cmk_check.cpp";
    fs::path obj = tmpdir / "cmk_check.o";
    std::ofstream(src.string()) << "int main(){return 0;}\n";
#ifdef _WIN32
    std::string out = quote((tmpdir / "cmk_check.obj").string());
    std::string cmd = quote(exe) + " " + flag + " /nologo /c " + quote(src.string()) + " /Fo" + out;
#else
    std::string cmd = quote(exe) + " -x c++ " + flag + " -c " + quote(src.string()) +
                      " -o " + quote(obj.string());
#endif
    int rc = run_silently(cmd);
    std::error_code ec; fs::remove(src, ec); fs::remove(obj, ec);
    return rc == 0;
}

struct Toolchain {
    std::string cxx;
    CompilerKind kind{};
    std::vector<std::string> cxx17_flag;
    std::vector<std::string> warn_squelch;
};

static Toolchain detect_toolchain()
{
    Toolchain tc;

    if (const char* env_cxx = std::getenv("CXX")) {
        tc.cxx = env_cxx;
    }

    if (tc.cxx.empty()) {
        tc.cxx = detect_cxx_exe();
    }

    if (tc.cxx.empty()) {
        error("Could not detect a C++ compiler. Please set CXX.");
    }

    tc.kind = detect_kind(tc.cxx);

    if (tc.kind == CompilerKind::MSVC) {
        tc.cxx17_flag = {"/std:c++17", "/EHsc"};
    } else {
        tc.cxx17_flag = {"-std=c++17"};
    }

    if (tc.kind != CompilerKind::MSVC &&
        compiler_supports_flag(tc.cxx,
                               "-Wno-inconsistent-missing-override"))
    {
        tc.warn_squelch.push_back(
            "-Wno-inconsistent-missing-override");
    }

    std::cout << "Detected C++ compiler: "
              << tc.cxx << "\n";

    return tc;
}

static std::string join(const std::vector<std::string>& v, const std::string& sep = " ") {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

constexpr std::string_view to_string(CompilerKind k) {
    switch (k) {
        case CompilerKind::GCC:    return "GCC";
        case CompilerKind::Clang:  return "Clang";
        case CompilerKind::MSVC:   return "MSVC";
        default:                   return "Unknown";
    }
}

static std::string with_exe_if_windows(std::string s) {
#ifdef _WIN32
    fs::path p = s;
    if (p.extension().empty()) p.replace_extension(".exe");
    return p.string();
#else
    return s;
#endif
}

int executeCppCode(ParsedSpec& spec) 
{
    std::string sourcefile = make_path(spec.modelpath , "Code2Cpp.cpp");                    
    std::string symengine_include = make_path(spec.symenginepath, "include");    
    std::string exefile = make_path(spec.modelpath, "code2cpp");

    Toolchain tc = detect_toolchain();

    std::cout<<"Compiling " + sourcefile + "\n";
    
    // Construct compile command using text2code_path
    std::stringstream cmd;
        
    // SymEngineFunctionWrappers.hpp lives at backend/Model/Text2codeGenerated/ and isn't
    // regenerated per-target — include it from the static location
    // (HOT.7.14 added per-target spec.modelpath; without this the
    // generated SymbolicFunctions.hpp can't find the wrapper header).
    std::string backend_model = make_path(spec.exasimpath, "backend/Model/Text2codeGenerated/");
    if (!fs::exists(make_path(backend_model, "SymEngineFunctionWrappers.hpp"))) {
      std::string installed_backend_model = make_path(spec.exasimpath, "include/backend/Model/Text2codeGenerated/");
      if (fs::exists(make_path(installed_backend_model, "SymEngineFunctionWrappers.hpp")))
        backend_model = installed_backend_model;
    }
    if (tc.kind == CompilerKind::MSVC) {    
      std::string symengine_lib = make_path(spec.symenginepath, "lib/symengine.lib");
      cmd << tc.cxx << " /std:c++17 /EHsc /W0 "
          << "/I" << quote(spec.symenginepath) << " "
          << "/I" << quote(symengine_include) << " "
          << "/I" << quote(backend_model) << " "
          << quote(sourcefile) << " "
          << quote(symengine_lib) << " "
          << "/Fe:" << quote(exefile);
    } else {
#if EXASIM_SYMENGINE_FOUND
      // Use the SymEngine resolved at build time (installed copy preferred,
      // vendored fallback) — baked into symengine_config.h.
      // -Wno-template-body: Red Hat GCC 13.3.1 backports this diagnostic from
      // GCC 14 and treats it as an error; the vendored Boost headers trigger it.
      // GCC silently accepts unknown -Wno-* flags, so this is safe everywhere.
      cmd << tc.cxx << " -std=c++17 -w -Wno-template-body "
          << EXASIM_SYMENGINE_INCFLAGS << " "
          << "-I" << quote(backend_model) << " "
          << quote(sourcefile) << " "
          << EXASIM_SYMENGINE_LIB << " -o "
          << quote(exefile);
#else
      std::string symengine_lib = make_path(spec.symenginepath, "lib/libsymengine.a");
      cmd << tc.cxx << " -std=c++17 -w -Wno-template-body "
          << "-I" << quote(spec.symenginepath) << " "
          << "-I" << quote(symengine_include) << " "
          << "-I" << quote(backend_model) << " "
          << quote(sourcefile) << " "
          << quote(symengine_lib) << " -o "
          << quote(exefile);
#endif
    }
                           
    std::cout<<cmd.str().c_str()<<std::endl;
    
    int status = std::system(cmd.str().c_str());
    if (status != 0)  
        error("Compiling Code2Cpp.cpp failed! Please compile text2code with -DUSE_CMAKE=ON and use cmake to run text2code.");        
    else
        std::cout<<"Compiling " + sourcefile + " successfully.\n";
                                  
    std::cout << "Running " << exefile << " ...\n";
    std::string run = with_exe_if_windows(exefile);    
    std::string run_cmd = quote(run) + " " + quote(spec.modelpath);
    status = std::system(run_cmd.c_str());
    if (status != 0) 
        error("code2cpp execution failed! Please compile text2code with -DUSE_CMAKE=ON and use cmake to run text2code.");            
    else
        std::cout<<"Running " + exefile + " successfully.\n";
    
    return 0;
}

std::string getSharedLibExtension() {
#if defined(_WIN32)
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else // Linux and other POSIX systems
    return ".so";
#endif
}

int buildDynamicLibraries(ParsedSpec& spec) 
{  
    std::cout << "text2code is building the shared library" << std::endl;

    // Kokkos location baked by the superbuild (it builds Kokkos out of the source
    // tree). The active backend's path points at the baked dir; the others stay
    // empty so only the matching libt2cmodel<backend> is built. Falls back to the
    // legacy in-source exasimpath/kokkos/build* layout for a standalone build.    
    const std::string _kk(EXASIM_KOKKOS_DIR), _kkb(EXASIM_KOKKOS_BACKEND);

    auto getenv_string = [](const char* name) {
        const char* v = std::getenv(name);
        return (v && std::string(v).size() > 0) ? std::string(v) : std::string{};
    };

    auto first_existing_dir = [](std::initializer_list<std::string> dirs) {
        for (const auto& dir : dirs) {
            if (!dir.empty() && fs::exists(dir) && fs::is_directory(dir)) {
                return dir;
            }
        }
        return std::string{};
    };

    const std::string kokkos_serial_path =
        !_kk.empty() ? (_kkb == "serial" ? _kk : "")
                     : first_existing_dir({
                           make_path(spec.exasimpath, "deps/kokkos/buildserial"),
                           make_path(spec.exasimpath, "build_local/deps/kokkos/buildserial"),
                           make_path(spec.exasimpath, "local/deps/kokkos/buildserial")});

    const std::string kokkos_cuda_path =
        !_kk.empty() ? (_kkb == "cuda" ? _kk : "")
                     : first_existing_dir({
                           make_path(spec.exasimpath, "deps/kokkos/buildcuda"),
                           make_path(spec.exasimpath, "build_local/deps/kokkos/buildcuda"),
                           make_path(spec.exasimpath, "local/deps/kokkos/buildcuda")});

    const std::string kokkos_hip_path =
        !_kk.empty() ? (_kkb == "hip" ? _kk : "")
                     : first_existing_dir({
                           make_path(spec.exasimpath, "deps/kokkos/buildhip"),
                           make_path(spec.exasimpath, "build_local/deps/kokkos/buildhip"),
                           make_path(spec.exasimpath, "local/deps/kokkos/buildhip")});
  
    std::string exasim_lib_path = getenv_string("EXASIMLIB_DIR");    
    if (exasim_lib_path.empty()) {
        const std::string exasim_prefix = getenv_string("EXASIM_PREFIX");
    
        if (!exasim_prefix.empty()) {
            exasim_lib_path = make_path(exasim_prefix, "lib");
        }
        else {
            exasim_lib_path = first_existing_dir({
                make_path(spec.exasimpath, "local/lib"),
                make_path(spec.exasimpath, "local/lib64"),
                make_path(spec.exasimpath, "lib"),
                make_path(spec.exasimpath, "lib64")});
            if (exasim_lib_path.empty()) {
                exasim_lib_path = make_path(spec.exasimpath, "local/lib");
            }
        }
    }    
    std::cout << "Using Exasim library directory: " << exasim_lib_path << "\n";
    fs::create_directories(exasim_lib_path);

    Toolchain tc = detect_toolchain();
    auto ext = getSharedLibExtension();
  
    auto out_name = [&](const std::string& stem) {
        return make_path(exasim_lib_path, stem) + ext;
    };

    auto inc = [&](const std::string& dir) {
        return (tc.kind == CompilerKind::MSVC)
               ? ("/I" + quote(dir))
               : ("-I" + quote(dir));
    };

    auto exasim_include_flags = [&]() -> std::string
    {
        std::stringstream flags;
        const std::string include_root = make_path(spec.exasimpath, "include");

        if (fs::exists(spec.exasimpath) && fs::is_directory(spec.exasimpath)) {
            flags << inc(spec.exasimpath) << " ";
        }

        if (include_root != spec.exasimpath &&
            fs::exists(include_root) && fs::is_directory(include_root)) {
            flags << inc(include_root) << " ";
        }

        return flags.str();
    };

    auto shlib_flags = [&] {
        if (tc.kind == CompilerKind::MSVC) {
            return std::string{"/LD /nologo /std:c++17 /EHsc /W0 "};
        }

#ifdef __APPLE__
        return std::string{"-dynamiclib -std=c++17 -fPIC "};
#else
        return std::string{"-shared -std=c++17 -fPIC "};
#endif
    }();

    auto install_name_flag = [&](const std::string& out) {
#ifdef __APPLE__
        if (tc.kind == CompilerKind::MSVC) {
            return std::string{};
        }

        return std::string{"-Wl,-install_name,@rpath/"}
             + fs::path(out).filename().string() + " ";
#else
        return std::string{};
#endif
    };

    auto kokkos_lib_dir = [&](const std::string& kokkos_path) {
        const std::string lib64_dir = make_path(kokkos_path, "lib64");
        const std::string lib_dir   = make_path(kokkos_path, "lib");

        if (fs::exists(lib64_dir) && fs::is_directory(lib64_dir)) {
            return lib64_dir;
        }

        return lib_dir;
    };

    auto require_kokkos_static_libs = [&](const std::string& kokkos_path,
                                          const std::string& backend,
                                          std::string& lib_dir,
                                          std::string& lib_core,
                                          std::string& lib_cont) {
        lib_dir = kokkos_lib_dir(kokkos_path);

        if (tc.kind == CompilerKind::MSVC) {
            lib_core = make_path(lib_dir, "kokkoscore.lib");
            lib_cont = make_path(lib_dir, "kokkoscontainers.lib");
        } else {
            lib_core = make_path(lib_dir, "libkokkoscore.a");
            lib_cont = make_path(lib_dir, "libkokkoscontainers.a");
        }

        if (!fs::exists(lib_core)) {
            std::cout << "Skipping Kokkos " << backend
                      << ": missing " << lib_core << "\n";
            return false;
        }

        if (!fs::exists(lib_cont)) {
            std::cout << "Skipping Kokkos " << backend
                      << ": missing " << lib_cont << "\n";
            return false;
        }

        return true;
    };

    auto text2code_model_source = [&]() {
        const std::string generated_src = make_path(spec.modelpath, "libt2cmodel.cpp");
        if (fs::exists(generated_src)) {
            return generated_src;
        }

        const std::string installed_src =
            make_path(spec.exasimpath, "include/backend/Model/Text2codeGenerated/libt2cmodel.cpp");
        if (fs::exists(installed_src)) {
            return installed_src;
        }

        return generated_src;
    };
        
    auto get_gpu_arch = [&](std::initializer_list<const char*> vars,
                            const std::string& backend) -> std::string
    {
        for (const char* var : vars) {
            std::string value = getenv_string(var);
            if (!value.empty()) {
                std::cout << "Using " << backend
                          << " GPU architecture from "
                          << var << ": "
                          << value << "\n";
                return value;
            }
        }
    
        std::cout << "Warning: no " << backend
                  << " GPU architecture specified. "
                  << "Using compiler default.\n";
    
        return {};
    };
    
    auto cuda_arch_flag = [&]() -> std::string
    {
        const std::string arch =
            get_gpu_arch({"EXASIM_CUDA_ARCH", "CUDAARCH", "EXASIM_GPU_ARCH"}, "CUDA");
    
        return arch.empty() ? std::string{} : "-arch=" + arch + " ";
    };
    
    auto hip_arch_flag = [&]() -> std::string
    {
        const std::string arch =
            get_gpu_arch({"EXASIM_HIP_ARCH", "HIPARCH", "EXASIM_GPU_ARCH"}, "HIP");
    
        return arch.empty() ? std::string{} : "--offload-arch=" + arch + " ";
    };

    int status = 0;
    int final_status = 0;
    int attempted_builds = 0;

    // -------------------- SERIAL --------------------
    if (fs::exists(kokkos_serial_path) && fs::is_directory(kokkos_serial_path)) {
        const std::string inc_dir = make_path(kokkos_serial_path, "include");
        const std::string src     = text2code_model_source();
        const std::string out     = out_name("libt2cmodelserial");

        std::string lib_dir, lib_core, lib_cont;

        if (!require_kokkos_static_libs(kokkos_serial_path,
                                        "serial",
                                        lib_dir,
                                        lib_core,
                                        lib_cont)) {
            final_status = 1;
        } else {
            ++attempted_builds;

            std::cout << "Compiling libt2cmodelserial\n";

            std::stringstream cmd;

            if (tc.kind == CompilerKind::MSVC) {
                cmd << tc.cxx << " " << shlib_flags
                    << inc(spec.modelpath) << " " << inc(inc_dir) << " "
                    << exasim_include_flags()
                    << quote(src) << " "
                    << "/Fe:" << quote(out) << " "
                    << "/link " << quote(lib_core) << " " << quote(lib_cont);
            } else {
                cmd << tc.cxx << " " << shlib_flags
                    << inc(spec.modelpath) << " " << inc(inc_dir) << " "
                    << exasim_include_flags()
                    << quote(src) << " "
                    << quote(lib_core) << " " << quote(lib_cont) << " "
                    << install_name_flag(out)
                    << "-o " << quote(out);
            }

            std::cout << cmd.str() << "\n";
            status = std::system(cmd.str().c_str());

            if (status != 0) {
                std::cout << "Compiling libt2cmodelserial failed!\n";
                final_status = 1;
            } else {
                std::cout << "libt2cmodelserial built: " << out << "\n";
            }
        }
    }

    // -------------------- CUDA --------------------
    if (fs::exists(kokkos_cuda_path) && fs::is_directory(kokkos_cuda_path)) {
        std::string cuda_cxx = tc.cxx;

        std::string nvccw = make_path(kokkos_cuda_path, "bin/nvcc_wrapper");
        if (fs::exists(nvccw)) {
            cuda_cxx = nvccw;
        }

        const std::string inc_dir = make_path(kokkos_cuda_path, "include");
        const std::string src     = text2code_model_source();
        const std::string out     = out_name("libt2cmodelcuda");

        std::string lib_dir, lib_core, lib_cont;

        if (!require_kokkos_static_libs(kokkos_cuda_path,
                                        "cuda",
                                        lib_dir,
                                        lib_core,
                                        lib_cont)) {
            final_status = 1;
        } else {
            ++attempted_builds;

            std::cout << "Compiling libt2cmodelcuda\n";

            std::stringstream cmd;

            cmd << quote(cuda_cxx)
                << " -shared -std=c++17 -Xcompiler -fPIC "
                << "--expt-extended-lambda --expt-relaxed-constexpr "
                << cuda_arch_flag()
                << inc(spec.modelpath) << " " << inc(inc_dir) << " "
                << exasim_include_flags()
                << quote(src) << " "
                << install_name_flag(out)
                << "-o " << quote(out) << " "
                << "-L" << quote(lib_dir) << " "
                << "-lkokkoscore -lkokkoscontainers";

            std::cout << cmd.str() << "\n";
            status = std::system(cmd.str().c_str());

            if (status != 0) {
                std::cout << "Compiling libt2cmodelcuda failed!\n";
                final_status = 1;
            } else {
                std::cout << "libt2cmodelcuda built: " << out << "\n";
            }
        }
    }

    // -------------------- HIP --------------------
    if (fs::exists(kokkos_hip_path) && fs::is_directory(kokkos_hip_path)) {
        std::string hip_cxx = tc.cxx;

        std::string hipccw = make_path(kokkos_hip_path, "bin/hipcc_wrapper");
        if (fs::exists(hipccw)) {
            hip_cxx = hipccw;
        } else {
            std::string hipcc = "hipcc";
            if (run_silently(quote(hipcc) + " --version") == 0) {
                hip_cxx = hipcc;
            }
        }

        const std::string inc_dir = make_path(kokkos_hip_path, "include");
        const std::string src     = text2code_model_source();
        const std::string out     = out_name("libt2cmodelhip");

        std::string lib_dir, lib_core, lib_cont;

        if (!require_kokkos_static_libs(kokkos_hip_path,
                                        "hip",
                                        lib_dir,
                                        lib_core,
                                        lib_cont)) {
            final_status = 1;
        } else {
            ++attempted_builds;

            std::cout << "Compiling libt2cmodelhip\n";

            std::stringstream cmd;

            cmd << quote(hip_cxx)
                << " -shared -std=c++17 -fPIC " << hip_arch_flag()
                << inc(spec.modelpath) << " " << inc(inc_dir) << " "
                << exasim_include_flags()
                << quote(src) << " "
                << install_name_flag(out)
                << "-o " << quote(out) << " "
                << "-L" << quote(lib_dir) << " "
                << "-lkokkoscore -lkokkoscontainers";

            std::cout << cmd.str() << "\n";
            status = std::system(cmd.str().c_str());

            if (status != 0) {
                std::cout << "Compiling libt2cmodelhip failed!\n";
                final_status = 1;
            } else {
                std::cout << "libt2cmodelhip built: " << out << "\n";
            }
        }
    }

    if (attempted_builds == 0) {
        std::cout << "No Kokkos backend libraries were built. "
                  << "No valid Kokkos build directory was found.\n";
        final_status = 1;
    }

    return final_status;
}
