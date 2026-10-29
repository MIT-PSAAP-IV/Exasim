/**
 * @file ExasimSolverSetup.hpp
 * @brief High-level helper API for selecting and attaching Exasim model providers.
 *
 * This header centralizes the provider-selection logic that was previously
 * duplicated in executable entry points. It selects the provider ABI from the
 * active compile-time mode and configures every model definition owned by an
 * ExasimSolver instance.
 *
 * The low-level ABI getter functions declared here are defined by the provider
 * modules for FrontendGenerated, Text2codeGenerated, UserDefined, and BuiltIn
 * models. This header intentionally declares them but does not include the
 * provider implementation translation units, so it remains install-safe.
 */

#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ExasimSolver.hpp"

// Low-level provider ABI entry points supplied by the provider modules.
const ExasimDriverABI& getSharedLibraryExasimDriverABI();
const ExasimDriverABI& getBuiltInLibraryExasimDriverABI();
const ExasimDriverABI& getBuiltInExasimDriverABI();
const ExasimDriverABI& getKokkosKernelExasimDriverABI();
const ExasimDriverABI& getText2codeGeneratedExasimDriverABI();
const ExasimDriverABI& getUserDefinedExasimDriverABI();
const ExasimDriverABI& getFrontendGeneratedExasimDriverABI();

inline const ExasimDriverABI& SelectExasimDriverABI()
{
#if defined(HAVE_SHAREDLIBRARY)
    return getSharedLibraryExasimDriverABI();  
#elif defined(HAVE_BUILTINLIBRARY)
    return getBuiltInLibraryExasimDriverABI();
#elif defined(HAVE_BUILTINMODEL)
    return getBuiltInExasimDriverABI();
#elif defined(HAVE_KOKKOSKERNEL)
    return getKokkosKernelExasimDriverABI();
#elif defined(HAVE_TEXT2CODE)
    return getText2codeGeneratedExasimDriverABI();
#elif defined(HAVE_UDERDEFINED) || defined(HAVE_USERDEFINED)
    return getUserDefinedExasimDriverABI();
#else
    return getFrontendGeneratedExasimDriverABI();
#endif
}

inline const char* SelectExasimDriverProviderName()
{
#if defined(HAVE_SHAREDLIBRARY)
    return "SharedLibrary";
#elif defined(HAVE_BUILTINLIBRARY)
    return "BuiltInLibrary";
#elif defined(HAVE_BUILTINMODEL)
    return "BuiltInModel";
#elif defined(HAVE_KOKKOSKERNEL)
    return "KokkosKernel";
#elif defined(HAVE_TEXT2CODE)
    return "Text2codeGenerated";
#elif defined(HAVE_UDERDEFINED) || defined(HAVE_USERDEFINED)
    return "UserDefined";
#else
    return "FrontendGenerated";
#endif
}

inline void PrintModelProvider(const int modelnumber, const int builtinmodelID)
{
    int rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(EXASIM_COMM_WORLD, &rank);
#endif

    if (rank == 0) {
        std::cout << "Model " << modelnumber
                  << ": provider = "
                  << SelectExasimDriverProviderName()
                  << ", builtinmodelID = " << builtinmodelID << std::endl;
    }
}

inline int ConfigureModelDefinitions(ExasimSolver& solver, const ExasimDriverABI& abi)
{
    const int numModelDefinitions = solver.NumModelDefinitions();

    for (int i = 0; i < numModelDefinitions; i++) {
        const int builtinmodelID = solver.BuiltinModelID(i);
        const int err = solver.SetModelDefinition(i, builtinmodelID, abi);
        if (err)
            return err;
        PrintModelProvider(i, builtinmodelID);
    }

    return 0;
}

inline int ConfigureModelDefinitions(ExasimSolver& solver,
                                     const ExasimDriverABI& abi,
                                     const std::vector<int>& vecBuiltinModelID)
{
    const int numModelDefinitions = solver.NumModelDefinitions();

    if (vecBuiltinModelID.empty())
        return 1;

    for (int i = 0; i < numModelDefinitions; i++) {
        int builtinmodelID = 0;
        if (static_cast<int>(vecBuiltinModelID.size()) == numModelDefinitions) {
            builtinmodelID = vecBuiltinModelID[i];
        }
        else if (vecBuiltinModelID.size() == 1) {
            // Legacy split-domain frontend apps pass one external provider ID
            // while the runtime creates one model definition per domain.
            builtinmodelID = vecBuiltinModelID[0];
        }
        else {
            return 1;
        }

        if (builtinmodelID <= 0)
            builtinmodelID = solver.BuiltinModelID(i);

        const int err = solver.SetModelDefinition(i, builtinmodelID, abi);
        if (err)
            return err;
        PrintModelProvider(i, builtinmodelID);
    }

    return 0;
}

inline int InitializeExasimSolver(ExasimSolver& solver, int argc, char** argv, MPI_Comm comm)
{
    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err)
        return err;

    const auto& abi = SelectExasimDriverABI();

    err = solver.ParseInputs(argc, argv, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err)
        return err;

    return 0;
}

inline int InitializeExasimPostprocessor(ExasimSolver& solver, int argc, char** argv, MPI_Comm comm)
{
    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err)
        return err;

    const auto& abi = SelectExasimDriverABI();

    err = solver.ParsePostprocessInputs(argc, argv);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err)
        return err;

    return 0;
}

inline int InitializeExasimPostprocessor(ExasimSolver& solver, int argc, char** argv,
                                         MPI_Comm comm, const std::vector<int>& vecBuiltinModelID)
{
    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err)
        return err;

    const auto& abi = SelectExasimDriverABI();

    err = solver.ParsePostprocessInputs(argc, argv);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver, abi, vecBuiltinModelID);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err)
        return err;

    return 0;
}

inline int InitializeExasimSolver(ExasimSolver& solver, int argc, char** argv,
                                  MPI_Comm comm, const std::vector<int>& vecBuiltinModelID)
{
    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err)
        return err;

    const auto& abi = SelectExasimDriverABI();

    err = solver.ParseInputs(argc, argv, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver, abi, vecBuiltinModelID);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err)
        return err;

    return 0;
}

inline int RunExasimSolver(ExasimSolver& solver, int argc, char** argv, MPI_Comm comm)
{
    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err)
        return err;

    const auto& abi = SelectExasimDriverABI();

    if (argc > 1 && std::string(argv[1]) == "--hdgldg") {
        err = solver.RunHdgLdg(argc, argv, abi);
        solver.Finalize();
        return err;
    }

    err = solver.ParseInputs(argc, argv, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err)
        return err;

    err = solver.Solve();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}

inline int RunExasimPostprocess(ExasimSolver& solver, int argc, char** argv, MPI_Comm comm)
{
    int err = InitializeExasimPostprocessor(solver, argc, argv, comm);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.Postprocess();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}

inline int RunExasimPostprocess(ExasimSolver& solver, int argc, char** argv,
                                MPI_Comm comm, const std::vector<int>& vecBuiltinModelID)
{
    int err = InitializeExasimPostprocessor(solver, argc, argv, comm, vecBuiltinModelID);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.Postprocess();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}

inline int RunExasimSolver(ExasimSolver& solver, int argc, char** argv,
                           MPI_Comm comm, const std::vector<int>& vecBuiltinModelID)
{
    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err)
        return err;

    const auto& abi = SelectExasimDriverABI();

    if (argc > 1 && std::string(argv[1]) == "--hdgldg") {
        err = solver.RunHdgLdg(argc, argv, abi);
        solver.Finalize();
        return err;
    }

    err = solver.ParseInputs(argc, argv, abi);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver, abi, vecBuiltinModelID);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err)
        return err;

    err = solver.Solve();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}

// The text2code executable used by RunText2CodeApp: EXASIM_TEXT2CODE_EXECUTABLE when it
// is defined and not empty; otherwise $EXASIM_PREFIX/bin/text2code, or <prefix>/bin/text2code
// of the Exasim installation the app was built against (EXASIM_INSTALL_PREFIX, set by
// find_package(Exasim)); otherwise text2code from PATH.
inline std::string Text2CodeExecutable()
{
    std::string text2code;
#ifdef EXASIM_TEXT2CODE_EXECUTABLE
    text2code = EXASIM_TEXT2CODE_EXECUTABLE;
#endif
    if (!text2code.empty())
        return text2code;
    if (const char* prefix = std::getenv("EXASIM_PREFIX"); prefix && *prefix)
        return std::string(prefix) + "/bin/text2code";
#ifdef EXASIM_INSTALL_PREFIX
    return std::string(EXASIM_INSTALL_PREFIX) + "/bin/text2code";
#else
    return "text2code";
#endif
}

/**
 * Entry point of a (coupled) Text2Code application:
 *
 *   mpirun -np <ranks1> exasimapp pdeapp0.txt <ranks1>
 *       Single-model solve of pdeapp0.txt (data in datain/, dataout/).
 *   mpirun -np <ranks1+ranks2> exasimapp pdeapp1.txt <ranks1> pdeapp2.txt <ranks2>
 *       Two-domain coupled solve: ranks [0, ranks1) solve model 1, the rest model 2.
 *   mpirun -np <total> exasimapp --interactions interaction.txt
 *       pdeapp1.txt <ranks1> ... pdeappN.txt <ranksN>
 *       Generalized-interaction input generation and N-domain launch.
 *
 * Rank 0 generates the datain folders with text2code (see Text2CodeExecutable).
 * MPI is initialized and finalized here unless the caller has already initialized it;
 * a caller that owns MPI can post-process the output (e.g. GatherSolution) after the
 * return, on every rank, before its own MPI_Finalize.
 */
inline int RunText2CodeApp(int argc, char** argv)
{
    namespace fs = std::filesystem;
#ifdef HAVE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
        MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto barrier = [] { MPI_Barrier(MPI_COMM_WORLD); };
    auto abort = [rank](const std::string& msg) {
        if (rank == 0) std::cerr << "exasimapp: " << msg << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    };
#else
    const int rank = 0, size = 1;
    auto barrier = [] {};
    auto abort = [](const std::string& msg) { std::cerr << "exasimapp: " << msg << "\n"; std::exit(1); };
#endif
    auto hasExactGraphKey = [](const fs::path& file) {
        std::ifstream in(file);
        std::string source;
        std::string line;
        while (std::getline(in, line)) {
            const std::size_t comment = line.find("//");
            if (comment != std::string::npos) line.erase(comment);
            source += line;
            source.push_back('\n');
        }
        std::size_t begin = 0;
        while (begin <= source.size()) {
            const std::size_t semicolon = source.find(';', begin);
            const std::size_t end = semicolon == std::string::npos ? source.size() : semicolon;
            const std::string statement = source.substr(begin, end - begin);
            const std::size_t equal = statement.find('=');
            if (equal != std::string::npos) {
                std::string key = statement.substr(0, equal);
                const auto first = key.find_first_not_of(" \t\r\n");
                const auto last = key.find_last_not_of(" \t\r\n");
                if (first != std::string::npos) {
                    key = key.substr(first, last - first + 1);
                    if (key == "interactionfile" || key == "numdomains") return true;
                }
            }
            if (semicolon == std::string::npos) break;
            begin = semicolon + 1;
        }
        return false;
    };
    auto quoteValue = [](const std::string& value) {
        std::string out = "\"";
        for (char c : value) {
            if (c == '\\' || c == '"') out.push_back('\\');
            out.push_back(c);
        }
        out.push_back('"');
        return out;
    };

    const bool graphForm = argc >= 7 && std::string(argv[1]) == "--interactions";
    if (!graphForm && argc != 3 && argc != 5)
        abort(std::string("usage: ") + argv[0] + " pdeapp0.txt ranks | pdeapp1.txt ranks1 pdeapp2.txt ranks2 | --interactions interaction.txt pdeapp1.txt ranks1 ... pdeappN.txt ranksN");

    if (graphForm) {
        if ((argc - 3) % 2 != 0)
            abort("--interactions requires pdeapp/rank pairs");
        const int domains = (argc - 3) / 2;
        if (domains < 2)
            abort("--interactions requires at least two domains");
        const fs::path interactionFile = fs::absolute(argv[2]);
        if (!fs::is_regular_file(interactionFile))
            abort("cannot read interaction file " + interactionFile.string());

        std::vector<fs::path> apps;
        std::string countsText;
        for (int i = 0; i < domains; ++i) {
            const fs::path app = fs::absolute(argv[3 + 2 * i]);
            if (!fs::is_regular_file(app)) abort("cannot read " + app.string());
            apps.push_back(app);
            if (i) countsText += ',';
            countsText += argv[4 + 2 * i];
        }
        exasim::DomainRankLayout layout;
        std::string diagnostic;
        if (!exasim::ParseDomainRankLayout(countsText, domains, size, layout, diagnostic))
            abort(diagnostic);

        std::vector<std::string> args{argv[0], std::to_string(domains)};
        const std::string text2code = Text2CodeExecutable();
        try {
            for (int i = 0; i < domains; ++i) {
                const fs::path dir = apps[static_cast<std::size_t>(i)].parent_path();
                if (rank == 0)
                    GenerateText2CodeDatain(
                        apps[static_cast<std::size_t>(i)].string(), layout.counts[static_cast<std::size_t>(i)],
                        i + 1, text2code,
                        {{"interactionfile", quoteValue(interactionFile.string())},
                         {"numdomains", std::to_string(domains)}});
                args.push_back((dir / ("datain" + std::to_string(i + 1))).string() + "/");
                args.push_back((dir / ("dataout" + std::to_string(i + 1)) / "out").string());
            }
        } catch (const std::exception& e) {
            abort(e.what());
        }
        args.push_back("--domain-ranks");
        args.push_back(countsText);
        barrier();

        std::vector<char*> cargs;
        for (auto& arg : args) cargs.push_back(arg.data());
        ExasimSolver solver;
#ifdef HAVE_MPI
        const int err = RunExasimSolver(solver, static_cast<int>(cargs.size()), cargs.data(), MPI_COMM_WORLD);
#else
        const int err = RunExasimSolver(solver, static_cast<int>(cargs.size()), cargs.data(), MPI_COMM_NULL);
#endif
        barrier();
#ifdef HAVE_MPI
        if (!initialized) MPI_Finalize();
#endif
        return err;
    }

    const fs::path app1 = fs::absolute(argv[1]), dir1 = app1.parent_path();
    if (hasExactGraphKey(app1) || (argc == 5 && hasExactGraphKey(fs::absolute(argv[3]))))
        abort("generalized interactions require RunText2CodeApp --interactions");
    const int ranks1 = std::stoi(argv[2]);
    const int ranks2 = argc == 5 ? std::stoi(argv[4]) : 0;
    if (ranks1 + ranks2 != size)
        abort("the rank counts must add up to the number of MPI ranks");

    std::vector<std::string> args{argv[0]};
    const std::string text2code = Text2CodeExecutable();
    try {
        if (argc == 3) {
            if (rank == 0)
                GenerateText2CodeDatain(app1.string(), ranks1, 0, text2code);
            args.insert(args.end(), {"1", (dir1 / "datain").string() + "/", (dir1 / "dataout" / "out").string()});
        } else {
            const fs::path app2 = fs::absolute(argv[3]), dir2 = app2.parent_path();
            if (rank == 0) {
                GenerateText2CodeDatain(app1.string(), ranks1, 1, text2code);
                GenerateText2CodeDatain(app2.string(), ranks2, 2, text2code);
            }
            args.insert(args.end(), {std::to_string(100 + ranks1),
                                     (dir1 / "datain1").string() + "/", (dir1 / "dataout1" / "out").string(),
                                     (dir2 / "datain2").string() + "/", (dir2 / "dataout2" / "out").string()});
        }
    } catch (const std::exception& e) {
        abort(e.what());
    }
    barrier();

    std::vector<char*> cargs;
    for (auto& arg : args)
        cargs.push_back(arg.data());
    int err = 0;
    {
        ExasimSolver solver;
#ifdef HAVE_MPI
        err = RunExasimSolver(solver, static_cast<int>(cargs.size()), cargs.data(), MPI_COMM_WORLD);
#else
        err = RunExasimSolver(solver, static_cast<int>(cargs.size()), cargs.data(), MPI_COMM_NULL);
#endif
    }
    barrier();   // every rank has written its output
#ifdef HAVE_MPI
    if (!initialized)
        MPI_Finalize();
#endif
    return err;
}
