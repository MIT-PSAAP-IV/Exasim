#include "include/ExasimSolver.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void accept(const char* text, int domains, int ranks,
            const std::vector<int>& counts, const std::vector<int>& offsets)
{
    exasim::DomainRankLayout layout;
    std::string diagnostic;
    require(exasim::ParseDomainRankLayout(text, domains, ranks, layout, diagnostic),
            diagnostic.c_str());
    require(layout.counts == counts, "unexpected domain counts");
    require(layout.offsets == offsets, "unexpected domain offsets");
    for (int rank = 0; rank < ranks; ++rank) {
        const int domain = layout.domainForRank(rank);
        require(domain >= 0 && domain < domains, "rank has no domain");
        require(rank >= offsets[domain] && rank < offsets[domain + 1],
                "rank mapped outside its domain interval");
    }
    require(layout.domainForRank(-1) == -1, "negative rank accepted");
    require(layout.domainForRank(ranks) == -1, "out-of-range rank accepted");
}

void reject(const char* text, int domains, int ranks)
{
    exasim::DomainRankLayout layout;
    std::string diagnostic;
    require(!exasim::ParseDomainRankLayout(text, domains, ranks, layout, diagnostic),
            "malformed layout accepted");
    require(!diagnostic.empty(), "rejection has no diagnostic");
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#ifdef PHASE2_SOLVER_PARSE_TEST
    if (argc >= 3 && (std::string(argv[1]) == "--parse" ||
                      std::string(argv[1]) == "--parse-twice")) {
        const bool twice = std::string(argv[1]) == "--parse-twice";
        const int solverArgc = argc - 1;
        char** solverArgv = argv + 1;
        ExasimSolver solver;
        int status = solver.InitializeEnvironment(solverArgc, solverArgv, MPI_COMM_WORLD);
        const ExasimDriverABI abi{};
        if (!status) status = solver.ParseInputs(solverArgc, solverArgv, abi);
        if (!status && twice) status = solver.ParseInputs(solverArgc, solverArgv, abi);
        int worldRank = 0, worldSize = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
        if (!status) {
            const auto& layout = solver.DomainRankLayout();
            const int domain = layout.domainForRank(worldRank);
            if (domain < 0 || solver.LocalDomain() != domain ||
                solver.NumModelDefinitions() != static_cast<int>(layout.counts.size()) ||
                layout.offsets.empty() || layout.offsets.back() != worldSize ||
                EXASIM_COMM_WORLD == MPI_COMM_NULL || EXASIM_COMM_LOCAL == MPI_COMM_NULL)
                status = 1;
            int localRank = -1, localSize = 0;
            if (!status) {
                MPI_Comm_rank(EXASIM_COMM_LOCAL, &localRank);
                MPI_Comm_size(EXASIM_COMM_LOCAL, &localSize);
                if (localRank != worldRank-layout.offsets[domain] ||
                    localSize != layout.counts[domain])
                    status = 1;
            }
        }
        // Every rank enters this probe even if a rank-local layout assertion
        // failed, so the lifecycle test cannot create a mismatched collective.
        int worldProbe = worldRank + 1, worldProbeSum = 0;
        MPI_Allreduce(&worldProbe, &worldProbeSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (worldProbeSum != worldSize*(worldSize+1)/2) status = 1;
        int globalStatus = 0;
        MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (worldRank == 0 && !globalStatus)
            std::cout << "PASS: production ParseInputs"
                      << (twice ? " communicator lifecycle" : "") << '\n';
        solver.Finalize();
        int finalizeProbe = 1, finalizeProbeSum = 0;
        MPI_Allreduce(&finalizeProbe, &finalizeProbeSum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (finalizeProbeSum != worldSize) globalStatus = 1;
        MPI_Finalize();
        return globalStatus;
    }
#endif
    if (argc == 3 && std::string(argv[1]) == "--emit") {
        int rank = 0, size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int domains = 1;
        for (char c : std::string(argv[2])) if (c == ',') ++domains;
        exasim::DomainRankLayout layout;
        std::string diagnostic;
        if (!exasim::ParseDomainRankLayout(argv[2], domains, size, layout, diagnostic)) {
            if (rank == 0) std::cerr << diagnostic << '\n';
            MPI_Finalize();
            return 1;
        }
        const int domain = layout.domainForRank(rank);
        const int offset = layout.offsets[static_cast<std::size_t>(domain)];
        std::cout << "rank=" << rank << " domain=" << domain
                  << " localRank=" << rank-offset << " fileoffset=" << offset
                  << " filein=datain" << domain+1 << "/"
                  << " fileout=dataout" << domain+1 << "/out"
                  << " modelDefinition=" << domain << '\n';
        MPI_Finalize();
        return 0;
    }
#endif
    accept("1,1", 2, 2, {1, 1}, {0, 1, 2});
    accept("1,2,3,1", 4, 7, {1, 2, 3, 1}, {0, 1, 3, 6, 7});
    accept("3,1,2,1", 4, 7, {3, 1, 2, 1}, {0, 3, 4, 6, 7});

    reject("", 2, 2);
    reject("1", 2, 1);
    reject("1,1", 1, 2);
    reject("0,2", 2, 2);
    reject("-1,3", 2, 2);
    reject("+1,1", 2, 2);
    reject("1, 1", 2, 2);
    reject("1,,1", 3, 2);
    reject("1,1,", 3, 2);
    reject("1,a", 2, 2);
    reject("1,1,1", 2, 3);
    reject("1,2", 2, 4);
    reject("2147483648,1", 2, 2);
    reject("2147483647,1", 2, 2);

    std::cout << "PASS: checked generalized domain-rank layouts\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
